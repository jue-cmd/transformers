import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Qwen3_5MoeBinaryConfig


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        H, HD = self.heads, self.head_dim

        # [B, N, D] -> [B, H, N, HD]
        q = self.q_proj(x).view(B, N, H, HD).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, HD).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, HD).transpose(1, 2)

        # Kernel feature map: ensure positivity for linear attention
        q = F.elu(q) + 1.0  # [B, H, N, HD]
        k = F.elu(k) + 1.0  # [B, H, N, HD]

        # Linear attention: (Q(K^T V)) / (Q sum(K))
        kv = torch.matmul(k.transpose(-2, -1), v)  # [B, H, HD, HD]
        k_sum = k.sum(dim=-2, keepdim=True)  # [B, H, 1, HD]

        num = torch.matmul(q, kv)  # [B, H, N, HD]
        denom = torch.matmul(q, k_sum.transpose(-2, -1))  # [B, H, N, 1]

        out = num / (denom + 1e-6)  # [B, H, N, HD]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class LinearAttentionBlock(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = LinearAttention(dim, heads=heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class PerceiverResampler(nn.Module):
    def __init__(self, config: Qwen3_5MoeBinaryConfig):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(config.num_queries, config.embed_dim))
        self.attn = nn.MultiheadAttention(config.embed_dim, config.num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 4),
            nn.GELU(),
            nn.Linear(config.embed_dim * 4, config.embed_dim)
        )

    def forward(self, x):
        B = x.shape[0]

        queries = self.queries.unsqueeze(0).repeat(B, 1, 1)

        attn_out, _ = self.attn(query=queries, key=x, value=x)
        x = queries + self.ln1(attn_out)
        x = x + self.mlp(self.ln2(x))
        return x


class BinaryByteModalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.downsample_factor = config.downsample_factor

        self.byte_embedding = nn.Embedding(256, config.encoder_dim)

        self.folding_proj = nn.Linear(config.encoder_dim * config.downsample_factor, config.encoder_dim)

        self.encoder_layers = nn.ModuleList([
            LinearAttentionBlock(dim=config.encoder_dim, heads=8) for _ in range(2)
        ])

        self.pos_conv = nn.Conv1d(config.encoder_dim, config.encoder_dim, kernel_size=5, padding=2,
                                  groups=config.encoder_dim)

        self.resampler = PerceiverResampler(config)

        self.projector = nn.Sequential(
            nn.Linear(config.encoder_dim, config.encoder_dim * 2),
            nn.GELU(),
            nn.Linear(config.encoder_dim * 2, config.llm_hidden_dim)
        )

    def forward(self, byte_ids):
        B, N, L = byte_ids.shape
        byte_ids_flat = byte_ids.view(B * N, L)
        pad_len = (self.downsample_factor - (L % self.downsample_factor)) % self.downsample_factor
        if pad_len > 0:
            byte_ids_flat = F.pad(byte_ids_flat, (0, pad_len), value=0)
            L = byte_ids_flat.shape[1]
        x = self.byte_embedding(byte_ids_flat)

        # 位置编码
        x = x.transpose(1, 2)
        x = self.pos_conv(x) + x
        x = x.transpose(1, 2).contiguous()

        E = x.shape[-1]
        x = x.view(B * N, L // self.downsample_factor, self.downsample_factor * E)
        x = self.folding_proj(x)
        for layer in self.encoder_layers:
            x = layer(x)
        compressed_matrix = self.resampler(x)
        num_queries = compressed_matrix.shape[1]
        llm_inputs_flat = self.projector(compressed_matrix)
        LLM_Dim = llm_inputs_flat.shape[-1]
        llm_inputs = llm_inputs_flat.view(B, N, num_queries, LLM_Dim)
        print(llm_inputs)
        return llm_inputs


if __name__ == "__main__":
    mock_binary_file = torch.randint(0, 256, (2, 1048576))
    encoder = BinaryByteModalEncoder(num_queries=256, encoder_dim=1024, downsample_factor=16, llm_hidden_dim=4096)

    with torch.no_grad():
        hidden_states = encoder(mock_binary_file)

    print(hidden_states.shape)
