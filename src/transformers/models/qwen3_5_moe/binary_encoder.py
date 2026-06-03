from transformers import Qwen3_5MoeBinaryConfig
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.g_proj = nn.Linear(dim, dim, bias=False)

        self.out_proj = nn.Linear(dim, dim)
        self.feature_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
        self._init_weights()

    def _init_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.g_proj, self.out_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, L, D = x.shape
        H, HD = self.heads, self.head_dim
        q = self.q_proj(x).view(B, L, H, HD).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, HD).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, HD).transpose(1, 2)
        g = F.silu(self.g_proj(x))
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        k_sum = k.sum(dim=-2, keepdim=True)
        denom = torch.matmul(q, k_sum.transpose(-2, -1)) + 1e-6
        kv = torch.matmul(k.transpose(-2, -1), v)
        out = torch.matmul(q, kv)
        out = out / denom
        out = out.transpose(1, 2).contiguous()

        out = self.feature_norm(out).view(B, L, D)

        return self.out_proj(out * g)


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
        self.queries = nn.Parameter(torch.randn(config.num_queries, config.encoder_dim))
        self.attn = nn.MultiheadAttention(config.encoder_dim, config.num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(config.encoder_dim)
        self.ln2 = nn.LayerNorm(config.encoder_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.encoder_dim, config.encoder_dim * 4),
            nn.GELU(),
            nn.Linear(config.encoder_dim * 4, config.encoder_dim)
        )

    def forward(self, x):
        B = x.shape[0]
        queries = self.queries.unsqueeze(0).repeat(B, 1, 1)
        attn_out, _ = self.attn(query=queries, key=x, value=x)
        x = queries + self.ln1(attn_out)
        x = x + self.mlp(self.ln2(x))
        return x


class BytePositionalConv(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(*args, **kwargs)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        out_t = self.conv(x_t)
        return out_t.transpose(1, 2).contiguous()


class BinaryByteModalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.byte_embedding = nn.Embedding(257, config.encoder_dim)
        self.encoder_layers = nn.ModuleList([
            LinearAttentionBlock(dim=config.encoder_dim, heads=config.num_heads) for _ in
            range(config.attn_nums)
        ])

        self.pos_conv = BytePositionalConv(config.encoder_dim, config.encoder_dim, kernel_size=5, padding=2,
                                           groups=config.encoder_dim)

    def forward(self, byte_ids):
        B, L = byte_ids.shape
        x = self.byte_embedding(byte_ids)

        x = self.pos_conv(x) + x

        for layer in self.encoder_layers:
            x = layer(x)
        return x


class BinaryEncoderForLLm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = BinaryByteModalEncoder(config)
        self.resampler = PerceiverResampler(config)

        self.projector = nn.Sequential(
            nn.Linear(config.encoder_dim, config.encoder_dim * 2),
            nn.GELU(),
            nn.Linear(config.encoder_dim * 2, config.llm_hidden_dim)
        )

    def forward(self, byte_ids):
        B, N, L = byte_ids.shape
        byte_ids_flat = byte_ids.view(B * N, L)
        x = self.model(byte_ids_flat)
        compressed_matrix = self.resampler(x)

        llm_inputs_flat = self.projector(compressed_matrix)
        llm_dim = llm_inputs_flat.shape[-1]

        num_queries = compressed_matrix.shape[1]
        llm_inputs = llm_inputs_flat.view(B, N, num_queries, llm_dim)
        return llm_inputs


class BinaryMLMPretrainWrapper(nn.Module):
    def __init__(self, encoder: BinaryByteModalEncoder, config):
        super().__init__()
        self.encoder = encoder
        self.encoder_dim = config.encoder_dim
        self.projector = nn.Sequential(
            nn.Linear(config.encoder_dim, 257)
        )

    def forward(self, byte_ids, labels=None):
        B, L = byte_ids.shape
        x = self.encoder(byte_ids)
        logits = self.projector(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 257), labels.view(-1))

        if loss is not None:
            return loss, logits
        return logits
