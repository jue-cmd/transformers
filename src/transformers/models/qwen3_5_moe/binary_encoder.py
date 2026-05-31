from transformers import Qwen3_5MoeBinaryConfig

import torch
from torch import nn
import torch.nn.functional as F

import torch
import torch.nn as nn


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=8, chunk_size=512):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        self.chunk_size = chunk_size

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
        B, N, D = x.shape
        H, HD = self.heads, self.head_dim
        C = self.chunk_size

        q = self.q_proj(x).view(B, N, H, HD).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, HD).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, HD).transpose(1, 2)
        g = torch.sigmoid(self.g_proj(x))

        q = torch.sigmoid(q)
        k = torch.sigmoid(k)
        kv_state = torch.zeros(B, H, HD, HD, dtype=x.dtype, device=x.device)
        output_chunks = []

        for i in range(0, N, C):
            q_chunk = q[:, :, i:i + C, :]
            k_chunk = k[:, :, i:i + C, :]
            v_chunk = v[:, :, i:i + C, :]
            chunk_kv = torch.matmul(k_chunk.transpose(-2, -1), v_chunk)
            if kv_state.dtype != chunk_kv.dtype:
                kv_state = kv_state.to(chunk_kv.dtype)
            out_chunk = torch.matmul(q_chunk, kv_state + chunk_kv)
            output_chunks.append(out_chunk)
            kv_state = kv_state + chunk_kv

        out = torch.cat(output_chunks, dim=2)
        out = self.feature_norm(out)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out.mul_(g)
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


class BinaryByteModalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.downsample_factor = config.downsample_factor

        self.byte_embedding = nn.Embedding(257, config.encoder_dim)

        self.folding_proj = nn.Linear(config.encoder_dim * config.downsample_factor, config.encoder_dim)

        self.encoder_layers = nn.ModuleList([
            LinearAttentionBlock(dim=config.encoder_dim, heads=8) for _ in range(config.attn_nums)
        ])

        self.pos_conv = nn.Conv1d(config.encoder_dim, config.encoder_dim, kernel_size=5, padding=2,
                                  groups=config.encoder_dim)

    def forward(self, byte_ids):
        B, L = byte_ids.shape
        pad_len = (self.downsample_factor - (L % self.downsample_factor)) % self.downsample_factor
        if pad_len > 0:
            byte_ids = F.pad(byte_ids, (0, pad_len), value=0)
            L = byte_ids.shape[1]
        x = self.byte_embedding(byte_ids)

        # 位置编码
        x = x.transpose(1, 2)
        x = self.pos_conv(x) + x
        x = x.transpose(1, 2).contiguous()

        E = x.shape[-1]
        x = x.view(B, L // self.downsample_factor, self.downsample_factor * E)
        x = self.folding_proj(x)
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
    """
    掩码/全文语言模型预训练包装器
    专门针对 PE/ELF 等具备全局关联性的二进制结构设计。
    支持对序列中的任意部分（包含尾部特定字符）进行多标签分类预测（0-255）。
    """

    def __init__(self, encoder: BinaryByteModalEncoder, config):
        super().__init__()
        self.encoder = encoder
        self.downsample_factor = config.downsample_factor

        self.mlm_head = nn.Sequential(
            nn.Linear(config.encoder_dim, config.encoder_dim * 2),
            nn.GELU(),
            nn.LayerNorm(config.encoder_dim * 2),
            nn.Linear(config.encoder_dim * 2, config.downsample_factor * 256)
        )

    def forward(self, byte_ids, labels=None):
        B, L = byte_ids.shape
        x = self.encoder(byte_ids)
        L_compressed = x.shape[1]
        logits = self.mlm_head(x)
        logits = logits.view(B, L_compressed * self.downsample_factor, 256)
        logits = logits[:, :L, :]
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            print(logits.shape)
            print(labels.shape)
            loss = loss_fct(logits.view(-1, 256), labels.view(-1))
        if loss is not None:
            return loss, logits
        return logits
