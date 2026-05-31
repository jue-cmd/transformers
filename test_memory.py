import os
import traceback

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from transformers import Qwen3_5MoeBinaryConfig
from transformers.models.qwen3_5_moe.binary_encoder import BinaryByteModalEncoder, BinaryMLMPretrainWrapper


def print_mem(label=""):
    """打印当前 GPU 占用的实际显存（单位：GB）"""
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"[{label}] 当前显存: {allocated:.2f} GB | 历史峰值: {max_allocated:.2f} GB")

@torch.no_grad()
def main():
    print("🚀 开始显存逐行追踪分析...")
    print_mem("初始状态")

    # 1. 初始化配置
    config = Qwen3_5MoeBinaryConfig()
    config.downsample_factor = 8
    config.attn_nums = 6
    config.encoder_dim = 1024

    # 2. 载入模型到 GPU
    encoder = BinaryByteModalEncoder(config)
    model = BinaryMLMPretrainWrapper(encoder, config).cuda()
    model.train()  # 开启训练模式，激活 Dropout 等状态
    print_mem("模型载入 GPU 后")

    # 3. 构造一个等效的虚拟 Batch（Batch=4, Length=512000）
    # 这样可以绕过漫长的 Dataset 加载，直接直击前向传播痛点
    B, N = 1, 1024*256
    print(f"\n📊 正在模拟构造 Batch... 形状: ({B}, {N})")
    byte_ids = torch.randint(0, 256, (B, N), dtype=torch.long).cuda()

    # 2. labels 范围在 [0, 255]，同时随机填入合法的忽略索引 -100
    # 先生成标准的 [0, 255] 标签
    print_mem("输入张量装载进 GPU 后")

    # 4. 显存打点追踪：进入前向传播
    print("\n🔥 开始前向传播逐层拆解...")
    try:
        # 使用自动混合精度（bf16/fp16），与你训练参数对齐
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"使用数据类型: {dtype}")

        with torch.amp.autocast(device_type="cuda", dtype=dtype):

            # --- 追踪 4.1: 字节嵌入层 (Byte Embedding + Linear Projection) ---
            print("\n[Step 1/4] 执行外部特征处理与 Embedding...")
            # 模拟等效的内部前向首阶段，观察特征降采样和映射
            hidden_states = model.encoder.embed_tokens(byte_ids) if hasattr(model.encoder, 'embed_tokens') else None
            print_mem("Embedding 层执行完毕")

            # --- 追踪 4.2: 核心注意力机制与 Transformer 层 (512K 的致命点) ---
            print("\n[Step 2/4] 进入核心 6 层 Attention 块层...")
            # 我们直接运行内部的组件，或者直接运行全流程
            # 如果在这一步直接闪退，说明是序列太长导致 Attention 里的激活值开辟直接冲爆了显存
            outputs = model(byte_ids=byte_ids)
            print_mem("模型 Forward 整体完成")
    except RuntimeError as e:
        print(f"\n❌ 捕获到标准的 PyTorch OOM 异常 喵：\n{e}")
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ 发生其他致命错误/闪退：\n{e}")


if __name__ == "__main__":
    main()