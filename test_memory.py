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
    model.train()  # 开启训练模式
    print_mem("模型载入 GPU 后")

    # 3. 构造一个等效的虚拟 Batch（Batch=1, Length=256K）
    B, N = 1, 1024*8
    print(f"\n📊 正在模拟构造 Batch... 形状: ({B}, {N})")
    byte_ids = torch.randint(0, 256, (B, N), dtype=torch.long).cuda()
    print_mem("输入张量装载进 GPU 后")

    # 4. 显存打点追踪：进入前向传播并开启 Profiler
    print("\n🔥 开始前向传播逐层拆解（伴随 Torch Profiler 性能采样）...")

    # 配置文件导出路径
    log_dir = "./log_profiler"
    os.makedirs(log_dir, exist_ok=True)

    try:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"使用数据类型: {dtype}")

        # 配置 Torch Profiler
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                # schedule 可以帮我们在连续运行多轮时进行热身和采样，这里单次运行直接在第1轮记录
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
                # record_shapes=True 极其重要，能帮你看到是多大尺度的 Tensor 挤爆了显存
                record_shapes=True,
                # profile_memory=True 开启显存追踪分配逻辑
                profile_memory=True,
                # with_stack=True 会关联到具体的 Python 代码行
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir)
        ) as prof:

            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                # --- 追踪 4.1: 字节嵌入层 ---
                print("\n[Step 1/4] 执行外部特征处理与 Embedding...")
                hidden_states = model.encoder.embed_tokens(byte_ids) if hasattr(model.encoder, 'embed_tokens') else None
                print_mem("Embedding 层执行完毕")

                # --- 追踪 4.2: 核心注意力机制与 Transformer 层 (512K/256K 的致命点) ---
                print("\n[Step 2/4] 进入核心 6 层 Attention 块层...")

                # 执行模型前向传播
                outputs = model(byte_ids=byte_ids)

                print_mem("模型 Forward 整体完成")

                # 显式步进 Profiler 计数器（触发保存）
                prof.step()

        print(f"\n🎉 Profiler 性能分析数据导出成功！")
        print(f"📂 导出目录: {os.path.abspath(log_dir)}")
    except RuntimeError as e:
        print(f"\n❌ 捕获到标准的 PyTorch OOM 异常：\n{e}")
        traceback.print_exc()
        print("\n⚠️ 警告: 如果在 `prof.step()` 之前发生了 OOM 闪退，Profiler 可能无法正常保存最后的 trace 文件。")
    except Exception as e:
        print(f"\n❌ 发生其他致命错误/闪退：\n{e}")


if __name__ == "__main__":
    main()