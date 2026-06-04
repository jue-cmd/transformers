import os

import wandb

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import tqdm
import os
from torch.utils.data import IterableDataset, DataLoader
from transformers import TrainingArguments, Trainer, Qwen3_5MoeBinaryConfig
from transformers.models.qwen3_5_moe.binary_encoder import BinaryByteModalEncoder, BinaryMLMPretrainWrapper, \
    LinearAttentionBlock
from torch.utils.data import Dataset
from torch.utils.checkpoint import checkpoint
from transformers import TrainerCallback


def make_block_checkpointed(block_module):
    orig_forward = block_module.forward

    def checkpointed_forward(*args, **kwargs):
        return checkpoint(orig_forward, *args, **kwargs, use_reentrant=False)

    block_module.forward = checkpointed_forward
    return block_module


class PyTorchProfilerCallback(TrainerCallback):
    def __init__(self, output_dir="./log/profiler"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.prof = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=1, repeat=1),
            profile_memory=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.output_dir)
        )
        self.prof.start()
        print(f"\n[Profiler] 性能分析器已启动！日志将保存至: {self.output_dir} 喵～")

    def on_step_end(self, args, state, control, **kwargs):
        if self.prof:
            self.prof.step()
        if state.global_step == 5:
            if self.prof:
                self.prof.stop()
                self.prof = None
                print("\n[Profiler] 性能分析完成并已成功导出！正在安全关闭 喵～")


class BinaryMLMDataCollator:
    def __init__(self, mask_prob=0.15, mask_token_id=256):
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id  # 占用 256 作为 MASK

    def __call__(self, examples):
        batch_byte_ids = [torch.tensor(e['byte_ids'], dtype=torch.long) for e in examples]
        byte_ids = torch.stack(batch_byte_ids, dim=0)
        labels = byte_ids.clone()
        masked_indices = torch.bernoulli(torch.full(byte_ids.shape, self.mask_prob)).bool()
        labels[~masked_indices] = -100  # 没被选中的不计算 loss
        indices_to_mask = torch.bernoulli(torch.full(byte_ids.shape, 0.8)).bool() & masked_indices
        byte_ids[indices_to_mask] = self.mask_token_id
        remaining_indices = masked_indices & ~indices_to_mask
        indices_random = torch.bernoulli(torch.full(byte_ids.shape, 0.5)).bool() & remaining_indices

        random_words = torch.randint(0, 256, byte_ids.shape, dtype=torch.long)
        byte_ids[indices_random] = random_words[indices_random]
        return {
            "byte_ids": byte_ids,
            "labels": labels
        }


def binary_chunks_generator(file_dir, chunk_size=1024):
    for root, _, files in os.walk(file_dir):
        for file in tqdm.tqdm(files, desc="Processing files"):
            if file.startswith('.'):
                continue

            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'rb') as f:
                    print(file_path)
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        if len(chunk) == chunk_size:
                            yield {"byte_ids": list(chunk)}
            except Exception as e:
                print(f"\n读取文件失败 {file_path}: {e}")


class BinaryMLMDataCollator:
    def __init__(self, mask_prob=0.15, mask_token_id=256):
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id

    def __call__(self, examples):
        batch_byte_ids = [torch.tensor(e['byte_ids'], dtype=torch.uint8) for e in examples]
        byte_ids = torch.stack(batch_byte_ids, dim=0).long()

        labels = byte_ids.clone()

        probability_matrix = torch.full(byte_ids.shape, self.mask_prob)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = -100

        byte_ids[masked_indices] = self.mask_token_id

        return {
            "byte_ids": byte_ids,
            "labels": labels
        }


class LabelDistributionCallback(TrainerCallback):
    def __init__(self, log_steps=1):
        self.log_steps = log_steps

    @torch.no_grad()
    def on_substep_end(self, args, state, control, model=None, **kwargs):
        # 仅在指定的步数进行打印，防止刷屏
        if state.global_step % self.log_steps != 0:
            return
        inputs_dict = kwargs.get("inputs", None)
        print(inputs_dict)
        if inputs_dict is None or "labels" not in inputs_dict:
            # 如果真的没拿到，不做静默退出，打印一行提示方便定位
            # print(" [Debug] 当前 Step 仍未捕获到 inputs 数据...")
            return
        labels = inputs_dict["labels"]
        # 2. 过滤掉不需要计算 loss 的 -100 填充符号
        active_labels = labels.view(-1)
        active_labels = active_labels[active_labels != -100]
        total_masked_count = active_labels.numel()

        print(f"\n📊 ====== [Step {state.global_step}] 被 MASK 的 Labels 真实分布统计 ======")
        print(f"当前 Batch 参与 Loss 计算的总有效字节数: {total_masked_count}")

        if total_masked_count == 0:
            print("⚠️ 警告: 当前 Batch 没有检测到任何有效被掩码的 Label！")
            print("=" * 60 + "\n")
            return

        # 3. 统计 0-255 范围内所有 Byte 的出现次数
        # 使用 long 类型的 active_labels 进行高效的 bincount 计数
        byte_counts = torch.bincount(active_labels.to(torch.long), minlength=256)

        # 4. 提取出现频率最高的前 5 个字节 (Top-5)
        topk_values, topk_indices = torch.topk(byte_counts, k=5)

        print("\n🔥 出现频率最高的前 5 个被掩码字节 (Top-5 High Freq):")
        for i in range(5):
            count = topk_values[i].item()
            byte_id = topk_indices[i].item()
            percentage = (count / total_masked_count) * 100
            print(f"   Byte ID: {byte_id:<3} (0x{byte_id:02X}) | 出现次数: {count:<6} | 占比: {percentage:.2f}%")

        # 5. 统计多样性：这个 Batch 一共抽到了多少种不同的 Byte
        unique_bytes = (byte_counts > 0).sum().item()
        print(f"\n🎲 标签多样性: 当前 Batch 共包含 {unique_bytes}/256 种不同的独立字节")

        # 6. 【极客功能】画一个简单的 ASCII 水平直方图，直观展现分布趋势
        print("\n📈 字节区间粗略分布直方图 (每 32 个字节一组):")
        chunk_size = 32
        for block in range(8):
            start = block * chunk_size
            end = start + chunk_size
            block_sum = byte_counts[start:end].sum().item()
            block_ratio = block_sum / total_masked_count

            # 用“#”号代表条形图，最大长度 30 个字符
            bar_length = int(block_ratio * 40)
            bar = "#" * bar_length

            print(f"   [{start:03d}-{end - 1:03d}]: {bar:<40} ({block_sum:<5}个, 占比 {block_ratio * 100:.1f}%)")

        # 7. 极端分布预警
        max_ratio = topk_values[0].item() / total_masked_count
        if max_ratio > 0.40:
            print(f"\n🚨 【重大警告】单一字节 0x{topk_indices[0].item():02X} 的占比高达 {max_ratio * 100:.1f}%！")
            print("   说明你的 EXE 数据集里依然残留大面积的单调字节（如全零），这会诱发线性注意力发生特征塌陷！")

        print("=" * 66 + "\n")

class DistributedBinaryDataset(Dataset):
    def __init__(self, file_dir, chunk_size=4096):
        self.chunk_size = chunk_size

        self.all_bytes = bytearray()

        is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

        if is_main_process:
            print(f"正在将 {file_dir} 加载喵～")

        file_paths = []
        for root, _, files in os.walk(file_dir):
            for file in files:
                if not file.startswith('.'):
                    file_paths.append(os.path.join(root, file))
        file_paths.sort()

        iterator = tqdm.tqdm(file_paths, desc="Loading files to Memory") if is_main_process else file_paths

        for file_path in iterator:
            try:
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                    file_len = len(file_data)
                    for i in range(0, file_len - chunk_size + 1, chunk_size):
                        chunk_view = file_data[i:i + chunk_size]
                        if chunk_view.count(b'\x00') > (chunk_size * 0.9):
                            #print("过滤了喵～")
                            continue
                        self.all_bytes.extend(chunk_view)

            except Exception as e:
                if is_main_process:
                    print(f"\n读取文件失败 {file_path}: {e} 喵～")

        self.num_total_chunks = len(self.all_bytes) // self.chunk_size

        if is_main_process:
            print(f"加载完成！全局内存大小: {len(self.all_bytes) / 1024 / 1024:.2f} MB")
            print(f"总计可切分 Chunk 数量 (Dataset 长度): {self.num_total_chunks}")

    def __len__(self):
        return self.num_total_chunks

    def __getitem__(self, idx):
        start_idx = idx * self.chunk_size
        end_idx = start_idx + self.chunk_size
        chunk = self.all_bytes[start_idx:end_idx]

        return {"byte_ids": list(chunk)}


def main():
    config = Qwen3_5MoeBinaryConfig()
    config.attn_nums = 4
    config.num_heads = 8
    config.encoder_dim = 1536

    encoder = BinaryByteModalEncoder(config)
    model = BinaryMLMPretrainWrapper(encoder, config)

    for module in model.modules():
        if isinstance(module, LinearAttentionBlock):
            make_block_checkpointed(module)

    data_collator = BinaryMLMDataCollator(mask_prob=0.25, mask_token_id=256)

    dataset = DistributedBinaryDataset(
        file_dir="/home/jue/文档/dataset/temp-dataset/test-train/",
        chunk_size=1024*2,
    )

    training_args = TrainingArguments(
        output_dir="./binary_mlm_output",
        num_train_epochs=20,
        per_device_train_batch_size=16,
        save_strategy="epoch",
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=1,

        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        dataloader_num_workers=1,
        dataloader_pin_memory=True,
        gradient_accumulation_steps=4,

        report_to="wandb",
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
        warmup_ratio=0.05
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        callbacks=[PyTorchProfilerCallback()]
    )
    trainer.train()
    if trainer.is_world_process_zero():
        unwrap_model = trainer.accelerator.unwrap_model(model)
        torch.save(unwrap_model.encoder.state_dict(), "./binary_encoder_pe_elf_8_dowm_8_hidden_1024_heads_16.pt")


if __name__ == "__main__":
    main()