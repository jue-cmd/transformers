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


class GradientMonitorCallback(TrainerCallback):
    def __init__(self, log_steps=10):
        self.log_steps = log_steps

    def on_substep_end(self, args, state, control, **kwargs):
        pass

    @torch.no_grad()
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.log_steps != 0 or model is None:
            return
        wandb_metrics = {}
        print(f"\n--- [Step {state.global_step}] 核心层梯度范数（Gradient Norm）全景扫描 喵～ ---")
        if hasattr(model, "module"):
            target_model = model
        else:
            target_model = model
        if hasattr(target_model.encoder,
                   "byte_embedding") and target_model.encoder.byte_embedding.weight.grad is not None:
            emb_grad = target_model.encoder.byte_embedding.weight.grad.norm().item()
            wandb_metrics['Embedding'] = emb_grad
            print(f"[Embedding] byte_embedding: {emb_grad:.4f}")
        if hasattr(target_model.encoder, "pos_conv") and target_model.encoder.pos_conv.conv.weight.grad is not None:
            conv_grad = target_model.encoder.pos_conv.conv.weight.grad.norm().item()
            wandb_metrics['PosConv'] = conv_grad
            print(f"[PosConv] positional_conv: {conv_grad:.4f}")
        if hasattr(target_model.encoder, "encoder_layers"):
            for i, layer in enumerate(target_model.encoder.encoder_layers):
                if layer.attn.q_proj.weight.grad is not None:
                    wandb_metrics[f"grad_norm/layer_{i}_attn_Q"] = layer.attn.q_proj.weight.grad.norm().item()
                if layer.attn.k_proj.weight.grad is not None:
                    wandb_metrics[f"grad_norm/layer_{i}_attn_K"] = layer.attn.k_proj.weight.grad.norm().item()
                if layer.attn.v_proj.weight.grad is not None:
                    wandb_metrics[f"grad_norm/layer_{i}_attn_V"] = layer.attn.v_proj.weight.grad.norm().item()
                if layer.attn.out_proj.weight.grad is not None:
                    wandb_metrics[f"grad_norm/layer_{i}_attn_Out"] = layer.attn.out_proj.weight.grad.norm().item()
                if layer.mlp[2].weight.grad is not None:
                    wandb_metrics[f"grad_norm/layer_{i}_mlp"] = layer.mlp[2].weight.grad.norm().item()
        if hasattr(target_model, "mlm_head"):
            head_grads = []
            for name, param in target_model.mlm_head.named_parameters():
                if param.grad is not None and "weight" in name:
                    head_grads.append(f"{name}: {param.grad.norm().item():.4f}")
            if head_grads:
                print(f"[MLM Head] " + " | ".join(head_grads))
                clean_name = name.replace(".", "_").replace("_weight", "")
                wandb_metrics[f"grad_norm/head_{clean_name}"] = param.grad.norm().item()
        print(wandb_metrics)
        if wandb_metrics and wandb is not None:
            wandb.log(wandb_metrics, step=state.global_step)

            print(
                f"[WandB Monitor] Step {state.global_step}: 已成功将 {len(wandb_metrics)} 个层的梯度范数推送到面板 喵～")
        print("-" * 60 + "\n")


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
        self.mask_token_id = mask_token_id

    def __call__(self, examples):
        batch_byte_ids = [torch.tensor(e['byte_ids'], dtype=torch.long) for e in examples]
        byte_ids = torch.stack(batch_byte_ids, dim=0)
        labels = byte_ids.clone()
        probability_matrix = torch.full(byte_ids.shape, self.mask_prob)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        indices_replaced = torch.bernoulli(torch.full(byte_ids.shape, 0.8)).bool() & masked_indices
        byte_ids[indices_replaced] = self.mask_token_id
        indices_random = torch.bernoulli(torch.full(byte_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
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
    config.downsample_factor = 8
    config.attn_nums = 6
    config.num_heads = 8
    config.encoder_dim = 2048

    encoder = BinaryByteModalEncoder(config)
    model = BinaryMLMPretrainWrapper(encoder, config)

    for module in model.modules():
        if isinstance(module, LinearAttentionBlock):
            make_block_checkpointed(module)

    data_collator = BinaryMLMDataCollator(mask_prob=0.5, mask_token_id=256)

    dataset = DistributedBinaryDataset(
        file_dir="/home/jue/文档/dataset/temp-dataset/",
        chunk_size=128,
    )

    training_args = TrainingArguments(
        output_dir="./binary_mlm_output",
        num_train_epochs=10,
        per_device_train_batch_size=1,
        save_strategy="epoch",
        learning_rate=3e-4,
        weight_decay=0.01,
        logging_steps=1,

        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_accumulation_steps=1,

        report_to="wandb",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        callbacks=[PyTorchProfilerCallback(), GradientMonitorCallback(log_steps=5)]
    )
    trainer.train()
    if trainer.is_world_process_zero():
        unwrap_model = trainer.accelerator.unwrap_model(model)
        torch.save(unwrap_model.encoder.state_dict(), "./binary_encoder_pe_elf_8_dowm_8_hidden_1024_heads_16.pt")


if __name__ == "__main__":
    main()