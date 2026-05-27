import os
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Qwen3_5MoeForConditionalGeneration,
    TrainingArguments,
    Trainer
)

# ==========================================
# 1. 基础配置与路径
# ==========================================
MODEL_PATH = "/home/jue/文档/model/qwen3.5-35b-a3b/"
OUTPUT_DIR = "./output_trainer_binary_auto"

BINARY_TOKEN = "<|binary_token|>"
BINARY_NUM_TOKENS = 256
MAX_BYTES = 4096  # 二进制文件读取的最大长度

# 纯净的原始数据集：没有任何 Tokenizer 侵入
mock_data = [
    {
        "file_path": "/home/jue/文档/hmcl/HMCL-3.12.2.sh",
        "instruction": "分析以下二进制文件内容并总结它的功能。",
        "target": "该文件是一个Shell脚本，用于在Linux环境下配置并启动HMCL启动器。"
    }
]


# ==========================================
# 2. 极简 Dataset（只读原始数据）
# ==========================================
class RawBinaryDataset(Dataset):
    def __init__(self, data_list, max_bytes=4096):
        self.data_list = data_list
        self.max_bytes = max_bytes

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        with open(item["file_path"], "rb") as f:
            raw_bytes = f.read(self.max_bytes)

        binary_tensor = torch.tensor(list(raw_bytes), dtype=torch.long)

        if len(binary_tensor) < self.max_bytes:
            binary_tensor = torch.cat(
                [binary_tensor, torch.zeros(self.max_bytes - len(binary_tensor), dtype=torch.long)])
        else:
            binary_tensor = binary_tensor[:self.max_bytes]

        return {
            "binary_tensors": binary_tensor,
            "instruction": item["instruction"],
            "target": item["target"]
        }


# ==========================================
# 3. 工业级万能 Data Collator（一站式处理）
# ==========================================
class MultimodalAutoCollator:
    def __init__(self, tokenizer, binary_token, binary_num_tokens):
        self.tokenizer = tokenizer
        self.binary_token = binary_token
        self.binary_num_tokens = binary_num_tokens

    def __call__(self, batch):
        # 1. 直接打包已经对齐好长度的二进制 Tensors
        binary_tensors = torch.stack([item["binary_tensors"] for item in batch])

        input_ids_list = []
        labels_list = []

        # 2. 逐条处理文本的拼接与 Tokenization
        for item in batch:
            # 动态构造包含 256 个占位符的完整 Prompt
            placeholder_str = self.binary_token * self.binary_num_tokens
            prompt_text = f"{item['instruction']}\n{placeholder_str}\n"
            target_text = f"{item['target']}{self.tokenizer.eos_token}"

            # 分别对 Prompt 和 Target 进行编码以计算边界
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
            target_ids = self.tokenizer(target_text, add_special_tokens=False).input_ids

            # 组合成完整的 input_ids
            full_input_ids = prompt_ids + target_ids

            # 构造精确的 Labels（Prompt 部分用 -100 屏蔽，不计算 Loss）
            full_labels = [-100] * len(prompt_ids) + target_ids

            input_ids_list.append(torch.tensor(full_input_ids, dtype=torch.long))
            labels_list.append(torch.tensor(full_labels, dtype=torch.long))

        # 3. 动态 Padding（将 Batch 内的所有文本补齐到当前 Batch 的最大长度）
        max_text_len = max(len(ids) for ids in input_ids_list)

        padded_input_ids = []
        padded_labels = []

        for ids, labels in zip(input_ids_list, labels_list):
            pad_len = max_text_len - len(ids)

            # input_ids 后面补齐 pad_token_id
            padded_ids = torch.cat([ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
            # labels 后面补齐 -100
            padded_lbs = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])

            padded_input_ids.append(padded_ids)
            padded_labels.append(padded_lbs)

        # 返回符合模型 forward 映射的标准字典
        return {
            "byte_ids": binary_tensors,
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels)
        }


# ==========================================
# 4. 主训练流程
# ==========================================
def main():
    # 1. 加载并初始化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    def rnn_and_linear_initializer(module):
        with torch.no_grad():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
                module.weight.data.normal_(mean=0.0, std=0.0002)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torch.nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.002)
            elif isinstance(module, (torch.nn.LayerNorm, torch.nn.GroupNorm)):
                if module.weight is not None:
                    module.weight.data.fill_(1.0)
                if module.bias is not None:
                    module.bias.data.zero_()
            for name, param in module.named_parameters(recurse=False):
                if "queries" in name or "query" in name:
                    param.data.normal_(mean=0.0, std=0.002)
                elif "projector" in name:
                    param.data.normal_(mean=0.0, std=0.0002)
                elif "bias" in name:
                    param.data.zero_()

    if hasattr(model.model, "binary_encoder"):
        print("-> 执行 binary_encoder 控噪初始化...")
        model.model.binary_encoder.apply(rnn_and_linear_initializer)

    model.requires_grad_(False)
    model.model.binary_encoder.requires_grad_(True)
    model.model.binary_encoder.train()

    train_dataset = RawBinaryDataset(mock_data, max_bytes=MAX_BYTES)
    data_collator = MultimodalAutoCollator(
        tokenizer=tokenizer,
        binary_token=BINARY_TOKEN,
        binary_num_tokens=BINARY_NUM_TOKENS
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        num_train_epochs=1,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    print("-> 启动训练（所有文本与 Tokenize 处理均在 Collator 中动态完成）...")
    trainer.train()

    # 8. 保存
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"-> 成功保存至 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()