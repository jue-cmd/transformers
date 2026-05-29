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
BINARY_START_TOKEN = "<|binary_start|>"
BINARY_END_TOKEN = "<|binary_end|>"
BINARY_NUM_TOKENS = 256
MAX_BYTES = 4096

mock_data = [
    {
        "file_path": "/home/jue/文档/hmcl/HMCL-3.12.2.sh",
        "target": "该文件是一个Shell脚本，用于在Linux环境下配置并启动HMCL启动器。"
    }
]


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

        byte_ids = torch.tensor(list(raw_bytes), dtype=torch.long)

        if len(byte_ids) > self.max_bytes:
            byte_ids = byte_ids[:self.max_bytes]

        return {
            "byte_ids": byte_ids,
            "target": item["target"]
        }


class MultimodalAutoCollator:
    def __init__(self, tokenizer, binary_token, binary_num_tokens):
        self.tokenizer = tokenizer
        self.binary_token = binary_token
        self.binary_num_tokens = binary_num_tokens

    def __call__(self, batch):
        byte_ids = torch.stack([item["byte_ids"] for item in batch])

        input_ids_list = []
        labels_list = []

        for item in batch:
            placeholder_str = BINARY_START_TOKEN + self.binary_token * self.binary_num_tokens + BINARY_END_TOKEN
            target_text = f"{item['target']}{self.tokenizer.eos_token}"
            input_str = placeholder_str + target_text
            placeholder_ids = self.tokenizer(input_str, add_special_tokens=False).input_ids
            target_ids = self.tokenizer(target_text, add_special_tokens=False).input_ids

            full_input_ids = placeholder_ids + target_ids

            full_labels = [-100] * len(placeholder_ids) + target_ids

            input_ids_list.append(torch.tensor(full_input_ids, dtype=torch.long))
            labels_list.append(torch.tensor(full_labels, dtype=torch.long))

        max_text_len = max(len(ids) for ids in input_ids_list)

        padded_input_ids = []
        padded_labels = []

        for ids, labels in zip(input_ids_list, labels_list):
            pad_len = max_text_len - len(ids)

            padded_ids = torch.cat([ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
            padded_lbs = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])

            padded_input_ids.append(padded_ids)
            padded_labels.append(padded_lbs)

        return {
            "byte_ids": byte_ids,
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels)
        }


def main():
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

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"-> 成功保存至 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
