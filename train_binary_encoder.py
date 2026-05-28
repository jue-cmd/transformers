import random
import torch
from torch import nn
import os
from datasets import Dataset

from transformers.models.qwen3_5_moe.binary_encoder import BinaryByteModalEncoder, BinaryMLMPretrainWrapper


class BinaryMLMDataCollator:
    def __init__(self, mask_prob=0.15, mask_token_id=256):
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id

    def __call__(self, examples):
        batch_byte_ids = [torch.tensor(e['byte_ids'], dtype=torch.long) for e in examples]
        byte_ids = torch.stack(batch_byte_ids, dim=0)
        labels = torch.full(byte_ids.shape, -100, dtype=torch.long)
        probability_matrix = torch.full(byte_ids.shape, self.mask_prob)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[masked_indices] = byte_ids[masked_indices]
        indices_replaced = torch.bernoulli(torch.full(byte_ids.shape, 0.8)).bool() & masked_indices
        byte_ids[indices_replaced] = self.mask_token_id
        indices_random = torch.bernoulli(torch.full(byte_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(0, 256, byte_ids.shape, dtype=torch.long)
        byte_ids[indices_random] = random_words[indices_random]
        return {
            "byte_ids": byte_ids,
            "labels": labels
        }


def load_binary_files(file_dir, chunk_size=1024):
    all_chunks = []
    for root, _, files in os.walk(file_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'rb') as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        if len(chunk) == chunk_size:
                            all_chunks.append(list(chunk))
            except Exception as e:
                print(f"读取文件失败 {file_path}: {e}")
    return Dataset.from_dict({"byte_ids": all_chunks})


from transformers import TrainingArguments, Trainer, Qwen3_5MoeBinaryConfig

config = Qwen3_5MoeBinaryConfig()
encoder = BinaryByteModalEncoder(config)
model = BinaryMLMPretrainWrapper(encoder, config)

data_collator = BinaryMLMDataCollator(mask_prob=0.15, mask_token_id=256)

dataset = load_binary_files("./train_datasets")

training_args = TrainingArguments(
    output_dir="./binary_mlm_output",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    save_strategy="epoch",
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=1,
    fp16=torch.cuda.is_available(),
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

torch.save(encoder.state_dict(), "./binary_encoder_pe_elf.pt")
