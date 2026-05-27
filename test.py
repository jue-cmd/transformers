from transformers import Qwen2_5_VLProcessor
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

processor = Qwen2_5_VLProcessor.from_pretrained("/home/jue/文档/model/qwen3.5-35b-a3b/")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "binary", "path": "/home/jue/文档/hmcl/HMCL-3.12.2.sh"},
            {"type": "text", "text": "hi"},
        ],
    }
]
inputs = processor.apply_chat_template(messages,
                                   tokenize=True,
                                   add_generation_prompt=True,
                                   return_dict=True,
                                   return_tensors="pt"
                                   )

import torch

model = Qwen3_5MoeForConditionalGeneration.from_pretrained("/home/jue/文档/model/qwen3.5-35b-a3b/")


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

model.model.binary_encoder.apply(rnn_and_linear_initializer)

model.save_pretrained("/home/jue/文档/model/qwen3.5-35b-a3b-binary_encoder/")

exit()

inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
a=processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(a)