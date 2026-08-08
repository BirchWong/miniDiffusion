from transformers import T5EncoderModel, T5Tokenizer
import torch
import numpy as np

import json

model = T5EncoderModel.from_pretrained("./t5-base-local")
tokenizer = T5Tokenizer.from_pretrained("./t5-base-local")
model.eval()

with open('prompts.json', 'r', encoding='utf-8') as f:
    prompts = json.load(f)
print(prompts)

inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    print(outputs.last_hidden_state.shape)

    prompt_embeddings = outputs.last_hidden_state # [B,T,D]

np.save("prompt_embedding_dictionary_T5.npy", prompt_embeddings.numpy())

print(f"编码完成！向量 shape: {prompt_embeddings.shape}")