import os
import glob
import shutil
import json
import numpy as np
from PIL import Image

with open('prompts.json', 'r', encoding='utf-8') as f:
    prompts = json.load(f)

dataset_path = "1"
root_path = os.path.join(".\data_processor", dataset_path)

paths = glob.glob(os.path.join(dataset_path, "*.png"))

dataset = []


for path in paths:

    name = os.path.basename(path)

    before_dash, after_dash = name.replace('.png', '').split('-')

    before_num = int(before_dash)
    after_num = int(after_dash)


    prompt = prompts[before_num]

    img_path = os.path.join(root_path, name)


    data_item = {
        "index": before_num,
        "image_path": img_path,
        "prompt": prompt,
    }

    dataset.append(data_item)

with open("dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"已保存 {len(dataset)} 条数据到 dataset.json")