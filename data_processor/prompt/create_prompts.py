import json

with open('appearance.json', 'r') as file:
    appearance = json.load(file)

hair_length = appearance['hair length']
hair_texture = appearance['hair texture']
hair_color = appearance['hair color']
hair_structure = appearance['hair structure']
top = appearance['top']

prompts = []

for a in hair_length:
    for b in hair_texture:
        for c in hair_color:
            for d in hair_structure:
                for e in top:
                    prompt = "a girl with " + a + " " + b + " " + c + " " + d + ", a " + e
                    #print(prompt)
                    prompts.append(prompt)
print(len(prompts))
with open("prompts.txt", "w", encoding="utf-8") as f:
    for p in prompts:
        f.write(p + "\n")

print("已生成 prompts.txt")

with open('prompts.json', 'w', encoding='utf-8') as f:
    json.dump(prompts, f, ensure_ascii=False, indent=4)

print("List saved to output.json")