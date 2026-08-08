# miniDiffusion

This project trains a Flow Matching / DDPM text-to-image model from scratch.

The training dataset consists of 10,000 image-text pairs.

All core algorithm implementations are located in the **`src`** folder for easy reading and learning.

## Explanation of Basic Principles

## Pre-trained Model Weights

VAE : https://huggingface.co/BirchWong/miniDiffusion/resolve/main/VAE_64.pth

DDPM Unet : https://huggingface.co/BirchWong/miniDiffusion/resolve/main/DDPM_Unet_256.pth

FlowMatching Unet : https://huggingface.co/BirchWong/miniDiffusion/resolve/main/FlowMatching_Unet_256.pth


## Results

![miniGPT training results](https://cdn.jsdelivr.net/gh/BirchWong/images@master/miniDiffusion-result.png)

---

    prompts = ["a girl with long straight brown hair, a light blue hoodie",
               "a girl with long wavy black hair, a light blue T-shirt",
               "a girl with short wavy brown hair, a light red hoodie",
               "a girl with short straight blonde hair, a brown jacket"] * 4

## Quick Inference (Zero Setup with Colab)

Colab: https://colab.research.google.com/drive/1YScHcc0aMKCND1wGzgth7nmNbdJi8KvZ?usp=sharing

## Training

Please read `docs/guide(-en).md` first.  
Then complete the dataset preparation and run `main_vae.py` and `main_ddpm.py`.   
Make sure to uncomment the line `# train(diffusion, dataloader, device, generator)` in `main_ddpm.py` to start training. 

You can use Stable Diffusion to generate text-image pairs, or feel free to email me for the dataset.  
My Gmail address has the same prefix as my GitHub username.
## Dependencies

• torch (GPU version)  
• numpy  
• transformers      
• json  
• PIL  
• tqdm   
• logging  
• matplotlib  
• copy  
• lpips  
