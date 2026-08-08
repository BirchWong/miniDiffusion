# miniDiffusion

## Explanation of Basic Principles

## Pre-trained Model Weights

VAE : https://huggingface.co/BirchWong/miniDiffusion/resolve/main/VAE_64.pth

DDPM Unet : https://huggingface.co/BirchWong/miniDiffusion/resolve/main/DDPM_Unet_256.pth

FlowMatching Unet : https://huggingface.co/BirchWong/miniDiffusion/resolve/main/FlowMatching_Unet_256.pth


## Results
![miniGPT training results](https://cdn.jsdelivr.net/gh/BirchWong/miniDiffusion@main/DDPM_output/DDPM_generation.png)
---

    prompts = ["a girl with long wavy white hair, a light blue T-shirt",
               "a girl with long straight black hair, a light green hoodie",
               "a girl with short wavy blonde hair, a brown jacket",
               "a girl with short straight brown hair, a black hoodie"] * 4

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
