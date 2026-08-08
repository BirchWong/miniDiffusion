from src.model_vae import *
from src.train_diffusion import *
from src.algorithm_diffusion import *
from src.model_Unet import Unet
from src.visualize import DiffusionGenerator
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import T5EncoderModel, T5Tokenizer


data = torch.load('dataset_for_diffusion.pt')
latents = data['latents']
labels = data['labels']

dataset = TensorDataset(latents, labels)

dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


unet = Unet(base_channels=256).to(device)
print(f"模型大小: {sum(p.numel() for p in unet.parameters()) * 4 / 1024 / 1024:.2f} MB")

# 这里选择 DDPM 或者 FlowMatching （模型推理时，记得载入对应的权重）
diffusion = FlowMatching(model=unet, device=device)

vae = VAE(base_channels=64).to(device)
vae.load_state_dict(torch.load("VAE_64.pth"))

prompt_embedding_dictionary = np.load("prompt_embedding_dictionary_T5.npy")
prompts_embedding = prompt_embedding_dictionary[16*[163]]
prompts_embedding = torch.from_numpy(prompts_embedding)
generator = DiffusionGenerator(vae, diffusion, device, prompts_embedding)

def train(diffusion, dataloader, device, generator):

    checkpoint_path = "FlowMatching_256.pth"

    # 可以继续之前的checkpoint训练。第一次训练，checkpoint就是None
    #checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint = None
    train_diffusion(diffusion, dataloader, device, epochs=400, lr=1e-4, save_path=checkpoint_path,generator=generator,checkpoint=checkpoint)

def inference(diffusion, generator):
    checkpoint = torch.load('FlowMatching_256.pth', weights_only=False)
    diffusion.model.load_state_dict(checkpoint['ema_model'])

    # T5模型需要自己提前下载，放到下面的对应位置
    T5 = T5EncoderModel.from_pretrained("./data_processor/prompt/t5-base-local")
    tokenizer = T5Tokenizer.from_pretrained("./data_processor/prompt/t5-base-local")
    T5.eval()

    prompts = ["a girl with long wavy white hair, a light blue T-shirt",
               "a girl with long straight black hair, a light green hoodie",
               "a girl with short wavy blonde hair, a brown jacket",
               "a girl with short straight brown hair, a black hoodie"] * 4

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=128)

    with torch.no_grad():
        outputs = T5(**inputs)
        #print(outputs.last_hidden_state.shape)
        prompts_embedding = outputs.last_hidden_state  # [B,T,D]

    generator.prompts_embedding = prompts_embedding
    if diffusion.__class__.__name__ == "DDPM":
        generator.generate(save_dir='DDPM_output', filename='DDPM_generation.png')
    elif diffusion.__class__.__name__ == "FlowMatching":
        generator.generate(save_dir='FlowMatching_output', filename='FlowMatching_generation.png')

if __name__ == "__main__":
    #train(diffusion, dataloader, device, generator)
    inference(diffusion, generator)
