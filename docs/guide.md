# 使用说明

## python文件

| Name          |                                  introduction                                  |
|:---|:------------------------------------------------------------------------------:|
| `main_vae.py`     |                               训练VAE得到 `vae_64.pth`                               |
| `main_diffusion.py`     |   训练DDPM或者FlowMatching<br/>得到 `diffusion_256.pth` <br/>( 需要用到先前的 `vae_64.pth` )    |
| `generate_dataset_for_diffusion.py` | 将图片提前用 VAE 编码，<br/>并和 T5 prompt embeddings 放在一起<br/>得到`dataset_for_diffusion.pt` |
| `extract_Unet.py`                   |        把 `diffusion_256.pth` 里<br/> Unet 模型参数单独提取出<br/>(其他的主要是optimizer)         |

## 文件夹

| Name                  |        introduction         |
|:----------------------|:-------------------------:|
| `VAE_output`          |      存放 VAE 重构的图片结果       |
| `DDPM_output`         |     存放 DDPM 算法生成的图片结果     |
| `FLowMatching_output` | 存放 FlowMatching 算法生成的图片结果 |
| `src`                 |         核心算法存放在这里         |
| `data_processor`      |        图片预处理（比较复杂）        |


### src/

| Name                     |              introduction                |
|:-------------------------|:--------------------------------------:|
| `algorithm_diffusion.py` |    FlowMatching 和 DDPM <br/> 的算法实现     |
| `model_Unet.py`          | FlowMatching 和 DDPM <br/> 的 Unet 模型的定义 |
| `model_vae.py`           |               VAE 模型的定义                |
| `train_diffusion.py`     |            Diffusion 的训练过程             |
| `train_vae.py`           |               VAE 的训练过程                |
| `visualize.py`           |                相关可视化工具                 |

### data_processor/

在 `prompt/` 文件夹里，需要有T5模型 `t5-base-local` (不习惯也可以改路径) <br/> 运行 `t5_embedding.py` 可以得到 `prompt_embedding_dictionary_T5.npy`

`dataset_process.py` 将 `prompts.json` 和 图片文件夹(1) 变成 `dataset.json`

`dataset.json` 和 `prompt_embedding_dictionary_T5.npy` 需要放到 `main_diffusion.py` 同目录，给 `generate_dataset_for_diffusion.py` 使用

图片文件夹等的太大了，这里没存