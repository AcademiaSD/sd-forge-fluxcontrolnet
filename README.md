# sd-forge-fluxcontrolnet
This is an implementation to use Canny, Depth, and Redux Flux1.dev ControlNet Forge WebUI Extension by [@AcademiaSD](https://github.com/AcademiaSD?tab=repositories)

## For Forge WebUI and ComfyUI tutorials
You can follow and see Spanish videos tutorials (English IA Dub and English Subtitles) on https://www.youtube.com/@Academia_SD

![image](https://github.com/user-attachments/assets/7a2bd67d-d8d6-4fd4-bcf6-88a56c80dd38)

## Install
Go to the Extensions tab > Install from URL > URL for this repository.

## Supports
✅ CUDA GPUs (tested with GPUs of 12GB VRAM, should work with 8GB VRAM).

❌ LoRa

## Requierements
> [!WARNING]  
> Easiest way to ensure necessary diffusers release is installed is to edit requirements_versions.txt in the webUI directory.
> 
> diffusers>=0.32.2,
> transformers>=4.48.3,
> Pillow>=9.5.0,
> tokenizers>=0.21,
> huggingface-hub>=0.28.1,
> controlnet-aux>=0.0.9,
> 
> Forge2 already has newer versions for all but diffusers. Be aware that updates to Forge2 may overwrite the requirements file.

## Set HF Access Token
> [!IMPORTANT] 
>
> Also needs a huggingface access token: Sign up / log in, go to your profile, create an access token. Read type is all you need, avoid the much more complicated Fine-grained option. Copy the token. Make a textfile called huggingface_access_token.txt in the main webui folder, e.g. {forge install directory}\webui, and paste the token in there. You will also need to accept the terms on the Flux1 Dev repositories pages.


## Downloads
> [!NOTE]  
> Download checkpoints and move to folder models/stable-diffusion
>
> - For Redux
>   (https://huggingface.co/Academia-SD/flux1-Dev-FP8/tree/main)
>
> - For Canny 
>   (https://huggingface.co/Academia-SD/flux1-Canny-Dev-FP8/tree/main)
>
> - For Depth
>   (https://huggingface.co/Academia-SD/flux1-Depth-Dev-FP8/tree/main)
