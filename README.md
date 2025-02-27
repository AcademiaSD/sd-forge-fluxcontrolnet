# Flux.1 Tools ControlNet Extension (sd-forge-fluxcontrolnet)
This is an implementation to use Canny, Depth, and Redux Flux1.dev ControlNet Forge WebUI Extension by [@AcademiaSD](https://github.com/AcademiaSD?tab=repositories)

## For Forge WebUI and ComfyUI tutorials
You can follow and see Spanish videos tutorials (English IA Dub and English Subtitles) on https://www.youtube.com/@Academia_SD

![image](https://github.com/user-attachments/assets/7a2bd67d-d8d6-4fd4-bcf6-88a56c80dd38)

## Install
Go to the Extensions tab > Install from URL > URL for this repository.

## Supports
✅ CUDA GPUs (tested with GPUs of 12GB VRAM, should work with 8GB VRAM).

❌ Custom LoRa (coming soon...)

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
> 
> You can download the requirements_versions.txt file for easier installation: https://drive.google.com/file/d/1nEpklfF7Ppflcq5HGFS9NI__ZfKmg-Yg/view?usp=sharing


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
>
> - Text encoders FP8 
>   (https://drive.google.com/file/d/1UwDD0bf2Y0upsFI9ZARNpvBUtL22Qxg3/view?usp=drive_link) Unzip into models/diffusers
    
