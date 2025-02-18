# sd-forge-fluxcontrolnet
Flux1.dev ControlNet Forge WebUI Extension


Install
Go to the Extensions tab, then Install from URL, use the URL for this repository.

Easiest way to ensure necessary diffusers release is installed is to edit requirements_versions.txt in the webUI directory.

diffusers>=0.32.2
transformers>=4.48.3
Pillow>=9.5.0
tokenizers>=0.21
huggingface-hub>=0.28.1
controlnet-aux>=0.0.9

Forge2 already has newer versions for all but diffusers. Be aware that updates to Forge2 may overwrite the requirements file.

Important

Also needs a huggingface access token: Sign up / log in, go to your profile, create an access token. Read type is all you need, avoid the much more complicated Fine-grained option. Copy the token. Make a textfile called huggingface_access_token.txt in the main webui folder, e.g. {forge install directory}\webui, and paste the token in there. You will also need to accept the terms on the SD3 repository page.

Note

Download checkpoints and move to folder models/stable-diffusion 
For Redux
https://civitai.com/models/1264072/flux1-dev-fp8 
For Canny 
https://civitai.com/models/1263977/flux1-canny-dev-fp8
For Depth
https://civitai.com/models/1264100/flux1-depth-dev-fp8

Texts Encoders models download:
https://drive.google.com/file/d/1li5UP3oKQWbdYHakOs4jvH94tuFQpPh2/view?usp=sharing

Unzip to models/diffusers keeping the cn folder

almost current UI screenshot
