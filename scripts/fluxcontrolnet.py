import gradio as gr
import torch
from diffusers import FluxControlPipeline, FluxControlNetModel, AutoencoderKL
from diffusers import FluxPriorReduxPipeline, FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer, TrainingArguments
from diffusers.utils import load_image
from PIL import Image
import gc
import bitsandbytes as bnb
import traceback
from functools import partial
import numpy as np
from controlnet_aux import CannyDetector
from controlnet_aux.processor import Processor
import os
from modules import script_callbacks
from modules.ui_components import ToolButton
import modules.generation_parameters_copypaste as parameters_copypaste
from modules.shared import opts, OptionInfo
from modules.ui_common import save_files
from huggingface_hub import hf_hub_download
import random
from datetime import datetime 
from collections import deque
from huggingface_hub import login
import threading
import json
import os
import io
import logging
import sys
from h11._util import LocalProtocolError

class CustomErrorFilter(logging.Filter):
    def filter(self, record):
        if record.exc_info:
            err_msg = str(record.exc_info[1])
            return not ("Too little data for declared Content-Length" in err_msg or 
                       "Too much data for declared Content-Length" in err_msg)
        return True

# Aplicar el filtro al logger de uvicorn
logging.getLogger("uvicorn.error").addFilter(CustomErrorFilter())

# También redirigir stderr para casos que no pase el logger
class ErrorFilter(io.StringIO):
    def write(self, message):
        if ("Too little data for declared Content-Length" not in message and
            "Too much data for declared Content-Length" not in message):
            sys.__stderr__.write(message)

sys.stderr = ErrorFilter()


# Academia-SD/flux1-dev-text_encoders-NF4
def load_huggingface_token():
    try:
        # Obtener el directorio raíz de ForgeWebUI
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        token_path = os.path.join(root_dir, "huggingface_access_token.txt")
        if not os.path.exists(token_path):
            raise FileNotFoundError(f"Token file not found at {token_path}")
        with open(token_path, 'r') as file:
            token = file.read().strip()
            if not token:
                raise ValueError("Token file is empty")
            return token
    except Exception as e:
        print(f"Error loading Hugging Face token: {str(e)}")
        return None
# Initialize Hugging Face login
try:
    token = load_huggingface_token()
    if token:
        login(token)
    else:
        print("Failed to load Hugging Face token. Some features may not be available.")
except Exception as e:
    print(f"Error during Hugging Face login: {str(e)}")
processor_id = 'canny'
menupro = "canny"
prompt_embeds_scale_1 = 1.0
prompt_embeds_scale_2 = 1.0
pooled_prompt_embeds_scale_1 = 1.0
pooled_prompt_embeds_scale_2 = 1.0
checkpoint_path = "./models/Stable-diffusion/"
current_model = "flux1-Canny-Dev_FP8.safetensors"

def debug_print(message, debug_enabled=False):
    if debug_enabled:
        print(message)

def print_memory_usage(message="", debug_enabled=False):
    if debug_enabled and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"{message} GPU Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")

def quantize_model_to_nf4(model, name="", debug_enabled=False):
    debug_print(f"\nQuantizing model {name} t...", debug_enabled)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            debug_print(f"\Transforming layer: {name}", debug_enabled)
            new_module = bnb.nn.Linear4bit(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                compute_dtype=torch.float32,
                compress_statistics=True,
                device=module.weight.device
            )
            with torch.no_grad():
                new_module.weight = bnb.nn.Params4bit(
                    module.weight.data,
                    requires_grad=False,
                    compress_statistics=True
                )
                if module.bias is not None:
                    new_module.bias = torch.nn.Parameter(module.bias.data)
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, new_module)
            else:
                setattr(model, child_name, new_module)
    return model

def clear_memory(debug_enabled=False):
    debug_print("\nEmptying memory", debug_enabled)
    if torch.cuda.is_available():
        print_memory_usage("Before cleaning:", debug_enabled)
        torch.cuda.empty_cache()
        gc.collect()

def update_dimensions(image, width_slider, height_slider):
    if image is not None:
        height, width = image.shape[:2]
        return width, height
    return width_slider, height_slider
    
class LogManager:
    def __init__(self, max_messages=6):
        self.messages = deque(maxlen=max_messages)
        self.log_box = None

    def log(self, message):
        if isinstance(message, tuple):
            message = " ".join(str(m) for m in message)
        print(message)
        self.messages.append(str(message))
        if self.log_box is not None:
            return "\n".join(self.messages)
        return None
        
def load_settings():
    try:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        extension_dir = os.path.join(root_dir, "extensions", "sd-forge-fluxcontrolnet", "utils")
        config_path = os.path.join(extension_dir, "extension_config.txt")
        
        # Valores por defecto
        settings = {
            "checkpoint_path": "./models/Stable-diffusion/",
            "output_dir": "./outputs/fluxcontrolnet/"
        }
        
        if os.path.exists(config_path):

            with open(config_path, 'r') as f:
                loaded_settings = json.load(f)

                settings.update(loaded_settings)
        
        return settings
        
        
    except Exception as e:
        print(f"Error loading settings: {str(e)}")
        return None
        

def save_settingsHF(hf_token_value):
    try:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        token_path = os.path.join(root_dir, "huggingface_access_token.txt")
        with open(token_path, 'w') as f:
            f.write(hf_token_value)
        return f"HF Token saved"
    except Exception as e:
        return f"Error saving HF Token: {str(e)}"

def save_settings(checkpoints_path, output_dir, debug_enabled=False):
    try:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        extension_dir = os.path.join(root_dir, "extensions", "sd-forge-fluxcontrolnet", "utils")
        os.makedirs(extension_dir, exist_ok=True)
        config_path = os.path.join(extension_dir, "extension_config.txt")
        config = {
            "checkpoint_path": checkpoints_path,  # Clave corregida
            "output_dir": output_dir
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        log_message = "Settings file saved"
        # Return all three expected values: log message and both path values
        return log_message, checkpoints_path, output_dir
    except Exception as e:
        error_message = f"Error saving settings file: {str(e)}"
        # In case of error, still return all three expected values
        return error_message, checkpoints_path, output_dir
        
class FluxControlNetTab:
    def __init__(self):
        self.pipe = None
        self.model_path = "Academia-SD/flux1-dev-text_encoders-NF4"
        self.current_processor = "canny"
        self.current_model = "flux1-Canny-Dev_FP8.safetensors"
        self.checkpoint_path = "./models/Stable-diffusion/"
        self.output_dir = "./outputs/fluxcontrolnet/"
        #self.default_image_path = "./extensions/sd-forge-fluxcontrolnet/assets/default.png" 
        self.default_processor_id = "depth_zoe"
        self.logger = LogManager()
        settings = load_settings()
        if settings:
            self.checkpoint_path = settings.get("checkpoint_path", "./models/Stable-diffusion/")
            self.output_dir = settings.get("output_dir", "./outputs/fluxcontrolnet/")
        else:
            self.checkpoint_path = "./models/Stable-diffusion/"
            self.output_dir = "./outputs/fluxcontrolnet/"
            
        
            
        #print("self.output_dir", self.output_dir)
        #print("self.checkpoint_path", self.checkpoint_path)

    def update_model_path(self, new_path, debug_enabled):
        if new_path and os.path.exists(new_path):
            self.model_path = new_path
            debug_print(f"Updated model path: {new_path}", debug_enabled)
            if self.pipe is not None:
                del self.pipe
                self.pipe = None
                clear_memory()
        return self.model_path

    def update_processor_and_model(self, processor_type):
        if processor_type == "canny":
            self.current_processor = "canny"
            self.current_model = "flux1-Canny-Dev_FP8.safetensors"
            self.default_processor_id = None
        elif processor_type == "depth":
            self.current_processor = "depth"
            self.current_model = "flux1-Depth-Dev_FP8.safetensors"
            self.default_processor_id = "depth_zoe"
        elif processor_type == "redux":
            self.current_processor = "redux"
            self.current_model = "flux1-Dev_FP8.safetensors"
            self.default_processor_id = None
        
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            clear_memory()
    
        return [
            gr.Button.update(variant="secondary" if processor_type == "canny" else "primary"),
            gr.Button.update(variant="secondary" if processor_type == "depth" else "primary"),
            gr.Button.update(variant="secondary" if processor_type == "redux" else "primary")
        ]

    def get_processor(self):
        if self.current_processor == "canny":
            return CannyDetector()
        elif self.current_processor == "depth":
            return
        elif self.current_processor == "redux":
            return 
        return # CannyDetector()  # Default to Canny

    def preprocess_image(self, input_image, low_threshold, high_threshold, detect_resolution, image_resolution, debug_enabled, processor_id=None):
        try:
            if input_image is None:
                return None
        
            debug_print("\nStarting preprocessing...", debug_enabled)
            
            # Cargar y validar imagen
            control_image = self.load_control_image(input_image)
            if not control_image:
                raise ValueError("Failed to load control image")
                
            # Convertir a RGB si es necesario
            if control_image.mode != 'RGB':
                control_image = control_image.convert('RGB')
                
            # Aplicar preprocesamiento según el modo
            if self.current_processor == "canny":
                processor = CannyDetector()
                processed_image = processor(
                    control_image, 
                    low_threshold=int(low_threshold),
                    high_threshold=int(high_threshold),
                    detect_resolution=int(detect_resolution),
                    image_resolution=int(image_resolution)
                )
            elif self.current_processor == "depth":
                actual_processor_id = processor_id or 'depth_zoe'
                processor = Processor(actual_processor_id)
                processed_image = processor(control_image)
            else:
                processed_image = control_image
                
            debug_print("\nPreprocess Done.", debug_enabled)
            return np.array(processed_image)
            
        except Exception as e:
            self.logger.log(f"\nError en el preprocesamiento: {str(e)}")
            self.logger.log(f"Stacktrace:\n{traceback.format_exc()}")
            return None

    def load_models(self, use_hyper_flux=True, debug_enabled=False):  #Text_encoders + vae
        debug_print("\nStarting model loading...", debug_enabled)
        dtype = torch.bfloat16
        
        debug_print("\nLoading CLIP text encoder...", debug_enabled)
        text_encoder = CLIPTextModel.from_pretrained(
            self.model_path, subfolder="text_encoder", torch_dtype=dtype
        )
        
        debug_print("\nLoading T5 text encoder...", debug_enabled)
        text_encoder_2 = T5EncoderModel.from_pretrained(
            self.model_path, subfolder="text_encoder_2", torch_dtype=dtype
        )
        
        debug_print("\nLoading VAE...", debug_enabled)
        vae = AutoencoderKL.from_pretrained(
            self.model_path, subfolder="vae", torch_dtype=dtype
        )
      
        debug_print("\nLoading tokenizers...", debug_enabled)
        tokenizer = CLIPTokenizer.from_pretrained(self.model_path, subfolder="tokenizer")
        tokenizer_2 = T5Tokenizer.from_pretrained(self.model_path, subfolder="tokenizer_2")
        
        clear_memory(debug_enabled)
        
        debug_print("\nLoading main Flux ControlNet Checkpoint...", debug_enabled)  #wip cambiar la ruta escrita por la ruta variable
        
        if self.current_processor == "canny":
            self.checkpoint_path = self.checkpoint_path or "./models/Stable-diffusion/"
            self.current_model = self.current_model or "flux1-Canny-Dev_FP8.safetensors"
            base_model = os.path.join(self.checkpoint_path, self.current_model)
        if self.current_processor == "depth":
            self.checkpoint_path = self.checkpoint_path or "./models/Stable-diffusion/"
            self.current_model = self.current_model or "flux1-Depth-Dev_FP8.safetensors"
            base_model = os.path.join(self.checkpoint_path, self.current_model)
        if self.current_processor == "redux":
            self.checkpoint_path = self.checkpoint_path or "./models/Stable-diffusion/"
            self.current_model = self.current_model or "flux1-Dev_FP8.safetensors"
            base_model = os.path.join(self.checkpoint_path, self.current_model)
           
        #debug_print("\nCargando modelo principal Flux ControlNet...", debug_enabled)
        #if self.current_processor == "canny":
        #    base_model = os.path.join("./models/Stable-diffusion/flux1-Canny-Dev_FP8.safetensors")
        #if self.current_processor == "depth":
        #    base_model = os.path.join("./models/Stable-diffusion/flux1-Depth-Dev_FP8.safetensors")
        #if self.current_processor == "redux":
        #    base_model = os.path.join("./models/Stable-diffusion/flux1-Dev_FP8.safetensors")
        
        if self.current_processor in ["canny", "depth"]:
            pipe = FluxControlPipeline.from_single_file(
                base_model,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                vae=vae,
                torch_dtype=dtype
            )
            if use_hyper_flux:
                pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", "Hyper-FLUX.1-dev-8steps-lora.safetensors"), lora_scale=0.125)
                pipe.fuse_lora(lora_scale=0.125)
        if self.current_processor == "redux":
            pipe = FluxPipeline.from_single_file(
                base_model,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                vae=vae,
                torch_dtype=dtype    
            )
            if use_hyper_flux:
                pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", "Hyper-FLUX.1-dev-8steps-lora.safetensors"), lora_scale=0.125)
                pipe.fuse_lora(lora_scale=0.125)
        
        debug_print("\nQuantizing main transformer...", debug_enabled)
        pipe.transformer = quantize_model_to_nf4(pipe.transformer, "Transformer principal", debug_enabled)
        
        debug_print("\nEnabling memory optimizations...", debug_enabled)
        if hasattr(torch.backends, 'memory_efficient_attention'):
            torch.backends.memory_efficient_attention.enabled = True
            debug_print("Memory efficient attention enabled", debug_enabled)
        
        pipe.enable_attention_slicing()
        debug_print("Attention slicing enabled", debug_enabled)
        
        pipe.enable_model_cpu_offload()
        debug_print("Model CPU offload enabled", debug_enabled)
        
        clear_memory(debug_enabled)
        debug_print("\nModels loaded and optimized correctly", debug_enabled)
        return pipe

    def load_control_image(self, input_image):
        try:
            if input_image is not None:
                # Convertir numpy array a PIL Image
                if isinstance(input_image, np.ndarray):
                    if input_image.size == 0:
                        raise ValueError("Empty numpy array")
                    image = Image.fromarray(input_image.astype('uint8'))
                    return image.convert('RGB')  # Asegurar modo RGB
                
                # Si ya es PIL Image
                if isinstance(input_image, Image.Image):
                    return input_image.convert('RGB')
                
            # Crear imagen blanca por defecto si todo falla
            return Image.new('RGB', (512, 512), color='white')
            
        except Exception as e:
            self.logger.log(f"Error loading control image: {str(e)}")
            return Image.new('RGB', (512, 512), color='white')
            #default_path = "./extensions/sd-forge-fluxcontrolnet/assets/default.png"
            
            #if os.path.exists(default_path):
            #    return load_image(default_path)
            
            # Crear imagen blanca si no existe el archivo
            #white_image = Image.new('RGB', (512, 512), color='white')
            
            #return white_image
            
            #return load_image(default_path)
            
            #if not os.path.exists(default_path):
            #    raise FileNotFoundError(f"Default image not found at {default_path}")
            #return load_image(default_path)
        except Exception as e:
            self.logger.log(f"Error loading control image: {str(e)}")
            return Image.new('RGB', (512, 512), color='white')

    def generate(
        self, prompt, input_image, width, height, steps, guidance, 
        low_threshold, high_threshold, detect_resolution, image_resolution, 
        reference_image, debug, processor_id, seed, randomize_seed, reference_scale,
        prompt_embeds_scale_1, prompt_embeds_scale_2, pooled_prompt_embeds_scale_1, 
        pooled_prompt_embeds_scale_2, use_hyper_flux, control_image2, text_encoder, text_encoder_2, 
        tokenizer, tokenizer_2, debug_enabled, output_dir):
        try:
            debug_print("\nStarting inference...", debug_enabled)
            
            if self.pipe is None:
                debug_print("Loading models for the first time...", debug_enabled)
                self.pipe = self.load_models(use_hyper_flux=use_hyper_flux, debug_enabled=debug_enabled)
            
            control_image = self.load_control_image(input_image)
            if self.current_processor == "canny":
                menupro = "canny"
                processor = self.get_processor()
                control_image = processor(
                    control_image, 
                    low_threshold=int(low_threshold),
                    high_threshold=int(high_threshold),
                    detect_resolution=int(detect_resolution),
                    image_resolution=int(image_resolution)
                )
            if self.current_processor == "depth":
                menupro = "depth"
                processor_id = 'depth_zoe'
                processor = Processor(processor_id)
                control_image = processor(
                    control_image, 
                )
            if self.current_processor == "redux":
                control_image = control_image 
                control_image2 = control_image2
                
            if randomize_seed:
                seed = random.randint(0, 999999999)
            seed_value = int(seed) if seed is not None else 0
            
            # Log inicial de parámetros
            #self.logger.log("\nIniciando generación con parámetros:")
            #self.logger.log(f"Width: {width}")
            #self.logger.log(f"Height: {height}")
            #self.logger.log(f"Steps: {steps}")
            #self.logger.log(f"Guidance Scale: {guidance}")
            #self.logger.log(f"Seed: {seed_value}")
            
            if self.current_processor == "canny":
                with torch.inference_mode():
                    self.logger.log("Starting generation process...")
                    result = self.pipe(
                        prompt=prompt,
                        control_image=control_image,
                        height=height,
                        width=width,
                        num_inference_steps=int(steps),
                        guidance_scale=guidance,
                        generator=torch.Generator("cpu").manual_seed(seed_value)
                    )
                    self.logger.log("Generation completed")
                    
            elif self.current_processor == "depth":
                with torch.inference_mode():
                    self.logger.log("Starting generation process...")
                    result = self.pipe(
                        prompt=prompt,
                        control_image=control_image,
                        height=height,
                        width=width,
                        num_inference_steps=int(steps),
                        guidance_scale=guidance,
                        generator=torch.Generator("cpu").manual_seed(seed_value)
                    )
                    self.logger.log("Generation completed")
            
            elif self.current_processor == "redux":
                self.logger.log("Starting Redux process...")
                pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-Redux-dev",
                    text_encoder=self.pipe.text_encoder,
                    text_encoder_2=self.pipe.text_encoder_2,
                    tokenizer=self.pipe.tokenizer,
                    tokenizer_2=self.pipe.tokenizer_2,
                    torch_dtype=torch.bfloat16
                ).to("cuda")
                
                my_prompt = prompt
                my_prompt2 = prompt
                
                if control_image2 is not None:
                    self.logger.log("Processing two images...")
                    pipe_prior_output = pipe_prior_redux([control_image, control_image2], 
                                                        prompt=[my_prompt, my_prompt2],
                                                        prompt_embeds_scale=[prompt_embeds_scale_1, prompt_embeds_scale_2],
                                                        pooled_prompt_embeds_scale=[pooled_prompt_embeds_scale_1, pooled_prompt_embeds_scale_2])
                else:
                    self.logger.log("Processing one image...")
                    pipe_prior_output = pipe_prior_redux(control_image, prompt=my_prompt, prompt_embeds_scale=[prompt_embeds_scale_1],
                                                        pooled_prompt_embeds_scale=[pooled_prompt_embeds_scale_1])
                
                cond_size = 729
                hidden_size = 4096
                max_sequence_length = 512
                full_attention_size = max_sequence_length + hidden_size + cond_size
                attention_mask = torch.zeros(
                    (full_attention_size, full_attention_size),
                    dtype=torch.bfloat16
                )
                bias = torch.log(
                    torch.tensor(reference_scale, dtype=torch.bfloat16).clamp(min=1e-5, max=1)
                )
                attention_mask[:, max_sequence_length : max_sequence_length + cond_size] = bias
                joint_attention_kwargs = dict(attention_mask=attention_mask)
                
                with torch.inference_mode():
                    self.logger.log("Generating final image...")
                    result = self.pipe(
                        guidance_scale=guidance,
                        num_inference_steps=int(steps),
                        generator=torch.Generator("cpu").manual_seed(seed_value),
                        joint_attention_kwargs=joint_attention_kwargs,
                        **pipe_prior_output,
                    )
                    self.logger.log("Generation completed")
            
            debug_print("\nGeneration completed", debug_enabled)
            clear_memory(debug_enabled)

            # Guardar la imagen
            output_directory = self.output_dir
            os.makedirs(output_directory, exist_ok=True)
            timestamp = datetime.now().strftime("%y_%m_%d")
            mode_map = {
                "canny": "canny",
                "depth": "depth",
                "redux": "redux"
            }
            filename = f"{mode_map[self.current_processor]}_{seed_value}_{timestamp}.png"
            file_path = os.path.join(output_directory, filename)
            
            result_image = result.images[0]
            result_image.save(file_path)
            self.logger.log(f"Image saved in: {file_path}")
            return result.images[0]
            
        except Exception as e:
            self.logger.log(f"\nError in generation: {str(e)}")
            self.logger.log("Stacktrace:" + traceback.format_exc())
            return None

def on_ui_tabs():
    flux_tab = FluxControlNetTab()
    settings = load_settings()
    if settings:
        initial_checkpoints = settings["checkpoint_path"]
        #initial_debug = settings["debug_mode"]
    else:
        initial_checkpoints = "./models/stable-diffusion/"
        initial_debug = False 
    
    #with gr.Blocks(analytics_enabled=False) as flux_interface:
    with gr.Blocks() as flux_interface:
        with gr.Row():
            extension_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            logo_path = os.path.join('file=', extension_path, 'assets', 'logo.png')

            gr.HTML(
                f"""
                <div style="text-align: center; max-width: 650px; margin: 0 auto">
                    <h3 style="display: flex; align-items: center; justify-content: center; gap: 10px;">
                        Flux.1 Tools ControlNet by Academia SD
                        <img src="file={logo_path}" style="height: 32px; width: auto;">
                        <a href="https://www.youtube.com/@Academia_SD" target="_blank" style="text-decoration: none; display: flex; align-items: center;">
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="red">
                                <path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z"/>
                            </svg>
                        </a>
                    </h3>
                </div>
                """
            )
        with gr.Row():
            gr.Column(scale=2)
            with gr.Column(scale=1):  # Este column ocupará 1/5 del espacio
                canny_btn = gr.Button("Canny", variant="secondary")
            with gr.Column(scale=1):  # Este column ocupará 1/5 del espacio
                depth_btn = gr.Button("Depth", variant="primary")    
            with gr.Column(scale=1):  # Este column ocupará 1/5 del espacio
                redux_btn = gr.Button("Redux", variant="primary")    
            gr.Column(scale=2) 

        with gr.Row():
            
            def handle_image_upload(image):
                try:
                    if image is None:
                        return None
                    # Asegurar que la imagen está en el formato correcto
                    if isinstance(image, np.ndarray):
                        if len(image.shape) == 2:  # Si es escala de grises
                            image = np.stack([image] * 3, axis=-1)
                        elif len(image.shape) == 3 and image.shape[2] == 4:  # Si es RGBA
                            image = image[:, :, :3]  # Convertir a RGB
                    return image
                except Exception as e:
                    print(f"Error loading image: {str(e)}")
                    return None

        # Luego definimos el componente input_image con los parámetros mejorados
            input_image = gr.Image(
                label="Control Image", 
                source="upload", 
                type="numpy",
                interactive=True,
                scale=1,
                #height=512,  # Aumentamos la altura
                #width=512,   # Especificamos el ancho
                every=1,
                container=True,  # Esto ayuda con el escalado
                image_mode='RGB'
            )
            control_image2 = gr.Image(
                label="Control Image 2", 
                source="upload", 
                type="numpy",
                interactive=True,
                visible=False,
                container=True,
                image_mode='RGB',
                every=1
            )
            reference_image = gr.Image(label="Reference Image", type="pil", interactive=False)
            output_gallery = gr.Gallery(label="Generated Images ", type="pil", elem_id="generated_image", show_label=True, interactive=False)
            selected_image = gr.State() 
              
        with gr.Row():
            
            with gr.Column(scale=1):
                get_dimensions_btn = gr.Button("Get Image Dimensions")
            with gr.Column(scale=0.1):
                use_default = gr.Button("Use Default", visible=False)
            with gr.Column(scale=1):
                preprocess_btn = gr.Button("Run Preprocessor", variant="secondary", visible=True)
            with gr.Column(scale=0.1):
                send_to_control_btn = gr.Button("Send to Control", variant="secondary", visible=False)
                batch = gr.Slider(label="Batch :", minimum=1, maximum=100, value=1, step=1)
            with gr.Column(scale=1):    
                generate_btn = gr.Button("Generate", variant="primary")
      
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...")
            with gr.Column(scale=1):  
                use_hyper_flux = gr.Checkbox(label="Use LoRA Hyper-FLUX1", value=False)
            with gr.Column(scale=1):
                
                progress_bar = gr.Textbox(
                    label="Progress", 
                    value="", 
                    interactive=False,
                    show_label=True,
                    visible=True
                )
               
        with gr.Row():
            width = gr.Slider(label="Width :", minimum=256, maximum=2048, value=1024, step=16)
            height = gr.Slider(label="Height :", minimum=256, maximum=2048, value=1024, step=16)
            steps = gr.Slider(label="Inference_Steps :", minimum=1, maximum=100, value=30, step=1)
            guidance = gr.Slider(label="Guidance_Scale:", minimum=1, maximum=100, value=30, step=0.1)
            with gr.Row():
                seed = gr.Slider(label="Seed_Value :", minimum=0, maximum=9999999999, value=0, step=1)
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        with gr.Row():
            low_threshold = gr.Slider(label="Low Threshold:", minimum=0, maximum=256, value=50, step=1, visible=True)
            high_threshold = gr.Slider(label="High Threshold:", minimum=0, maximum=256, value=200, step=1, visible=True)
            detect_resolution = gr.Slider(label="Detect Resolution:", minimum=256, maximum=2048, value=1024, step=16, visible=True)
            image_resolution = gr.Slider(label="Image Resolution:", minimum=256, maximum=2048, value=1024, step=16, visible=True)
            processor_id = gr.Dropdown(
                choices=["depth_leres", "depth_leres++", "depth_midas", "depth_zoe"],
                label="Depth processor:",
                value='depth_zoe',
                visible=False
            )
            reference_scale = gr.Slider(
                info="lower to enhance prompt adherence",
                label="Masking Scale:",
                minimum=0.01,
                maximum=0.08,
                step=0.001,
                value=0.03,
                visible=False
            )
            prompt_embeds_scale_1 = gr.Slider(
                label="Prompt embeds scale 1st image",
                info=" ",
                minimum=0,
                maximum=1.5,
                step=0.01,
                value=1,
                visible=False
            )
            prompt_embeds_scale_2 = gr.Slider(
                label="Prompt embeds scale 2nd image",
                info=" ",
                minimum=0,
                maximum=1.5,
                step=0.01,
                value=1,
                visible=False
            )
            pooled_prompt_embeds_scale_1 = gr.Slider(
                label="Pooled prompt embeds scale 1nd image",
                info=" ",
                minimum=0,
                maximum=1.5,
                step=0.01,
                value=1,
                visible=False
            )
            pooled_prompt_embeds_scale_2 = gr.Slider(
                label="Pooled prompt embeds scale 2nd image:",
                info=" ",
                minimum=0,
                maximum=1.5,
                step=0.01,
                value=1,
                visible=False
            )
        with gr.Row(elem_id="image_buttons", elem_classes="image-buttons"):
            buttons = {
                'img2img': ToolButton('🖼️', elem_id='_send_to_img2img', tooltip="Send image to img2img tab."),
                'inpaint': ToolButton('🎨️', elem_id='_send_to_inpaint', tooltip="Send image to img2img inpaint tab."),
                'extras': ToolButton('📐', elem_id='_send_to_extras', tooltip="Send image to extras tab."),
            }
            for paste_tabname, paste_button in buttons.items():
                parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                    paste_button=paste_button, tabname=paste_tabname, source_tabname=None, source_image_component=output_gallery,
                    paste_field_names=[]
                ))
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                with gr.Column(scale=5):
                    hf_token = gr.Textbox(
                        label="Hugging Face Read Token:",
                        placeholder="Enter your Hugging Face token...",
                        type="password"
                    )
            with gr.Row():
                gr.Column(scale=1)
                gr.Column(scale=1)
                with gr.Column(scale=1):
                    save_settingsHF_btn = gr.Button("Save HF Token:", variant="primary")
                gr.Column(scale=1)
                gr.Column(scale=1)        
            
            settings = load_settings() or {}
            checkpoint_value = settings.get("checkpoint_path", "./models/Stable-diffusion/")
            output_value = settings.get("output_dir", "./outputs/fluxcontrolnet/")
            
            with gr.Row():
                ckpt_display = gr.Markdown(f"Latest checkpoints path: `{checkpoint_value}`")
                outp_display = gr.Markdown(f"Latest images output dir: `{output_value}`")
       
     
            with gr.Row():
                
                checkpoint_path = gr.Textbox(
                    label="Checkpoints_Path :",
                    value=checkpoint_value,  # Valor inicial desde la instancia
                    placeholder="Enter model path..."
                )
                
                #output_dir = output_dir if 'output_dir' in globals() and output_dir else "./outputs/fluxcontrolnet/"
                output_dir = gr.Textbox(
                    label="Output_Images_Path :",
                    value=output_value,  # Valor inicial desde la instancia
                    placeholder="Enter output path..."
                )
                            
            with gr.Row():
                with gr.Column(scale=1):
                    update_path_btn = gr.Button(" Update_Path: ", size="sm", value=False, visible=False)
                gr.Column(scale=1)
                with gr.Column(scale=1):
                    save_settings_btn = gr.Button("Save Settings", variant="primary")
                gr.Column(scale=1)
                with gr.Column(scale=1):
                    debug = gr.Checkbox(label="Debug Mode", value=False)
            
            with gr.Row():
                log_box = gr.Textbox(
                    label="Latest Logs",
                    interactive=False,
                    lines=6,
                    value="Waiting for operations..."
                )
                flux_tab.logger.log_box = log_box
            
            #botones
            
            # Función wrapper para actualizar la instancia al guardar
            def save_settings_wrapper(checkpoint_path_val, output_dir_val, debug_val):
                log_msg, new_cp, new_od = save_settings(checkpoint_path_val, output_dir_val, debug_val)
                flux_tab.checkpoint_path = new_cp
                flux_tab.output_dir = new_od

                
                return log_msg, new_cp, new_od
            
            save_settings_btn.click(
                
                fn=save_settings_wrapper,
                inputs=[checkpoint_path, output_dir, debug],
                outputs=[log_box, checkpoint_path, output_dir]
            )

            save_settingsHF_btn.click(
                fn=save_settingsHF,
                inputs=[hf_token],
                outputs=[log_box]
            )
       
            #use_default.click(fn=lambda: load_image("./extensions/sd-forge-fluxcontrolnet/assets/default.png"), outputs=[input_image])
            get_dimensions_btn.click(
                fn=update_dimensions,
                inputs=[input_image, width, height],
                outputs=[width, height]
            )
        def on_processor_change(mode, use_hyper_flux, input_image, low_threshold, high_threshold, detect_resolution, image_resolution, debug, processor_id):
            # Actualizar el modelo
            flux_tab.update_processor_and_model(mode)
            
            # Forzar el processor_id según el modo
            if mode == "depth":
                new_processor_id = 'depth_zoe'
            elif mode == "canny":
                new_processor_id = 'canny'
            else:
                new_processor_id = None

            # Realizar el preprocesamiento si hay una imagen
            processed_image = None
            if input_image is not None and mode != "redux":
                try:
                    if mode == "depth":
                        processor = Processor('depth_zoe')
                        temp_image = flux_tab.load_control_image(input_image)
                        processed_image = processor(temp_image)
                    elif mode == "canny":
                        processor = CannyDetector()
                        temp_image = flux_tab.load_control_image(input_image)
                        processed_image = processor(
                            temp_image,
                            low_threshold=int(low_threshold),
                            high_threshold=int(high_threshold),
                            detect_resolution=int(detect_resolution),
                            image_resolution=int(image_resolution)
                        )
                    
                    if processed_image is not None:
                        processed_image = np.array(processed_image)
                except Exception as e:
                    print(f"Error en preprocesamiento: {str(e)}")
                    processed_image = None

            if mode != "redux":
                control_image2_update = gr.update(value=None, visible=False)
            else:
                control_image2_update = gr.update(visible=True)

            # Configurar actualizaciones de UI
            if mode == "canny":
                default_steps = 30 if not use_hyper_flux else 8
                ctrl_updates = [
                    gr.update(visible=True),    # low_threshold
                    gr.update(visible=True),    # high_threshold
                    gr.update(visible=True),    # detect_resolution
                    gr.update(visible=True),    # image_resolution
                    gr.update(visible=False),   # processor_id dropdown
                    gr.update(visible=False),   # reference_scale
                    gr.update(visible=False),   # prompt_embeds_scale_1
                    gr.update(visible=False),   # prompt_embeds_scale_2
                    gr.update(visible=False),   # pooled_prompt_embeds_scale_1
                    gr.update(visible=False),   # pooled_prompt_embeds_scale_2
                    gr.update(value=default_steps),  # steps
                    gr.update(value=30),        # guidance
                    gr.update(value=processed_image, visible=True, interactive=False),  # reference_image
                    gr.update(visible=False),   # control_image2
                    control_image2_update
                ]
            elif mode == "depth":
                default_steps = 30 if not use_hyper_flux else 8
                ctrl_updates = [
                    gr.update(visible=False),   # low_threshold
                    gr.update(visible=False),   # high_threshold
                    gr.update(visible=False),   # detect_resolution
                    gr.update(visible=False),   # image_resolution
                    gr.update(visible=True, value='depth_zoe'),  # processor_id dropdown
                    gr.update(visible=False),   # reference_scale
                    gr.update(visible=False),   # prompt_embeds_scale_1
                    gr.update(visible=False),   # prompt_embeds_scale_2
                    gr.update(visible=False),   # pooled_prompt_embeds_scale_1
                    gr.update(visible=False),   # pooled_prompt_embeds_scale_2
                    gr.update(value=default_steps),  # steps
                    gr.update(value=30),        # guidance
                    gr.update(value=processed_image, visible=True, interactive=False),  # reference_image
                    gr.update(visible=False),   # control_image2
                    control_image2_update
                ]
            else:  # redux
                default_steps = 30 if not use_hyper_flux else 8
                ctrl_updates = [
                    gr.update(visible=False),   # low_threshold
                    gr.update(visible=False),   # high_threshold
                    gr.update(visible=False),   # detect_resolution
                    gr.update(visible=False),   # image_resolution
                    gr.update(visible=False),   # processor_id dropdown
                    gr.update(visible=True),    # reference_scale
                    gr.update(visible=True),    # prompt_embeds_scale_1
                    gr.update(visible=True),    # prompt_embeds_scale_2
                    gr.update(visible=True),    # pooled_prompt_embeds_scale_1
                    gr.update(visible=True),    # pooled_prompt_embeds_scale_2
                    gr.update(value=default_steps),  # steps
                    gr.update(value=3.5),       # guidance
                    gr.update(visible=False),   # reference_image
                    gr.update(visible=True),    # control_image2
                    control_image2_update
                ]

            return ctrl_updates
                    
            
        def safe_load_image(img):
            try:
                if img is None:
                    return None
                # Asegurarnos de que la imagen esté en el formato correcto
                if isinstance(img, np.ndarray):
                    # Hacer una copia para evitar problemas de memoria compartida
                    return img.copy()
                return img
            except Exception as e:
                print(f"Error loading image: {e}")
                return None

        # Separar el evento de carga del evento de preprocesamiento
        input_image.upload(
            fn=safe_load_image,
            inputs=[input_image],
            outputs=[input_image],
            queue=False
        )
        
        control_image2.upload(
            fn=safe_load_image,  # Usa la misma función safe_load_image
            inputs=[control_image2],
            outputs=[control_image2],
            queue=False
        )
        
        # Evento separado para el preprocesamiento
        input_image.change(
            fn=flux_tab.preprocess_image,
            inputs=[
                input_image, 
                low_threshold, 
                high_threshold, 
                detect_resolution, 
                image_resolution, 
                debug, 
                processor_id
            ],
            outputs=[reference_image],
            queue=False
        )
        
        low_threshold.release(
            fn=flux_tab.preprocess_image,
            inputs=[input_image, low_threshold, high_threshold, detect_resolution, image_resolution, debug, processor_id],
            outputs=[reference_image]
        )
        high_threshold.release(
            fn=flux_tab.preprocess_image,
            inputs=[input_image, low_threshold, high_threshold, detect_resolution, image_resolution, debug, processor_id],
            outputs=[reference_image]
        )
        image_resolution.release(
            fn=flux_tab.preprocess_image,
            inputs=[input_image, low_threshold, high_threshold, detect_resolution, image_resolution, debug, processor_id],
            outputs=[reference_image]
        )
        detect_resolution.release(
            fn=flux_tab.preprocess_image,
            inputs=[input_image, low_threshold, high_threshold, detect_resolution, image_resolution, debug, processor_id],
            outputs=[reference_image]
        )
        
        #use_default.click(fn=lambda: load_image("./extensions/sd-forge-fluxcontrolnet/assets/default.png"), outputs=[input_image])
        get_dimensions_btn.click(
            fn=update_dimensions,
            inputs=[input_image, width, height],
            outputs=[width, height]
        )
        preprocess_btn.click(
            fn=lambda: gr.update(interactive=False),
            outputs=[preprocess_btn]
        ).then(
            fn=flux_tab.preprocess_image,
            inputs=[
                input_image, 
                low_threshold, 
                high_threshold, 
                detect_resolution, 
                image_resolution, 
                debug, 
                processor_id
            ],
            outputs=[reference_image]
        ).then(
            fn=lambda: gr.update(interactive=True),
            outputs=[preprocess_btn]
        )
        
        update_path_btn.click(
            fn=flux_tab.preprocess_image,
            inputs=[checkpoint_path, debug],
            outputs=[checkpoint_path]
        )
        use_hyper_flux.change(
        
            fn=lambda x: gr.update(value=8 if x else 30),
            inputs=[use_hyper_flux],
            outputs=[steps]
        )
        
        def update_preprocessing(input_image, low_threshold, high_threshold, detect_resolution, image_resolution, debug, processor_id):
            if input_image is not None:
                return flux_tab.preprocess_image(input_image, low_threshold, high_threshold, detect_resolution, image_resolution, debug, processor_id)
            return None
            
        for slider in [low_threshold, high_threshold, detect_resolution, image_resolution]:
            slider.release(
                fn=update_preprocessing,
                inputs=[input_image, low_threshold, high_threshold, detect_resolution, image_resolution, debug, processor_id],
                outputs=[reference_image]
            )
        def pre_generate():
            return gr.Button.update(value="Generating...", variant="secondary", interactive=False)
        def post_generate(result):
            return result, gr.Button.update(value="Generate", variant="primary", interactive=True)
        # Se ha modificado la función generate_with_state para quitar los parámetros innecesarios
        def generate_with_state(
            prompt, input_image, width, height, steps, guidance,
            low_threshold, high_threshold, detect_resolution, image_resolution,
            reference_image, debug, processor_id, seed, randomize_seed, reference_scale,
            prompt_embeds_scale_1, prompt_embeds_scale_2, pooled_prompt_embeds_scale_1,
            pooled_prompt_embeds_scale_2, use_hyper_flux, control_image2, batch
        ):
            try:
                results = []
                # Primer update con lista vacía e inicialización del progreso
                yield results, flux_tab.logger.log("Starting batch generation..."), "Starting..."
                
                for i in range(int(batch)):
                    # Actualizar progreso de batch
                    batch_progress = f"Processing image {i+1} of {batch}"
                    yield results, flux_tab.logger.log(batch_progress), batch_progress
                    
                    local_seed = random.randint(0, 999999999) if randomize_seed else seed
                    result = flux_tab.generate(
                        prompt=prompt,
                        input_image=input_image,
                        width=width,
                        height=height,
                        steps=steps,
                        guidance=guidance,
                        low_threshold=low_threshold,
                        high_threshold=high_threshold,
                        detect_resolution=detect_resolution,
                        image_resolution=image_resolution,
                        reference_image=reference_image,
                        debug=debug,
                        processor_id=processor_id,
                        seed=local_seed,
                        randomize_seed=randomize_seed,
                        reference_scale=reference_scale,
                        prompt_embeds_scale_1=prompt_embeds_scale_1,
                        prompt_embeds_scale_2=prompt_embeds_scale_2,
                        pooled_prompt_embeds_scale_1=pooled_prompt_embeds_scale_1,
                        pooled_prompt_embeds_scale_2=pooled_prompt_embeds_scale_2,
                        text_encoder=None,
                        text_encoder_2=None,
                        tokenizer=None,
                        tokenizer_2=None,
                        debug_enabled=debug,
                        use_hyper_flux=use_hyper_flux,
                        control_image2=control_image2,
                        output_dir=output_dir
                    )
                    
                    if result is not None:
                        results.append(result)
                        # Actualizar galería y progreso después de cada imagen generada
                        complete_msg = f"Image {i+1}/{batch} completed"
                        yield results, flux_tab.logger.log(complete_msg), complete_msg
                
                # Mensaje final
                final_msg = "Batch generation completed!"
                yield results, flux_tab.logger.log(final_msg), final_msg
        
            except Exception as e:
                error_msg = f"Error in generation: {str(e)}"
                yield None, flux_tab.logger.log(error_msg), "Error: " + str(e)
                
        #progress_bar = gr.Textbox(
        #    label="Progress", 
        #    value="", 
        #    interactive=False,
        #    show_label=True,
        #    visible=True
        #)

        generate_btn.click(
            fn=pre_generate,
            inputs=None,
            outputs=[generate_btn],
        ).then(
            fn=generate_with_state,
            inputs=[
                prompt, input_image, width, height, steps, guidance,
                low_threshold, high_threshold, detect_resolution, image_resolution,
                reference_image, debug, processor_id, seed, randomize_seed, reference_scale,
                prompt_embeds_scale_1, prompt_embeds_scale_2, pooled_prompt_embeds_scale_1,
                pooled_prompt_embeds_scale_2, use_hyper_flux, control_image2, batch
            ],
            outputs=[output_gallery, log_box, progress_bar],
            show_progress=True
        ).then(
            fn=post_generate,
            inputs=[output_gallery],
            outputs=[output_gallery, generate_btn]
        )
        
        #------------------
        
        output_gallery.select(
            fn=lambda evt: evt,  # Captura la imagen seleccionada
            outputs=[selected_image]
        )
        
        send_to_control_btn.click(
            #fn=lambda generated: generated,
            #inputs=[output_gallery],
            #outputs=[input_image]
        #    output_gallery.select(
        #    fn=lambda evt: evt,
        #    inputs=[output_gallery],
        #    outputs=[input_image]
            fn=lambda img: img,
            inputs=[selected_image],
            outputs=[input_image]
        
        )
        
#)      
        depth_btn.click(
            # Primero actualizamos solo los colores de los botones
            fn=lambda: (
                gr.Button.update(variant="primary"),    # canny
                gr.Button.update(variant="secondary"),  # depth
                gr.Button.update(variant="primary"),    # redux
            ),
            outputs=[canny_btn, depth_btn, redux_btn]
        ).then(
            # Luego hacemos el resto del procesamiento
            fn=lambda use_hyper, img, lt, ht, dr, ir, debug, pid: on_processor_change(
                "depth", use_hyper, img, lt, ht, dr, ir, debug, pid
            ),
            inputs=[
                use_hyper_flux, input_image, low_threshold, high_threshold,
                detect_resolution, image_resolution, debug, processor_id
            ],
            outputs=[
                low_threshold, high_threshold, detect_resolution, image_resolution,
                processor_id, reference_scale,
                prompt_embeds_scale_1, prompt_embeds_scale_2,
                pooled_prompt_embeds_scale_1, pooled_prompt_embeds_scale_2,
                steps, guidance,
                reference_image, control_image2
            ]
        )
        
        # Manejador para el botón de Canny
        canny_btn.click(
            # Primero actualizamos solo los colores de los botones
            fn=lambda: (
                gr.Button.update(variant="secondary"),  # canny
                gr.Button.update(variant="primary"),    # depth
                gr.Button.update(variant="primary"),    # redux
            ),
            outputs=[canny_btn, depth_btn, redux_btn]
        ).then(
            # Luego hacemos el resto del procesamiento
            fn=lambda use_hyper, img, lt, ht, dr, ir, debug, pid: on_processor_change(
                "canny", use_hyper, img, lt, ht, dr, ir, debug, pid
            ),
            inputs=[
                use_hyper_flux, input_image, low_threshold, high_threshold,
                detect_resolution, image_resolution, debug, processor_id
            ],
            outputs=[
                low_threshold, high_threshold, detect_resolution, image_resolution,
                processor_id, reference_scale,
                prompt_embeds_scale_1, prompt_embeds_scale_2,
                pooled_prompt_embeds_scale_1, pooled_prompt_embeds_scale_2,
                steps, guidance,
                reference_image, control_image2
            ]
        )

        # Manejador para el botón de Redux
        redux_btn.click(
            # Primero actualizamos solo los colores de los botones
            fn=lambda: (
                gr.Button.update(variant="primary"),    # canny
                gr.Button.update(variant="primary"),    # depth
                gr.Button.update(variant="secondary"),  # redux
            ),
            outputs=[canny_btn, depth_btn, redux_btn]
        ).then(
            # Luego hacemos el resto del procesamiento
            fn=lambda use_hyper, img, lt, ht, dr, ir, debug, pid: on_processor_change(
                "redux", use_hyper, img, lt, ht, dr, ir, debug, pid
            ),
            inputs=[
                use_hyper_flux, input_image, low_threshold, high_threshold,
                detect_resolution, image_resolution, debug, processor_id
            ],
            outputs=[
                low_threshold, high_threshold, detect_resolution, image_resolution,
                processor_id, reference_scale,
                prompt_embeds_scale_1, prompt_embeds_scale_2,
                pooled_prompt_embeds_scale_1, pooled_prompt_embeds_scale_2,
                steps, guidance,
                reference_image, control_image2
            ]
        )


    return [(flux_interface, "Flux.1 Tools", "flux_controlnet_tab")]

script_callbacks.on_ui_tabs(on_ui_tabs)
