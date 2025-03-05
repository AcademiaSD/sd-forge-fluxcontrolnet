import gradio as gr
import torch
from diffusers import FluxControlPipeline, FluxControlNetModel, AutoencoderKL
from diffusers import FluxPriorReduxPipeline, FluxPipeline
from diffusers import FluxFillPipeline
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
import glob
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
import io
import logging
import sys
from h11._util import LocalProtocolError
from modules_forge.forge_canvas.canvas import ForgeCanvas, canvas_head
from PIL import Image, ImageDraw, ImageChops

# Verificar CUDA (GPU NVIDIA)
has_cuda = torch.cuda.is_available()

# Verificar MPS (GPU Apple M1/M2/M3)
has_mps = hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available') and torch.mps.is_available()

# Determinar el mejor dispositivo disponible
if has_cuda:
    device = torch.device("cuda")
elif has_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

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

def verify_huggingface_token(token):
    """Verifica si el token de Hugging Face es válido."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # Intenta una operación simple para verificar el token
        api.whoami(token=token)
        return True
    except Exception as e:
        print(f"Error al verificar token de HF: {str(e)}")
        return False

def check_existing_token():
    """Verifica si ya existe un token válido."""
    token = load_huggingface_token()
    if token and verify_huggingface_token(token):
        return "✅ You have a correct HF Token!"
    return ""


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



def list_lora_files(lora_dir="./models/lora/"):
    if not os.path.exists(lora_dir):
        os.makedirs(lora_dir, exist_ok=True)
        
    lora_files = []
    # Recorrer todas las subcarpetas
    for root, dirs, files in os.walk(lora_dir):
        for file in files:
            if file.endswith('.safetensors'):
                # Calcular la ruta relativa respecto a lora_dir
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, lora_dir)
                lora_files.append(rel_path)
                
    return ["None"] + sorted(lora_files)

def debug_print(message, debug_enabled=False):
    if debug_enabled:
        print(message)
        
def create_generator(seed_value):
    """Crea un generador compatible con el dispositivo actual."""
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.Generator(device="cuda").manual_seed(seed_value)
    elif device.type == "mps" and hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available') and torch.mps.is_available():
        # MPS no soporta generadores directamente, usamos el de CPU
        return torch.Generator("cpu").manual_seed(seed_value)
    else:
        return torch.Generator("cpu").manual_seed(seed_value)

def print_memory_usage(message="", debug_enabled=False):
    if not debug_enabled:
        return
        
    if device.type == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"{message} GPU Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")
    elif device.type == "mps" and hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available') and torch.mps.is_available():
        # MPS (Metal Performance Shaders para Apple Silicon) tiene diferentes APIs
        if hasattr(torch.mps, 'current_allocated_memory'):
            allocated = torch.mps.current_allocated_memory() / 1024**2
            print(f"{message} MPS Memory: {allocated:.2f}MB allocated")
        else:
            print(f"{message} MPS Memory: usage stats unavailable")
    elif device.type == "cpu":
        print(f"{message} Using CPU (no memory statistics available)")

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
    if device.type == "cuda" and torch.cuda.is_available():
        print_memory_usage("Before cleaning:", debug_enabled)
        torch.cuda.empty_cache()
        gc.collect()
    elif device.type == "mps" and hasattr(torch, 'mps') and torch.mps.is_available():
        # MPS también puede necesitar limpieza específica
        gc.collect()
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    else:
        # Para CPU, solo podemos hacer la recolección de basura
        gc.collect()

def update_dimensions(image, canvas_background, current_mode, width_slider, height_slider):
    """Actualiza las dimensiones basándose en la imagen activa según el modo"""
    
    # Determinar qué imagen usar según el modo
    actual_image = canvas_background if current_mode == "fill" else image
    
    if actual_image is not None:
        if isinstance(actual_image, np.ndarray):
            height, width = actual_image.shape[:2]
            return width, height
        elif hasattr(actual_image, "size"):
            # Para imágenes PIL
            width, height = actual_image.size
            return width, height
    
    # Si no hay imagen, mantener los valores de los sliders
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
            "output_dir": "./outputs/fluxcontrolnet/",
            "lora_dir": "./models/lora/",  # Valor por defecto para loras
            "text_encoders_path": "./models/Stable-diffusion/text_encoders_FP8"  # Default for text encoders
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
        
        # Verificar si el token es válido utilizando la nueva función
        if verify_huggingface_token(hf_token_value):
            return f"✅ HF Token saved and verified!", "✅ You have a correct HF Token!"
        else:
            return f"⚠️ HF Token saved but could not be verified!", ""
            
    except Exception as e:
        return f"❌ Error saving HF Token: {str(e)}", ""

def save_settings(checkpoints_path, output_dir, lora_dir, text_encoders_path, debug_enabled=False):
    try:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        extension_dir = os.path.join(root_dir, "extensions", "sd-forge-fluxcontrolnet", "utils")
        os.makedirs(extension_dir, exist_ok=True)
        config_path = os.path.join(extension_dir, "extension_config.txt")
        config = {
            "checkpoint_path": checkpoints_path,
            "output_dir": output_dir,
            "lora_dir": lora_dir,
            "text_encoders_path": text_encoders_path
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        log_message = "Settings file saved"
        return log_message, checkpoints_path, output_dir, lora_dir, text_encoders_path
    except Exception as e:
        error_message = f"Error saving settings file: {str(e)}"
        return error_message, checkpoints_path, output_dir, lora_dir, text_encoders_path
        
class FluxControlNetTab:
    def __init__(self):
        self.pipe = None
        #self.model_path = "Academia-SD/flux1-dev-text_encoders-NF4
        self.model_path = "./models/diffusers/text_encoders_FP8"  # Default path
        self.text_encoders_path = "./models/Stable-diffusion/text_encoders_FP8"  # New default path
        

        self.current_processor = "canny"
        self.current_model = "flux1-Canny-Dev_FP8.safetensors"
        self.checkpoint_path = "./models/Stable-diffusion/"
        self.output_dir = "./outputs/fluxcontrolnet/"
        self.lora_dir = "./models/lora/"  # Añadimos la ruta de loras
        #self.default_image_path = "./extensions/sd-forge-fluxcontrolnet/assets/default.png" 
        self.default_processor_id = "depth_zoe"
        self.logger = LogManager()
        # Atributos para los LoRAs
        self.lora1_model = None
        self.lora1_scale = 1.0
        self.lora2_model = None
        self.lora2_scale = 1.0
        self.lora3_model = None
        self.lora3_scale = 1.0
        # Atributos para rastrear el estado del pipeline
        self.loaded_processor = None
        self.loaded_hyper_flux = None
        self.loaded_lora1 = None
        self.loaded_lora1_scale = None
        self.loaded_lora2 = None
        self.loaded_lora2_scale = None
        self.loaded_lora3 = None
        self.loaded_lora3_scale = None
        settings = load_settings()
        if settings:
        
            self.checkpoint_path = settings.get("checkpoint_path", "./models/Stable-diffusion/")
            self.output_dir = settings.get("output_dir", "./outputs/fluxcontrolnet/")
            self.lora_dir = settings.get("lora_dir", "./models/lora/")  # Cargar la ruta de loras
            self.text_encoders_path = settings.get("text_encoders_path", "./models/Stable-diffusion/text_encoders_FP8")  # Load from settings
        else:
            self.checkpoint_path = "./models/Stable-diffusion/"
            self.output_dir = "./outputs/fluxcontrolnet/"
            self.lora_dir = "./models/lora/"
            self.text_encoders_path = "./models/Stable-diffusion/text_encoders_FP8"  # Default path
            
    def expand_canvas(self, background_image, expand_up=False, expand_down=False, expand_left=False, expand_right=False, expand_range="128"):
        """
        Expande el lienzo de la imagen en las direcciones seleccionadas, añadiendo píxeles
        de espacio en blanco en cada dirección marcada según el rango seleccionado.
        """
        try:
            self.logger.log("Expanding canvas...")
            
            # Convertir el rango a un número entero
            expand_pixels = int(expand_range)
            self.logger.log(f"Expansion range: {expand_pixels} pixels")
            
            # Convertir a PIL Image si es un numpy array
            if background_image is None:
                self.logger.log("No image to expand")
                return None, None
                
            if isinstance(background_image, np.ndarray):
                pil_img = Image.fromarray(background_image.astype('uint8'))
            else:
                pil_img = background_image
                
            # Obtener dimensiones originales
            width, height = pil_img.size
            self.logger.log(f"Original size: {width}x{height}")
            
            # Calcular nuevas dimensiones
            new_width = width + (expand_pixels if expand_left else 0) + (expand_pixels if expand_right else 0)
            new_height = height + (expand_pixels if expand_up else 0) + (expand_pixels if expand_down else 0)
            
            # Si no hay cambios, devolver la imagen original
            if new_width == width and new_height == height:
                self.logger.log("Inpaint without expansion")
                return pil_img, Image.new('L', (width, height), 0)  # Máscara vacía
            
            self.logger.log(f"New size: {new_width}x{new_height}")
            
            # Crear nuevo lienzo (con fondo blanco)
            new_img = Image.new('RGB', (new_width, new_height), (255, 255, 255))
            
            # Calcular la posición para pegar la imagen original
            paste_x = expand_pixels if expand_left else 0
            paste_y = expand_pixels if expand_up else 0
            
            # Pegar la imagen original en el nuevo lienzo
            new_img.paste(pil_img, (paste_x, paste_y))
            
            # Crear una máscara correspondiente (blanca en las áreas nuevas)
            mask = Image.new('L', (new_width, new_height), 0)  # Inicialmente toda negra
            
            # Marcar las áreas expandidas como blancas (áreas a procesar)
            draw = ImageDraw.Draw(mask)
            
            # Arriba
            if expand_up:
                draw.rectangle([0, 0, new_width, paste_y - 1], fill=255)
                self.logger.log(f"Expanding {expand_pixels}px to up")
                
            # Abajo
            if expand_down:
                draw.rectangle([0, height + paste_y, new_width, new_height], fill=255)
                self.logger.log(f"Expanding {expand_pixels}px to down")
                
            # Izquierda
            if expand_left:
                draw.rectangle([0, 0, paste_x - 1, new_height], fill=255)
                self.logger.log(f"Expanding {expand_pixels}px to left")
                
            # Derecha
            if expand_right:
                draw.rectangle([width + paste_x, 0, new_width, new_height], fill=255)
                self.logger.log(f"Expanding {expand_pixels}px to right")
            
            self.logger.log("Canvas expansion completed")
            return new_img, mask
            
        except Exception as e:
            error_msg = f"Error expanding canvas: {str(e)}"
            self.logger.log(error_msg)
            # En caso de error, devolver la imagen original sin cambios
            if isinstance(background_image, np.ndarray):
                return Image.fromarray(background_image.astype('uint8')), None
            return background_image, None
                    
            
    def fill_canvas_to_mask(self, background, foreground, expand_up, expand_down, expand_left, expand_right, expand_range, mask_blur):
        """
        Transfiere tanto la máscara como la imagen base del canvas, expandiendo según las direcciones seleccionadas.
        Aplica blur a los bordes de la máscara para mejorar las transiciones.
        """
        # Variables para almacenar resultados
        mask = None
        new_width = 0
        new_height = 0
        
        # 1. Primero, procesar la máscara dibujada por el usuario (si existe)
        if foreground is not None:
            if isinstance(foreground, np.ndarray):
                user_mask = Image.fromarray(foreground).convert('L')
            else:
                user_mask = foreground.convert('L')
            
            # Inicializamos la máscara con la del usuario
            mask = user_mask
        else:
            # Si no hay máscara de usuario, crear una vacía
            if background is not None:
                if isinstance(background, np.ndarray):
                    h, w = background.shape[:2]
                    user_mask = Image.new('L', (w, h), 0)  # Máscara vacía
                else:
                    w, h = background.size
                    user_mask = Image.new('L', (w, h), 0)  # Máscara vacía
                mask = user_mask
        
        # 2. Luego, expandir el lienzo si se ha seleccionado alguna dirección
        if (expand_up or expand_down or expand_left or expand_right) and background is not None:
            # Convertir el rango a un número entero
            expand_pixels = int(expand_range)
            self.logger.log(f"Expansion range: {expand_pixels} pixels")
            
            # Convertir background a PIL Image si es un numpy array
            if isinstance(background, np.ndarray):
                bg_img = Image.fromarray(background.astype('uint8'))
            else:
                bg_img = background
                
            # Obtener dimensiones originales
            original_width, original_height = bg_img.size
            self.logger.log(f"Original dimensions: {original_width}x{original_height}")
            
            # Calcular nuevas dimensiones
            new_width = original_width + (expand_pixels if expand_left else 0) + (expand_pixels if expand_right else 0)
            new_height = original_height + (expand_pixels if expand_up else 0) + (expand_pixels if expand_down else 0)
            
            # Si hay cambios en las dimensiones
            if new_width != original_width or new_height != original_height:
                self.logger.log(f"New dimensions: {new_width}x{new_height}")
                
                # Crear nuevo lienzo (con fondo blanco)
                new_img = Image.new('RGB', (new_width, new_height), (255, 255, 255))
                
                # Calcular la posición para pegar la imagen original
                paste_x = expand_pixels if expand_left else 0
                paste_y = expand_pixels if expand_up else 0
                
                # Pegar la imagen original en el nuevo lienzo
                new_img.paste(bg_img, (paste_x, paste_y))
                
                # Crear una nueva máscara del tamaño extendido
                extended_mask = Image.new('L', (new_width, new_height), 0)  # Inicialmente toda negra
                
                # Pegar la máscara del usuario en la posición correcta de la máscara extendida
                if mask is not None:
                    extended_mask.paste(mask, (paste_x, paste_y))
                
                # Marcar las áreas expandidas como blancas (áreas a procesar)
                draw = ImageDraw.Draw(extended_mask)
                
                # Arriba
                if expand_up:
                    draw.rectangle([0, 0, new_width, paste_y - 1], fill=255)
                    self.logger.log(f"Expanding {expand_pixels}px to UP")
                    
                # Abajo
                if expand_down:
                    draw.rectangle([0, original_height + paste_y, new_width, new_height], fill=255)
                    self.logger.log(f"Expanding {expand_pixels}px to DOWN")
                    
                # Izquierda
                if expand_left:
                    draw.rectangle([0, 0, paste_x - 1, new_height], fill=255)
                    self.logger.log(f"Expanding {expand_pixels}px to LEFT")
                    
                # Derecha
                if expand_right:
                    draw.rectangle([original_width + paste_x, 0, new_width, new_height], fill=255)
                    self.logger.log(f"Expanding {expand_pixels}px to RIGHT")
                
                # Actualizar la imagen de fondo y la máscara
                background = new_img
                mask = extended_mask
                
                self.logger.log("Canvas expansion completed")
            else:
                # Si no hay cambio en dimensiones, usar las originales
                new_width = original_width
                new_height = original_height
        else:
            # Si no hay expansión, usar las dimensiones originales
            if background is not None:
                if isinstance(background, np.ndarray):
                    new_height, new_width = background.shape[:2]
                else:
                    new_width, new_height = background.size
        
        # 3. Aplicar blur a la máscara si se especificó un valor de blur
        if mask is not None and mask_blur > 0:
            self.logger.log(f"Applying {mask_blur} pixel blur to the mask")
            try:
                # Primero convertimos a numpy para mayor flexibilidad
                mask_np = np.array(mask)
                
                # Aplicar filtro gaussiano para suavizar los bordes
                from scipy.ndimage import gaussian_filter
                
                # Crear una versión borrosa de la máscara
                blurred_mask = gaussian_filter(mask_np.astype(float), sigma=mask_blur/2)
                
                # Normalizar a 0-255
                blurred_mask = (blurred_mask / blurred_mask.max() * 255).astype(np.uint8)
                
                # Crear una nueva imagen PIL con la máscara borrosa
                mask = Image.fromarray(blurred_mask)
                
                self.logger.log("Blur applied correctly to the mask")
            except Exception as e:
                self.logger.log(f"Error applying blur to mask: {str(e)}")
                # Continuar con la máscara original si hay un error
        
        # IMPORTANTE: Ahora devolvemos la máscara, el fondo y las nuevas dimensiones
        return mask, background, new_width, new_height
    
    
            
    def switch_mode(self, new_mode, use_hyper_flux=False):
        """Cambia al modo seleccionado y actualiza la interfaz."""
        # Actualizar modo actual
        old_mode = self.current_processor
        self.current_processor = new_mode
        
        # Establecer modelo correspondiente
        if new_mode == "canny":
            self.current_model = "flux1-Canny-Dev_FP8.safetensors"
            self.default_processor_id = None
        elif new_mode == "depth":
            self.current_model = "flux1-Depth-Dev_FP8.safetensors"
            self.default_processor_id = "depth_zoe"
        elif new_mode == "redux":
            self.current_model = "flux1-Dev_FP8.safetensors" 
            self.default_processor_id = None
        elif new_mode == "fill":
            self.current_model = "flux1-Fill-Dev_FP8.safetensors"
            self.default_processor_id = None
        
        # Limpiar pipeline si es necesario
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            clear_memory()
        
        # Actualizaciones de botones
        canny_update = gr.Button.update(variant="secondary" if new_mode == "canny" else "primary", visible=True)
        depth_update = gr.Button.update(variant="secondary" if new_mode == "depth" else "primary", visible=True)
        redux_update = gr.Button.update(variant="secondary" if new_mode == "redux" else "primary", visible=True)
        fill_update = gr.Button.update(variant="secondary" if new_mode == "fill" else "primary", visible=True)
        
        # Actualizaciones comunes
        steps_update = gr.update(value=10 if use_hyper_flux else 30)
        
        # CLAVE: Establecer la visibilidad de reference_image
        reference_image_visible = new_mode in ["canny", "depth"]
        
        # Devolver lista de actualizaciones según el modo
        if new_mode == "canny":
            return [
                canny_update,                   # canny_btn
                depth_update,                   # depth_btn
                redux_update,                   # redux_btn
                fill_update,                    # fill_btn
                gr.update(visible=True, label="Control Image"), # input_image
                gr.update(visible=False),       # fill_canvas_group
                new_mode,                       # current_mode
                gr.update(visible=True),        # low_threshold
                gr.update(visible=True),        # high_threshold
                gr.update(visible=True),        # detect_resolution
                gr.update(visible=True),        # image_resolution
                gr.update(visible=False),       # processor_id
                gr.update(visible=False),       # reference_scale
                gr.update(visible=False),       # prompt_embeds_scale_1
                gr.update(visible=False),       # prompt_embeds_scale_2
                gr.update(visible=False),       # pooled_prompt_embeds_scale_1
                gr.update(visible=False),       # pooled_prompt_embeds_scale_2
                steps_update,                   # steps
                gr.update(value=30),            # guidance
                gr.update(visible=False),       # control_image2
                gr.update(visible=False),       # prompt2
                gr.update(visible=False),       # mask_image
                gr.update(visible=False),       # fill_controls
                gr.update(visible=False),       # transfer_mask_btn
                gr.update(visible=True)         # reference_image - VISIBLE
            ]
        elif new_mode == "depth":
            return [
                canny_update,                   # canny_btn
                depth_update,                   # depth_btn
                redux_update,                   # redux_btn
                fill_update,                    # fill_btn
                gr.update(visible=True, label="Control Image"), # input_image
                gr.update(visible=False),       # fill_canvas_group
                new_mode,                       # current_mode
                gr.update(visible=False),       # low_threshold
                gr.update(visible=False),       # high_threshold
                gr.update(visible=False),       # detect_resolution
                gr.update(visible=False),       # image_resolution
                gr.update(visible=True),        # processor_id
                gr.update(visible=False),       # reference_scale
                gr.update(visible=False),       # prompt_embeds_scale_1
                gr.update(visible=False),       # prompt_embeds_scale_2
                gr.update(visible=False),       # pooled_prompt_embeds_scale_1
                gr.update(visible=False),       # pooled_prompt_embeds_scale_2
                steps_update,                   # steps
                gr.update(value=30),            # guidance
                gr.update(visible=False),       # control_image2
                gr.update(visible=False),       # prompt2
                gr.update(visible=False),       # mask_image
                gr.update(visible=False),       # fill_controls
                gr.update(visible=False),       # transfer_mask_btn
                gr.update(visible=True, value=None)         # reference_image - VISIBLE
            ]
        elif new_mode == "redux":
            return [
                canny_update,                   # canny_btn
                depth_update,                   # depth_btn
                redux_update,                   # redux_btn
                fill_update,                    # fill_btn
                gr.update(visible=True, label="Control Image"), # input_image
                gr.update(visible=False),       # fill_canvas_group
                new_mode,                       # current_mode
                gr.update(visible=False),       # low_threshold
                gr.update(visible=False),       # high_threshold
                gr.update(visible=False),       # detect_resolution
                gr.update(visible=False),       # image_resolution
                gr.update(visible=False),       # processor_id
                gr.update(visible=True),        # reference_scale
                gr.update(visible=True),        # prompt_embeds_scale_1
                gr.update(visible=True),        # prompt_embeds_scale_2
                gr.update(visible=True),        # pooled_prompt_embeds_scale_1
                gr.update(visible=True),        # pooled_prompt_embeds_scale_2
                steps_update,                   # steps
                gr.update(value=3.5),           # guidance
                gr.update(visible=True),        # control_image2
                gr.update(visible=True),        # prompt2
                gr.update(visible=False),       # mask_image
                gr.update(visible=False),       # fill_controls
                gr.update(visible=False),       # transfer_mask_btn
                gr.update(visible=False)        # reference_image - OCULTO
            ]
        elif new_mode == "fill":
            return [
                canny_update,                   # canny_btn
                depth_update,                   # depth_btn
                redux_update,                   # redux_btn
                fill_update,                    # fill_btn
                gr.update(visible=False),       # input_image
                gr.update(visible=True),        # fill_canvas_group
                new_mode,                       # current_mode
                gr.update(visible=False),       # low_threshold
                gr.update(visible=False),       # high_threshold
                gr.update(visible=False),       # detect_resolution
                gr.update(visible=False),       # image_resolution
                gr.update(visible=False),       # processor_id
                gr.update(visible=False),       # reference_scale
                gr.update(visible=False),       # prompt_embeds_scale_1
                gr.update(visible=False),       # prompt_embeds_scale_2
                gr.update(visible=False),       # pooled_prompt_embeds_scale_1
                gr.update(visible=False),       # pooled_prompt_embeds_scale_2
                steps_update,                   # steps
                gr.update(value=30),           # guidance
                gr.update(visible=False),       # control_image2
                gr.update(visible=False),       # prompt2
                gr.update(visible=True),        # mask_image
                gr.update(visible=True),        # fill_controls
                gr.update(visible=True),        # transfer_mask_btn
                gr.update(visible=True)        # reference_image - OCULTO True solo para test
            ]
            
    def prepare_mask_image(self, mask_image, for_visualization=False):
        """Procesa una imagen de máscara para inpainting/outpainting."""
        try:
            if mask_image is None:
                return None
                
            # Convertir a imagen PIL si es numpy array
            # if isinstance(mask_image, np.ndarray):
                # # Si tiene canal alpha (RGBA), usar ese como máscara
                # if mask_image.shape[-1] == 4:
                    # # Extraer el canal alpha
                    # alpha = mask_image[:, :, 3]
                    # mask = Image.fromarray(alpha)
                # else:
                    # # Convertir a escala de grises
                    # mask = Image.fromarray(mask_image.astype('uint8')).convert('L')
            # else:
                # # Si ya es una imagen PIL
                # mask = mask_image.convert('L')
                
            # Binarizar la máscara: blanco (255) donde se debe aplicar inpainting
            threshold = 128
            mask = mask.point(lambda p: 255 if p > threshold else 0)
            
            # Versión de numpy para procesamiento
            processed_mask = np.array(mask)
            
            # Para visualización, convertir a RGB
            if for_visualization and len(processed_mask.shape) == 2:
                visual_mask = np.zeros((*processed_mask.shape, 3), dtype=np.uint8)
                visual_mask[processed_mask > 0] = [255, 255, 255]
                return visual_mask
            
            # Para el modelo, devolver directamente la máscara en escala de grises
            return mask  # Devolver la imagen PIL en escala de grises
            
        except Exception as e:
            self.logger.log(f"Error processing mask: {str(e)}")
            return None
            
    # def update_reference_from_canvas(self, canvas_background):
        # """Transfiere la imagen del canvas a reference_image automáticamente."""
        # try:
            # if canvas_background is not None:
                # # Procesar como imagen si es necesario
                # return canvas_background
            # return None
        # except Exception as e:
            # self.logger.log(f"Error al actualizar reference_image: {str(e)}")
            # return None
            
    def toggle_reference_visibility(self, visible, current_processor):
        new_visible = not visible        
        button_text = "🙈 Hide" if new_visible else "👁️ Show"
                
        if self.current_processor == "canny" or self.current_processor == "depth":
            reference_image_update = gr.update(visible=new_visible)
            control_image2_update = gr.update(visible=False)
            prompt2_update = gr.update(visible=False)
            mask_image_update = gr.update(visible=False)
        elif self.current_processor == "redux":
            reference_image_update = gr.update(visible=False)
            control_image2_update = gr.update(visible=new_visible)
            prompt2_update = gr.update(visible=new_visible)
            mask_image_update = gr.update(visible=False)
        elif self.current_processor == "fill":
            reference_image_update = gr.update(visible=new_visible)
            control_image2_update = gr.update(visible=False)
            prompt2_update = gr.update(visible=False)
            mask_image_update = gr.update(visible=new_visible)
        
        # IMPORTANTE: Añadir el botón actualizado de nuevo
        button_update = gr.Button.update(value=button_text, variant="primary")
        
        return (
            new_visible,
            reference_image_update,
            control_image2_update,
            button_update,  # Este valor es necesario para toggle_reference_btn
            prompt2_update,
            mask_image_update
        )

    def update_model_path(self, new_path, debug_enabled):
        if new_path and os.path.exists(new_path):
            self.model_path = new_path
            debug_print(f"Updated model path: {new_path}", debug_enabled)
            if self.pipe is not None:
                del self.pipe
                self.pipe = None
                clear_memory()
        return self.model_path

    # def update_processor_and_model(self, processor_type):
        # if processor_type == "canny":
            # self.current_processor = "canny"
            # self.current_model = "flux1-Canny-Dev_FP8.safetensors"
            # self.default_processor_id = None
        # elif processor_type == "depth":
            # self.current_processor = "depth"
            # self.current_model = "flux1-Depth-Dev_FP8.safetensors"
            # self.default_processor_id = "depth_zoe"
        # elif processor_type == "redux":
            # self.current_processor = "redux"
            # self.current_model = "flux1-Dev_FP8.safetensors"
            # self.default_processor_id = None
        # elif processor_type == "fill":
            # self.current_processor = "fill"
            # self.current_model = "flux1-Fill-Dev_FP8.safetensors"
            # self.default_processor_id = None
        
        # if self.pipe is not None:
            # del self.pipe
            # self.pipe = None
            # clear_memory()
        
        # # Asegurar que todos los botones están visibles y actualizar variantes
        # return [
            # gr.Button.update(variant="secondary" if processor_type == "canny" else "primary", visible=True),
            # gr.Button.update(variant="secondary" if processor_type == "depth" else "primary", visible=True),
            # gr.Button.update(variant="secondary" if processor_type == "redux" else "primary", visible=True),
            # gr.Button.update(variant="secondary" if processor_type == "fill" else "primary", visible=True)
        # ]

    def get_processor(self):
        if self.current_processor == "canny":
            return CannyDetector()
        elif self.current_processor == "depth":
            return
        elif self.current_processor == "redux":
            return 
        elif self.current_processor == "fill":
            return
        return # CannyDetector()  # Default to Canny

    def preprocess_image(self, input_image, low_threshold, high_threshold, detect_resolution, image_resolution, debug_enabled, processor_id=None, width=None, height=None):
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
                
            # Guardar las dimensiones originales o usar las especificadas
            original_width, original_height = control_image.size
            target_width = width if width is not None else original_width
            target_height = height if height is not None else original_height
                
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
                from PIL import Image  # Importar aquí para evitar errores
                actual_processor_id = processor_id or 'depth_zoe'
                processor = Processor(actual_processor_id)
                processed_image = processor(control_image)
                
                # Redimensionar usando BICUBIC para mejor calidad
                if isinstance(processed_image, Image.Image):
                    processed_image = processed_image.resize((target_width, target_height), Image.BICUBIC)
                elif isinstance(processed_image, np.ndarray):
                    pil_img = Image.fromarray(processed_image)
                    pil_img = pil_img.resize((target_width, target_height), Image.BICUBIC)
                    processed_image = np.array(pil_img)
                    
                debug_print(f"Depth image resized to: {target_width}x{target_height} using BICUBIC", debug_enabled)
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
            self.text_encoders_path, subfolder="text_encoder", torch_dtype=dtype
        )
        
        debug_print("\nLoading T5 text encoder...", debug_enabled)
        text_encoder_2 = T5EncoderModel.from_pretrained(
            self.text_encoders_path, subfolder="text_encoder_2", torch_dtype=dtype
        )
        
        debug_print("\nLoading VAE...", debug_enabled)
        vae = AutoencoderKL.from_pretrained(
            self.text_encoders_path, subfolder="vae", torch_dtype=dtype
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
        elif self.current_processor == "depth":
            self.checkpoint_path = self.checkpoint_path or "./models/Stable-diffusion/"
            self.current_model = self.current_model or "flux1-Depth-Dev_FP8.safetensors"
            base_model = os.path.join(self.checkpoint_path, self.current_model)
        elif self.current_processor == "redux":
            self.checkpoint_path = self.checkpoint_path or "./models/Stable-diffusion/"
            self.current_model = self.current_model or "flux1-Dev_FP8.safetensors"
            base_model = os.path.join(self.checkpoint_path, self.current_model)
        elif self.current_processor == "fill":
            self.checkpoint_path = self.checkpoint_path or "./models/Stable-diffusion/"
            self.current_model = self.current_model or "flux1-Fill-Dev_FP8.safetensors"
            base_model = os.path.join(self.checkpoint_path, self.current_model)
        else:
            # Por defecto, usar Canny
            self.checkpoint_path = self.checkpoint_path or "./models/Stable-diffusion/"
            self.current_model = "flux1-Canny-Dev_FP8.safetensors"
            base_model = os.path.join(self.checkpoint_path, self.current_model)
           
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
            
            # Cargar LoRAs adicionales para Canny y Depth
            try:
                if hasattr(self, 'lora1_model') and self.lora1_model and self.lora1_model != "None":
                    lora1_path = os.path.join(self.lora_dir, self.lora1_model)
                    if os.path.exists(lora1_path):
                        debug_print(f"\nLoading Custom LoRA 1: {self.lora1_model} with strength {self.lora1_scale}", debug_enabled)
                        self.logger.log(f"Loading Custom LoRA 1: {self.lora1_model} with strength {self.lora1_scale}")
                        pipe.load_lora_weights(lora1_path, lora_scale=float(self.lora1_scale))
                        pipe.fuse_lora(lora_scale=float(self.lora1_scale))
                
                if hasattr(self, 'lora2_model') and self.lora2_model and self.lora2_model != "None":
                    lora2_path = os.path.join(self.lora_dir, self.lora2_model)
                    if os.path.exists(lora2_path):
                        debug_print(f"\nLoading Custom LoRA 2: {self.lora2_model} with strength {self.lora2_scale}", debug_enabled)
                        self.logger.log(f"Loading Custom LoRA 2: {self.lora2_model} with strength {self.lora2_scale}")
                        pipe.load_lora_weights(lora2_path, lora_scale=float(self.lora2_scale))
                        pipe.fuse_lora(lora_scale=float(self.lora2_scale))
                
                if hasattr(self, 'lora3_model') and self.lora3_model and self.lora3_model != "None":
                    lora3_path = os.path.join(self.lora_dir, self.lora3_model)
                    if os.path.exists(lora3_path):
                        debug_print(f"\nLoading Custom LoRA 3: {self.lora3_model} with strength {self.lora3_scale}", debug_enabled)
                        self.logger.log(f"Loading Custom LoRA 3: {self.lora3_model} with strength {self.lora3_scale}")
                        pipe.load_lora_weights(lora3_path, lora_scale=float(self.lora3_scale))
                        pipe.fuse_lora(lora_scale=float(self.lora3_scale))
            except Exception as e:
                error_msg = f"Error loading custom LoRAs: {str(e)}"
                debug_print(error_msg, debug_enabled)
                self.logger.log(error_msg)
            
        elif self.current_processor == "fill":   
            pipe = FluxFillPipeline.from_single_file(
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
                
            try:
                if hasattr(self, 'lora1_model') and self.lora1_model and self.lora1_model != "None":
                    lora1_path = os.path.join(self.lora_dir, self.lora1_model)
                    if os.path.exists(lora1_path):
                        debug_print(f"\nLoading Custom LoRA 1: {self.lora1_model} with strength {self.lora1_scale}", debug_enabled)
                        self.logger.log(f"Loading Custom LoRA 1: {self.lora1_model} with strength {self.lora1_scale}")
                        pipe.load_lora_weights(lora1_path, lora_scale=float(self.lora1_scale))
                        pipe.fuse_lora(lora_scale=float(self.lora1_scale))
                
                if hasattr(self, 'lora2_model') and self.lora2_model and self.lora2_model != "None":
                    lora2_path = os.path.join(self.lora_dir, self.lora2_model)
                    if os.path.exists(lora2_path):
                        debug_print(f"\nLoading Custom LoRA 2: {self.lora2_model} with strength {self.lora2_scale}", debug_enabled)
                        self.logger.log(f"Loading Custom LoRA 2: {self.lora2_model} with strength {self.lora2_scale}")
                        pipe.load_lora_weights(lora2_path, lora_scale=float(self.lora2_scale))
                        pipe.fuse_lora(lora_scale=float(self.lora2_scale))
                
                if hasattr(self, 'lora3_model') and self.lora3_model and self.lora3_model != "None":
                    lora3_path = os.path.join(self.lora_dir, self.lora3_model)
                    if os.path.exists(lora3_path):
                        debug_print(f"\nLoading Custom LoRA 3: {self.lora3_model} with strength {self.lora3_scale}", debug_enabled)
                        self.logger.log(f"Loading Custom LoRA 3: {self.lora3_model} with strength {self.lora3_scale}")
                        pipe.load_lora_weights(lora3_path, lora_scale=float(self.lora3_scale))
                        pipe.fuse_lora(lora_scale=float(self.lora3_scale))
            except Exception as e:
                error_msg = f"Error loading custom LoRAs: {str(e)}"
                debug_print(error_msg, debug_enabled)
                self.logger.log(error_msg)
            
        elif self.current_processor == "redux":
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
                
                
            try:
                if hasattr(self, 'lora1_model') and self.lora1_model and self.lora1_model != "None":
                    lora1_path = os.path.join(self.lora_dir, self.lora1_model)
                    if os.path.exists(lora1_path):
                        debug_print(f"\nLoading Custom LoRA 1: {self.lora1_model} with strength {self.lora1_scale}", debug_enabled)
                        self.logger.log(f"Loading Custom LoRA 1: {self.lora1_model} with strength {self.lora1_scale}")
                        pipe.load_lora_weights(lora1_path, lora_scale=float(self.lora1_scale))
                        pipe.fuse_lora(lora_scale=float(self.lora1_scale))
                
                if hasattr(self, 'lora2_model') and self.lora2_model and self.lora2_model != "None":
                    lora2_path = os.path.join(self.lora_dir, self.lora2_model)
                    if os.path.exists(lora2_path):
                        debug_print(f"\nLoading Custom LoRA 2: {self.lora2_model} with strength {self.lora2_scale}", debug_enabled)
                        self.logger.log(f"Loading Custom LoRA 2: {self.lora2_model} with strength {self.lora2_scale}")
                        pipe.load_lora_weights(lora2_path, lora_scale=float(self.lora2_scale))
                        pipe.fuse_lora(lora_scale=float(self.lora2_scale))
                
                if hasattr(self, 'lora3_model') and self.lora3_model and self.lora3_model != "None":
                    lora3_path = os.path.join(self.lora_dir, self.lora3_model)
                    if os.path.exists(lora3_path):
                        debug_print(f"\nLoading Custom LoRA 3: {self.lora3_model} with strength {self.lora3_scale}", debug_enabled)
                        self.logger.log(f"Loading Custom LoRA 3: {self.lora3_model} with strength {self.lora3_scale}")
                        pipe.load_lora_weights(lora3_path, lora_scale=float(self.lora3_scale))
                        pipe.fuse_lora(lora_scale=float(self.lora3_scale))
            except Exception as e:
                error_msg = f"Error al cargar LoRAs personalizados: {str(e)}"
                debug_print(error_msg, debug_enabled)
                self.logger.log(error_msg)
        
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

    def generate(
        self, prompt, prompt2, input_image, width, height, steps, guidance, 
        low_threshold, high_threshold, detect_resolution, image_resolution, 
        reference_image, debug, processor_id, seed, randomize_seed, reference_scale,
        prompt_embeds_scale_1, prompt_embeds_scale_2, pooled_prompt_embeds_scale_1, 
        pooled_prompt_embeds_scale_2, use_hyper_flux, control_image2, text_encoder, text_encoder_2, 
        tokenizer, tokenizer_2, debug_enabled, output_dir, lora1_model, lora1_scale, 
        lora2_model, lora2_scale, lora3_model, lora3_scale, mask_image, fill_mode="Inpaint"):
        try:
            debug_print("\nStarting inference...", debug_enabled)
            from PIL import Image
            # Guardar los valores de LoRA como atributos de clase
            self.lora1_model = lora1_model
            self.lora1_scale = lora1_scale
            self.lora2_model = lora2_model
            self.lora2_scale = lora2_scale
            self.lora3_model = lora3_model
            self.lora3_scale = lora3_scale
            
            # Verificar si necesitamos recargar el modelo
            need_reload = False
            
            # Si el pipe no existe, necesitamos cargarlo
            if self.pipe is None:
                need_reload = True
                debug_print("Cargando modelos por primera vez", debug_enabled)
            # Si el procesador cambió, necesitamos recargar
            elif self.loaded_processor != self.current_processor:
                need_reload = True
                debug_print(f"Procesador cambió de {self.loaded_processor} a {self.current_processor}", debug_enabled)
            # Si el estado de Hyper-Flux cambió
            elif self.loaded_hyper_flux != use_hyper_flux:
                need_reload = True
                debug_print(f"Estado de Hyper-Flux cambió", debug_enabled)
            # Si algún LoRA cambió (modelo o escala)
            elif (self.loaded_lora1 != lora1_model or 
                  (lora1_model != "None" and self.loaded_lora1_scale != lora1_scale) or
                  self.loaded_lora2 != lora2_model or 
                  (lora2_model != "None" and self.loaded_lora2_scale != lora2_scale) or
                  self.loaded_lora3 != lora3_model or 
                  (lora3_model != "None" and self.loaded_lora3_scale != lora3_scale)):
                need_reload = True
                debug_print("LoRAs changed, reloading models", debug_enabled)
            
            if need_reload:
                # Si tenemos que recargar, liberar memoria primero
                if self.pipe is not None:
                    del self.pipe
                    self.pipe = None
                    clear_memory(debug_enabled)
                
                # Cargar el nuevo modelo
                self.pipe = self.load_models(use_hyper_flux=use_hyper_flux, debug_enabled=debug_enabled)
                
                # Actualizar estado del pipeline para futuras comparaciones
                self.loaded_processor = self.current_processor
                self.loaded_hyper_flux = use_hyper_flux
                self.loaded_lora1 = lora1_model
                self.loaded_lora1_scale = lora1_scale if lora1_model != "None" else None
                self.loaded_lora2 = lora2_model
                self.loaded_lora2_scale = lora2_scale if lora2_model != "None" else None
                self.loaded_lora3 = lora3_model
                self.loaded_lora3_scale = lora3_scale if lora3_model != "None" else None
                self.logger.log("Model loaded with new parameters")
            else:
                debug_print("Reutilizando modelo cargado previamente", debug_enabled)
            
            control_image = self.load_control_image(input_image)
            
            # La generación de semilla se manejará en generate_with_state para cada imagen
            seed_value = int(seed) if seed is not None else 0
            
            if self.current_processor == "fill":
                # Modo Fill para inpainting/outpainting
                menupro = "fill"
                
                with torch.inference_mode():
                    self.logger.log("Starting generation process...")
                    
                    # Asegurarnos que la imagen base y la máscara están en el formato correcto
                    base_image = self.load_control_image(reference_image)
                    
                    # Convertir la máscara a PIL si no lo es ya
                    
                    mask_img = mask_image
                    if isinstance(mask_img, np.ndarray):
                        mask_img = Image.fromarray(mask_img.astype('uint8'))
                    if mask_img.mode != 'L':
                        mask_img = mask_img.convert('L')
                    
                    # Verificar que ambas tienen el mismo tamaño
                    if base_image.size != mask_img.size:
                        self.logger.log(f"Resizing mask to match image: {base_image.size}")
                        mask_img = mask_img.resize(base_image.size)
                        
                    self.logger.log(f"Image mode: {base_image.mode}, Mask mode: {mask_img.mode}")
                    
                    result = self.pipe(
                        prompt=prompt,
                        image=base_image,       # Usa reference_image como base
                        mask_image=mask_img,    # Usa mask_image como máscara
                        height=height,
                        width=width,
                        num_inference_steps=int(steps),
                        guidance_scale=guidance,
                        generator=create_generator(seed_value)
                    )
                    self.logger.log("Generation completed")
                
            elif self.current_processor == "canny":
                menupro = "canny"
                processor = self.get_processor()
                control_image = processor(
                    control_image, 
                    low_threshold=int(low_threshold),
                    high_threshold=int(high_threshold),
                    detect_resolution=int(detect_resolution),
                    image_resolution=int(image_resolution)
                )
                with torch.inference_mode():
                    self.logger.log("Starting generation process...")
                    result = self.pipe(
                        prompt=prompt,
                        control_image=control_image,
                        height=height,
                        width=width,
                        num_inference_steps=int(steps),
                        guidance_scale=guidance,
                        generator=create_generator(seed_value)
                    )
                    self.logger.log("Generation completed")
                        
            elif self.current_processor == "depth":
                menupro = "depth"
                from PIL import Image
                
                # Log de debug para ver las dimensiones originales
                if isinstance(control_image, Image.Image):
                    self.logger.log(f"Original control_image dimensions: {control_image.size}")
                elif isinstance(control_image, np.ndarray):
                    self.logger.log(f"Original control_image dimensions: {control_image.shape}")
                
                # Procesar la imagen de forma controlada
                processor = Processor(processor_id)
                processed_control = processor(control_image).convert("RGB")
                
                # Asegurarnos de que la imagen procesada tenga el tamaño exacto deseado
                if isinstance(processed_control, Image.Image):
                    # Si ya está en formato PIL, redimensionar con alta calidad
                    current_size = processed_control.size
                    self.logger.log(f"Processed depth image size before resize: {current_size}")
                    if current_size != (width, height):
                        control_image = processed_control.resize((width, height), Image.BICUBIC)
                        self.logger.log(f"Resized to {width}x{height} using BICUBIC")
                    else:
                        control_image = processed_control
                        self.logger.log("No resize needed - image already correct size")
                elif isinstance(processed_control, np.ndarray):
                    # Si es numpy array, convertir a PIL para mejor redimensionamiento
                    pil_img = Image.fromarray(processed_control)
                    current_size = pil_img.size
                    self.logger.log(f"Processed depth image size before resize: {current_size}")
                    if current_size != (width, height):
                        pil_img = pil_img.resize((width, height), Image.BICUBIC)
                        self.logger.log(f"Resized to {width}x{height} using BICUBIC")
                    control_image = pil_img  # Mantener como PIL Image
                

                self.logger.log(f"Final image type: {type(control_image)}")
                if isinstance(control_image, Image.Image):
                    self.logger.log(f"Final image size: {control_image.size}")
                elif isinstance(control_image, np.ndarray):
                    self.logger.log(f"Final image shape: {control_image.shape}")
                
                
                with torch.inference_mode():
                    self.logger.log("Starting generation process...")
                    result = self.pipe(
                        prompt=prompt,
                        control_image=control_image,
                        height=height,
                        width=width,
                        num_inference_steps=int(steps),
                        guidance_scale=guidance,
                        generator=create_generator(seed_value)
                    )
                    self.logger.log("Generation completed")
            
                
            elif self.current_processor == "redux":
                self.logger.log("Starting Redux process...")
                pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
                    "Runware/FLUX.1-Redux-dev",
                    text_encoder=self.pipe.text_encoder,
                    text_encoder_2=self.pipe.text_encoder_2,
                    tokenizer=self.pipe.tokenizer,
                    tokenizer_2=self.pipe.tokenizer_2,
                    torch_dtype=torch.bfloat16
                ).to(device)
                    
                    
                my_prompt = prompt
                my_prompt2 = prompt2 if prompt2 else prompt  # Aseguramos que prompt2 tenga un valor
                    
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
            timestamp = datetime.now().strftime("%y_%m_%d_%H%M%S") 
            mode_map = {
                "canny": "canny",
                "depth": "depth",
                "redux": "redux",
                "fill": "fill"
            }
            filename = f"{mode_map[self.current_processor]}_{seed_value}_{timestamp}.png"
            file_path = os.path.join(output_directory, filename)
            
            result_image = result.images[0]
            result_image.save(file_path)
            self.logger.log(f"Image saved in: {file_path}")
            
            if torch.cuda.is_available():
                # Limpieza suave para evitar problemas con generaciones futuras
                torch.cuda.empty_cache()
            gc.collect()
                        
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
        initial_output = settings["output_dir"]
        initial_lora = settings["lora_dir"]
        initial_text_encoders = settings.get("text_encoders_path", "./models/Stable-diffusion/text_encoders_FP8")
    else:
        initial_checkpoints = "./models/stable-diffusion/"
        initial_output = "./outputs/fluxcontrolnet/"
        initial_lora = "./models/lora/"
        initial_text_encoders = "./models/Stable-diffusion/text_encoders_FP8"
    
    css = """
        /* Reducir el ancho de los checkboxes y etiquetas */
        .fill-controls-container .gr-checkbox-container {
            min-width: 60px !important;
            max-width: 60px !important;
        }
        
        /* Reducir el ancho del dropdown */
        .fill-controls-container .gr-dropdown {
            min-width: 80px !important;
            max-width: 80px !important;
        }
        
        /* Reducir el ancho del botón */
        .fill-controls-container .gr-button {
            min-width: 70px !important;
            max-width: 70px !important;
            padding: 2px 8px !important;
        }
        
        /* Alinear las etiquetas y reducir tamaño de texto */
        .fill-controls-container label {
            font-size: 0.85em !important;
            margin-bottom: 2px !important;
        }
    """
   
    with gr.Blocks(analytics_enabled=False, head=canvas_head) as flux_interface: 
        with gr.Row():
            extension_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            logo_path = os.path.join('file=', extension_path, 'assets', 'logo.png')

            gr.HTML(
                f"""
                <div style="text-align: center; max-width: 650px; margin: 0 auto">
                    <h3 style="display: flex; align-items: center; justify-content: center; gap: 10px;">
                        Flux.1 Tools ControlNet by Academia SD
                        <img src="file={logo_path}" style="height: 40px; width: auto;">
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
            #gr.Column(scale=0.2)
            with gr.Column(scale=1):
                canny_btn = gr.Button("Canny", variant="secondary", visible=True)
            with gr.Column(scale=1):
                depth_btn = gr.Button("Depth", variant="primary", visible=True)
            with gr.Column(scale=1):
                redux_btn = gr.Button("Redux", variant="primary", visible=True)
            with gr.Column(scale=1):
                fill_btn = gr.Button("Fill", variant="primary", visible=True)    

        reference_visible = gr.State(value=True)

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

            with gr.Group(elem_id="input_image_container"):
                # Input image para los modos que no son "fill"
                input_image = gr.Image(
                    label="Control Image", 
                    sources=["upload", "clipboard"],  
                    type="numpy",
                    interactive=True,
                    scale=1,
                    every=1,
                    container=True,
                    image_mode='RGB',
                    visible=True,  # Inicialmente visible
                    elem_id="normal_input_image"
                )
                
                # Canvas para el modo "fill"
                #fill_canvas_label = gr.Markdown("Base Image (Draw mask)", visible=False)
                with gr.Group(visible=False, elem_id="fill_canvas_container") as fill_canvas_group:
                    fill_canvas = ForgeCanvas(
                        elem_id="Fill_Mode_Canvas", 
                        #height=512, 
                        scribble_color="#FFFFFF",
                        scribble_color_fixed=True, 
                        scribble_alpha=100, 
                        scribble_alpha_fixed=True, 
                        scribble_softness_fixed=True
                    )
            
            # Imagen para máscara en el modo Fill
            mask_image = gr.Image(
                label="Mask Image (white areas will be inpaint/outpaint)",
                sources=["upload", "clipboard"],
                type="numpy",
                interactive=False,
                visible=False,
                scale=1,
                elem_id="mask_image"
            )
            
            control_image2 = gr.Image(
                label="Control Image 2", 
                sources=["upload", "clipboard"],
                type="numpy",
                interactive=True,
                visible=False,
                container=True,
                image_mode='RGB',
                every=1
            )
            reference_image = gr.Image(label="Reference Image", type="pil", interactive=False)
            output_gallery = gr.Gallery(
                label="Generated Images", 
                type="pil", 
                elem_id="generated_image", 
                show_label=True, 
                interactive=False,
                object_fit="contain",
                columns=1,
                rows=1,
            )
            
            selected_image = gr.State() 
            current_mode = gr.State("canny")
            
             
        with gr.Row():
            with gr.Column(scale=1):
                get_dimensions_btn = gr.Button("📐 Get Image Dimensions")
            with gr.Column(scale=1):
               
                preprocess_btn = gr.Button("💥 Run Preprocessor", variant="secondary", visible=True)
            with gr.Column(scale=1):
                toggle_reference_btn = gr.Button("🙈 Hide", variant="primary", interactive=True)
                
            with gr.Column(scale=0.1):
                send_to_control_btn = gr.Button("Send to Control", variant="secondary", visible=False)
                batch = gr.Slider(label=" Batch :", minimum=1, maximum=100, value=1, step=1)
            with gr.Column(scale=1):    
                generate_btn = gr.Button("⚡ Generate", variant="primary")
                
        with gr.Row(visible=False, elem_classes="fill-controls-container") as fill_controls:
        
            with gr.Column(scale=0, visible=False):
                fill_mode = gr.Dropdown(
                    label="Fill Mode",
                    choices=["Inpaint", "Outpaint"],
                    value="Inpaint",
                    visible=False
                )

            with gr.Column(scale=0.4, min_width=40):
                expand_range = gr.Slider(
                    label="Expansion Range:",  # Etiqueta más corta
                    minimum=0,
                    maximum=512,
                    
                    value=0,
                    step=64
                )
            
            with gr.Column(scale=0.4, min_width=40):
                mask_blur = gr.Slider(
                    label="Mask Blur",
                    minimum=0,
                    maximum=128,
                    value=16,
                    step=1
                )
            
            with gr.Column(scale=0.2, min_width=40):
                expand_up = gr.Checkbox(label="Expand Up ↑", value=False)
            
            with gr.Column(scale=0.2, min_width=40):
                expand_down = gr.Checkbox(label="Expand Down ↓", value=False) 
            
            with gr.Column(scale=0.2, min_width=40):
                expand_left = gr.Checkbox(label="Expand Left ←", value=False) 
            
            with gr.Column(scale=0.2, min_width=40):
                expand_right = gr.Checkbox(label="Expand Right →", value=False) 
                
            with gr.Column(scale=0.3, min_width=40):    
                transfer_mask_btn = gr.Button("Preview Mask", visible=False)                 
      
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...", scale=1)
                    prompt2 = gr.Textbox(label="Prompt for 2nd Image", placeholder="Enter prompt for second image...", visible=False, scale=1)
            with gr.Column(scale=1):  
                use_hyper_flux = gr.Checkbox(label="Use LoRA Hyper-FLUX1", value=False)
            with gr.Column(scale=1):
                progress_bar = gr.Textbox(
                    label=" Progress:", 
                    value="", 
                    interactive=False,
                    show_label=True,
                    #show_progress=True,
                    visible=True
                )
        
               
        with gr.Row():
            width = gr.Slider(label="Width :", minimum=256, maximum=2048, value=1024, step=16)
            height = gr.Slider(label="Height :", minimum=256, maximum=2048, value=1024, step=16)
            steps = gr.Slider(label="Inference_Steps :", minimum=1, maximum=100, value=30, step=1)
            guidance = gr.Slider(label="Guidance_Scale:", minimum=1, maximum=100, value=30, step=0.1)
            with gr.Row():
                seed = gr.Number(label="Seed :", minimum=0, maximum=999999999, value=0, step=1)
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                
        with gr.Row():
            # LoRA 1
            lora1_model = gr.Dropdown(
                choices=list_lora_files(flux_tab.lora_dir), 
                label="Custom LoRA 1:", 
                value="None", 
                scale=2
            )
            lora1_scale = gr.Slider(label="Strength L1", minimum=-2.0, maximum=3.0, value=1.0, step=0.1, scale=1)
            
            # LoRA 2
            lora2_model = gr.Dropdown(
                choices=list_lora_files(flux_tab.lora_dir), 
                label="Custom LoRA 2:", 
                value="None", 
                scale=2
            )
            lora2_scale = gr.Slider(label="Strength L2", minimum=-2.0, maximum=3.0, value=1.0, step=0.1, scale=1)
            
            # LoRA 3
            lora3_model = gr.Dropdown(
                choices=list_lora_files(flux_tab.lora_dir), 
                label="Custom LoRA 3:", 
                value="None", 
                scale=2
            )
            lora3_scale = gr.Slider(label="Strength L3", minimum=-2.0, maximum=3.0, value=1.0, step=0.1, scale=1)
            
            refresh_all_loras_btn = gr.Button("🔄 Refresh LoRa folder", size="sm", scale=0.5)
                
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
                gr.Column(scale=1)
                with gr.Column(scale=1):
                    # Añadir un elemento de texto para mostrar el estado del token
                    hf_token = gr.Textbox(
                        label="Hugging Face Read Token :",
                        placeholder="Enter your Hugging Face token...",
                        type="password"
                    )
                with gr.Column(scale=1):    
                    
                    hf_token_status = gr.Markdown(
                        value=check_existing_token(),  # Usamos la función directamente al inicializar
                        elem_id="hf_token_status"
                    )
                gr.Column(scale=1)
                    
                    
            with gr.Row():
                
                gr.Column(scale=1)
                with gr.Column(scale=1):
                    save_settingsHF_btn = gr.Button("Save HF Token:", variant="primary")
                with gr.Column(scale=1):
                    check_token_btn = gr.Button("Check Current Token", variant="primary")
                gr.Column(scale=1)        
            
            settings = load_settings() or {}
            checkpoint_value = settings.get("checkpoint_path", "./models/Stable-diffusion/")
            output_value = settings.get("output_dir", "./outputs/fluxcontrolnet/")
            lora_value = settings.get("lora_dir", "./models/lora/")
            text_encoders_value = settings.get("text_encoders_path", "./models/Stable-diffusion/text_encoders_FP8")
            
            with gr.Row():
                ckpt_display = gr.Markdown(f"Current checkpoints path: `{checkpoint_value}`")
                outp_display = gr.Markdown(f"Current images output dir: `{output_value}`")
                lora_display = gr.Markdown(f"Current LoRA path: `{lora_value}`")
                text_encoders_display = gr.Markdown(f"Current text encoders path: `{text_encoders_value}`")
       
            with gr.Row():
                
                checkpoint_path = gr.Textbox(
                    label="Checkpoints_Path :",
                    value=checkpoint_value,
                    placeholder="Enter model path..."
                )
                
                output_dir = gr.Textbox(
                    label="Output_Images_Path :",
                    value=output_value,
                    placeholder="Enter output path..."
                )
                
                lora_dir = gr.Textbox(
                    label="LoRA_Path :",
                    value=lora_value,
                    placeholder="Enter LoRA path..."
                )
                
                text_encoders_path = gr.Textbox(
                    label="Text_Encoders_Path :",
                    value=text_encoders_value,
                    placeholder="Enter text encoders path..."
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
            def save_settings_wrapper(checkpoint_path_val, output_dir_val, lora_dir_val, text_encoders_path_val, debug_val):
                log_msg, new_cp, new_od, new_lora, new_text_encoders = save_settings(
                    checkpoint_path_val, 
                    output_dir_val,
                    lora_dir_val,
                    text_encoders_path_val,
                    debug_val
                )
                flux_tab.checkpoint_path = new_cp
                flux_tab.output_dir = new_od
                flux_tab.lora_dir = new_lora
                flux_tab.text_encoders_path = new_text_encoders
                
                # Actualizar las listas de LoRAs
                new_lora_choices = list_lora_files(new_lora)
                
                return [
                    log_msg,           # log_box
                    new_cp,            # checkpoint_path
                    new_od,            # output_dir
                    new_lora,          # lora_dir
                    new_text_encoders, # text_encoders_path
                    gr.update(choices=new_lora_choices),  # lora1_model
                    gr.update(choices=new_lora_choices),  # lora2_model
                    gr.update(choices=new_lora_choices),  # lora3_model
                    gr.Markdown.update(value=f"Latest checkpoints path: `{new_cp}`"),
                    gr.Markdown.update(value=f"Latest images output dir: `{new_od}`"),
                    gr.Markdown.update(value=f"Latest LoRA path: `{new_lora}`"),
                    gr.Markdown.update(value=f"Latest text encoders path: `{new_text_encoders}`")
                ]
            
            save_settings_btn.click(
                fn=save_settings_wrapper,
                inputs=[checkpoint_path, output_dir, lora_dir, text_encoders_path, debug],
                outputs=[
                    log_box, checkpoint_path, output_dir, lora_dir, text_encoders_path,
                    lora1_model, lora2_model, lora3_model,
                    ckpt_display, outp_display, lora_display, text_encoders_display
                ]
            )

            save_settingsHF_btn.click(
                fn=save_settingsHF,
                inputs=[hf_token],
                outputs=[log_box]
            )
       
            get_dimensions_btn.click(
                fn=lambda img, canvas_bg, mode, w, h: update_dimensions(img, canvas_bg, mode, w, h),
                inputs=[input_image, fill_canvas.background, current_mode, width, height],
                outputs=[width, height]
            )
                               
            
        def safe_load_image(img):
            try:
                if img is None:
                    return None
                
                if isinstance(img, np.ndarray):
                    
                    return img.copy()
                return img
            except Exception as e:
                print(f"Error loading image: {e}")
                return None
        
        
        def send_to_canvas(image, current_mode):
            if current_mode == "fill":
                return image, None  # background, foreground
            return None, None
        
        transfer_mask_btn.click(
            fn=lambda bg, fg, up, down, left, right, range, blur: flux_tab.fill_canvas_to_mask(bg, fg, up, down, left, right, range, blur),
            inputs=[fill_canvas.background, fill_canvas.foreground, expand_up, expand_down, expand_left, expand_right, expand_range, mask_blur],
            outputs=[mask_image, reference_image, width, height]
        )

      
        input_image.upload(
            fn=safe_load_image,
            inputs=[input_image],
            outputs=[input_image],
            queue=False
        )
        
        mask_image.upload(
            fn=flux_tab.prepare_mask_image,
            inputs=[mask_image],
            outputs=[reference_image],
            queue=False
        )
        
        control_image2.upload(
            fn=safe_load_image, 
            inputs=[control_image2],
            outputs=[control_image2],
            queue=False
        )
        
        
        def conditional_preprocess(img, lt, ht, dr, ir, dbg, pid, current_mode):
            # Solo procesar automáticamente para modos que no sean "depth"
            if img is not None and current_mode != "depth":
                return flux_tab.preprocess_image(img, lt, ht, dr, ir, dbg, pid)
            return None

        input_image.change(
            fn=conditional_preprocess,
            inputs=[
                input_image, 
                low_threshold, 
                high_threshold, 
                detect_resolution, 
                image_resolution, 
                debug, 
                processor_id,
                current_mode  # Añadir el modo actual como entrada
            ],
            outputs=[reference_image],
            queue=False
        )
        
        low_threshold.release(
            fn=flux_tab.preprocess_image,
            inputs=[input_image, low_threshold, high_threshold, detect_resolution, image_resolution, debug, processor_id, width, height],
            outputs=[reference_image]
        )
        high_threshold.release(
            fn=flux_tab.preprocess_image,
            inputs=[input_image, low_threshold, high_threshold, detect_resolution, image_resolution, debug, processor_id, width, height],
            outputs=[reference_image]
        )
        image_resolution.release(
            fn=flux_tab.preprocess_image,
            inputs=[input_image, low_threshold, high_threshold, detect_resolution, image_resolution, debug, processor_id, width, height],
            outputs=[reference_image]
        )
        detect_resolution.release(
            fn=flux_tab.preprocess_image,
            inputs=[input_image, low_threshold, high_threshold, detect_resolution, image_resolution, debug, processor_id, width, height],
            outputs=[reference_image]
        )
        
        def check_current_token():
            result = check_existing_token()
            if result:
                return f"✅ Token verification: {result}", result
            else:
                return "❌ No valid token found or token could not be verified", ""

       
        check_token_btn.click(
            fn=check_current_token,
            inputs=[],
            outputs=[log_box, hf_token_status]
        )

        
        def auto_transfer_then_pre_generate(current_mode, bg, fg, up, down, left, right, range, blur):
            # Actualizar el botón primero
            btn_update = gr.Button.update(value="Generating...", variant="secondary", interactive=False)
            
            # Solo ejecutar transfer si estamos en modo fill
            if current_mode == "fill":
                # Ejecutar la transferencia
                mask, ref_img, new_width, new_height = flux_tab.fill_canvas_to_mask(bg, fg, up, down, left, right, range, blur)
                # Devolver los resultados de la transferencia y el botón actualizado
                return mask, ref_img, new_width, new_height, btn_update
            else:
                # Si NO estamos en modo fill, devolver gr.update() para mantener los valores actuales
                return gr.update(), gr.update(), gr.update(), gr.update(), btn_update
        
        
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
                processor_id,
                width,
                height
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
        
            fn=lambda x: gr.update(value=10 if x else 30),
            inputs=[use_hyper_flux],
            outputs=[steps]
        )
        
        def update_preprocessing(input_image, low_threshold, high_threshold, detect_resolution, image_resolution, debug, processor_id, width, height):
            if input_image is not None:
                return flux_tab.preprocess_image(input_image, low_threshold, high_threshold, detect_resolution, image_resolution, debug, processor_id, width, height)
            return None
                    
        for slider in [low_threshold, high_threshold, detect_resolution, image_resolution]:
            slider.release(
                fn=update_preprocessing,
                inputs=[input_image, low_threshold, high_threshold, detect_resolution, image_resolution, debug, processor_id, width, height],
                outputs=[reference_image]
            )
        def pre_generate():
            return gr.Button.update(value="Generating...", variant="secondary", interactive=False)
        def post_generate(result):
            return result, gr.Button.update(value="Generate", variant="primary", interactive=True)
        
        def generate_with_state(
            current_mode, prompt, prompt2, input_image, fill_canvas_background, width, height, 
            steps, guidance, low_threshold, high_threshold, detect_resolution, image_resolution,
            reference_image, debug, processor_id, seed, randomize_seed, reference_scale,
            prompt_embeds_scale_1, prompt_embeds_scale_2, pooled_prompt_embeds_scale_1,
            pooled_prompt_embeds_scale_2, use_hyper_flux, control_image2, batch,
            lora1_model, lora1_scale, lora2_model, lora2_scale, lora3_model, lora3_scale,
            mask_image, fill_mode="Inpaint"
        ):
            # Usar la imagen correcta según el modo
            if current_mode == "fill":
                # En modo "fill", pasamos None ya que usaremos reference_image directamente en generate()
                actual_input_image = None
            else:
                actual_input_image = input_image
            
            try:
                results = []
                total_batch = int(batch) if batch is not None else 1
                
                # Track the last used seed
                last_used_seed = None
                
                status_msg = "Starting generation..."
                yield results, flux_tab.logger.log(status_msg), status_msg, gr.update()
                
                for i in range(total_batch):
                    # Generate the seed just before generating the image
                    if randomize_seed:
                        current_seed = random.randint(0, 999999999)
                    elif i > 0:  
                        # If not randomized but in batch > 1, increment the seed
                        current_seed = int(seed) + i
                    else:
                        # First batch with non-randomized seed
                        current_seed = int(seed)
                    
                    # Keep track of the last used seed
                    last_used_seed = current_seed
                                        
                    if i > 0:
                        # Limpieza suave de memoria para evitar OOM
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                    
                    status_msg = f"Generating image {i+1} of {total_batch} with seed: {current_seed}"
                    yield results, flux_tab.logger.log(status_msg), status_msg, gr.update(value=current_seed)
                    
                    # Generate the image with the current seed
                    result = flux_tab.generate(
                        prompt=prompt,
                        prompt2=prompt2,
                        input_image=actual_input_image,  # Usamos la imagen según el modo actual
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
                        seed=current_seed,
                        randomize_seed=False,
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
                        output_dir=output_dir,
                        lora1_model=lora1_model,
                        lora1_scale=lora1_scale,
                        lora2_model=lora2_model,
                        lora2_scale=lora2_scale,
                        lora3_model=lora3_model,
                        lora3_scale=lora3_scale,
                        mask_image=mask_image,
                        fill_mode=fill_mode
                    )
                    
                    if result is not None:
                        results.append(result)
                        status_msg = f"Completed {i+1} of {total_batch} images"
                        yield results, flux_tab.logger.log(status_msg), status_msg, gr.update(value=current_seed)
                
                # Always update the UI with the last seed we actually used
                final_msg = "Generation completed successfully!"
                yield results, flux_tab.logger.log(final_msg), final_msg, gr.update(value=last_used_seed)
                
            except Exception as e:
                error_msg = f"Error in generation: {str(e)}"
                print(error_msg)
                yield None, flux_tab.logger.log(error_msg), error_msg, gr.update()

#----------------------------

        
        # Evento click para el botón de alternar
        toggle_reference_btn.click(
            fn=flux_tab.toggle_reference_visibility,
            inputs=[reference_visible, gr.State(flux_tab.current_processor)],
            outputs=[reference_visible, reference_image, control_image2, toggle_reference_btn, prompt2, mask_image]
        )
        
        # Función para actualizar la lista de LoRAs
        def refresh_all_loras():
            choices = list_lora_files(flux_tab.lora_dir)
            return [
                gr.Dropdown.update(choices=choices),
                gr.Dropdown.update(choices=choices),
                gr.Dropdown.update(choices=choices)
            ]

        generate_btn.click(
            fn=auto_transfer_then_pre_generate,
            inputs=[
                current_mode,
                fill_canvas.background,
                fill_canvas.foreground,
                expand_up,
                expand_down,
                expand_left,
                expand_right,
                expand_range,
                mask_blur  # Añadir el slider de blur
            ],
            outputs=[
                mask_image,
                reference_image,
                width,
                height,
                generate_btn
            ]
        ).then(
            fn=generate_with_state,
            inputs=[
                current_mode, prompt, prompt2, input_image, fill_canvas.background, width, height, 
                steps, guidance, low_threshold, high_threshold, detect_resolution, image_resolution,
                reference_image, debug, processor_id, seed, randomize_seed, reference_scale,
                prompt_embeds_scale_1, prompt_embeds_scale_2, pooled_prompt_embeds_scale_1,
                pooled_prompt_embeds_scale_2, use_hyper_flux, control_image2, batch,
                lora1_model, lora1_scale, lora2_model, lora2_scale, lora3_model, lora3_scale,
                mask_image
            ],
            outputs=[output_gallery, log_box, progress_bar, seed],
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
            fn=lambda img: img,
            inputs=[selected_image],
            outputs=[input_image]
        )
            

        # Botón Canny
        canny_btn.click(
            fn=lambda use_flux: flux_tab.switch_mode("canny", use_flux),
            inputs=[use_hyper_flux],
            outputs=[
                canny_btn,
                depth_btn,
                redux_btn,
                fill_btn,
                input_image,
                fill_canvas_group,
                current_mode,
                low_threshold,
                high_threshold,
                detect_resolution,
                image_resolution,
                processor_id,
                reference_scale,
                prompt_embeds_scale_1,
                prompt_embeds_scale_2,
                pooled_prompt_embeds_scale_1,
                pooled_prompt_embeds_scale_2,
                steps,
                guidance,
                control_image2,
                prompt2,
                mask_image,
                fill_controls,
                transfer_mask_btn,
                reference_image  
            ]
        )

        # Botón Depth
        depth_btn.click(
            fn=lambda use_flux: flux_tab.switch_mode("depth", use_flux),
            inputs=[use_hyper_flux],
            outputs=[
                canny_btn,
                depth_btn,
                redux_btn,
                fill_btn,
                input_image,
                fill_canvas_group,
                current_mode,
                low_threshold,
                high_threshold,
                detect_resolution,
                image_resolution,
                processor_id,
                reference_scale,
                prompt_embeds_scale_1,
                prompt_embeds_scale_2,
                pooled_prompt_embeds_scale_1,
                pooled_prompt_embeds_scale_2,
                steps,
                guidance,
                control_image2,
                prompt2,
                mask_image,
                fill_controls,
                transfer_mask_btn,
                reference_image  
            ]
        )

        # Botón Redux
        redux_btn.click(
            fn=lambda use_flux: flux_tab.switch_mode("redux", use_flux),
            inputs=[use_hyper_flux],
            outputs=[
                canny_btn,
                depth_btn,
                redux_btn,
                fill_btn,
                input_image,
                fill_canvas_group,
                current_mode,
                low_threshold,
                high_threshold,
                detect_resolution,
                image_resolution,
                processor_id,
                reference_scale,
                prompt_embeds_scale_1,
                prompt_embeds_scale_2,
                pooled_prompt_embeds_scale_1,
                pooled_prompt_embeds_scale_2,
                steps,
                guidance,
                control_image2,
                prompt2,
                mask_image,
                fill_controls,
                transfer_mask_btn,
                reference_image  
            ]
        )

        # Botón Fill
        fill_btn.click(
            fn=lambda use_flux: flux_tab.switch_mode("fill", use_flux),
            inputs=[use_hyper_flux],
            outputs=[
                canny_btn,
                depth_btn,
                redux_btn,
                fill_btn,
                input_image,
                fill_canvas_group,
                current_mode,
                low_threshold,
                high_threshold,
                detect_resolution,
                image_resolution,
                processor_id,
                reference_scale,
                prompt_embeds_scale_1,
                prompt_embeds_scale_2,
                pooled_prompt_embeds_scale_1,
                pooled_prompt_embeds_scale_2,
                steps,
                guidance,
                control_image2,
                prompt2,
                mask_image,
                fill_controls,
                transfer_mask_btn,
                reference_image  
            ]
        )

    return [(flux_interface, "Flux.1 Tools", "flux_controlnet_tab")]

script_callbacks.on_ui_tabs(on_ui_tabs)