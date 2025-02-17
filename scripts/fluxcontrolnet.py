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
#from image_gen_aux import DepthPreprocessor
import os
from modules import script_callbacks
from modules.ui_components import ToolButton
import modules.generation_parameters_copypaste as parameters_copypaste
#import modules.scripts as scripts
from modules.shared import opts, OptionInfo
#from modules import script_callbacks
#from modules.ui_components import ToolButton
from modules.ui_common import save_files
from huggingface_hub import hf_hub_download
import random
from datetime import datetime 
from collections import deque
#from modules_forge.forge_canvas.canvas import ForgeCanvas, canvas_head
#from importlib import reload
from huggingface_hub import login

# Inicia sesión con tu token
login("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")  # Reemplaza con tu token
 
# ["canny", "depth_leres", "depth_leres++", "depth_midas", "depth_zoe", "lineart_anime",
#  "lineart_coarse", "lineart_realistic", "mediapipe_face", "mlsd", "normal_bae", "normal_midas",
#  "openpose", "openpose_face", "openpose_faceonly", "openpose_full", "openpose_hand",
#  "scribble_hed, "scribble_pidinet", "shuffle", "softedge_hed", "softedge_hedsafe",
#  "softedge_pidinet", "softedge_pidsafe", "dwpose"]
processor_id = 'canny'
#processor = Processor(processor_id)
menupro = "canny"


prompt_embeds_scale_1 = 1.0
prompt_embeds_scale_2 = 1.0
pooled_prompt_embeds_scale_1 = 1.0
pooled_prompt_embeds_scale_2 = 1.0

def debug_print(message, debug_enabled=False):
    if debug_enabled:
        self.logger.log(message)

def print_memory_usage(message="", debug_enabled=False):
    if debug_enabled and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        self.logger.log(f"{message} GPU Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")

def quantize_model_to_nf4(model, name="", debug_enabled=False):
    debug_print(f"\nCuantizando modelo {name} a NF4...", debug_enabled)
    print_memory_usage("Antes de cuantizacion:", debug_enabled)
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            debug_print(f"Convirtiendo capa: {name}", debug_enabled)
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
    
    print_memory_usage("Despues de cuantizacion:", debug_enabled)
    return model

def clear_memory(debug_enabled=False):
    debug_print("\nLimpiando memoria...", debug_enabled)
    if torch.cuda.is_available():
        print_memory_usage("Antes de limpiar:", debug_enabled)
        torch.cuda.empty_cache()
        gc.collect()
        print_memory_usage("Despues de limpiar:", debug_enabled)

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
        print(message)  # Mantener el print original en consola
        self.messages.append(str(message))
        if self.log_box is not None:
            return "\n".join(self.messages)
        return None
        
class FluxControlNetTab:
    def __init__(self):
        self.pipe = None
        self.model_path = "./models/diffusers/cn"
        self.current_processor = "canny"
        self.current_model = "flux1CannyDevFp8_v10.safetensors"
        self.default_processor_id = "depth_zoe"
        self.logger = LogManager()  # Inicializar el logger aquí

    def update_model_path(self, new_path, debug_enabled):
        if new_path and os.path.exists(new_path):
            self.model_path = new_path
            debug_print(f"Updated model path: {new_path}", debug_enabled)
            # Clear existing pipeline to force reload with new path
            if self.pipe is not None:
                del self.pipe
                self.pipe = None
                clear_memory()
        return self.model_path

    def update_processor_and_model(self, processor_type):
        if processor_type == "canny":
            self.current_processor = "canny"
            self.current_model = "flux1CannyDevFp8_v10.safetensors"
            self.default_processor_id = None
        elif processor_type == "depth":
            self.current_processor = "depth"
            self.current_model = "flux1DepthDevFp8_v10.safetensors"
            self.default_processor_id = "depth_zoe"
        elif processor_type == "redux":
            self.current_processor = "redux"
            self.current_model = "flux1DevFp8_v10.safetensors"
            self.default_processor_id = None
        
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            clear_memory()
    
        print(f"Processor changed to: {processor_type}")
        
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
        return CannyDetector()  # Default to Canny

    def preprocess_image(self, input_image, low_threshold, high_threshold, detect_resolution, image_resolution, debug_enabled, processor_id=None):
        try:
            debug_print("\nIniciando preprocesamiento...", debug_enabled)
            
            control_image = self.load_control_image(input_image)
            
            if self.current_processor == "canny":
                processor = self.get_processor()
                control_image = processor(
                    control_image, 
                    low_threshold=int(low_threshold),
                    high_threshold=int(high_threshold),
                    detect_resolution=int(detect_resolution),
                    image_resolution=int(image_resolution)
                )
            
            elif self.current_processor == "depth":
                # Usar el processor_id del dropdown si está disponible, sino usar el default
                actual_processor_id = processor_id if processor_id is not None else self.default_processor_id
                processor = Processor(actual_processor_id)
                control_image = processor(
                    control_image,
                )
            
            elif self.current_processor == "redux":
                control_image = control_image
            
            os.makedirs("extensions/sd-forge-fluxcontrolnet/maps", exist_ok=True)
            control_image.save("extensions/sd-forge-fluxcontrolnet/maps/controlmap.png")
            
            control_image_np = np.array(control_image)
            
            debug_print("\nPreprocesamiento completado", debug_enabled)
            return control_image_np
            
        except Exception as e:
            self.logger.log(f"\nError en el preprocesamiento: {str(e)}")
            self.logger.log("Stacktrace:", traceback.format_exc())
            return None

    def load_models(self, debug_enabled=False):
        debug_print("\nIniciando carga de modelos...", debug_enabled)
        
        dtype = torch.bfloat16
        
        debug_print("\nCargando CLIP text encoder...", debug_enabled)
        text_encoder = CLIPTextModel.from_pretrained(
            self.model_path, subfolder="text_encoder", torch_dtype=dtype
        )
        #text_encoder = quantize_model_to_nf4(text_encoder, "CLIP text encoder", debug_enabled)
        
        debug_print("\nCargando T5 text encoder...", debug_enabled)
        text_encoder_2 = T5EncoderModel.from_pretrained(
            self.model_path, subfolder="text_encoder_2", torch_dtype=dtype
        )
        #text_encoder_2 = quantize_model_to_nf4(text_encoder_2, "T5 text encoder", debug_enabled)
        
        debug_print("\nCargando VAE...", debug_enabled)
        vae = AutoencoderKL.from_pretrained(
            self.model_path, subfolder="vae", torch_dtype=dtype
            #"./models/", subfolder="vae", torch_dtype=dtype
        )
      
        #vae = quantize_model_to_nf4(vae, "VAE", debug_enabled)
        
        debug_print("\nCargando tokenizers...", debug_enabled)
        tokenizer = CLIPTokenizer.from_pretrained(self.model_path, subfolder="tokenizer")
        tokenizer_2 = T5Tokenizer.from_pretrained(self.model_path, subfolder="tokenizer_2")
        
        #debug_print("\nCargando scheduler...", debug_enabled)
        #scheduler = TrainingArguments.from_pretrained(self.model_path, subfolder="scheduler")
        
        clear_memory(debug_enabled)
        
        debug_print("\nCargando modelo principal Flux ControlNet...", debug_enabled)
        
        if self.current_processor == "canny":
            base_model = os.path.join("./models/Stable-diffusion/flux1CannyDevFp8_v10.safetensors") #aqui
        if self.current_processor == "depth":
            base_model = os.path.join("./models/Stable-diffusion/flux1DepthDevFp8_v10.safetensors") #aqui
        if self.current_processor == "redux":
            base_model = os.path.join("./models/Stable-diffusion/flux1DevFp8_v10.safetensors") #aqui
        
        if self.current_processor == "canny" or self.current_processor == "depth":
            pipe = FluxControlPipeline.from_single_file(
                base_model,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                vae=vae,
                torch_dtype=dtype
            )
            
        if self.current_processor == "redux":
        
   
            pipe = FluxPipeline.from_single_file(
                base_model,
                text_encoder=None,
                text_encoder_2=None,
                tokenizer=None,
                tokenizer_2=None,
                vae=vae,
                torch_dtype=dtype    
            ).to("cuda")
        
            # test lora
            pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", "Hyper-FLUX.1-dev-8steps-lora.safetensors"), lora_scale=0.125)
            pipe.fuse_lora(lora_scale=0.125)
            #pipe.to(torch_dtype=dtype )
        
        
        debug_print("\nCuantizando transformer principal...", debug_enabled)
        pipe.transformer = quantize_model_to_nf4(pipe.transformer, "Transformer principal", debug_enabled)
        
        debug_print("\nActivando optimizaciones de memoria...", debug_enabled)
        if hasattr(torch.backends, 'memory_efficient_attention'):
            torch.backends.memory_efficient_attention.enabled = True
            debug_print("Memory efficient attention activado", debug_enabled)
        
        pipe.enable_attention_slicing()
        debug_print("Attention slicing activado", debug_enabled)
        
        pipe.enable_model_cpu_offload()
        debug_print("Model CPU offload activado", debug_enabled)
        
        clear_memory(debug_enabled)
        debug_print("\nModelos cargados y optimizados correctamente", debug_enabled)

        return pipe

    def load_control_image(self, input_image):
        if input_image is not None:
            return Image.fromarray(input_image.astype('uint8'))
        return load_image("./models/diffusers/cn")

    def generate(
        self, prompt, input_image, width, height, steps, guidance, 
        low_threshold, high_threshold, detect_resolution, image_resolution, 
        reference_image, debug, processor_id, seed, randomize_seed, reference_scale,
        prompt_embeds_scale_1, prompt_embeds_scale_2, pooled_prompt_embeds_scale_1, 
        pooled_prompt_embeds_scale_2, text_encoder, text_encoder_2, tokenizer, 
        tokenizer_2, debug_enabled,
        
        ):
        try:
            self.logger.log(f"Seed value: {seed}")
            self.logger.log(f"Prompt value: {prompt}")
            self.logger.log(f"Reference_scale value: {reference_scale}")
            self.logger.log(f"Steps value: {steps}")
            self.logger.log(f"Guide value: {guidance}")
            self.logger.log(f"Prompt_embeds_scale value: {prompt_embeds_scale_1}")
            
            debug_print("\nIniciando generacion de imagen...", debug_enabled)
            print_memory_usage("Memoria inicial:", debug_enabled)
            
            if self.pipe is None:
                debug_print("Cargando modelos por primera vez...", debug_enabled)
                self.pipe = self.load_models(debug_enabled)
            
            control_image = self.load_control_image(input_image)
            
            
            if self.current_processor == "canny":
                #actualprocessor = "canny"
                menupro = "canny"
                processor = self.get_processor()
                control_image = processor(
                    control_image, 
                    low_threshold=int(low_threshold),
                    high_threshold=int(high_threshold),
                    detect_resolution=int(detect_resolution),
                    image_resolution=int(image_resolution)
                )
            
            #preprocesador de generacion
            if self.current_processor == "depth":
                menupro = "depth"
                #actualprocessor = "depth"
                processor_id = 'depth_zoe'
                processor = Processor(processor_id)
                control_image = processor(
                    control_image, 
                    
                )

            if self.current_processor == "redux":
                #actualprocessor = "canny"
                #processor = self.get_processor()
                control_image = control_image 
                    #low_threshold=int(low_threshold),
          
            #control_image.save("./extensions/sd-forge-fluxcontrolnet/maps/controlmap.png")
            
            #debug_print("\nImagen de control cargada", debug_enabled)
            
            if randomize_seed:
                seed = random.randint(0, 999999999)
            
            # Ensure seed is an integer
            seed_value = int(seed) if seed is not None else 1234
            
            debug_print(f"\nGenerando con parametros:", debug_enabled)
            debug_print(f"Width: {width}", debug_enabled)
            debug_print(f"Height: {height}", debug_enabled)
            debug_print(f"Steps: {steps}", debug_enabled)
            debug_print(f"Orientation Scale: {guidance}", debug_enabled)
            debug_print(f"Low Threshold: {low_threshold}", debug_enabled)
            debug_print(f"High Threshold: {high_threshold}", debug_enabled)
            debug_print(f"Detect Resolution: {detect_resolution}", debug_enabled)
            debug_print(f"Image Resolution: {image_resolution}", debug_enabled)
            debug_print(f"Seed: {seed_value}", debug_enabled)
            debug_print(f"Prompt: {prompt}", debug_enabled)
            
            #render
            
            if self.current_processor == "canny" or self.current_processor == "depth":
                with torch.inference_mode():
                    result = self.pipe(
                        prompt=prompt,
                        control_image=control_image,
                        height=height,
                        width=width,
                        num_inference_steps=int(steps),
                        guidance_scale=guidance,
                        generator=torch.Generator("cpu").manual_seed(seed_value),
                    )
                
            if self.current_processor == "redux":
                #pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained("black-forest-labs/FLUX.1-Redux-dev", torch_dtype=torch.bfloat16).to("cuda")
                pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-Redux-dev",
                    
                    text_encoder=self.pipe.text_encoder,
                    text_encoder_2=self.pipe.text_encoder_2,
                    tokenizer=self.pipe.tokenizer,
                    tokenizer_2=self.pipe.tokenizer_2,
                    torch_dtype=torch.bfloat16
                ).to("cuda")
                
                #prompt_embeds_scale_1=1.0
                #pooled_prompt_embeds_scale_1=1.0
                #reference_scale=0.03
                #my_image=control_image
                #my_prompt=prompt
                pipe_prior_output = pipe_prior_redux(control_image, prompt=prompt, prompt_embeds_scale = [prompt_embeds_scale_1],
                                            pooled_prompt_embeds_scale = [pooled_prompt_embeds_scale_1])
                                            
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
                joint_attention_kwargs=dict(attention_mask=attention_mask)                            
                #pipe_prior_output = pipe_prior_redux(control_image, prompt)
                #my_image = control_image
                
                
            
                with torch.inference_mode():
                    result = self.pipe(
                        #prompt=prompt,
                        #control_image=control_image,
                        #height=height,
                        #width=width,
                        guidance_scale=guidance,
                        num_inference_steps=int(steps),
                        generator=torch.Generator("cpu").manual_seed(seed_value),
                        joint_attention_kwargs=joint_attention_kwargs,
                        **pipe_prior_output,
                    )
                    
                
            
            debug_print("\nGeneracion completada", debug_enabled)
            clear_memory(debug_enabled)
            
            output_dir = "./extensions/sd-forge-fluxcontrolnet/outputs"
            os.makedirs(output_dir, exist_ok=True)
        
                # Generar nombre de archivo
            timestamp = datetime.now().strftime("%y_%m_%d")
            mode_map = {
                "canny": "canny",
                "depth": "depth",
                "redux": "redux"
            }
            filename = f"{mode_map[self.current_processor]}_{seed_value}_{timestamp}.png"
            file_path = os.path.join(output_dir, filename)
        
            # Guardar imagen
            result_image = result.images[0]
            result_image.save(file_path)
            self.logger.log(f"Imagen guardada en: {file_path}")
            
            return result.images[0]
            
            
        except Exception as e:
            self.logger.log(f"\nError en la generacion: {str(e)}")
            self.logger.log("Stacktrace:" + traceback.format_exc())
            return None

        
def on_ui_tabs():
    flux_tab = FluxControlNetTab()
    
    with gr.Blocks(analytics_enabled=False) as flux_interface:
        with gr.Row():
            gr.HTML(
                """
                <div style="text-align: center; max-width: 650px; margin: 0 auto">
                    <h1>Unofficial Flux1 Dev Controlnet</h1>
                    <p>By Academia SD</p>
                </div>
                """
            )
        with gr.Row():
            canny_btn = gr.Button("Canny", variant="secondary")
            depth_btn = gr.Button("Depth", variant="primary")
            redux_btn = gr.Button("Redux", variant="primary")
            
        
        with gr.Row():
            input_image = gr.Image(label="Control Image", source="upload", type="numpy", interactive=True)
            reference_image = gr.Image(label="Reference Image", type="numpy", interactive=False)
            output_image = gr.Image(label="Generated Image", elem_id="generated_image", type="pil", interactive=False)
        with gr.Row():
            get_dimensions_btn = gr.Button("Get Image Dimensions")
            use_default = gr.Button("Use Default Image")
            preprocess_btn = gr.Button("Run Preprocessor", variant="secondary", visible=True)
            generate_btn = gr.Button("Generate", variant="primary")
            
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...")
        with gr.Row():
            width = gr.Slider(label="Width :", minimum=256, maximum=2048, value=1024, step=16)
            height = gr.Slider(label="Height :", minimum=256, maximum=2048, value=1024, step=16)
            steps = gr.Slider(label="Inference_Steps :", minimum=1, maximum=100, value=30, step=1)
            guidance = gr.Slider(label="Guidance_Scale:", minimum=1, maximum=100, value=30, step=0.1)
            with gr.Row():
                seed = gr.Slider(label="Seed noise:", minimum=0, maximum=9999999999, value=0, step=1)
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        
        
        # CONTROLES DE PREPROCESAMIENTO: Se definen una única vez, con visibilidad según el modo
        with gr.Row():
            low_threshold = gr.Slider(label="Low Threshold:", minimum=0, maximum=256, value=50, step=1, visible=True)
            high_threshold = gr.Slider(label="High Threshold:", minimum=0, maximum=256, value=200, step=1, visible=True)
            detect_resolution = gr.Slider(label="Detect Resolution:", minimum=128, maximum=2048, value=1024, step=16, visible=True)
            image_resolution = gr.Slider(label="Image Resolution:", minimum=128, maximum=2048, value=1024, step=16, visible=True)
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
                label="prompt embeds scale 1st image",
                info="info sobre esto",
                minimum=0,
                maximum=1.5,
                step=0.01,
                value=1,
                visible=False
            )
            prompt_embeds_scale_2 = gr.Slider(
                label="prompt embeds scale 2nd image",
                info="info sobre esto",
                minimum=0,
                maximum=1.5,
                step=0.01,
                value=1,
                visible=False
            )
            pooled_prompt_embeds_scale_1 = gr.Slider(
                label="pooled prompt embeds scale 1nd image",
                info="info sobre esto",
                minimum=0,
                maximum=1.5,
                step=0.01,
                value=1,
                visible=False
            )
            pooled_prompt_embeds_scale_2 = gr.Slider(
                label="pooled prompt embeds scale 2nd image",
                info="info sobre esto",
                minimum=0,
                maximum=1.5,
                step=0.01,
                value=1,
                visible=False
            )
            
        with gr.Row(elem_id=f"image_buttons", elem_classes="image-buttons"):
            buttons = {
                'img2img': ToolButton('🖼️', elem_id=f'_send_to_img2img', tooltip="Send image to img2img tab."),
                'inpaint': ToolButton('🎨️', elem_id=f'_send_to_inpaint', tooltip="Send image to img2img inpaint tab."),
                'extras': ToolButton('📐', elem_id=f'_send_to_extras', tooltip="Send image to extras tab."),
            }
            for paste_tabname, paste_button in buttons.items():
                parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                    paste_button=paste_button, tabname=paste_tabname, source_tabname=None, source_image_component=output_image,
                    paste_field_names=[]
                ))

        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                # Añadir el log box
                log_box = gr.Textbox(
                    label="Latest Logs",
                    interactive=False,
                    lines=6,
                    value="Waiting for operations..."
                )
                flux_tab.logger.log_box = log_box
                
            with gr.Row():
                model_path = gr.Textbox(
                    label="Models Path:", 
                    value="./models/diffusers/cn",
                    interactive=True,
                    placeholder="Enter model path..."
                )
            with gr.Row():
                update_path_btn = gr.Button("?? Update Path", size="sm")
                debug = gr.Checkbox(label="Debug Mode", value=False)

        # Función auxiliar para actualizar botones y controles de preprocesamiento
        def on_processor_change(mode):
            # Actualiza el procesador y modelo internamente
            btn_updates = flux_tab.update_processor_and_model(mode)
            if mode == "canny":
                ctrl_updates = [
                    gr.update(visible=True),   # low_threshold
                    gr.update(visible=True),   # high_threshold
                    gr.update(visible=True),   # detect_resolution
                    gr.update(visible=True),   # image_resolution
                    gr.update(visible=False),   # processor_id dropdown
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(value=30),       # steps
                    gr.update(value=30),       # guidance
        
                ]
            elif mode == "depth":
                ctrl_updates = [
                    gr.update(visible=False),  # low_threshold
                    gr.update(visible=False),  # high_threshold
                    gr.update(visible=False),  # detect_resolution
                    gr.update(visible=False),  # image_resolution
                    gr.update(visible=True),    # processor_id dropdown
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(value=25),       # steps
                    gr.update(value=30),       # guidance
        
                ]
            elif mode == "redux":
                ctrl_updates = [
                    gr.update(visible=False),  # low_threshold
                    gr.update(visible=False),  # high_threshold
                    gr.update(visible=False),  # detect_resolution
                    gr.update(visible=False),  # image_resolution
                    gr.update(visible=False),  # processor_id dropdown
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(value=8),        # steps
            gr.update(value=3.5),      # guidance
        
                ]
            # Retornamos la actualización de 3 botones + 5 controles = 8 salidas en total
            return btn_updates + ctrl_updates

        # Event listeners for auto-preprocessing
         # Listeners para cambios en la imagen y sliders (preprocesador)
        input_image.change(
            fn=flux_tab.preprocess_image,
            inputs=[input_image, low_threshold, high_threshold, detect_resolution, image_resolution, debug, processor_id],
            outputs=[reference_image]
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
        
        use_default.click(fn=lambda: load_image("./models/diffusers/cn/default.png"), outputs=[input_image])
        get_dimensions_btn.click(
            fn=update_dimensions,
            inputs=[input_image, width, height],
            outputs=[width, height]
        )
        preprocess_btn.click(
            fn=flux_tab.preprocess_image,
            inputs=[input_image, low_threshold, high_threshold, detect_resolution, image_resolution, debug, processor_id],
            outputs=[reference_image]
        )
        update_path_btn.click(
            fn=flux_tab.preprocess_image,
            inputs=[model_path, debug],
            outputs=[model_path]
        )
        
      
        def pre_generate():
            """Función que se ejecuta antes de la generación"""
            return gr.Button.update(value="Generating...", variant="secondary", interactive=False)

        def post_generate(result):
            """Función que se ejecuta después de la generación"""
            return result, gr.Button.update(value="Generate", variant="primary", interactive=True)

        def generate_with_state(*args):
            try:
                result = flux_tab.generate(
                    *args,
                    text_encoder=None,
                    text_encoder_2=None,
                    tokenizer=None,
                    tokenizer_2=None,
                    debug_enabled=args[11]
                )
                log_text = flux_tab.logger.log("Generation completed")
                return result, log_text
            except Exception as e:
                error_msg = f"Error during generation: {str(e)}"
                log_text = flux_tab.logger.log(error_msg)
                return None, log_text

        # Configurar el evento del botón con pre y post procesamiento
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
                pooled_prompt_embeds_scale_2
            ],
            outputs=[output_image, log_box],
        ).then(
            fn=post_generate,
            inputs=[output_image],
            outputs=[output_image, generate_btn]
        )
        
        # Actualización de botones y visibilidad de controles al cambiar de modo
        # Actualización de botones y visibilidad de controles al cambiar de modo
        canny_btn.click(
            fn=lambda: on_processor_change("canny"),
            inputs=[], 
            outputs=[
                canny_btn, depth_btn, redux_btn,
                low_threshold, high_threshold, detect_resolution, image_resolution,
                processor_id, reference_scale,
                prompt_embeds_scale_1, prompt_embeds_scale_2,
                pooled_prompt_embeds_scale_1, pooled_prompt_embeds_scale_2,
                steps, guidance
            ]
        )
        depth_btn.click(
            fn=lambda: on_processor_change("depth"),
            inputs=[], 
            outputs=[
                canny_btn, depth_btn, redux_btn,
                low_threshold, high_threshold, detect_resolution, image_resolution,
                processor_id, reference_scale,
                prompt_embeds_scale_1, prompt_embeds_scale_2,
                pooled_prompt_embeds_scale_1, pooled_prompt_embeds_scale_2,
                steps, guidance
            ]
        )
        redux_btn.click(
            fn=lambda: on_processor_change("redux"),
            inputs=[], 
            outputs=[
                canny_btn, depth_btn, redux_btn,
                low_threshold, high_threshold, detect_resolution, image_resolution,
                processor_id, reference_scale,
                prompt_embeds_scale_1, prompt_embeds_scale_2,
                pooled_prompt_embeds_scale_1, pooled_prompt_embeds_scale_2,
                steps, guidance
            ]
        )
    return [(flux_interface, "Flux.1 ControlNet", "flux_controlnet_tab")]

# Register the tab
script_callbacks.on_ui_tabs(on_ui_tabs)        