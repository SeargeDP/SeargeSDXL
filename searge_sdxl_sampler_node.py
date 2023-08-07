"""

Custom nodes for SDXL in ComfyUI

MIT License

Copyright (c) 2023 Searge

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import datetime

import comfy.samplers
import comfy_extras.nodes_upscale_model
import comfy_extras.nodes_post_processing
import folder_paths
import json
import nodes
import numpy as np
import os

from comfy.cli_args import args
from datetime import datetime
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .modules.sampling import SeargeSDXLSampler2
from .modules.sampling import SeargeSDXLImage2ImageSampler2
# from .modules.sampling import SeargeSDXLSamplerV3
# from .modules.sampling import SeargeSDXLImage2ImageSamplerV3

from .modules.prompting import SeargeSDXLPromptEncoder
from .modules.prompting import SeargeSDXLBasePromptEncoder
from .modules.prompting import SeargeSDXLRefinerPromptEncoder

from .modules.prompting import SeargePromptText
from .modules.prompting import SeargePromptCombiner

from .modules.processing import SeargeParameterProcessor
from .modules.processing import SeargeStylePreprocessor

from .modules.ui import SeargeInput1
from .modules.ui import SeargeOutput1
from .modules.ui import SeargeInput2
from .modules.ui import SeargeOutput2
from .modules.ui import SeargeInput3
from .modules.ui import SeargeOutput3
from .modules.ui import SeargeInput4
from .modules.ui import SeargeOutput4
from .modules.ui import SeargeInput5
from .modules.ui import SeargeOutput5
from .modules.ui import SeargeInput6
from .modules.ui import SeargeOutput6
from .modules.ui import SeargeInput7
from .modules.ui import SeargeOutput7
from .modules.ui import SeargeGenerated1

from .modules.legacy import SeargeSDXLSampler
from .modules.legacy import SeargeSDXLImage2ImageSampler


# Input: sampler inputs

class SeargeSamplerInputs:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "ddim"}),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "ddim_uniform"}),
                    },
                }

    RETURN_TYPES = ("SAMPLER_NAME", "SCHEDULER_NAME", )
    RETURN_NAMES = ("sampler_name", "scheduler", )
    FUNCTION = "get_value"

    CATEGORY = "Searge/Inputs"

    def get_value(self, sampler_name, scheduler, ):
        return (sampler_name, scheduler, )


# Input: enabler inputs

class SeargeEnablerInputs:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "state": (SeargeParameterProcessor.STATES, {"default": SeargeParameterProcessor.STATES[1]}),
                    },
                }

    RETURN_TYPES = ("ENABLE_STATE", )
    RETURN_NAMES = ("state", )
    FUNCTION = "get_value"

    CATEGORY = "Searge/Inputs"

    def get_value(self, state, ):
        return (state, )


# Input: save folder inputs

class SeargeSaveFolderInputs:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "save_to": (SeargeParameterProcessor.SAVE_TO, {"default": SeargeParameterProcessor.SAVE_TO[0]}),
                    },
                }

    RETURN_TYPES = ("SAVE_FOLDER", )
    RETURN_NAMES = ("save_to", )
    FUNCTION = "get_value"

    CATEGORY = "Searge/Inputs"

    def get_value(self, save_to, ):
        return (save_to, )


# Tool: integer constant

class SeargeIntegerConstant:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "value": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    },
                }

    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("value", )
    FUNCTION = "get_value"

    CATEGORY = "Searge/Integers"

    def get_value(self, value, ):
        return (value,)


# Tool: integer pair

class SeargeIntegerPair:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "value1": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "value2": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    },
                }

    RETURN_TYPES = ("INT", "INT", )
    RETURN_NAMES = ("value 1", "value 2", )
    FUNCTION = "get_value"

    CATEGORY = "Searge/Integers"

    def get_value(self, value1, value2, ):
        return (value1,value2,)


# Tool: integer math

class SeargeIntegerMath:
    OPERATIONS = ["a * b + c", "a + c", "a - c", "a * b", "a / b"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "op": (SeargeIntegerMath.OPERATIONS, {"default": "a * b + c"}),
                    "a": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "b": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
                    "c": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    },
                }

    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("result", )
    FUNCTION = "get_value"

    CATEGORY = "Searge/Integers"

    def get_value(self, op, a, b, c, ):
        res = 0
        if op == "a * b + c":
            res = a * b + c
        elif op == "a + c":
            res = a + c
        elif op == "a - c":
            res = a - c
        elif op == "a * b":
            res = a * b
        elif op == "a / b":
            res = a // b
        return (int(res),)


# Tool: integer scaler

class SeargeIntegerScaler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "value": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "factor": ("FLOAT", {"default": 1.0, "step": 0.01}),
                    "multiple_of": ("INT", {"default": 1, "min": 0, "max": 65536}),
                    },
                }

    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("value", )
    FUNCTION = "get_value"

    CATEGORY = "Searge/Integers"

    def get_value(self, value, factor, multiple_of, ):
        return (int(value * factor // multiple_of) * multiple_of, )


# Tool: float constant

class SeargeFloatConstant:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "value": ("FLOAT", {"default": 0.0, "step": 0.01}),
                    },
                }

    RETURN_TYPES = ("FLOAT", )
    RETURN_NAMES = ("value", )
    FUNCTION = "get_value"

    CATEGORY = "Searge/Floats"

    def get_value(self, value, ):
        return (value,)


# Tool: float pair

class SeargeFloatPair:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "value1": ("FLOAT", {"default": 0.0, "step": 0.01}),
                    "value2": ("FLOAT", {"default": 0.0, "step": 0.01}),
                    },
                }

    RETURN_TYPES = ("FLOAT", "FLOAT", )
    RETURN_NAMES = ("value 1", "value 2", )
    FUNCTION = "get_value"

    CATEGORY = "Searge/Floats"

    def get_value(self, value1, value2, ):
        return (value1,value2,)


# Tool: float math

class SeargeFloatMath:
    OPERATIONS = ["a * b + c", "a + c", "a - c", "a * b", "a / b"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "op": (SeargeFloatMath.OPERATIONS, {"default": "a * b + c"}),
                    "a": ("FLOAT", {"default": 0.0, "step": 0.01}),
                    "b": ("FLOAT", {"default": 1.0, "step": 0.01}),
                    "c": ("FLOAT", {"default": 0.0, "step": 0.01}),
                    },
                }

    RETURN_TYPES = ("FLOAT", )
    RETURN_NAMES = ("result", )
    FUNCTION = "get_value"

    CATEGORY = "Searge/Floats"

    def get_value(self, op, a, b, c, ):
        res = 0.0
        if op == "a * b + c":
            res = a * b + c
        elif op == "a + c":
            res = a + c
        elif op == "a - c":
            res = a - c
        elif op == "a * b":
            res = a * b
        elif op == "a / b":
            res = a / b
        return (res,)


# Util: custom save node (without preview)

class SeargeImageSave:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE", ),
                    "filename_prefix": ("STRING", {"default": "SeargeSDXL-%date%/Image"}),
                    "state": ("ENABLE_STATE", {"default": SeargeParameterProcessor.STATES[1]}),
                    "save_to": ("SAVE_FOLDER", {"default": SeargeParameterProcessor.SAVE_TO[0]}),
                    },
                "hidden": {
                    "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
                    },
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "Searge/Files"

    def save_images(self, images, filename_prefix, state, save_to, prompt=None, extra_pnginfo=None):
        # "disabled"
        if state == SeargeParameterProcessor.STATES[0]:
            return {}

        # "input folder"
        if save_to == SeargeParameterProcessor.SAVE_TO[1]:
            output_dir = folder_paths.get_input_directory()
            filename_prefix = "output-%date%"
        # incl. SeargeParameterProcessor.SAVE_TO[0] -> "output folder"
        else:
            output_dir = folder_paths.get_output_directory()

        filename_prefix = filename_prefix.replace("%date%", datetime.now().strftime("%Y-%m-%d"))

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir, images[0].shape[1], images[0].shape[0])

        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if args.disable_metadata is None or not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)

            counter += 1

        return {}


# Util: custom checkpoint loader

class SeargeCheckpointLoader:
    def __init__(self):
        self.chkp_loader = nodes.CheckpointLoaderSimple()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "ckpt_name": ("CHECKPOINT_NAME", ),
                    },
                }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", )
    FUNCTION = "load_checkpoint"

    CATEGORY = "Searge/Files"

    def load_checkpoint(self, ckpt_name, ):
        return self.chkp_loader.load_checkpoint(ckpt_name)


# Util: custom vae loader

class SeargeVAELoader:
    def __init__(self):
        self.vae_loader = nodes.VAELoader()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae_name": ("VAE_NAME", ),
                    },
                }
    RETURN_TYPES = ("VAE", )
    FUNCTION = "load_vae"

    CATEGORY = "Searge/Files"

    def load_vae(self, vae_name, ):
        return self.vae_loader.load_vae(vae_name)


# Util: custom upscale model loader

class SeargeUpscaleModelLoader:
    def __init__(self):
        self.upscale_model_loader = comfy_extras.nodes_upscale_model.UpscaleModelLoader()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "upscaler_name": ("UPSCALER_NAME", ),
                    },
                }
    RETURN_TYPES = ("UPSCALE_MODEL", )
    FUNCTION = "load_upscaler"

    CATEGORY = "Searge/Files"

    def load_upscaler(self, upscaler_name, ):
        return self.upscale_model_loader.load_model(upscaler_name)


# Util: custom upscale model loader

class SeargeLoraLoader:
    def __init__(self):
        self.lora_loader = nodes.LoraLoader()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    "clip": ("CLIP",),
                    "lora_name": ("LORA_NAME",),
                    "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    },
                }
    RETURN_TYPES = ("MODEL", "CLIP", )
    FUNCTION = "load_lora"

    CATEGORY = "Searge/Files"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip, ):
        return self.lora_loader.load_lora(model, clip, lora_name, strength_model, strength_clip)


# Tool: Muxer for selecting between 3 latent inputs

class SeargeLatentMuxer3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "input0": ("LATENT", ),
                    "input1": ("LATENT", ),
                    "input2": ("LATENT", ),
                    "input_selector": ("INT", {"default": 0, "min": 0, "max": 2}),
                    },
                }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("output", )
    FUNCTION = "mux"

    CATEGORY = "Searge/FlowControl"

    def mux(self, input0, input1, input2, input_selector, ):
        if input_selector == 1:
            return (input1,)
        elif input_selector == 2:
            return (input2,)
        else:
            return (input0, )


# Tool: Muxer for selecting between 5 conditioning inputs

class SeargeConditioningMuxer2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "input0": ("CONDITIONING", ),
                    "input1": ("CONDITIONING", ),
                    "input_selector": ("INT", {"default": 0, "min": 0, "max": 1}),
                    },
                }

    RETURN_TYPES = ("CONDITIONING", )
    RETURN_NAMES = ("output", )
    FUNCTION = "mux"

    CATEGORY = "Searge/FlowControl"

    def mux(self, input0, input1, input_selector, ):
        if input_selector == 1:
            return (input1,)
        else:
            return (input0, )


# Tool: Muxer for selecting between 5 conditioning inputs

class SeargeConditioningMuxer5:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "input0": ("CONDITIONING", ),
                    "input1": ("CONDITIONING", ),
                    "input2": ("CONDITIONING", ),
                    "input3": ("CONDITIONING", ),
                    "input4": ("CONDITIONING", ),
                    "input_selector": ("INT", {"default": 0, "min": 0, "max": 4}),
                    },
                }

    RETURN_TYPES = ("CONDITIONING", )
    RETURN_NAMES = ("output", )
    FUNCTION = "mux"

    CATEGORY = "Searge/FlowControl"

    def mux(self, input0, input1, input2, input3, input4, input_selector, ):
        if input_selector == 1:
            return (input1,)
        elif input_selector == 2:
            return (input2,)
        elif input_selector == 3:
            return (input3,)
        elif input_selector == 4:
            return (input4,)
        else:
            return (input0, )


# Register nodes in ComfyUI

NODE_CLASS_MAPPINGS = {
    "SeargeSDXLSampler2": SeargeSDXLSampler2,
    "SeargeSDXLImage2ImageSampler2": SeargeSDXLImage2ImageSampler2,
#    "SeargeSDXLSamplerV3": SeargeSDXLSamplerV3,
#    "SeargeSDXLImage2ImageSamplerV3": SeargeSDXLImage2ImageSamplerV3,

    "SeargeSamplerInputs": SeargeSamplerInputs,
    "SeargeEnablerInputs": SeargeEnablerInputs,
    "SeargeSaveFolderInputs": SeargeSaveFolderInputs,

    "SeargeSDXLPromptEncoder": SeargeSDXLPromptEncoder,
    "SeargeSDXLBasePromptEncoder": SeargeSDXLBasePromptEncoder,
    "SeargeSDXLRefinerPromptEncoder": SeargeSDXLRefinerPromptEncoder,

    "SeargePromptText": SeargePromptText,
    "SeargePromptCombiner": SeargePromptCombiner,

    "SeargeIntegerConstant": SeargeIntegerConstant,
    "SeargeIntegerPair": SeargeIntegerPair,
    "SeargeIntegerMath": SeargeIntegerMath,
    "SeargeIntegerScaler": SeargeIntegerScaler,

    "SeargeFloatConstant": SeargeFloatConstant,
    "SeargeFloatPair": SeargeFloatPair,
    "SeargeFloatMath": SeargeFloatMath,

    "SeargeImageSave": SeargeImageSave,
    "SeargeCheckpointLoader": SeargeCheckpointLoader,
    "SeargeVAELoader": SeargeVAELoader,
    "SeargeUpscaleModelLoader": SeargeUpscaleModelLoader,
    "SeargeLoraLoader": SeargeLoraLoader,

    "SeargeLatentMuxer3": SeargeLatentMuxer3,
    "SeargeConditioningMuxer2": SeargeConditioningMuxer2,
    "SeargeConditioningMuxer5": SeargeConditioningMuxer5,

    "SeargeParameterProcessor": SeargeParameterProcessor,
    "SeargeStylePreprocessor": SeargeStylePreprocessor,

    "SeargeInput1": SeargeInput1,
    "SeargeOutput1": SeargeOutput1,
    "SeargeInput2": SeargeInput2,
    "SeargeOutput2": SeargeOutput2,
    "SeargeInput3": SeargeInput3,
    "SeargeOutput3": SeargeOutput3,
    "SeargeInput4": SeargeInput4,
    "SeargeOutput4": SeargeOutput4,
    "SeargeInput5": SeargeInput5,
    "SeargeOutput5": SeargeOutput5,
    "SeargeInput6": SeargeInput6,
    "SeargeOutput6": SeargeOutput6,
    "SeargeInput7": SeargeInput7,
    "SeargeOutput7": SeargeOutput7,
    "SeargeGenerated1": SeargeGenerated1,

    "SeargeSDXLSampler": SeargeSDXLSampler,
    "SeargeSDXLImage2ImageSampler": SeargeSDXLImage2ImageSampler,
}


# Human readable names for the nodes

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeargeSDXLSampler2": "SDXL Sampler v2 (Searge)",
    "SeargeSDXLImage2ImageSampler2": "Image2Image Sampler v2 (Searge)",
#    "SeargeSDXLSamplerV3": "SDXL Sampler v3 (Searge)",
#    "SeargeSDXLImage2ImageSamplerV3": "Image2Image Sampler v3 (Searge)",

    "SeargeSamplerInputs": "Sampler Settings",
    "SeargeEnablerInputs": "Enable / Disable",
    "SeargeSaveFolderInputs": "Save Folder",

    "SeargeSDXLPromptEncoder": "SDXL Prompt Encoder (Searge)",
    "SeargeSDXLBasePromptEncoder": "SDXL Base Prompt Encoder (Searge)",
    "SeargeSDXLRefinerPromptEncoder": "SDXL Refiner Prompt Encoder (Searge)",

    "SeargePromptText": "Prompt text input",
    "SeargePromptCombiner": "Prompt combiner",

    "SeargeIntegerConstant": "Integer Constant",
    "SeargeIntegerPair": "Integer Pair",
    "SeargeIntegerMath": "Integer Math",
    "SeargeIntegerScaler": "Integer Scaler",

    "SeargeFloatConstant": "Float Constant",
    "SeargeFloatPair": "Float Pair",
    "SeargeFloatMath": "Float Math",

    "SeargeImageSave": "Save Image (Searge)",
    "SeargeCheckpointLoader": "Checkpoint Loader (Searge)",
    "SeargeVAELoader": "VAE Loader (Searge)",
    "SeargeUpscaleModelLoader": "Upscale Model Loader (Searge)",
    "SeargeLoraLoader": "Lora Loader (Searge)",

    "SeargeLatentMuxer3": "3-Way Muxer for Latents",
    "SeargeConditioningMuxer2": "2-Way Muxer for Conditioning",
    "SeargeConditioningMuxer5": "5-Way Muxer for Conditioning",

    "SeargeParameterProcessor": "Parameter Processor",
    "SeargeStylePreprocessor": "Style Preprocessor (wip)",

    "SeargeInput1": "Prompts",
    "SeargeOutput1": "Prompts",
    "SeargeInput2": "Generation Parameters",
    "SeargeOutput2": "Generation Parameters",
    "SeargeInput3": "Advanced Parameters",
    "SeargeOutput3": "Advanced Parameters",
    "SeargeInput4": "Model Names",
    "SeargeOutput4": "Model Names",
    "SeargeInput5": "Prompt Processing",
    "SeargeOutput5": "Prompt Processing",
    "SeargeInput6": "HiResFix Parameters",
    "SeargeOutput6": "HiResFix Parameters",
    "SeargeInput7": "Misc Parameters",
    "SeargeOutput7": "Misc Parameters",
    "SeargeGenerated1": "Flow Control Parameters",

    "SeargeSDXLSampler": "Legacy SDXL Sampler (Searge)",
    "SeargeSDXLImage2ImageSampler": "Legacy Image2Image Sampler (Searge)",
}
