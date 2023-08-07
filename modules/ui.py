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

import comfy.samplers
import folder_paths
import nodes

from .processing import SeargeParameterProcessor


# UI: Prompt Inputs

class SeargeInput1:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "main_prompt": ("STRING", {"multiline": True, "default": ""}),
                    "secondary_prompt": ("STRING", {"multiline": True, "default": ""}),
                    "style_prompt": ("STRING", {"multiline": True, "default": ""}),
                    "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                    "negative_style": ("STRING", {"multiline": True, "default": ""}),
                    },
                "optional": {
                    "inputs": ("PARAMETER_INPUTS", ),
                    "image": ("IMAGE",),
                    "mask": ("MASK",),
                    },
                }

    RETURN_TYPES = ("PARAMETER_INPUTS", )
    RETURN_NAMES = ("inputs", )
    FUNCTION = "mux"

    CATEGORY = "Searge/UI/Inputs"

    def mux(self, main_prompt, secondary_prompt, style_prompt, negative_prompt, negative_style, inputs=None, image=None, mask=None):
        if inputs is None:
            parameters = {}
        else:
            parameters = inputs

        parameters["main_prompt"] = main_prompt
        parameters["secondary_prompt"] = secondary_prompt
        parameters["style_prompt"] = style_prompt
        parameters["negative_prompt"] = negative_prompt
        parameters["negative_style"] = negative_style
        parameters["image"] = image
        parameters["mask"] = mask

        return (parameters, )


# UI: Prompt Outputs

class SeargeOutput1:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "parameters": ("PARAMETERS", ),
                    },
                }

    RETURN_TYPES = ("PARAMETERS", "STRING", "STRING", "STRING", "STRING", "STRING", "IMAGE", "MASK", )
    RETURN_NAMES = ("parameters", "main_prompt", "secondary_prompt", "style_prompt", "negative_prompt", "negative_style", "image", "mask", )
    FUNCTION = "demux"

    CATEGORY = "Searge/UI/Outputs"

    def demux(self, parameters):
        main_prompt = parameters["main_prompt"]
        secondary_prompt = parameters["secondary_prompt"]
        style_prompt = parameters["style_prompt"]
        negative_prompt = parameters["negative_prompt"]
        negative_style = parameters["negative_style"]
        image = parameters["image"]
        mask = parameters["mask"]

        return (parameters, main_prompt, secondary_prompt, style_prompt, negative_prompt, negative_style, image, mask, )


# UI: Generation Parameters Input

class SeargeInput2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "image_width": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "image_height": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "steps": ("INT", {"default": 20, "min": 0, "max": 200}),
                    "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.5}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "ddim"}),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "ddim_uniform"}),
                    "save_image": (SeargeParameterProcessor.STATES, {"default": SeargeParameterProcessor.STATES[1]}),
                    "save_directory": (SeargeParameterProcessor.SAVE_TO, {"default": SeargeParameterProcessor.SAVE_TO[0]}),
                    },
                "optional": {
                    "inputs": ("PARAMETER_INPUTS", ),
                    },
                }

    RETURN_TYPES = ("PARAMETER_INPUTS", )
    RETURN_NAMES = ("inputs", )
    FUNCTION = "mux"

    CATEGORY = "Searge/UI/Inputs"

    def mux(self, seed, image_width, image_height, steps, cfg, sampler_name, scheduler, save_image, save_directory, inputs=None):
        if inputs is None:
            parameters = {}
        else:
            parameters = inputs

        parameters["seed"] = seed
        parameters["image_width"] = image_width
        parameters["image_height"] = image_height
        parameters["steps"] = steps
        parameters["cfg"] = cfg
        parameters["sampler_name"] = sampler_name
        parameters["scheduler"] = scheduler
        parameters["save_image"] = save_image
        parameters["save_directory"] = save_directory

        return (parameters, )


# UI: Generation Parameters Output

class SeargeOutput2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "parameters": ("PARAMETERS", ),
                    },
                }

    RETURN_TYPES = ("PARAMETERS", "INT", "INT", "INT", "INT", "FLOAT", "SAMPLER_NAME", "SCHEDULER_NAME", "ENABLE_STATE", "SAVE_FOLDER", )
    RETURN_NAMES = ("parameters", "seed", "image_width", "image_height", "steps", "cfg", "sampler_name", "scheduler", "save_image", "save_directory", )
    FUNCTION = "demux"

    CATEGORY = "Searge/UI/Outputs"

    def demux(self, parameters):
        seed = parameters["seed"]
        image_width = parameters["image_width"]
        image_height = parameters["image_height"]
        steps = parameters["steps"]
        cfg = parameters["cfg"]
        sampler_name = parameters["sampler_name"]
        scheduler = parameters["scheduler"]
        save_image = parameters["save_image"]
        save_directory = parameters["save_directory"]

        return (parameters, seed, image_width, image_height, steps, cfg, sampler_name, scheduler, save_image, save_directory, )


# UI: Advanced Parameters Input

class SeargeInput3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "base_ratio": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "refiner_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
                    "refiner_intensity": (SeargeParameterProcessor.REFINER_INTENSITY, {"default": SeargeParameterProcessor.REFINER_INTENSITY[1]}),
                    "precondition_steps": ("INT", {"default": 0, "min": 0, "max": 10}),
                    "batch_size": ("INT", {"default": 1, "min": 1, "max": 4}),
                    "upscale_resolution_factor": ("FLOAT", {"default": 2.0, "min": 0.25, "max": 4.0, "step": 0.25}),
                    "save_upscaled_image": (SeargeParameterProcessor.STATES, {"default": SeargeParameterProcessor.STATES[1]}),
                    },
                "optional": {
                    "inputs": ("PARAMETER_INPUTS", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    },
                }

    RETURN_TYPES = ("PARAMETER_INPUTS", )
    RETURN_NAMES = ("inputs", )
    FUNCTION = "mux"

    CATEGORY = "Searge/UI/Inputs"

    def mux(self, base_ratio, refiner_strength, refiner_intensity, precondition_steps, batch_size, upscale_resolution_factor, save_upscaled_image, inputs=None, denoise=None):
        if inputs is None:
            parameters = {}
        else:
            parameters = inputs

        parameters["denoise"] = denoise
        parameters["base_ratio"] = base_ratio
        parameters["refiner_strength"] = refiner_strength
        parameters["refiner_intensity"] = refiner_intensity
        parameters["precondition_steps"] = precondition_steps
        parameters["batch_size"] = batch_size
        parameters["upscale_resolution_factor"] = upscale_resolution_factor
        parameters["save_upscaled_image"] = save_upscaled_image

        return (parameters, )


# UI: Advanced Parameters Output

class SeargeOutput3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "parameters": ("PARAMETERS", ),
                    },
                }

    RETURN_TYPES = ("PARAMETERS", "FLOAT", "FLOAT", "FLOAT", "INT", "INT", "INT", "FLOAT", "ENABLE_STATE", )
    RETURN_NAMES = ("parameters", "denoise", "base_ratio", "refiner_strength", "noise_offset", "precondition_steps", "batch_size", "upscale_resolution_factor", "save_upscaled_image", )
    FUNCTION = "demux"

    CATEGORY = "Searge/UI/Outputs"

    def demux(self, parameters):
        denoise = parameters["denoise"]
        base_ratio = parameters["base_ratio"]
        refiner_strength = parameters["refiner_strength"]
        noise_offset = parameters["noise_offset"]
        precondition_steps = parameters["precondition_steps"]
        batch_size = parameters["batch_size"]
        upscale_resolution_factor = parameters["upscale_resolution_factor"]
        save_upscaled_image = parameters["save_upscaled_image"]

        return (parameters, denoise, base_ratio, refiner_strength, noise_offset, precondition_steps, batch_size, upscale_resolution_factor, save_upscaled_image, )


# UI: Model Selector Input

class SeargeInput4:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "base_model": (folder_paths.get_filename_list("checkpoints"), ),
                    "refiner_model": (folder_paths.get_filename_list("checkpoints"), ),
                    "vae_model": (folder_paths.get_filename_list("vae"), ),
                    "main_upscale_model": (folder_paths.get_filename_list("upscale_models"),),
                    "support_upscale_model": (folder_paths.get_filename_list("upscale_models"),),
                    "lora_model": (folder_paths.get_filename_list("loras"),),
                    },
                "optional": {
                    "model_settings": ("MODEL_SETTINGS", ),
                    },
                }

    RETURN_TYPES = ("MODEL_NAMES", )
    RETURN_NAMES = ("model_names", )
    FUNCTION = "mux"

    CATEGORY = "Searge/UI/Inputs"

    def mux(self, base_model, refiner_model, vae_model, main_upscale_model, support_upscale_model, lora_model, model_settings=None):
        if model_settings is None:
            model_names = {}
        else:
            model_names = model_settings

        model_names["base_model"] = base_model
        model_names["refiner_model"] = refiner_model
        model_names["vae_model"] = vae_model
        model_names["main_upscale_model"] = main_upscale_model
        model_names["support_upscale_model"] = support_upscale_model
        model_names["lora_model"] = lora_model

        return (model_names, )


# UI: Model Selector

class SeargeOutput4:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model_names": ("MODEL_NAMES", ),
                    },
                }

    RETURN_TYPES = ("MODEL_NAMES", "CHECKPOINT_NAME", "CHECKPOINT_NAME", "VAE_NAME", "UPSCALER_NAME", "UPSCALER_NAME", "LORA_NAME", )
    RETURN_NAMES = ("model_names", "base_model", "refiner_model", "vae_model", "main_upscale_model", "support_upscale_model", "lora_model", )
    FUNCTION = "demux"

    CATEGORY = "Searge/UI/Outputs"

    def demux(self, model_names):
        base_model = model_names["base_model"]
        refiner_model = model_names["refiner_model"]
        vae_model = model_names["vae_model"]
        main_upscale_model = model_names["main_upscale_model"]
        support_upscale_model = model_names["support_upscale_model"]
        lora_model = model_names["lora_model"]

        return (model_names, base_model, refiner_model, vae_model, main_upscale_model, support_upscale_model, lora_model,)


# UI: Prompt Processing Input

class SeargeInput5:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "base_conditioning_scale": ("FLOAT", {"default": 2.0, "min": 0.25, "max": 4.0, "step": 0.25}),
                    "refiner_conditioning_scale": ("FLOAT", {"default": 2.0, "min": 0.25, "max": 4.0, "step": 0.25}),
                    "style_prompt_power": ("FLOAT", {"default": 0.33, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "negative_style_power": ("FLOAT", {"default": 0.67, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "style_template": (SeargeParameterProcessor.STYLE_TEMPLATE, {"default": SeargeParameterProcessor.STYLE_TEMPLATE[0]}),
                    },
                "optional": {
                    "inputs": ("PARAMETER_INPUTS", ),
                    },
                }

    RETURN_TYPES = ("PARAMETER_INPUTS", )
    RETURN_NAMES = ("inputs", )
    FUNCTION = "mux"

    CATEGORY = "Searge/UI/Inputs"

    def mux(self, base_conditioning_scale, refiner_conditioning_scale, style_prompt_power, negative_style_power, style_template, inputs=None):
        if inputs is None:
            parameters = {}
        else:
            parameters = inputs

        parameters["base_conditioning_scale"] = base_conditioning_scale
        parameters["refiner_conditioning_scale"] = refiner_conditioning_scale
        parameters["style_prompt_power"] = style_prompt_power
        parameters["negative_style_power"] = negative_style_power
        parameters["style_template"] = style_template

        return (parameters, )


# UI: Prompt Processing Output

class SeargeOutput5:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "parameters": ("PARAMETERS", ),
                    },
                }

    RETURN_TYPES = ("PARAMETERS", "FLOAT", "FLOAT", "FLOAT", "FLOAT", )
    RETURN_NAMES = ("parameters", "base_conditioning_scale", "refiner_conditioning_scale", "style_prompt_power", "negative_style_power", )
    FUNCTION = "demux"

    CATEGORY = "Searge/UI/Outputs"

    def demux(self, parameters):
        base_conditioning_scale = parameters["base_conditioning_scale"]
        refiner_conditioning_scale = parameters["refiner_conditioning_scale"]
        style_prompt_power = parameters["style_prompt_power"]
        negative_style_power = parameters["negative_style_power"]

        return (parameters, base_conditioning_scale, refiner_conditioning_scale, style_prompt_power, negative_style_power, )


# UI: HiResFix Parameters Input

class SeargeInput6:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "hires_fix": (SeargeParameterProcessor.STATES, {"default": SeargeParameterProcessor.STATES[1]}),
                    "hrf_steps": ("INT", {"default": 0, "min": 0, "max": 100}),
                    "hrf_denoise": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "hrf_upscale_factor": ("FLOAT", {"default": 1.5, "min": 0.25, "max": 4.0, "step": 0.25}),
                    "hrf_intensity": (SeargeParameterProcessor.REFINER_INTENSITY, {"default": SeargeParameterProcessor.REFINER_INTENSITY[1]}),
                    "hrf_seed_offset": (SeargeParameterProcessor.HRF_SEED_OFFSET, {"default": SeargeParameterProcessor.HRF_SEED_OFFSET[1]}),
                    "hrf_smoothness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                    },
                "optional": {
                    "inputs": ("PARAMETER_INPUTS", ),
                    },
                }

    RETURN_TYPES = ("PARAMETER_INPUTS", )
    RETURN_NAMES = ("inputs", )
    FUNCTION = "mux"

    CATEGORY = "Searge/UI/Inputs"

    def mux(self, hires_fix, hrf_steps, hrf_denoise, hrf_upscale_factor, hrf_intensity, hrf_seed_offset, hrf_smoothness, inputs=None):
        if inputs is None:
            parameters = {}
        else:
            parameters = inputs

        parameters["hires_fix"] = hires_fix
        parameters["hrf_steps"] = hrf_steps
        parameters["hrf_denoise"] = hrf_denoise
        parameters["hrf_upscale_factor"] = hrf_upscale_factor
        parameters["hrf_intensity"] = hrf_intensity
        parameters["hrf_seed_offset"] = hrf_seed_offset
        parameters["hrf_smoothness"] = hrf_smoothness

        return (parameters, )


# UI: HiResFix Parameters Output

class SeargeOutput6:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "parameters": ("PARAMETERS", ),
                    },
                }

    RETURN_TYPES = ("PARAMETERS", "INT", "FLOAT", "FLOAT", "INT", "INT", "ENABLE_STATE", "FLOAT", )
    RETURN_NAMES = ("parameters", "hrf_steps", "hrf_denoise", "hrf_upscale_factor", "hrf_noise_offset", "hrf_seed", "hires_fix", "hrf_smoothness", )
    FUNCTION = "demux"

    CATEGORY = "Searge/UI/Outputs"

    def demux(self, parameters):
        hrf_steps = parameters["hrf_steps"]
        hrf_denoise = parameters["hrf_denoise"]
        hrf_upscale_factor = parameters["hrf_upscale_factor"]
        hrf_noise_offset = parameters["hrf_noise_offset"]
        hrf_seed = parameters["hrf_seed"]
        hires_fix = parameters["hires_fix"]
        hrf_smoothness = parameters["hrf_smoothness"]

        return (parameters, hrf_steps, hrf_denoise, hrf_upscale_factor, hrf_noise_offset, hrf_seed, hires_fix, hrf_smoothness, )


# UI: Misc Inputs

class SeargeInput7:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "lora_strength": ("FLOAT", {"default": 0.2, "min": -10.0, "max": 10.0, "step": 0.05}),
                    "operation_mode": (SeargeParameterProcessor.OPERATION_MODE, {"default": SeargeParameterProcessor.OPERATION_MODE[0]}),
                    "prompt_style": (SeargeParameterProcessor.PROMPT_STYLE, {"default": SeargeParameterProcessor.PROMPT_STYLE[0]}),
                    },
                "optional": {
                    "inputs": ("PARAMETER_INPUTS", ),
                    },
                }

    RETURN_TYPES = ("PARAMETER_INPUTS", )
    RETURN_NAMES = ("inputs", )
    FUNCTION = "mux"

    CATEGORY = "Searge/UI/Inputs"

    def mux(self, lora_strength, operation_mode, prompt_style, inputs=None):
        if inputs is None:
            parameters = {}
        else:
            parameters = inputs

        parameters["lora_strength"] = lora_strength
        parameters["operation_mode"] = operation_mode
        parameters["prompt_style"] = prompt_style

        return (parameters, )


# UI: Misc Outputs

class SeargeOutput7:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "parameters": ("PARAMETERS", ),
                    },
                }

    RETURN_TYPES = ("PARAMETERS", "FLOAT", )
    RETURN_NAMES = ("parameters", "lora_strength", )
    FUNCTION = "demux"

    CATEGORY = "Searge/UI/Outputs"

    def demux(self, parameters):
        lora_strength = parameters["lora_strength"]

        return (parameters, lora_strength, )


# UI: Generated outputs for flow control

class SeargeGenerated1:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "parameters": ("PARAMETERS", ),
                    },
                }

    RETURN_TYPES = ("PARAMETERS", "INT", "INT", "INT", )
    RETURN_NAMES = ("parameters", "operation_selector", "prompt_style_selector", "prompt_style_group", )
    FUNCTION = "demux"

    CATEGORY = "Searge/UI/Generated"

    def demux(self, parameters):
        operation_selector = parameters["operation_selector"]
        prompt_style_selector = parameters["prompt_style_selector"]
        prompt_style_group = parameters["prompt_style_group"]

        return (parameters, operation_selector, prompt_style_selector, prompt_style_group, )
