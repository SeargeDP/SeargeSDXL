"""

Custom nodes for SDXL in ComfyUI

MIT License

Copyright (c) 2023 SeargeDP

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
import comfy_extras.nodes_upscale_model
import folder_paths
import nodes


# SDXL Sampler with base and refiner support

class SeargeSDXLSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "base_model": ("MODEL",),
                    "base_positive": ("CONDITIONING", ),
                    "base_negative": ("CONDITIONING", ),
                    "refiner_model": ("MODEL",),
                    "refiner_positive": ("CONDITIONING", ),
                    "refiner_negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xfffffffffffffff0}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                    "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.5}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "ddim"}),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "ddim_uniform"}),
                    "base_ratio": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    },
                "optional": {
                    "refiner_prep_steps": ("INT", {"default": 0, "min": 0, "max": 10}),
                    "noise_offset": ("INT", {"default": 1, "min": 0, "max": 1}),
                    "refiner_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.1}),
                    },
                }

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "sample"

    CATEGORY = "Searge/Sampling"

    def sample(self, base_model, base_positive, base_negative, refiner_model, refiner_positive, refiner_negative, latent_image, noise_seed, steps, cfg, sampler_name, scheduler, base_ratio, denoise, refiner_prep_steps=None, noise_offset=None, refiner_strength=None):
        base_steps = int(steps * base_ratio)

        if noise_offset is None:
            noise_offset = 1

        if refiner_strength is None:
            refiner_strength = 1.0

        if refiner_strength < 0.01:
            refiner_strength = 0.01

        if denoise < 0.01:
            return (latent_image, )

        start_at_step = 0
        input_latent = latent_image

        if refiner_prep_steps is not None:
            if refiner_prep_steps >= base_steps:
                refiner_prep_steps = base_steps - 1

            if refiner_prep_steps > 0:
                start_at_step = refiner_prep_steps
                precondition_result = nodes.common_ksampler(refiner_model, noise_seed + 2, steps, cfg, sampler_name, scheduler, refiner_positive, refiner_negative, latent_image, denoise=denoise, disable_noise=False, start_step=steps - refiner_prep_steps, last_step=steps, force_full_denoise=False)
                input_latent = precondition_result[0]

        if base_steps >= steps:
            return nodes.common_ksampler(base_model, noise_seed, steps, cfg, sampler_name, scheduler, base_positive, base_negative, input_latent, denoise=denoise, disable_noise=False, start_step=start_at_step, last_step=steps, force_full_denoise=True)

        base_result = nodes.common_ksampler(base_model, noise_seed, steps, cfg, sampler_name, scheduler, base_positive, base_negative, input_latent, denoise=denoise, disable_noise=False, start_step=start_at_step, last_step=base_steps, force_full_denoise=True)

        return nodes.common_ksampler(refiner_model, noise_seed + noise_offset, steps, cfg, sampler_name, scheduler, refiner_positive, refiner_negative, base_result[0], denoise=denoise * refiner_strength, disable_noise=False, start_step=base_steps, last_step=steps, force_full_denoise=True)


# SDXL Image2Image Sampler (incl. HiRes Fix)

class SeargeSDXLImage2ImageSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "base_model": ("MODEL",),
                    "base_positive": ("CONDITIONING", ),
                    "base_negative": ("CONDITIONING", ),
                    "refiner_model": ("MODEL",),
                    "refiner_positive": ("CONDITIONING",),
                    "refiner_negative": ("CONDITIONING",),
                    "image": ("IMAGE", ),
                    "vae": ("VAE",),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xfffffffffffffff0}),
                    "steps": ("INT", {"default": 20, "min": 0, "max": 200}),
                    "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.5}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "ddim"}),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "ddim_uniform"}),
                    "base_ratio": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "denoise": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                    },
                "optional": {
                    "upscale_model": ("UPSCALE_MODEL",),
                    "scaled_width": ("INT", {"default": 1536, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "scaled_height": ("INT", {"default": 1536, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "noise_offset": ("INT", {"default": 1, "min": 0, "max": 1}),
                    "refiner_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.1}),
                    },
                }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "sample"

    CATEGORY = "Searge/Sampling"

    def sample(self, base_model, base_positive, base_negative, refiner_model, refiner_positive, refiner_negative, image, vae, noise_seed, steps, cfg, sampler_name, scheduler, base_ratio, denoise, upscale_model=None, scaled_width=None, scaled_height=None, noise_offset=None, refiner_strength=None):
        base_steps = int(steps * base_ratio)

        if noise_offset is None:
            noise_offset = 1

        if refiner_strength is None:
            refiner_strength = 1.0

        if refiner_strength < 0.01:
            refiner_strength = 0.01

        if steps < 1:
            return (image, )

        scaled_image = image

        if upscale_model is not None:
            upscale_result = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel().upscale(upscale_model, image)
            scaled_image = upscale_result[0]

        if scaled_width is not None and scaled_height is not None:
            upscale_result = nodes.ImageScale().upscale(scaled_image, "bicubic", scaled_width, scaled_height, "center")
            scaled_image = upscale_result[0]

        if denoise < 0.01:
            return (scaled_image, )

        vae_encode_result = nodes.VAEEncode().encode(vae, scaled_image)
        input_latent = vae_encode_result[0]

        if base_steps >= steps:
            result_latent = nodes.common_ksampler(base_model, noise_seed, steps, cfg, sampler_name, scheduler, base_positive, base_negative, input_latent, denoise=denoise, disable_noise=False, start_step=0, last_step=steps, force_full_denoise=True)
        else:
            base_result = nodes.common_ksampler(base_model, noise_seed, steps, cfg, sampler_name, scheduler, base_positive, base_negative, input_latent, denoise=denoise, disable_noise=False, start_step=0, last_step=base_steps, force_full_denoise=True)
            result_latent = nodes.common_ksampler(refiner_model, noise_seed + noise_offset, steps, cfg, sampler_name, scheduler, refiner_positive, refiner_negative, base_result[0], denoise=denoise * refiner_strength, disable_noise=False, start_step=base_steps, last_step=steps, force_full_denoise=True)

        vae_decode_result = nodes.VAEDecode().decode(vae, result_latent[0])
        output_image = vae_decode_result[0]

        return (output_image, )


# SDXL CLIP Text Encoder for prompts with base and refiner support

class SeargeSDXLPromptEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "base_clip": ("CLIP", ),
                    "refiner_clip": ("CLIP", ),
                    "pos_g": ("STRING", {"multiline": True, "default": "POS_G"}),
                    "pos_l": ("STRING", {"multiline": True, "default": "POS_L"}),
                    "pos_r": ("STRING", {"multiline": True, "default": "POS_R"}),
                    "neg_g": ("STRING", {"multiline": True, "default": "NEG_G"}),
                    "neg_l": ("STRING", {"multiline": True, "default": "NEG_L"}),
                    "neg_r": ("STRING", {"multiline": True, "default": "NEG_R"}),
                    "base_width": ("INT", {"default": 4096, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "base_height": ("INT", {"default": 4096, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "crop_w": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "crop_h": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "target_width": ("INT", {"default": 4096, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "target_height": ("INT", {"default": 4096, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "pos_ascore": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                    "neg_ascore": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 1000.0, "step": 0.01}),
                    "refiner_width": ("INT", {"default": 2048, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "refiner_height": ("INT", {"default": 2048, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    },
                }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "CONDITIONING", )
    RETURN_NAMES = ("base_positive", "base_negative", "refiner_positive", "refiner_negative", )
    FUNCTION = "encode"

    CATEGORY = "Searge/ClipEncoding"

    def encode(self, base_clip, refiner_clip, pos_g, pos_l, pos_r, neg_g, neg_l, neg_r, base_width, base_height, crop_w, crop_h, target_width, target_height, pos_ascore, neg_ascore, refiner_width, refiner_height, ):
        empty = base_clip.tokenize("")

        # positive base prompt
        tokens1 = base_clip.tokenize(pos_g)
        tokens1["l"] = base_clip.tokenize(pos_l)["l"]

        if len(tokens1["l"]) != len(tokens1["g"]):
            while len(tokens1["l"]) < len(tokens1["g"]):
                tokens1["l"] += empty["l"]
            while len(tokens1["l"]) > len(tokens1["g"]):
                tokens1["g"] += empty["g"]

        cond1, pooled1 = base_clip.encode_from_tokens(tokens1, return_pooled=True)
        res1 = [[cond1, {"pooled_output": pooled1, "width": base_width, "height": base_height, "crop_w": crop_w, "crop_h": crop_h, "target_width": target_width, "target_height": target_height}]]

        # negative base prompt
        tokens2 = base_clip.tokenize(neg_g)
        tokens2["l"] = base_clip.tokenize(neg_l)["l"]

        if len(tokens2["l"]) != len(tokens2["g"]):
            while len(tokens2["l"]) < len(tokens2["g"]):
                tokens2["l"] += empty["l"]
            while len(tokens2["l"]) > len(tokens2["g"]):
                tokens2["g"] += empty["g"]

        cond2, pooled2 = base_clip.encode_from_tokens(tokens2, return_pooled=True)
        res2 = [[cond2, {"pooled_output": pooled2, "width": base_width, "height": base_height, "crop_w": crop_w, "crop_h": crop_h, "target_width": target_width, "target_height": target_height}]]


        # positive refiner prompt
        tokens3 = refiner_clip.tokenize(pos_r)
        cond3, pooled3 = refiner_clip.encode_from_tokens(tokens3, return_pooled=True)
        res3 = [[cond3, {"pooled_output": pooled3, "aesthetic_score": pos_ascore, "width": refiner_width, "height": refiner_height}]]

        # negative refiner prompt
        tokens4 = refiner_clip.tokenize(neg_r)
        cond4, pooled4 = refiner_clip.encode_from_tokens(tokens4, return_pooled=True)
        res4 = [[cond4, {"pooled_output": pooled4, "aesthetic_score": neg_ascore, "width": refiner_width, "height": refiner_height}]]

        return (res1, res2, res3, res4, )


# SDXL CLIP Text Encoder for base prompts

class SeargeSDXLBasePromptEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "base_clip": ("CLIP", ),
                    "pos_g": ("STRING", {"multiline": True, "default": "POS_G"}),
                    "pos_l": ("STRING", {"multiline": True, "default": "POS_L"}),
                    "neg_g": ("STRING", {"multiline": True, "default": "NEG_G"}),
                    "neg_l": ("STRING", {"multiline": True, "default": "NEG_L"}),
                    "base_width": ("INT", {"default": 4096, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "base_height": ("INT", {"default": 4096, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "crop_w": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "crop_h": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "target_width": ("INT", {"default": 4096, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "target_height": ("INT", {"default": 4096, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    },
                }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", )
    RETURN_NAMES = ("base_positive", "base_negative", )
    FUNCTION = "encode"

    CATEGORY = "Searge/ClipEncoding"

    def encode(self, base_clip, pos_g, pos_l, neg_g, neg_l, base_width, base_height, crop_w, crop_h, target_width, target_height, ):
        empty = base_clip.tokenize("")

        # positive base prompt
        tokens1 = base_clip.tokenize(pos_g)
        tokens1["l"] = base_clip.tokenize(pos_l)["l"]

        if len(tokens1["l"]) != len(tokens1["g"]):
            while len(tokens1["l"]) < len(tokens1["g"]):
                tokens1["l"] += empty["l"]
            while len(tokens1["l"]) > len(tokens1["g"]):
                tokens1["g"] += empty["g"]

        cond1, pooled1 = base_clip.encode_from_tokens(tokens1, return_pooled=True)
        res1 = [[cond1, {"pooled_output": pooled1, "width": base_width, "height": base_height, "crop_w": crop_w, "crop_h": crop_h, "target_width": target_width, "target_height": target_height}]]

        # negative base prompt
        tokens2 = base_clip.tokenize(neg_g)
        tokens2["l"] = base_clip.tokenize(neg_l)["l"]

        if len(tokens2["l"]) != len(tokens2["g"]):
            while len(tokens2["l"]) < len(tokens2["g"]):
                tokens2["l"] += empty["l"]
            while len(tokens2["l"]) > len(tokens2["g"]):
                tokens2["g"] += empty["g"]

        cond2, pooled2 = base_clip.encode_from_tokens(tokens2, return_pooled=True)
        res2 = [[cond2, {"pooled_output": pooled2, "width": base_width, "height": base_height, "crop_w": crop_w, "crop_h": crop_h, "target_width": target_width, "target_height": target_height}]]

        return (res1, res2, )


# SDXL CLIP Text Encoder for prompts with base and refiner support

class SeargeSDXLRefinerPromptEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "refiner_clip": ("CLIP", ),
                    "pos_r": ("STRING", {"multiline": True, "default": "POS_R"}),
                    "neg_r": ("STRING", {"multiline": True, "default": "NEG_R"}),
                    "pos_ascore": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                    "neg_ascore": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 1000.0, "step": 0.01}),
                    "refiner_width": ("INT", {"default": 2048, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "refiner_height": ("INT", {"default": 2048, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    },
                }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", )
    RETURN_NAMES = ("refiner_positive", "refiner_negative", )
    FUNCTION = "encode"

    CATEGORY = "Searge/ClipEncoding"

    def encode(self, refiner_clip, pos_r, neg_r, pos_ascore, neg_ascore, refiner_width, refiner_height, ):

        # positive refiner prompt
        tokens1 = refiner_clip.tokenize(pos_r)
        cond1, pooled1 = refiner_clip.encode_from_tokens(tokens1, return_pooled=True)
        res1 = [[cond1, {"pooled_output": pooled1, "aesthetic_score": pos_ascore, "width": refiner_width, "height": refiner_height}]]

        # negative refiner prompt
        tokens2 = refiner_clip.tokenize(neg_r)
        cond2, pooled2 = refiner_clip.encode_from_tokens(tokens2, return_pooled=True)
        res2 = [[cond2, {"pooled_output": pooled2, "aesthetic_score": neg_ascore, "width": refiner_width, "height": refiner_height}]]

        return (res1, res2, )


# Tool: text input node for prompt text

class SeargePromptText:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "prompt": ("STRING", {"default": "", "multiline": True}),
                    },
                }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("prompt", )
    FUNCTION = "get_value"

    CATEGORY = "Searge/Prompting"

    def get_value(self, prompt):
        return (prompt,)


# Tool: text input node for prompt text

class SeargePromptCombiner:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "prompt1": ("STRING", {"default": "", "multiline": True}),
                    "separator": ("STRING", {"default": ", ", "multiline": False}),
                    "prompt2": ("STRING", {"default": "", "multiline": True}),
                    },
                }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("combined prompt", )
    FUNCTION = "get_value"

    CATEGORY = "Searge/Prompting"

    def get_value(self, prompt1, separator, prompt2, ):
        len1 = len(prompt1)
        len2 = len(prompt2)
        prompt = ""
        if len1 > 0 and len2 > 0:
            prompt = prompt1 + separator + prompt2
        elif len1 > 0:
            prompt = prompt1
        elif len2 > 0:
            prompt = prompt2
        return (prompt,)


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


# UI: Parameter Processor

class SeargeParameterProcessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "inputs": ("PARAMETER_INPUTS", ),
                    },
                "optional": {
                    },
                }

    RETURN_TYPES = ("PARAMETERS", )
    RETURN_NAMES = ("parameters", )
    FUNCTION = "process"

    CATEGORY = "Searge/UI"

    def process(self, inputs):
        if inputs is None:
            parameters = {}
        else:
            parameters = inputs

        if parameters["denoise"] is None:
            parameters["denoise"] = 1.0

        saturation = parameters["refiner_saturation"]
        if saturation is not None:
            if saturation == SeargeInput3.REFINER_SATURATION[0]:
                parameters["noise_offset"] = 0
            elif saturation == SeargeInput3.REFINER_SATURATION[1]:
                parameters["noise_offset"] = 1
            else:
                parameters["noise_offset"] = 1

        return (parameters, )


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
                    },
                }

    RETURN_TYPES = ("PARAMETER_INPUTS", )
    RETURN_NAMES = ("inputs", )
    FUNCTION = "mux"

    CATEGORY = "Searge/UI/Inputs"

    def mux(self, main_prompt, secondary_prompt, style_prompt, negative_prompt, negative_style, inputs=None):
        if inputs is None:
            parameters = {}
        else:
            parameters = inputs

        parameters["main_prompt"] = main_prompt
        parameters["secondary_prompt"] = secondary_prompt
        parameters["style_prompt"] = style_prompt
        parameters["negative_prompt"] = negative_prompt
        parameters["negative_style"] = negative_style

        return (parameters, )


# UI: Prompt Outputs

class SeargeOutput1:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "parameters": ("PARAMETERS", ),
                    },
                }

    RETURN_TYPES = ("PARAMETERS", "STRING", "STRING", "STRING", "STRING", "STRING", )
    RETURN_NAMES = ("parameters", "main_prompt", "secondary_prompt", "style_prompt", "negative_prompt", "negative_style", )
    FUNCTION = "demux"

    CATEGORY = "Searge/UI/Outputs"

    def demux(self, parameters):
        main_prompt = parameters["main_prompt"]
        secondary_prompt = parameters["secondary_prompt"]
        style_prompt = parameters["style_prompt"]
        negative_prompt = parameters["negative_prompt"]
        negative_style = parameters["negative_style"]

        return (parameters, main_prompt, secondary_prompt, style_prompt, negative_prompt, negative_style, )


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
                    },
                "optional": {
                    "inputs": ("PARAMETER_INPUTS", ),
                    },
                }

    RETURN_TYPES = ("PARAMETER_INPUTS", )
    RETURN_NAMES = ("inputs", )
    FUNCTION = "mux"

    CATEGORY = "Searge/UI/Inputs"

    def mux(self, seed, image_width, image_height, steps, cfg, sampler_name, scheduler, inputs=None):
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

        return (parameters, )


# UI: Generation Parameters Output

class SeargeOutput2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "parameters": ("PARAMETERS", ),
                    },
                }

    RETURN_TYPES = ("PARAMETERS", "INT", "INT", "INT", "INT", "FLOAT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, )
    RETURN_NAMES = ("parameters", "seed", "image_width", "image_height", "steps", "cfg", "sampler_name", "scheduler", )
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

        return (parameters, seed, image_width, image_height, steps, cfg, sampler_name, scheduler, )


# UI: Advanced Parameters Input

class SeargeInput3:
    REFINER_SATURATION = ["hard", "soft"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "base_ratio": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "refiner_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.1}),
                    "refiner_saturation": (SeargeInput3.REFINER_SATURATION, {"default": "soft"}),
                    "precondition_steps": ("INT", {"default": 0, "min": 0, "max": 10}),
                    "batch_size": ("INT", {"default": 1, "min": 1, "max": 4}),
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

    def mux(self, base_ratio, refiner_strength, refiner_saturation, precondition_steps, batch_size, inputs=None, denoise=None):
        if inputs is None:
            parameters = {}
        else:
            parameters = inputs

        parameters["denoise"] = denoise
        parameters["base_ratio"] = base_ratio
        parameters["refiner_strength"] = refiner_strength
        parameters["refiner_saturation"] = refiner_saturation
        parameters["precondition_steps"] = precondition_steps
        parameters["batch_size"] = batch_size

        return (parameters, )


# UI: Advanced Parameters Output

class SeargeOutput3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "parameters": ("PARAMETERS", ),
                    },
                }

    RETURN_TYPES = ("PARAMETERS", "FLOAT", "FLOAT", "INT", "INT", "INT", )
    RETURN_NAMES = ("parameters", "denoise", "base_ratio", "noise_offset", "precondition_steps", "batch_size", )
    FUNCTION = "demux"

    CATEGORY = "Searge/UI/Outputs"

    def demux(self, parameters):
        denoise = parameters["denoise"]
        base_ratio = parameters["base_ratio"]
        refiner_strength = parameters["refiner_strength"]
        noise_offset = parameters["noise_offset"]
        precondition_steps = parameters["precondition_steps"]
        batch_size = parameters["batch_size"]

        return (parameters, denoise, base_ratio, refiner_strength, noise_offset, precondition_steps, batch_size, )


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
                    "inputs": ("PARAMETER_INPUTS", ),
                    },
                }

    RETURN_TYPES = ("PARAMETER_INPUTS", )
    RETURN_NAMES = ("inputs", )
    FUNCTION = "mux"

    CATEGORY = "Searge/UI/Inputs"

    def mux(self, base_model, refiner_model, vae_model, main_upscale_model, support_upscale_model, lora_model, inputs=None):
        if inputs is None:
            parameters = {}
        else:
            parameters = inputs

        parameters["base_model"] = base_model
        parameters["refiner_model"] = refiner_model
        parameters["vae_model"] = vae_model
        parameters["main_upscale_model"] = main_upscale_model
        parameters["support_upscale_model"] = support_upscale_model
        parameters["lora_model"] = lora_model

        return (parameters, )


# UI: Model Selector

class SeargeOutput4:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "parameters": ("PARAMETERS", ),
                    },
                }

    RETURN_TYPES = ("PARAMETERS", (folder_paths.get_filename_list("checkpoints"),), (folder_paths.get_filename_list("checkpoints"),), (folder_paths.get_filename_list("vae"),), (folder_paths.get_filename_list("upscale_models"),), (folder_paths.get_filename_list("upscale_models"),), (folder_paths.get_filename_list("loras"),), )
    RETURN_NAMES = ("parameters", "base_model", "refiner_model", "vae_model", "main_upscale_model", "support_upscale_model", "lora_model", )
    FUNCTION = "demux"

    CATEGORY = "Searge/UI/Outputs"

    def demux(self, parameters):
        base_model = parameters["base_model"]
        refiner_model = parameters["refiner_model"]
        vae_model = parameters["vae_model"]
        main_upscale_model = parameters["main_upscale_model"]
        support_upscale_model = parameters["support_upscale_model"]
        lora_model = parameters["lora_model"]

        return (parameters, base_model, refiner_model, vae_model, main_upscale_model, support_upscale_model, lora_model, )


# Register nodes in ComfyUI

NODE_CLASS_MAPPINGS = {
    "SeargeSDXLSampler": SeargeSDXLSampler,
    "SeargeSDXLImage2ImageSampler": SeargeSDXLImage2ImageSampler,

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

    "SeargeParameterProcessor": SeargeParameterProcessor,
    "SeargeInput1": SeargeInput1,
    "SeargeOutput1": SeargeOutput1,
    "SeargeInput2": SeargeInput2,
    "SeargeOutput2": SeargeOutput2,
    "SeargeInput3": SeargeInput3,
    "SeargeOutput3": SeargeOutput3,
    "SeargeInput4": SeargeInput4,
    "SeargeOutput4": SeargeOutput4,
}


# Human readable names for the nodes

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeargeSDXLSampler": "SDXL Sampler (SeargeSDXL)",
    "SeargeSDXLImage2ImageSampler": "Image2Image Sampler (SeargeSDXL)",

    "SeargeSDXLPromptEncoder": "SDXL Prompt Encoder (SeargeSDXL)",
    "SeargeSDXLBasePromptEncoder": "SDXL Base Prompt Encoder (SeargeSDXL)",
    "SeargeSDXLRefinerPromptEncoder": "SDXL Refiner Prompt Encoder (SeargeSDXL)",

    "SeargePromptText": "Prompt text input (SeargeSDXL)",
    "SeargePromptCombiner": "Prompt combiner (SeargeSDXL)",

    "SeargeIntegerConstant": "Integer Constant (SeargeSDXL)",
    "SeargeIntegerPair": "Integer Pair (SeargeSDXL)",
    "SeargeIntegerMath": "Integer Math (SeargeSDXL)",
    "SeargeIntegerScaler": "Integer Scaler (SeargeSDXL)",

    "SeargeFloatConstant": "Float Constant (SeargeSDXL)",
    "SeargeFloatPair": "Float Pair (SeargeSDXL)",
    "SeargeFloatMath": "Float Math (SeargeSDXL)",

    "SeargeParameterProcessor": "Parameter Processor",
    "SeargeInput1": "Prompts",
    "SeargeOutput1": "Prompts",
    "SeargeInput2": "Generation Parameters",
    "SeargeOutput2": "Generation Parameters",
    "SeargeInput3": "Advanced Parameters",
    "SeargeOutput3": "Advanced Parameters",
    "SeargeInput4": "Model Names",
    "SeargeOutput4": "Model Names",
}
