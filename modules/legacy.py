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
import comfy_extras.nodes_post_processing
import comfy_extras.nodes_upscale_model
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
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 30, "min": 1, "max": 1000}),
                    "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_2m"}),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras"}),
                    "base_ratio": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    },
                }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("", )
    FUNCTION = "sample"

    CATEGORY = "Searge/Legacy"

    def sample(self, base_model, base_positive, base_negative, refiner_model, refiner_positive, refiner_negative, latent_image, noise_seed, steps, cfg, sampler_name, scheduler, base_ratio, denoise):
        base_steps = int(steps * base_ratio)

        if denoise < 0.01:
            return (latent_image, )

        if base_steps >= steps:
            return nodes.common_ksampler(base_model, noise_seed, steps, cfg, sampler_name, scheduler, base_positive, base_negative, latent_image, denoise=denoise, disable_noise=False, start_step=0, last_step=steps, force_full_denoise=True)

        base_result = nodes.common_ksampler(base_model, noise_seed, steps, cfg, sampler_name, scheduler, base_positive, base_negative, latent_image, denoise=denoise, disable_noise=False, start_step=0, last_step=base_steps, force_full_denoise=False)
        return nodes.common_ksampler(refiner_model, noise_seed, steps, cfg, sampler_name, scheduler, refiner_positive, refiner_negative, base_result[0], denoise=1.0, disable_noise=True, start_step=base_steps, last_step=steps, force_full_denoise=True)


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
                    "sampler_name": ("SAMPLER_NAME", {"default": "ddim"}),
                    "scheduler": ("SCHEDULER_NAME", {"default": "ddim_uniform"}),
                    "base_ratio": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "denoise": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                    },
                "optional": {
                    "upscale_model": ("UPSCALE_MODEL",),
                    "scaled_width": ("INT", {"default": 1536, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "scaled_height": ("INT", {"default": 1536, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "noise_offset": ("INT", {"default": 1, "min": 0, "max": 1}),
                    "refiner_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
                    "softness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                    },
                }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "sample"

    CATEGORY = "Searge/Legacy"

    def sample(self, base_model, base_positive, base_negative, refiner_model, refiner_positive, refiner_negative, image, vae, noise_seed, steps, cfg, sampler_name, scheduler, base_ratio, denoise, softness, upscale_model=None, scaled_width=None, scaled_height=None, noise_offset=None, refiner_strength=None):
        base_steps = int(steps * (base_ratio + 0.0001))

        if noise_offset is None:
            noise_offset = 1

        if refiner_strength is None:
            refiner_strength = 1.0

        if refiner_strength < 0.01:
            refiner_strength = 0.01

        if steps < 1:
            return (image, )

        scaled_image = image

        use_upscale_model = upscale_model is not None and softness < 0.9999
        if use_upscale_model:
            upscale_result = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel().upscale(upscale_model, image)
            scaled_image = upscale_result[0]

        if scaled_width is not None and scaled_height is not None:
            upscale_result = nodes.ImageScale().upscale(scaled_image, "bicubic", scaled_width, scaled_height, "center")
            scaled_image = upscale_result[0]

        if use_upscale_model and softness > 0.0001:
            upscale_result = nodes.ImageScale().upscale(image, "bicubic", scaled_width, scaled_height, "center")
            scaled_original = upscale_result[0]

            blend_result = comfy_extras.nodes_post_processing.Blend().blend_images(scaled_image, scaled_original, softness, "normal")
            scaled_image = blend_result[0]

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
