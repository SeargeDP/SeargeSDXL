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

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import comfy_extras.nodes_post_processing
import comfy_extras.nodes_upscale_model
import latent_preview
import nodes
import torch


def sdxl_sample(modelB, modelR, noiseO, stepsB, stepsR, cfg, sampler_name, scheduler, positiveB, negativeB, positiveR, negativeR, latent_image, denoise=1.0, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None, callbackB=None, callbackR=None, disable_pbar=False, seed=None):
    device = comfy.model_management.get_torch_device()

    if noise_mask is not None:
        noise_mask = comfy.sample.prepare_mask(noise_mask, noiseO.shape, device)

    steps = stepsB + stepsR

    comfy.model_management.load_model_gpu(modelB)
    real_modelB = modelB.model

    noise = noiseO.to(device)
    latent_image = latent_image.to(device)

    positive_copyB = comfy.sample.broadcast_cond(positiveB, noise.shape[0], device)
    negative_copyB = comfy.sample.broadcast_cond(negativeB, noise.shape[0], device)

    modelsB = comfy.sample.load_additional_models(positiveB, negativeB, modelB.model_dtype())

    samplerB = comfy.samplers.KSampler(real_modelB, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=modelB.model_options)

    samplesB = samplerB.sample(noise, positive_copyB, negative_copyB, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=stepsB, force_full_denoise=False, denoise_mask=noise_mask, sigmas=sigmas, callback=callbackB, disable_pbar=disable_pbar, seed=seed)

    comfy.sample.cleanup_additional_models(modelsB)

    noise = torch.zeros(samplesB.size(), dtype=samplesB.dtype, layout=samplesB.layout, device=device)

    if noise_mask is not None:
        latent_for_refiner = samplesB * noise_mask + latent_image * (1.0 - noise_mask)
    else:
        latent_for_refiner = samplesB

    comfy.model_management.load_model_gpu(modelR)
    real_modelR = modelR.model

    positive_copyR = comfy.sample.broadcast_cond(positiveR, noise.shape[0], device)
    negative_copyR = comfy.sample.broadcast_cond(negativeR, noise.shape[0], device)

    modelsR = comfy.sample.load_additional_models(positiveR, negativeR, modelR.model_dtype())

    samplerR = comfy.samplers.KSampler(real_modelR, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=modelR.model_options)

    samples = samplerR.sample(noise, positive_copyR, negative_copyR, cfg=cfg, latent_image=latent_for_refiner, start_step=stepsB, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callbackR, disable_pbar=disable_pbar, seed=seed)
    samples = samples.cpu()

    comfy.sample.cleanup_additional_models(modelsR)

    return samples


def sdxl_ksampler(modelB, modelR, seed, stepsB, stepsR, cfg, sampler_name, scheduler, positiveB, negativeB, positiveR, negativeR, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    device = comfy.model_management.get_torch_device()
    latent_image = latent["samples"]

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    previewerB = latent_preview.get_previewer(device, modelB.model.latent_format)
    previewerR = latent_preview.get_previewer(device, modelR.model.latent_format)

    steps = stepsB + stepsR
    pbar = comfy.utils.ProgressBar(steps)

    def callbackB(step, x0, x, total_steps):
        preview_bytes = None
        if previewerB:
            preview_bytes = previewerB.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)

    def callbackR(step, x0, x, total_steps):
        preview_bytes = None
        if previewerR:
            preview_bytes = previewerR.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)

    samples = sdxl_sample(modelB, modelR, noise, stepsB, stepsR, cfg, sampler_name, scheduler, positiveB, negativeB, positiveR, negativeR, latent_image,
                                  denoise=denoise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callbackB=callbackB, callbackR=callbackR, seed=seed)

    out = latent.copy()
    out["samples"] = samples
    return (out, )


# SDXL Sampler with base and refiner support

class SeargeSDXLSampler2:
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
                    "sampler_name": ("SAMPLER_NAME", {"default": "ddim"}),
                    "scheduler": ("SCHEDULER_NAME", {"default": "ddim_uniform"}),
                    "base_ratio": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    },
                "optional": {
                    "refiner_prep_steps": ("INT", {"default": 0, "min": 0, "max": 10}),
                    "noise_offset": ("INT", {"default": 1, "min": 0, "max": 1}),
                    "refiner_strength": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.05}),
                    },
                }

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "sample"

    CATEGORY = "Searge/Sampling"

    def sample(self, base_model, base_positive, base_negative, refiner_model, refiner_positive, refiner_negative, latent_image, noise_seed, steps, cfg, sampler_name, scheduler, base_ratio, denoise, refiner_prep_steps=None, noise_offset=None, refiner_strength=None):
        base_steps = int(steps * (base_ratio + 0.0001))

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

class SeargeSDXLImage2ImageSampler2:
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

    CATEGORY = "Searge/Sampling"

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


# SDXL Sampler with base and refiner support

class SeargeSDXLSamplerV3:
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
                    "sampler_name": ("SAMPLER_NAME", {"default": "ddim"}),
                    "scheduler": ("SCHEDULER_NAME", {"default": "ddim_uniform"}),
                    "base_ratio": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    },
                "optional": {
                    "refiner_prep_steps": ("INT", {"default": 0, "min": 0, "max": 10}),
#                    "noise_offset": ("INT", {"default": 1, "min": 0, "max": 1}),
#                    "refiner_strength": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.05}),
                    },
                }

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "sample"

    CATEGORY = "Searge/Sampling"

    def sample(self, base_model, base_positive, base_negative, refiner_model, refiner_positive, refiner_negative, latent_image, noise_seed, steps, cfg, sampler_name, scheduler, base_ratio, denoise, refiner_prep_steps=None, noise_offset=None, refiner_strength=None):
        base_steps = int(steps * (base_ratio + 0.0001))
        refiner_steps = max(0, steps - base_steps)

#        if noise_offset is None:
#            noise_offset = 1

#        if refiner_strength is None:
#            refiner_strength = 1.0

#        if refiner_strength < 0.01:
#            refiner_strength = 0.01

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

        return sdxl_ksampler(base_model, refiner_model, noise_seed, base_steps, refiner_steps, cfg, sampler_name, scheduler, base_positive, base_negative, refiner_positive, refiner_negative, input_latent, denoise=denoise, disable_noise=False, start_step=start_at_step, last_step=steps, force_full_denoise=True)

#        base_result = nodes.common_ksampler(base_model, noise_seed, steps, cfg, sampler_name, scheduler, base_positive, base_negative, input_latent, denoise=denoise, disable_noise=False, start_step=start_at_step, last_step=base_steps, force_full_denoise=True)
#        return nodes.common_ksampler(refiner_model, noise_seed + noise_offset, steps, cfg, sampler_name, scheduler, refiner_positive, refiner_negative, base_result[0], denoise=denoise * refiner_strength, disable_noise=False, start_step=base_steps, last_step=steps, force_full_denoise=True)


# SDXL Image2Image Sampler (incl. HiRes Fix)

class SeargeSDXLImage2ImageSamplerV3:
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
#                    "noise_offset": ("INT", {"default": 1, "min": 0, "max": 1}),
#                    "refiner_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
                    "softness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                    },
                }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "sample"

    CATEGORY = "Searge/Sampling"

    def sample(self, base_model, base_positive, base_negative, refiner_model, refiner_positive, refiner_negative, image, vae, noise_seed, steps, cfg, sampler_name, scheduler, base_ratio, denoise, softness, upscale_model=None, scaled_width=None, scaled_height=None, noise_offset=None, refiner_strength=None):
        base_steps = int(steps * (base_ratio + 0.0001))
        refiner_steps = max(0, steps - base_steps)

#        if noise_offset is None:
#            noise_offset = 1

#        if refiner_strength is None:
#            refiner_strength = 1.0

#        if refiner_strength < 0.01:
#            refiner_strength = 0.01

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
            result_latent = sdxl_ksampler(base_model, refiner_model, noise_seed, base_steps, refiner_steps, cfg, sampler_name, scheduler, base_positive, base_negative, refiner_positive, refiner_negative, input_latent, denoise=denoise, disable_noise=False, start_step=0, last_step=steps, force_full_denoise=True)

#            base_result = nodes.common_ksampler(base_model, noise_seed, steps, cfg, sampler_name, scheduler, base_positive, base_negative, input_latent, denoise=denoise, disable_noise=False, start_step=0, last_step=base_steps, force_full_denoise=True)
#            result_latent = nodes.common_ksampler(refiner_model, noise_seed + noise_offset, steps, cfg, sampler_name, scheduler, refiner_positive, refiner_negative, base_result[0], denoise=denoise * refiner_strength, disable_noise=False, start_step=base_steps, last_step=steps, force_full_denoise=True)

        vae_decode_result = nodes.VAEDecode().decode(vae, result_latent[0])
        output_image = vae_decode_result[0]

        return (output_image, )
