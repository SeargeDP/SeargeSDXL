# SeargeSDXL
Custom nodes extension and workflows for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
including workflows to use SDXL 1.0 with both the base and refiner checkpoints.



# Installing and Updating:

### Recommended Installation:
- Navigate to your `ComfyUI/custom_nodes/` directory
- Open a command line window in the *custom_nodes* directory
- Run `git clone https://github.com/SeargeDP/SeargeSDXL.git`
- Restart ComfyUI

### Alternative Installation (not recommended):
- Download and unpack the latest release from the [Searge SDXL CivitAI page](https://civitai.com/models/111463)
- Drop the `SeargeSDXL` folder into the `ComfyUI/custom_nodes` directory and restart ComfyUI.

### Updating an Existing Installation
- Navigate to your `ComfyUI/custom_nodes/` directory
- If you installed via `git clone` before
  - Open a command line window in the *custom_nodes* directory
  - Run `git pull`
- If you installed from a zip file
  - Unpack the `SeargeSDXL` folder from the latest release into `ComfyUI/custom_nodes`, overwrite existing files
- Restart ComfyUI



## Checkpoints and Models for these Workflows

### Direct Downloads
(from Huggingface)

- [SDXL 1.0 base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors)
- [SDXL 1.0 refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors)
- [Fixed SDXL 0.9 vae](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors)
- [4x_NMKD-Siax_200k upscaler](https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKD-Siax_200k.pth)
- [4x-UltraSharp upscaler](https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x-UltraSharp.pth)



# More Information
Now 4 workflows are included in the examples folder.
They are called *reborn*, *image2image*, *inpainting*, and *simple*.

The simple workflow is included as a bonus, it is great if you are just starting with SDXL and don't need
an advanced workflow.

### Positive Prompts
Prompting has been improved. The main prompt and secondary prompt influence the two CLIP models
similar to the way my original workflow worked.

A third positive prompt was added for style description and artist references.
This one influences both CLIP models.

### Negative Prompts
For the negative prompt the way it is now designed expects objects and subjects you don't want in the negative
prompt and styles that you don't want in the negative style.

### Prompt Power
For both, the positive style & reference prompt and the negative style prompt, you can now change their
*power*, a weight of how much they should be applied.

The defaults are *0.333* for positive style power and *0.667* for negative style power.

The positive style power should be in the range *0.1* to *0.5*, at higher values the style is applied rather strong
and can "take over the image". You should try it to understand better what too large values will do.



## Workflow Descriptions

### Reborn
The *reborn* workflow is a new workflow, created from scratch. It requires the latest additions to the
SeargeSDXL custom node extension, because it makes use of some new node types.

The interface for using this new workflow is also designed in a different way, with all parameters that
are usually tweaked to generate images tighly packed together. This should make it easier to have every
important element on the screen at the same time without scrolling.

### Image to Image
In this workflow you should first copy an image into the `ConfyUI/input` directory.

Then select that image as the *Source Image* (next to the prompt inputs).
If it does not show up, press the *Refresh* button on the Comfy UI control box.

For image to image the parameter *Denoise* will determine how much the source image should be changed
according to the prompt.
Ranges are from *0.0* for "no change" to *1.0* for "completely change".

Good values to try are probably in the *0.2* to *0.8* range.
With examples of *0.25* for "very little change", *0.5* for "some changes", or *0.75* for "a lot of changes"

### Inpainting
This is similar to the image to image workflow.
But it also lets you define a mask for selective changes of only parts of the image.

To use this workflow, prepare a source image the same way as described in the image to image workflow.
Then **right click** on the *Inpainting Mask* image (the bottom one next to the input prompts) and select
**Open in Mask Editor**.

Paint your mask and then press the *Save to node* button when you are done.
The *Denoise* parameter works the same way as in image to image, but only masked areas will be changed.

### Simple
The main difference between *simple* and *reborn* is how the prompts are used through the CLIP models.
The *reborn* workflow has advanced options to prompt styles independently of subjects.
In the simple workflow you only have to worry about positive and negative prompts.

For the easiest way to get started I recommend to just use the same prompt as main and secondary,
and leave the negative prompt empty.



# Available Example Workflows
These the workflows included in the examples folder.

### Reborn Workflow

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL-workflow-1.png" width="768">

### Image to Image Workflow

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL-workflow-2.png" width="768">

### Inpainting Workflow

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL-workflow-3.png" width="768">

### Simple Workflow

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL-workflow-4.png" width="768">



## Example images generated with these workflows

### Reborn Workflow

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL-reborn.png" width="768">

### Image to Image Workflow

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL-img2img.png" width="768">

### Inpainting Workflow

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL-inpaint.png" width="768">

### Simple Workflow

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL-simple.png" width="768">



# Custom Nodes
These custom node types are available in the extension.

The details about them are only important if you want to use them in your own workflow or if you want to
understand better how the included workflows work.

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL-Nodetypes.png" width="768">


## SDXL Sampler Node
<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL-Node-1.png" width="407">

### Inputs
- **base_model** - connect the SDXL base model here, provided via a `Load Checkpoint` node 
- **base_positive** - recommended to use a `CLIPTextEncodeSDXL` with 4096 for `width`, `height`, `target_width`, and `target_height`
- **base_negative** - recommended to use a `CLIPTextEncodeSDXL` with 4096 for `width`, `height`, `target_width`, and `target_height`
- **refiner_model** - connect the SDXL refiner model here, provided via a `Load Checkpoint` node 
- **refiner_positive** - recommended to use a `CLIPTextEncodeSDXLRefiner` with 2048 for `width`, and `height`
- **refiner_negative** - recommended to use a `CLIPTextEncodeSDXLRefiner` with 2048 for `width`, and `height`
- **latent_image** - either an empty latent image or a VAE-encoded latent from a source image for img2img
- **noise_seed** - the random seed for generating the image
- **steps** - total steps for the sampler, it will internally be split into base steps and refiner steps
- **cfg** - CFG scale (classifier free guidance), values between 3.0 and 12.0 are most commonly used
- **sampler_name** - the noise sampler _(I prefer dpmpp_2m with the karras scheduler, sometimes ddim with the ddim_uniform scheduler)_
- **scheduler** - the scheduler to use with the sampler selected in `sampler_name`
- **base_ratio** - the ratio between base model steps and refiner model steps _(0.8 = 80% base model and 20% refiner model, with 30 total steps that's 24 base steps and 6 refiner steps)_
- **denoise** - denoising factor, keep this at 1.0 when creating new images from an empty latent and between 0.0-1.0 in the img2img workflow

### Outputs
- **LATENT** - the generated latent image


## SDXL Prompt Node
<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL-Node-2.png" width="434">

### Inputs
- **base_clip** - connect the SDXL base CLIP here, provided via a `Load Checkpoint` node 
- **refiner_clip** - connect the SDXL refiner CLIP here, provided via a `Load Checkpoint` node 
- **pos_g** - the text for the positive base prompt G 
- **pos_l** - the text for the positive base prompt L
- **pos_r** - the text for the positive refiner prompt
- **neg_g** - the text for the negative base prompt G
- **neg_l** - the text for the negative base prompt L
- **neg_r** - the text for the negative refiner prompt
- **base_width** - the width for the base conditioning
- **base_height** - the height for the base conditioning
- **crop_w** - crop width for the base conditioning
- **crop_h** - crop height for the base conditioning
- **target_width** - the target width for the base conditioning
- **target_height** - the target height for the base conditioning
- **pos_ascore** - the positive aesthetic score for the refiner conditioning
- **neg_ascore** - the negative aesthetic score for the refiner conditioning
- **refiner_width** - the width for the refiner conditioning
- **refiner_height** - the height for the refiner conditioning

### Outputs
- **CONDITIONING** 1 - the positive base prompt conditioning
- **CONDITIONING** 2 - the negative base prompt conditioning
- **CONDITIONING** 3 - the positive refiner prompt conditioning
- **CONDITIONING** 4 - the negative refiner prompt conditioning
