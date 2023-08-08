# Searge-SDXL v3.x - "Truly Reborn"
*Custom nodes extension* for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
including *a workflow* to use *SDXL 1.0* with both the *base and refiner* checkpoints.


# Version 3.4
Instead of having separate workflows for different tasks, everything is now integrated in **one workflow file**.

### Always use the latest version of the workflow json file with the latest version of the custom nodes!

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL-Example.png" width="768">


## What's new in v3.4?
- Minor tweaks and fixes and the beginnings of some code restructuring, nothing user should notice in the workflows
- Preparations for more upcoming improvements in a compatible way
- Added compatibility with v1.x workflows, these have been used in some tutorials and did not work anymore with newer
versions of the extension
- *(backwards compatibility with v2.x and older v3.x version - before v3.3 - is unfortunately not possible)*

## What about v3.3?
- Starting from v3.3 the custom node extension will always be compatible with workflows created with v3.3 or later
- *(backwards compatibility with v2.x, v3.0, v3.1. and v3.2 workflows is unfortunately not possible)*
- Going forward, older versions of workflow will remain in the `workflow` folder, I still highly recommend to **always 
use the latest version** and loading it **from the JSON file** instead of the example images 
- *Version 3.3 has never been publicly released*

## What's new in v3.2?
- More prompting modes, including the "3-prompt" style that's common in other workflows
using separate prompts for the 2 CLIP models in SDXL (CLIP G & CLIP L) and a negative prompt
  - **3-Prompt G+L-N** - Similar to simple mode, but cares about *a main, a secondary, and a negative prompt*
and **ignores** the *additional style prompting fields*, this is great to get similar results as on other
workflows and makes it easier to compare the images
  - **Subject - Style** - The *subject focused* positives with the *style focused* negatives
  - **Style - Subject** - The *style focused* positives with the *subject focused* negatives
  - **Style Only** - **Only** the positive and negative **style prompts** are used and *main/secondary/negative are ignored*
  - **Weighted - Overlay** - The positive prompts are *weighted* and the negative prompts are *overlaid*
  - **Overlay - Weighted** - The positive prompts are *overlaid* and the negative prompts are *weighted*
- Better bug fix for the "exploding" the search box issue, should finally be fixed *(for real)* now
- Some additional node types to make it easier to still use my nodes in other custom workflows
- The custom node extension should now also work on **Python 3.9** again, it required 3.10 before

## What's new in v3.1?
- Fixed the issue with "exploding" the search box when this extension is installed
- Loading of Checkpoints, VAE, Upscalers, and Loras through custom nodes
- Updated workflow to make use of the added node types
- Adjusted the default settings for some parameters in the workflow
- Fixed some reported issues with the workflow and custom nodes
- Prepared the workflow for an upcoming feature

## What's new in v3.0?
- Completely overhauled **user interface**, now even easier to use than before
- More organized workflow graph - if you want to understand how it is designed "under the hood", it should now be
easier to figure out what is where and how things are connected
- New settings that help to tweak the generated images *without changing the composition*
  - Quickly iterate between *sharper* results and *softer* results of the same image without changing the composition
or subject
  - Easily make colors pop where needed, or render a softer image where it fits the mood better
- Three operating modes in **ONE** workflow
  - **text-to-image**
  - **image-to-image**
  - **inpainting**
- Different prompting modes (**5 modes** available)
  - **Simple** - Just cares about **a positive and a negative prompt** and *ignores the additional prompting fields*, this
is great to get started with SDXL, ComfyUI, and this workflow
  - **Subject Focus** - In this mode the *main/secondary prompts* are more important than the *style prompts*
  - **Style Focus** - In this mode the *style prompts* are more important than the *main/secondary prompts*
  - **Weighted** - In this mode the balance between *main/secondary prompts* and *style prompts* can be influenced with
the *style prompt power* and *negative prompt power* option
  - **Overlay** - In this mode the main*/secondary prompts* and the *style prompts* are competing with each other
- Greatly *improved Hires-Fix* - now with more options to influence the results
- A (rather limited for now) alpha test for *style templates*, this is work in progress and only includes one
style for now (called *test*)
- Options to change the **intensity of the refiner** when used together with the base model,
separate for *main pass* and *hires-fix pass*
- *(... many more things probably, since the workflow was almost completely re-made)*

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL-UI.png" width="768">



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

- download [SDXL 1.0 base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors) and copy it into `ComfyUI/models/checkpoints`
- download [SDXL 1.0 refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors) and copy it into `ComfyUI/models/checkpoints`
- download [Fixed SDXL 0.9 vae](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors) and copy it into `ComfyUI/models/vae`
- download [SDXL Offset Noise LoRA](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors)
and copy it into `ComfyUI/models/loras`
- download [4x_NMKD-Siax_200k upscaler](https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKD-Siax_200k.pth) and copy it into `ComfyUI/models/upscale_models`
- download [4x-UltraSharp upscaler](https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x-UltraSharp.pth) and copy it into `ComfyUI/models/upscale_models`



# More Information
Now **3** operating modes are included in the workflow, the *.json-file* for it is in the `workflow` folder.
They are called *text2image*, *image2image*, and *inpainting*.

The simple workflow has not returned as a separate workflow, but is now also *fully integrated*.

To enable it, switch the **prompt mode** option to **simple** and it will only pay attention to the *main prompt*
and the *negative prompt*.

Or switch the **prompt mode** to **3 prompts** and only the *main prompt*, the *secondary prompt*, and the
*negative prompt* are used.



# The Workflow
The workflow is included in the `workflow` folder.

**After updating Searge SDXL, always make sure to load the latest version of the json file. Older versions of the
workflow are often not compatible anymore with the updated node extension.**

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL-Overview.png" width="768">



# Searge SDXL Reborn Workflow Description
The **Reborn v3.x** workflow is a new workflow, created from scratch. It requires the latest additions to the
SeargeSDXL custom node extension, because it makes use of some new node types.

The interface for using this new workflow is also designed in a different way, with all parameters that
are usually tweaked to generate images tightly packed together. This should make it easier to have every
important element on the screen at the same time without scrolling.

Starting from version 3.0 all 3 operating modes (text-to-image, image-to-image, and inpainting) are available
from the same workflow and can be switched with an option.

## Reborn Workflow v3.x Operating Modes

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/UI-operation-mode.png" width="512">

### Text to Image Mode
In this mode you can generate images from text descriptions. The source image and the mask (next to the prompt inputs)
are not used in this mode.

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL-workflow-1.png" width="768">
<br>
<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL-reborn.png" width="512">

### Image to Image Mode
In this mode you should first copy an image into the `ConfyUI/input` directory.
Alternatively you can change the option for the **save directory** to **input folder** when generating images, in that
case you have to press the ComfyUI *Refresh* button and it should show up in the image loader node.

Then select that image as the *Source Image* (next to the prompt inputs).
If it does not show up, press the *Refresh* button on the Comfy UI control box.

For image to image the parameter *Denoise* will determine how much the source image should be changed
according to the prompt.
Ranges are from *0.0* for "no change" to *1.0* for "completely change".

Good values to try are probably in the *0.2* to *0.8* range.
With examples of *0.25* for "very little change", *0.5* for "some changes", or *0.75* for "a lot of changes"

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL-workflow-2.png" width="768">
<br>
<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL-img2img.png" width="512">

### Inpainting Mode
This is similar to the image to image mode.
But it also lets you define a mask for selective changes of only parts of the image.

To use this mode, prepare a source image the same way as described in the image to image workflow.
Then **right click** on the *Inpainting Mask* image (the bottom one next to the input prompts) and select
**Open in Mask Editor**.

Paint your mask and then press the *Save to node* button when you are done.
The *Denoise* parameter works the same way as in image to image, but only masked areas will be changed.

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL-workflow-3.png" width="768">
<br>
<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL-inpaint.png" width="512">



# Prompting Modes

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/UI-prompt-style.png" width="512">

## Reborn Workflow v3.x Prompting Modes

### Simple
Just cares about the **main** and the **negative** prompt and **ignores** the *additional prompting fields*, this
is great to get started with SDXL, ComfyUI, and this workflow

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/01-simple.jpg" width="512">

### 3-Prompt G+L-N
Similar to simple mode, but cares about the **main & secondary** and the **negative** prompt
and **ignores** the *additional style prompting fields*, this is great to get similar results as on other
workflows and makes it easier to compare the images

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/02-3_prompts.jpg" width="512">

### Subject Focus
In this mode the *main & secondary* prompts are **more important** than the *style* prompts

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/03-subject_focus.jpg" width="512">

### Style Focus
In this mode the *style* prompts are **more important** than the *main & secondary* prompts

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/04-style_focus.jpg" width="512">

### Weighted
In this mode the **balance** between *main & secondary* prompts and *style prompts* can be influenced with
the **style prompt power** and **negative prompt power** option

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/05-weighted.jpg" width="512">

### Overlay
In this mode the *main & secondary* prompts and the *style* prompts are **competing with each other**

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/06-overlay.jpg" width="512">

### Subject - Style
The *main & secondary* positives with the *style* negatives

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/07-subject-style.jpg" width="512">

### Style - Subject
The *style* positives with the *main & secondary* negatives

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/08-style-subject.jpg" width="512">

### Style Only
**Only** the *style* prompt and *negative style* prompt are used, the *main & secondary* and *negative* are ignored

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/09-style_only.jpg" width="512">

### Weighted - Overlay
The *main & secondary* and *style* prompts are **weighted**, the *negative* and *negative style* prompts are **overlaid**

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/10-weighted-overlay.jpg" width="512">

### Overlay - Weighted
The *main & secondary* and *style* prompts are **overlaid**, the *negative* and *negative style* prompts are **weighted**

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/11-overlay-weighted.jpg" width="512">



# Custom Nodes
These custom node types are available in the extension.

The details about them are only important if you want to use them in your own workflow or if you want to
understand better how the included workflows work.

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL-Nodetypes.png" width="768">


## SDXL Sampler Node
<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL-Node-1.png" width="407">

### Inputs
- **base_model** - connect the SDXL base model here, provided via a `Load Checkpoint` node 
- **base_positive** - recommended to use a `CLIPTextEncodeSDXL` with 4096 for `width`, `height`,
`target_width`, and `target_height`
- **base_negative** - recommended to use a `CLIPTextEncodeSDXL` with 4096 for `width`, `height`,
`target_width`, and `target_height`
- **refiner_model** - connect the SDXL refiner model here, provided via a `Load Checkpoint` node 
- **refiner_positive** - recommended to use a `CLIPTextEncodeSDXLRefiner` with 2048 for `width`, and `height`
- **refiner_negative** - recommended to use a `CLIPTextEncodeSDXLRefiner` with 2048 for `width`, and `height`
- **latent_image** - either an empty latent image or a VAE-encoded latent from a source image for img2img
- **noise_seed** - the random seed for generating the image
- **steps** - total steps for the sampler, it will internally be split into base steps and refiner steps
- **cfg** - CFG scale (classifier free guidance), values between 3.0 and 12.0 are most commonly used
- **sampler_name** - the noise sampler _(I prefer dpmpp_2m with the karras scheduler, sometimes ddim
with the ddim_uniform scheduler)_
- **scheduler** - the scheduler to use with the sampler selected in `sampler_name`
- **base_ratio** - the ratio between base model steps and refiner model steps _(0.8 = 80% base model and 20% refiner
model, with 30 total steps that's 24 base steps and 6 refiner steps)_
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
