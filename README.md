
# Searge-SDXL: EVOLVED ~~v4.x~~ v3.991 for ComfyUI

*Custom nodes extension* for [ComfyUI](https://github.com/comfyanonymous/ComfyUI),
**including a workflow** to use *SDXL 1.0* with both the *base and refiner* checkpoints.

# Public test version 3.991

This version is the first public test version of the huge update to version 4.0 and is 95% feature complete.

## Missing features

Right now the following features are still missing and are planned for the complete v4.0 release:

- **Prompt Styles** - loading and applying style templates from a file
- **More Prompting Modes** - many of the unique prompting modes from v3.x are still missing and need to be
re-implemented in the new architecture of this extension
- **Condition Mixing** - this is the foundation for re-introducing the v3.x prompting modes, but it's planned
to have an even more flexible system to design your own custom prompting modes



# Table of Content

<!-- TOC -->
* [Searge-SDXL: EVOLVED ~~v4.x~~ v3.991 for ComfyUI](#searge-sdxl-evolved-v4x-v3991-for-comfyui)
* [Public test version 3.991](#public-test-version-3991)
  * [Missing features](#missing-features)
* [Table of Content](#table-of-content)
* [Version ~~4.0~~ 3.991](#version-40-3991)
  * [Always use the latest version of the workflow json file with the latest version of the custom nodes!](#always-use-the-latest-version-of-the-workflow-json-file-with-the-latest-version-of-the-custom-nodes)
* [Installing and Updating](#installing-and-updating)
  * [Recommended Installation of the Test Version](#recommended-installation-of-the-test-version)
    * [Recommended Update of the Test Version](#recommended-update-of-the-test-version)
  * [Checkpoints and Models for these Workflows](#checkpoints-and-models-for-these-workflows)
    * [Direct Downloads](#direct-downloads)
* [Updates](#updates)
  * [What's new in ~~v4.0~~ 3.991?](#whats-new-in-v40-3991)
    * [Major Highlights](#major-highlights)
    * [Smaller Changes and Additions](#smaller-changes-and-additions-)
* [The Workflow File](#the-workflow-file)
  * [Documentation](#documentation)
* [Workflow Details](#workflow-details)
  * [Operating Modes](#operating-modes)
    * [Text to Image Mode](#text-to-image-mode)
    * [Image to Image Mode](#image-to-image-mode)
    * [Inpainting Mode](#inpainting-mode)
* [More Example Images](#more-example-images)
<!-- TOC -->



# Version ~~4.0~~ 3.991

Instead of having separate workflows for different tasks, everything is integrated in **one workflow file**.

## Always use the latest version of the workflow json file with the latest version of the custom nodes!

<img src="docs/img/main_readme/banner.png" width="768">



# Installing and Updating

## Recommended Installation of the Test Version

- Download and unpack the latest test release from the [Searge SDXL CivitAI page](https://civitai.com/models/111463) or
the [Github releases page for this project](https://github.com/SeargeDP/SeargeSDXL/releases).
- Drop the `SeargeSDXL-Test` folder into the `ComfyUI/custom_nodes` directory and restart ComfyUI.

### Recommended Update of the Test Version

- When new test versions are released, before the final v4.0 update release, repeat the steps from
the [Recommended Installation of the Test Version](#recommended-installation-of-the-test-version) section
and overwrite existing files in the process.


## Checkpoints and Models for these Workflows

This workflow depends on certain checkpoint files to be installed in ComfyUI, here is a list of the necessary
files that the workflow expects to be available.

If any of the mentioned folders does not exist in `ComfyUI/models`, **create** the missing folder and put the
downloaded file into it.

I recommend to **download and copy all** these files *(the required, recommended, and optional ones)* to make
**full use of all features** included in the workflow!

### Direct Downloads

(from Huggingface)

- **(required)** download [SDXL 1.0 Base with 0.9 VAE (7 GB)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors) and copy it into `ComfyUI/models/checkpoints`
  - *(this should be pre-selected as the base model on the workflow already)*


- **(recommended)** download [SDXL 1.0 Refiner with 0.9 VAE (6 GB)](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors) and copy it into `ComfyUI/models/checkpoints`
  - *(you should select this as the refiner model on the workflow)*


- *(optional)* download [Fixed SDXL 0.9 vae (335 MB)](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors) and copy it into `ComfyUI/models/vae`
  - *(instead of using the VAE that's embedded in SDXL 1.0, this one has been fixed to work in fp16 and should **fix the issue with generating black images**)*


- *(optional)* download [SDXL Offset Noise LoRA (50 MB)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors) and copy it into `ComfyUI/models/loras`
  - *(the example lora that was released alongside SDXL 1.0, it can add more contrast through offset-noise)*
 

- **(recommended)** download [4x-UltraSharp (67 MB)](https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x-UltraSharp.pth) and copy it into `ComfyUI/models/upscale_models`
  - *(you should select this as the primary upscaler on the workflow)*
 

- **(recommended)** download [4x_NMKD-Siax_200k (67 MB)](https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKD-Siax_200k.pth) and copy it into `ComfyUI/models/upscale_models`
  - *(you should select this as the secondary upscaler on the workflow)*


- **(recommended)** download [4x_Nickelback_70000G (67 MB)](https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_Nickelback_70000G.pth) and copy it into `ComfyUI/models/upscale_models`
  - *(you should select this as the high-res upscaler on the workflow)*


- *(optional)* download [1x-ITF-SkinDiffDetail-Lite-v1 (20 MB)](https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_Nickelback_70000G.pth) and copy it into `ComfyUI/models/upscale_models`
  - *(you can select this as the detail processor on the workflow)*


- **(required)** download [ControlNetHED (30 MB)](https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth) and copy it into `ComfyUI/models/annotators`
  - *(this will be used by the controlnet nodes)*


- **(required)** download [res101 (531 MB)](https://huggingface.co/lllyasviel/Annotators/resolve/main/res101.pth) and copy it into `ComfyUI/models/annotators`
  - *(this will be used by the controlnet nodes)*


- **(recommended)** download [clip_vision_g (3.7 GB)](https://huggingface.co/stabilityai/control-lora/resolve/main/revision/clip_vision_g.safetensors) and copy it into `ComfyUI/models/clip_vision`
  - *(you should select this as the clip vision model on the workflow)*


- **(recommended)** download [control-lora-canny-rank256 (774 MB)](https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-canny-rank256.safetensors) and copy it into `ComfyUI/models/controlnet`
  - *(you should select this as the canny checkpoint on the workflow)*


- **(recommended)** download [control-lora-depth-rank256 (774 MB)](https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-depth-rank256.safetensors) and copy it into `ComfyUI/models/controlnet`
  - *(you should select this as the depth checkpoint on the workflow)*


- **(recommended)** download [control-lora-recolor-rank256 (774 MB)](https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-recolor-rank256.safetensors) and copy it into `ComfyUI/models/controlnet`
  - *(you should select this as the recolor checkpoint on the workflow)*


- **(recommended)** download [control-lora-sketch-rank256 (774 MB)](https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-sketch-rank256.safetensors) and copy it into `ComfyUI/models/controlnet`
  - *(you should select this as the sketch checkpoint on the workflow)*


- *(optional)* download [OpenPoseXL2 (5 GB)](https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0/resolve/main/OpenPoseXL2.safetensors) and copy it into `ComfyUI/models/controlnet`
  - *(you can select this as the custom controlnet checkpoint on the workflow)*


Now everything should be prepared, but you may to have to adjust some file names in the different model selector boxes
on the workflow. Do so by clicking on the filename in the workflow UI and selecting the correct file from the list.

<img src="docs/img/main_readme/full_graph.png" width="768">



# Updates

Find information about the latest changes here.


## What's new in ~~v4.0~~ 3.991?

### Major Highlights
- A **complete re-write** of the custom node extension and the SDXL workflow 
- **Highly optimized** processing pipeline, now **up to 20% faster** than in older workflow versions
- Support for **Controlnet and Revision**, up to 5 can be applied together
- **Multi-LoRA** support with up to 5 LoRA's at once
- ... (TODO: list more major highlights)

### Smaller Changes and Additions 
- Workflows created with this extension and metadata embeddings in generated images are forward-compatible with
future updates of this project
- The custom node extension included in this project is backward-compatible with every workflow since version v3.3
- ... (TODO: list more smaller changes)

<br><img src="docs/img/main_readme/ui-3.png" width="768">

*(5 multi-purpose image inputs for revision and controlnet)*



# The Workflow File

The workflow is included as a `.json` file in the `workflow` folder.

**After updating Searge SDXL, always make sure to load the latest version of the json file if you want to benefit
from the latest features, updates, and bugfixes.**

(you can check the version of the workflow that you are using by looking at the workflow information box)

![Workflow Version](docs/img/main_readme/workflow_version.png)


## Documentation

[Click this link to see the documentation](docs/readme.md)

<img src="docs/img/main_readme/ui-1.png" width="768">

*(the main UI of the workflow)*



# Workflow Details

The **EVOLVED v4.x** workflow is a new workflow, created from scratch. It requires the latest additions to the
SeargeSDXL custom node extension, because it makes use of some new node types.

The interface for using this new workflow is also designed in a different way, with all parameters that
are usually tweaked to generate images tightly packed together. This should make it easier to have every
important element on the screen at the same time without scrolling.

<img src="docs/img/main_readme/ui-2.png" width="768">

*(more advanced UI elements right next to the main UI)*



## Operating Modes

![Workflow Version](docs/img/main_readme/operating_mode.png)

### Text to Image Mode

In this mode you can generate images from text descriptions. The source image and the mask (next to the prompt inputs)
are not used in this mode.

<img src="docs/img/main_readme/ui_txt2img.png" width="768">

*(example of using text-to-image in the workflow)*

<br>

<img src="docs/img/main_readme/result_txt2img.png" width="512">

*(result of the text-to-image example)*



### Image to Image Mode

In this mode you can generate images from text descriptions and a source image. The mask (next to the prompt inputs)
is not used in this mode.

<img src="docs/img/main_readme/ui_img2img.png" width="768">

*(example of using image-to-image in the workflow)*

<br>

<img src="docs/img/main_readme/result_img2img.png" width="512">

*(result of the image-to-image example)*



### Inpainting Mode

In this mode you can generate images from text descriptions and a source image. Both, the source image and the mask
(next to the prompt inputs) are used in this mode.

This is similar to the image to image mode, but it also lets you define a mask for selective changes of only parts
of the image.

<img src="docs/img/main_readme/ui_inpainting.png" width="768">

*(example of using inpainting in the workflow)*

<br>

<img src="docs/img/main_readme/result_inpainting.png" width="512">

*(result of the inpainting example)*

# More Example Images

A small collection of example images (with embedded workflow) can be found in the `examples` folder. [Here is an
overview of the included images.](examples/readme.md)
