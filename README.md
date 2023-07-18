# SeargeSDXL
Custom nodes for easier use of SDXL in [ComfyUI](https://github.com/comfyanonymous/ComfyUI) including an img2img workflow that utilizes both the base and refiner checkpoints.

# Install:

### Recommended Installation:
- Navigate to your `ComfyUI/custom_nodes/` folder
- Run `git clone https://github.com/SeargeDP/SeargeSDXL.git`
- Restart ComfyUI

### Alternative Installation:
- Drop the `SeargeSDXL` folder into the `ComfyUI/custom_nodes` directory and restart ComfyUI.

# Custom Nodes

<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL-Nodetypes.png" width="768">

## SDXL Sampler Node
<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL-Node.png" width="407">

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
<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL-Node2.png" width="434">

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

# Examples

### Workflow
<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL-img2img.png" width="768">

### Result
<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL_00001.png" width="768">
<img src="https://github.com/SeargeDP/SeargeSDXL/blob/main/example/Searge-SDXL_00002.png" width="768">
