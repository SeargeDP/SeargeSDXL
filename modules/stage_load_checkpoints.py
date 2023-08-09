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

from .data_utils import retrieve_parameter
from .mb_pipeline import PipelineAccess
from .names import Names
from .node_wrapper import NodeWrapper
from .ui import UI


# --------------------------------------------------------------------------------
# Stage: Load Checkpoints
# --------------------------------------------------------------------------------

class SeargeStageLoadCheckpoints:
    def __init__(self):
        pass

    def get_input(self, data, stage_data):
        # if we still don't have stage data,
        if stage_data is None and data is not None:
            stage_data = {
                PipelineAccess.NAME: retrieve_parameter(PipelineAccess.NAME, data),
            }

        return stage_data

    def process(self, data, stage_input):
        access = PipelineAccess(stage_input)

        # TODO: this stage will always execute, even if the pipeline is disabled and in that case unload all models
        if not access.is_pipeline_enabled():
            pass

        base_changed = access.setting_changed(UI.S_CHECKPOINTS, UI.F_BASE_CHECKPOINT)
        if base_changed:
            base_name = access.get_active_setting(UI.S_CHECKPOINTS, UI.F_BASE_CHECKPOINT)

            base_checkpoint = NodeWrapper.checkpoint_loader.load_checkpoint(base_name)

            access.update_in_cache(Names.C_BASE_CHECKPOINT, [base_name], base_checkpoint)
        else:
            base_checkpoint = access.get_from_cache(Names.C_BASE_CHECKPOINT)

        base_model = base_checkpoint[0]
        base_clip = base_checkpoint[1]
        base_vae = base_checkpoint[2]

        if base_changed:
            access.update_in_pipeline(Names.P_BASE_MODEL, base_model)
            access.update_in_pipeline(Names.P_BASE_CLIP, base_clip)
            access.update_in_pipeline(Names.P_BASE_VAE, base_vae)
        else:
            access.restore_in_pipeline(Names.P_BASE_MODEL, base_model)
            access.restore_in_pipeline(Names.P_BASE_CLIP, base_clip)
            access.restore_in_pipeline(Names.P_BASE_VAE, base_vae)

        refiner_changed = access.setting_changed(UI.S_CHECKPOINTS, UI.F_REFINER_CHECKPOINT)
        if refiner_changed:
            refiner_name = access.get_active_setting(UI.S_CHECKPOINTS, UI.F_REFINER_CHECKPOINT, UI.NONE)

            if refiner_name == UI.NONE:
                refiner_checkpoint = (None, None, None,)
            else:
                refiner_checkpoint = NodeWrapper.checkpoint_loader.load_checkpoint(refiner_name)

            access.update_in_cache(Names.C_REFINER_CHECKPOINT, [refiner_name], refiner_checkpoint)
        else:
            refiner_checkpoint = access.get_from_cache(Names.C_REFINER_CHECKPOINT)

        refiner_model = refiner_checkpoint[0]
        refiner_clip = refiner_checkpoint[1]
        refiner_vae = refiner_checkpoint[2]

        if refiner_changed:
            access.update_in_pipeline(Names.P_REFINER_MODEL, refiner_model)
            access.update_in_pipeline(Names.P_REFINER_CLIP, refiner_clip)
            access.update_in_pipeline(Names.P_REFINER_VAE, refiner_vae)
        else:
            access.restore_in_pipeline(Names.P_REFINER_MODEL, refiner_model)
            access.restore_in_pipeline(Names.P_REFINER_CLIP, refiner_clip)
            access.restore_in_pipeline(Names.P_REFINER_VAE, refiner_vae)

        vae_changed = access.setting_changed(UI.S_CHECKPOINTS, UI.F_VAE_CHECKPOINT)
        if vae_changed:
            vae_name = access.get_active_setting(UI.S_CHECKPOINTS, UI.F_VAE_CHECKPOINT, UI.VAE_FROM_BASE_MODEL)

            if vae_name == UI.VAE_FROM_REFINER_MODEL:
                if refiner_vae is not None:
                    vae_checkpoint = refiner_vae
                else:
                    vae_checkpoint = base_vae

            elif vae_name == UI.VAE_FROM_BASE_MODEL:
                vae_checkpoint = base_vae

            else:
                vae_checkpoint = NodeWrapper.vae_loader.load_vae(vae_name)[0]

            access.update_in_cache(Names.C_VAE_CHECKPOINT, [vae_name], vae_checkpoint)
            access.update_in_pipeline(Names.P_VAE_MODEL, vae_checkpoint)
        else:
            vae_checkpoint = access.get_from_cache(Names.C_VAE_CHECKPOINT)
            access.restore_in_pipeline(Names.P_VAE_MODEL, vae_checkpoint)

        vae_model = vae_checkpoint

        hires_upscaler_changed = access.setting_changed(UI.S_UPSCALE_MODELS, UI.F_HIGH_RES_UPSCALER)
        if hires_upscaler_changed:
            hires_upscaler_name = access.get_active_setting(UI.S_UPSCALE_MODELS, UI.F_HIGH_RES_UPSCALER, UI.NONE)

            if hires_upscaler_name != UI.NONE:
                hires_upscaler_model = NodeWrapper.upscale_loader.load_model(hires_upscaler_name)[0]
            else:
                hires_upscaler_model = None

            access.update_in_cache(Names.C_HIRES_UPSCALE_MODEL, [hires_upscaler_name], hires_upscaler_model)
            access.update_in_pipeline(Names.P_HIRES_UPSCALER, hires_upscaler_model)
        else:
            hires_upscaler_model = access.get_from_cache(Names.C_HIRES_UPSCALE_MODEL)
            access.restore_in_pipeline(Names.P_HIRES_UPSCALER, hires_upscaler_model)

        hires_upscaler = hires_upscaler_model

        primary_upscaler_changed = access.setting_changed(UI.S_UPSCALE_MODELS, UI.F_PRIMARY_UPSCALER)
        if primary_upscaler_changed:
            primary_upscaler_name = access.get_active_setting(UI.S_UPSCALE_MODELS, UI.F_PRIMARY_UPSCALER, UI.NONE)

            if primary_upscaler_name != UI.NONE:
                primary_upscaler_model = NodeWrapper.upscale_loader.load_model(primary_upscaler_name)[0]
            else:
                primary_upscaler_model = None

            access.update_in_cache(Names.C_PRIMARY_UPSCALE_MODEL, [primary_upscaler_name], primary_upscaler_model)
            access.update_in_pipeline(Names.P_PRIMARY_UPSCALER, primary_upscaler_model)
        else:
            primary_upscaler_model = access.get_from_cache(Names.C_PRIMARY_UPSCALE_MODEL)
            access.restore_in_pipeline(Names.P_PRIMARY_UPSCALER, primary_upscaler_model)

        primary_upscaler = primary_upscaler_model

        secondary_upscaler_changed = access.setting_changed(UI.S_UPSCALE_MODELS, UI.F_SECONDARY_UPSCALER)
        if secondary_upscaler_changed:
            secondary_upscaler_name = access.get_active_setting(UI.S_UPSCALE_MODELS, UI.F_SECONDARY_UPSCALER, UI.NONE)

            if secondary_upscaler_name != UI.NONE:
                secondary_upscaler_model = NodeWrapper.upscale_loader.load_model(secondary_upscaler_name)[0]
            else:
                secondary_upscaler_model = None

            access.update_in_cache(Names.C_SECONDARY_UPSCALE_MODEL, [secondary_upscaler_name], secondary_upscaler_model)
            access.update_in_pipeline(Names.P_SECONDARY_UPSCALER, secondary_upscaler_model)
        else:
            secondary_upscaler_model = access.get_from_cache(Names.C_SECONDARY_UPSCALE_MODEL)
            access.restore_in_pipeline(Names.P_SECONDARY_UPSCALER, secondary_upscaler_model)

        secondary_upscaler = secondary_upscaler_model

        detail_processor_changed = access.setting_changed(UI.S_UPSCALE_MODELS, UI.F_DETAIL_PROCESSOR)
        if detail_processor_changed:
            detail_processor_name = access.get_active_setting(UI.S_UPSCALE_MODELS, UI.F_DETAIL_PROCESSOR, UI.NONE)

            if detail_processor_name != UI.NONE:
                detail_processor_model = NodeWrapper.upscale_loader.load_model(detail_processor_name)[0]
            else:
                detail_processor_model = None

            access.update_in_cache(Names.C_DETAIL_PROCESSOR_MODEL, [detail_processor_name], detail_processor_model)
            access.update_in_pipeline(Names.P_DETAIL_PROCESSOR, detail_processor_model)
        else:
            detail_processor_model = access.get_from_cache(Names.C_DETAIL_PROCESSOR_MODEL)
            access.restore_in_pipeline(Names.P_DETAIL_PROCESSOR, detail_processor_model)

        detail_processor = detail_processor_model

        clip_vision_checkpoint_changed = access.setting_changed(UI.S_CONTROLNET_MODELS, UI.F_CLIP_VISION_CHECKPOINT)
        if clip_vision_checkpoint_changed:
            clip_vision_checkpoint_name = access.get_active_setting(UI.S_CONTROLNET_MODELS, UI.F_CLIP_VISION_CHECKPOINT, UI.NONE)

            if clip_vision_checkpoint_name != UI.NONE:
                clip_vision_model = NodeWrapper.clipvision_loader.load_clip(clip_vision_checkpoint_name)[0]
            else:
                clip_vision_model = None

            access.update_in_cache(Names.C_CLIP_VISION_MODEL, [clip_vision_checkpoint_name], clip_vision_model)
            access.update_in_pipeline(Names.P_CLIP_VISION_MODEL, clip_vision_model)
        else:
            clip_vision_model = access.get_from_cache(Names.C_CLIP_VISION_MODEL)
            access.restore_in_pipeline(Names.P_CLIP_VISION_MODEL, clip_vision_model)

        clip_vision = clip_vision_model

        canny_checkpoint_changed = access.setting_changed(UI.S_CONTROLNET_MODELS, UI.F_CANNY_CHECKPOINT)
        if canny_checkpoint_changed:
            canny_checkpoint_name = access.get_active_setting(UI.S_CONTROLNET_MODELS, UI.F_CANNY_CHECKPOINT, UI.NONE)

            if canny_checkpoint_name != UI.NONE:
                canny_model = NodeWrapper.controlnet_loader.load_controlnet(canny_checkpoint_name)[0]
            else:
                canny_model = None

            access.update_in_cache(Names.C_CN_CANNY_MODEL, [canny_checkpoint_name], canny_model)
            access.update_in_pipeline(Names.P_CN_CANNY_MODEL, canny_model)
        else:
            canny_model = access.get_from_cache(Names.C_CN_CANNY_MODEL)
            access.restore_in_pipeline(Names.P_CN_CANNY_MODEL, canny_model)

        cn_canny = canny_model

        depth_checkpoint_changed = access.setting_changed(UI.S_CONTROLNET_MODELS, UI.F_DEPTH_CHECKPOINT)
        if depth_checkpoint_changed:
            depth_checkpoint_name = access.get_active_setting(UI.S_CONTROLNET_MODELS, UI.F_DEPTH_CHECKPOINT, UI.NONE)

            if depth_checkpoint_name != UI.NONE:
                depth_model = NodeWrapper.controlnet_loader.load_controlnet(depth_checkpoint_name)[0]
            else:
                depth_model = None

            access.update_in_cache(Names.C_CN_DEPTH_MODEL, [depth_checkpoint_name], depth_model)
            access.update_in_pipeline(Names.P_CN_DEPTH_MODEL, depth_model)
        else:
            depth_model = access.get_from_cache(Names.C_CN_DEPTH_MODEL)
            access.restore_in_pipeline(Names.P_CN_DEPTH_MODEL, depth_model)

        cn_depth = depth_model

        recolor_checkpoint_changed = access.setting_changed(UI.S_CONTROLNET_MODELS, UI.F_RECOLOR_CHECKPOINT)
        if recolor_checkpoint_changed:
            recolor_checkpoint_name = access.get_active_setting(UI.S_CONTROLNET_MODELS, UI.F_RECOLOR_CHECKPOINT, UI.NONE)

            if recolor_checkpoint_name != UI.NONE:
                recolor_model = NodeWrapper.controlnet_loader.load_controlnet(recolor_checkpoint_name)[0]
            else:
                recolor_model = None

            access.update_in_cache(Names.C_CN_RECOLOR_MODEL, [recolor_checkpoint_name], recolor_model)
            access.update_in_pipeline(Names.P_CN_RECOLOR_MODEL, recolor_model)
        else:
            recolor_model = access.get_from_cache(Names.C_CN_RECOLOR_MODEL)
            access.restore_in_pipeline(Names.P_CN_RECOLOR_MODEL, recolor_model)

        cn_recolor = recolor_model

        sketch_checkpoint_changed = access.setting_changed(UI.S_CONTROLNET_MODELS, UI.F_SKETCH_CHECKPOINT)
        if sketch_checkpoint_changed:
            sketch_checkpoint_name = access.get_active_setting(UI.S_CONTROLNET_MODELS, UI.F_SKETCH_CHECKPOINT, UI.NONE)

            if sketch_checkpoint_name != UI.NONE:
                sketch_model = NodeWrapper.controlnet_loader.load_controlnet(sketch_checkpoint_name)[0]
            else:
                sketch_model = None

            access.update_in_cache(Names.C_CN_SKETCH_MODEL, [sketch_checkpoint_name], sketch_model)
            access.update_in_pipeline(Names.P_CN_SKETCH_MODEL, sketch_model)
        else:
            sketch_model = access.get_from_cache(Names.C_CN_SKETCH_MODEL)
            access.restore_in_pipeline(Names.P_CN_SKETCH_MODEL, sketch_model)

        cn_sketch = sketch_model

        custom_checkpoint_changed = access.setting_changed(UI.S_CONTROLNET_MODELS, UI.F_CUSTOM_CHECKPOINT)
        if custom_checkpoint_changed:
            custom_checkpoint_name = access.get_active_setting(UI.S_CONTROLNET_MODELS, UI.F_CUSTOM_CHECKPOINT, UI.NONE)

            if custom_checkpoint_name != UI.NONE:
                custom_model = NodeWrapper.controlnet_loader.load_controlnet(custom_checkpoint_name)[0]
            else:
                custom_model = None

            access.update_in_cache(Names.C_CN_CUSTOM_MODEL, [custom_checkpoint_name], custom_model)
            access.update_in_pipeline(Names.P_CN_CUSTOM_MODEL, custom_model)
        else:
            custom_model = access.get_from_cache(Names.C_CN_CUSTOM_MODEL)
            access.restore_in_pipeline(Names.P_CN_CUSTOM_MODEL, custom_model)

        cn_custom = custom_model

        loaded_models = {
            Names.F_BASE_MODEL: base_model,
            Names.F_BASE_CLIP: base_clip,
            Names.F_BASE_VAE: base_vae,
            Names.F_REFINER_MODEL: refiner_model,
            Names.F_REFINER_CLIP: refiner_clip,
            Names.F_REFINER_VAE: refiner_vae,
            Names.F_VAE_MODEL: vae_model,
            Names.F_HIRES_UPSCALER: hires_upscaler,
            Names.F_PRIMARY_UPSCALER: primary_upscaler,
            Names.F_SECONDARY_UPSCALER: secondary_upscaler,
            Names.F_DETAIL_PROCESSOR: detail_processor,
            Names.F_CLIP_VISION_MODEL: clip_vision,
            Names.F_CN_CANNY_MODEL: cn_canny,
            Names.F_CN_DEPTH_MODEL: cn_depth,
            Names.F_CN_RECOLOR_MODEL: cn_recolor,
            Names.F_CN_SKETCH_MODEL: cn_sketch,
            Names.F_CN_CUSTOM_MODEL: cn_custom,
        }

        if data is not None:
            data[Names.S_LOADED_MODELS] = loaded_models

        stage_output = {
            Names.S_LOADED_MODELS: loaded_models,
        }

        return (data, stage_output,)
