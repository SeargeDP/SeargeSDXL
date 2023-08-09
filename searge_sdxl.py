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

from .modules.ui import Defs

from .modules.prompt_text_input import SeargeTextInputV2
from .modules.prompt_adapter import SeargePromptAdapterV2
from .modules.image_adapter import SeargeImageAdapterV2
from .modules.controlnet_adapter import SeargeControlnetAdapterV2

from .modules.ui_separator import SeargeSeparator
from .modules.ui_preview_image import SeargePreviewImage

from .modules.ui_model_selector import SeargeModelSelector
from .modules.ui_upscale_models import SeargeUpscaleModels
from .modules.ui_loras import SeargeLoras
from .modules.ui_controlnet_models import SeargeControlnetModels

from .modules.ui_generation_parameters import SeargeGenerationParameters
from .modules.ui_conditioning_parameters import SeargeConditioningParameters
from .modules.ui_advanced_parameters import SeargeAdvancedParameters
from .modules.ui_image_saving import SeargeImageSaving
from .modules.ui_operating_mode import SeargeOperatingMode
from .modules.ui_img2img_inpaint import SeargeImage2ImageAndInpainting
from .modules.ui_custom_prompt_mode import SeargeCustomPromptMode
from .modules.ui_prompt_styles import SeargePromptStyles
from .modules.ui_high_resolution import SeargeHighResolution
from .modules.ui_condition_mixing import SeargeConditionMixing

from .modules.magic_box import SeargeMagicBox
from .modules.mb_pipeline_start import SeargePipelineStart
from .modules.mb_pipeline_terminator import SeargePipelineTerminator

from .modules.after_vae_decode import SeargeCustomAfterVaeDecode
from .modules.after_upscaling import SeargeCustomAfterUpscaling

from .modules.debug_printer import SeargeDebugPrinter


# ====================================================================================================
# Register nodes in ComfyUI
# ====================================================================================================

SEARGE_CLASS_MAPPINGS = {
    f"SeargeTextInputV2{Defs.CLASS_POSTFIX}": SeargeTextInputV2,
    f"SeargePromptAdapterV2{Defs.CLASS_POSTFIX}": SeargePromptAdapterV2,
    f"SeargeImageAdapterV2{Defs.CLASS_POSTFIX}": SeargeImageAdapterV2,
    f"SeargeControlnetAdapterV2{Defs.CLASS_POSTFIX}": SeargeControlnetAdapterV2,

    f"SeargeSeparator{Defs.CLASS_POSTFIX}": SeargeSeparator,
    f"SeargePreviewImage{Defs.CLASS_POSTFIX}": SeargePreviewImage,

    f"SeargeAdvancedParameters{Defs.CLASS_POSTFIX}": SeargeAdvancedParameters,
    f"SeargeConditioningParameters{Defs.CLASS_POSTFIX}": SeargeConditioningParameters,
    f"SeargeConditionMixing{Defs.CLASS_POSTFIX}": SeargeConditionMixing,
    f"SeargeControlnetModels{Defs.CLASS_POSTFIX}": SeargeControlnetModels,
    f"SeargeCustomPromptMode{Defs.CLASS_POSTFIX}": SeargeCustomPromptMode,
    f"SeargeGenerationParameters{Defs.CLASS_POSTFIX}": SeargeGenerationParameters,
    f"SeargeHighResolution{Defs.CLASS_POSTFIX}": SeargeHighResolution,
    f"SeargeImage2ImageAndInpainting{Defs.CLASS_POSTFIX}": SeargeImage2ImageAndInpainting,
    f"SeargeImageSaving{Defs.CLASS_POSTFIX}": SeargeImageSaving,
    f"SeargeLoras{Defs.CLASS_POSTFIX}": SeargeLoras,
    f"SeargeModelSelector{Defs.CLASS_POSTFIX}": SeargeModelSelector,
    f"SeargeOperatingMode{Defs.CLASS_POSTFIX}": SeargeOperatingMode,
    f"SeargePromptStyles{Defs.CLASS_POSTFIX}": SeargePromptStyles,
    f"SeargeUpscaleModels{Defs.CLASS_POSTFIX}": SeargeUpscaleModels,

    f"SeargeMagicBox{Defs.CLASS_POSTFIX}": SeargeMagicBox,
    f"SeargePipelineStart{Defs.CLASS_POSTFIX}": SeargePipelineStart,
    f"SeargePipelineTerminator{Defs.CLASS_POSTFIX}": SeargePipelineTerminator,

    f"SeargeCustomAfterVaeDecode{Defs.CLASS_POSTFIX}": SeargeCustomAfterVaeDecode,
    f"SeargeCustomAfterUpscaling{Defs.CLASS_POSTFIX}": SeargeCustomAfterUpscaling,

    f"SeargeDebugPrinter{Defs.CLASS_POSTFIX}": SeargeDebugPrinter,
}

# ====================================================================================================
# Human readable names for the nodes
# ====================================================================================================

SEARGE_DISPLAY_NAME_MAPPINGS = {
    f"SeargeTextInputV2{Defs.CLASS_POSTFIX}": "Text Input v2",
    f"SeargePromptAdapterV2{Defs.CLASS_POSTFIX}": "Prompt Adapter v2",
    f"SeargeImageAdapterV2{Defs.CLASS_POSTFIX}": "Image Adapter v2",
    f"SeargeControlnetAdapterV2{Defs.CLASS_POSTFIX}": "Controlnet Adapter v2",

    f"SeargeSeparator{Defs.CLASS_POSTFIX}": "Separator",
    f"SeargePreviewImage{Defs.CLASS_POSTFIX}": "SeargePreviewImage",

    f"SeargeAdvancedParameters{Defs.CLASS_POSTFIX}": "Advanced Parameters v2",
    f"SeargeConditioningParameters{Defs.CLASS_POSTFIX}": "Conditioning Parameters v2",
    f"SeargeConditionMixing{Defs.CLASS_POSTFIX}": "Condition Mixing v2",
    f"SeargeControlnetModels{Defs.CLASS_POSTFIX}": "Controlnet Models Selector v2",
    f"SeargeCustomPromptMode{Defs.CLASS_POSTFIX}": "Custom Prompt Mode v2",
    f"SeargeGenerationParameters{Defs.CLASS_POSTFIX}": "Generation Parameters v2",
    f"SeargeHighResolution{Defs.CLASS_POSTFIX}": "High Resolution v2",
    f"SeargeImage2ImageAndInpainting{Defs.CLASS_POSTFIX}": "Image to Image and Inpainting v2",
    f"SeargeImageSaving{Defs.CLASS_POSTFIX}": "Image Saving v2",
    f"SeargeLoras{Defs.CLASS_POSTFIX}": "Lora Selector v2",
    f"SeargeModelSelector{Defs.CLASS_POSTFIX}": "Model Selector v2",
    f"SeargeOperatingMode{Defs.CLASS_POSTFIX}": "Operating Mode v2",
    f"SeargePromptStyles{Defs.CLASS_POSTFIX}": "Prompt Styles v2",
    f"SeargeUpscaleModels{Defs.CLASS_POSTFIX}": "Upscale Models Selector v2",

    f"SeargeMagicBox{Defs.CLASS_POSTFIX}": "Searge's Magic Box for SDXL",
    f"SeargePipelineStart{Defs.CLASS_POSTFIX}": "Magic Box Pipeline Start",
    f"SeargePipelineTerminator{Defs.CLASS_POSTFIX}": "Magic Box Pipeline Terminator",

    f"SeargeCustomAfterVaeDecode{Defs.CLASS_POSTFIX}": "After VAE Decode",
    f"SeargeCustomAfterUpscaling{Defs.CLASS_POSTFIX}": "After Upscaling",

    f"SeargeDebugPrinter{Defs.CLASS_POSTFIX}": "Debug Printer",
}
