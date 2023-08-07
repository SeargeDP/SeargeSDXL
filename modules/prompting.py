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

import nodes


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
