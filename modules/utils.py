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

import torch


def next_multiple_of(value, factor):
    return int(int((value + factor - 1) // factor) * factor)


def get_image_size(image):
    if image is None:
        return (None, None,)

    (_, height, width, _) = image.shape

    return (width, height,)


def get_mask_size(mask):
    if mask is None:
        return (None, None,)

    (height, width) = mask.shape

    return (width, height,)


def get_latent_size(latent):
    if latent is None or "samples" not in latent:
        return (None, None,)

    samples = latent["samples"]

    (_, _, height, width) = samples.shape

    return (width, height,)


def get_latent_pixel_size(latent):
    (width, height) = get_latent_size(latent)

    if width is None or height is None:
        return (None, None,)

    return (width * 8, height * 8,)


def slerp(factor, input1, input2):
    dims = input1.shape

    input1 = input1.reshape(dims[0], -1)
    input2 = input2.reshape(dims[0], -1)

    input1_norm = input1 / torch.norm(input1, dim=1, keepdim=True)
    input2_norm = input2 / torch.norm(input2, dim=1, keepdim=True)

    input1_norm[input1_norm != input1_norm] = 0.0
    input2_norm[input2_norm != input2_norm] = 0.0

    omega = torch.acos((input1_norm * input2_norm).sum(1))
    sin_omega = torch.sin(omega)

    result = ((torch.sin((1.0 - factor) * omega) / sin_omega).unsqueeze(1) * input1
              + (torch.sin(factor * omega) / sin_omega).unsqueeze(1) * input2)

    return result.reshape(dims)


def slerp_latents(latent1, latent2, factor):
    result = slerp(factor, latent1.clone(), latent2.clone())
    return result
