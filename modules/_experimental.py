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


def gaussian_latent_noise(width=128, height=128, seed=-1, fac=0.5, batch_size=1, nul=0.0, srnd=False, ver="xl"):
    limit = {
        "v1": {
            "min": {"A": -5.5618, "B": -17.1368, "C": -10.3445, "D": -8.6218},
            "max": {"A": 13.5369, "B": 11.1997, "C": 16.3043, "D": 10.6343},
            "nul": {"A": -5.3870, "B": -14.2931, "C": 6.2738, "D": 7.1220},
        },
        "xl": {
            "min": {"A": -22.2127, "B": -20.0131, "C": -17.7673, "D": -14.9434},
            "max": {"A": 17.9334, "B": 26.3043, "C": 33.1648, "D": 8.9380},
            "nul": {"A": -21.9287, "B": 3.8783, "C": 2.5879, "D": 2.5435},
        }
    }

    # seed
    if seed >= 0:
        torch.manual_seed(seed)

    limit = limit[ver]

    out = []
    for i in range(batch_size):
        if srnd:  # shared random
            rand = torch.rand([height, width])
            lat = torch.stack([
                (limit["min"]["A"] + torch.clone(rand) * (limit["max"]["A"] - limit["min"]["A"])),
                (limit["min"]["B"] + torch.clone(rand) * (limit["max"]["B"] - limit["min"]["B"])),
                (limit["min"]["C"] + torch.clone(rand) * (limit["max"]["C"] - limit["min"]["C"])),
                (limit["min"]["D"] + torch.clone(rand) * (limit["max"]["D"] - limit["min"]["D"])),
            ])

        else:  # separate random
            lat = torch.stack([
                (limit["min"]["A"] + torch.rand([height, width]) * (limit["max"]["A"] - limit["min"]["A"])),
                (limit["min"]["B"] + torch.rand([height, width]) * (limit["max"]["B"] - limit["min"]["B"])),
                (limit["min"]["C"] + torch.rand([height, width]) * (limit["max"]["C"] - limit["min"]["C"])),
                (limit["min"]["D"] + torch.rand([height, width]) * (limit["max"]["D"] - limit["min"]["D"])),
            ])

        tnul = torch.stack([  # black image
            torch.ones([height, width]) * limit["nul"]["A"],
            torch.ones([height, width]) * limit["nul"]["B"],
            torch.ones([height, width]) * limit["nul"]["C"],
            torch.ones([height, width]) * limit["nul"]["D"],
        ])

        out.append(((lat * fac) * (1.0 - nul) + tnul * nul) / 2)

    return {"samples": torch.stack(out)}
