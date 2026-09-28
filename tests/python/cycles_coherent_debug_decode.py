"""Read raw linear EXR channel differences from the temporary v16 connector probe."""

import json
from pathlib import Path
import sys

import bpy
import numpy as np


path = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
image = bpy.data.images.load(str(path), check_existing=False)
height, width = int(image.size[1]), int(image.size[0])
rgba = np.asarray(image.pixels[:], dtype=np.float64).reshape(height, width, image.channels)
rgb = rgba[:, :, :3]
roi = rgb[height // 8:height * 7 // 8, width // 8:width * 7 // 8]
halves = (roi[:roi.shape[0] // 2], roi[roi.shape[0] // 2:])
result = {"source": str(path), "resolution": [width, height], "channels": {}}
for name, half in zip(("first", "second"), halves):
    red_minus_green = half[:, :, 0] - half[:, :, 1]
    blue_minus_green = half[:, :, 2] - half[:, :, 1]
    result["channels"][name] = {
        "mean_rgb": np.mean(half, axis=(0, 1)).tolist(),
        "mean_red_minus_green": float(np.mean(red_minus_green)),
        "mean_blue_minus_green": float(np.mean(blue_minus_green)),
        "stage_marker_fraction": float(np.mean(blue_minus_green > 0.125)),
        "accepted_fewer_than_solved_fraction": float(np.mean(red_minus_green > 0.03125)),
        "center_rgb": half[half.shape[0] // 2, half.shape[1] // 2].tolist(),
    }
print(json.dumps(result, indent=2), flush=True)
