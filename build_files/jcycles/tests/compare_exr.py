# Compare two multilayer EXRs channel by channel (run with any Blender's Python: OIIO + numpy).
#   blender -b --factory-startup --python compare_exr.py -- a.exr b.exr
import sys

import numpy as np
import OpenImageIO as oiio

a_path, b_path = sys.argv[sys.argv.index("--") + 1:][:2]


def load(path):
    """All channels of all parts (Blender writes one part per pass)."""
    image = oiio.ImageInput.open(path)
    channels = {}
    first = image.spec()
    part = 0
    while image.seek_subimage(part, 0):
        spec = image.spec()
        pixels = np.asarray(image.read_image(part, 0, 0, spec.nchannels, oiio.FLOAT)).reshape(
            spec.height, spec.width, -1)
        for i, name in enumerate(spec.channelnames):
            channels[name] = pixels[:, :, i]
        part += 1
    image.close()
    return channels, first


a, sa = load(a_path)
b, sb = load(b_path)
skip = ("Composite.", "Image.")
names_a = {n for n in a if not n.startswith(skip)}
names_b = {n for n in b if not n.startswith(skip)}
print("COMPARE size", (sa.width, sa.height), (sb.width, sb.height))
print("COMPARE only in first:", sorted(names_a - names_b))
print("COMPARE only in second:", sorted(names_b - names_a))
worst = 0.0
for name in sorted(names_a & names_b):
    diff = float(np.max(np.abs(a[name] - b[name])))
    worst = max(worst, diff)
    if diff > 0:
        print("COMPARE differs", name, diff)
print("COMPARE channels", len(names_a & names_b), "max abs diff", worst)
print("COMPARE RESULT", "IDENTICAL" if worst == 0.0 and names_a == names_b else "DIFFERENT")
