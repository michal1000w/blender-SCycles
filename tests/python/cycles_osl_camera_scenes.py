# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Render custom (OSL) camera scenes on the CPU or Metal to EXR.

Metal has no OSL back end: it runs a translation of the compiled camera shader, see
intern/cycles/device/metal/osl_camera.h. These scenes compare that translation with the OSL
runtime of the CPU device.

Probe scenes evaluate one group of OSL operations per render and write the results as the
throughput of a camera looking at a uniform white world, so the image holds the computed values:
with one sample per pixel both devices must agree to floating point accuracy. Derivative probes do
the same for the automatic differentiation behind the ray differentials.

Render scenes use the camera shader templates that ship with Blender and a shader that traces
rays through a stack of spherical lens surfaces, with arrays, loops and early exits. Those compare
statistically.

  blender -b --factory-startup --python tests/python/cycles_osl_camera_scenes.py -- \\
      --output DIRECTORY [--device CPU|METAL|METAL_CPU] [--scenes a,b,c] [--samples N] \\
      [--resolution N] [--session]

METAL_CPU renders with the GPU and the CPU together, where each runs its own implementation of
the camera shader. --session instead renders a sequence of edits in one render session: parameter
changes, other shaders, a shader that Metal cannot translate, a regular camera and an animation.

Compare two output directories with cycles_osl_camera_compare.py.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy


# Values are encoded so that they stay positive and survive the film: v * 0.25 + 4.
PROBE_SHADER = r"""
struct Surface {
    float radius;
    vector tint;
    float weights[3];
};

float polynomial(float x, float coefficients[5])
{
    float result = 0.0;
    for (int i = arraylength(coefficients) - 1; i >= 0; i--) {
        result = result * x + coefficients[i];
    }
    return result;
}

/* Early returns from nested loops. */
float first_above(float threshold, float values[8])
{
    for (int pass = 0; pass < 2; pass++) {
        for (int i = 0; i < 8; i++) {
            if (values[i] < 0.0) {
                continue;
            }
            if (values[i] > threshold) {
                return values[i] + pass;
            }
            if (i == 6 && pass == 1) {
                break;
            }
        }
    }
    return -1.0;
}

int bisect(float x, float table[8])
{
    int left = 0;
    int right = 7;
    while (left < right - 1) {
        int middle = (left + right) / 2;
        if (table[middle] <= x) {
            left = middle;
        }
        else {
            right = middle;
        }
    }
    return left;
}

shader probe(int test = 0,
             float scale = 1.0,
             vector tint = vector(1.0, 1.0, 1.0),
             int flag = 0 [[ string widget = "checkBox" ]],
             string space = "camera",
             float derived = scale * 2.0 + 1.0,
             output point position = point(0.0),
             output vector direction = vector(0.0, 0.0, 1.0),
             output color throughput = color(0.0))
{
    point raster = camera_shader_raster_position();
    float x = raster[0] * 4.0 - 2.0;
    float y = raster[1] * 4.0 - 2.0;
    /* Exactly representable coordinates, for functions that amplify rounding. */
    float xi = floor(raster[0] * 64.0);
    float yi = floor(raster[1] * 64.0);
    vector p = vector(x, y, x * y * 0.5 + 0.25);
    vector q = vector(y - 0.3, 0.7 - x, 1.5);
    vector r = vector(0.0);

    if (test == 0) {
        r = vector(sin(x * 3.0), cos(y * 3.0), tan(x * y * 0.3));
    }
    else if (test == 1) {
        r = vector(asin(x), acos(y), atan(x * 3.0));
    }
    else if (test == 2) {
        r = vector(atan2(y, x), sinh(x), cosh(y));
    }
    else if (test == 3) {
        r = vector(tanh(x * y), exp(x), exp2(y));
    }
    else if (test == 4) {
        r = vector(expm1(x * 0.01) * 50.0, max(log(x), -8.0), log2(fabs(x) + 0.1));
    }
    else if (test == 5) {
        r = vector(log10(fabs(y) + 0.1), logb(x * 10.0 + 0.3), sqrt(x));
    }
    else if (test == 6) {
        r = vector(min(inversesqrt(x + 0.5), 12.0), cbrt(x * 3.0), pow(x, y));
    }
    else if (test == 7) {
        r = vector(pow(x, 3.0), pow(x, 2.0), pow(fabs(x) + 0.1, y));
    }
    else if (test == 8) {
        r = vector(fmod(x * 3.0, y), mod(x * 3.0, y + 0.01), floor(x * 3.0 + 0.11));
    }
    else if (test == 9) {
        r = vector(ceil(y * 3.0 + 0.11), round(x * y * 4.0 + 0.2), trunc(x * 3.0 + 0.11));
    }
    else if (test == 10) {
        r = vector(sign(x + 0.013), fabs(x * y), step(0.31, x));
    }
    else if (test == 11) {
        r = vector(smoothstep(-1.0, 1.0, x), mix(x, y, 0.3), clamp(x, -0.5, 0.5));
    }
    else if (test == 12) {
        r = vector(min(x, y), max(x, y), select(x, y, x > y));
    }
    else if (test == 13) {
        /* Division by zero is zero. */
        r = vector(erf(x), erfc(y), x / floor(y + 0.5));
    }
    else if (test == 14) {
        int a = (int)(x * 10.0 + 0.3);
        int b = (int)(y * 3.0 + 0.3);
        r = vector(a % 4, a / b, (a & 5) | (b << 2));
        r += vector((a ^ b) & 15, (~a) & 7, (a >> 1) & 7) * 0.001;
    }
    else if (test == 15) {
        r = vector(dot(p, q), length(p), distance(p, q));
    }
    else if (test == 16) {
        r = normalize(p) + normalize(vector(0.0)) + cross(p, q) * 0.1;
    }
    else if (test == 17) {
        vector n = normalize(q);
        r = reflect(normalize(p), n) + refract(normalize(p), n, 0.8) * 0.5 +
            faceforward(n, p, n) * 0.25;
    }
    else if (test == 18) {
        r = rotate(p, x * 2.0, point(0.1, 0.2, 0.3), point(1.0, 0.5, 0.25));
    }
    else if (test == 19) {
        matrix m = matrix(1.0, 0.2, 0.0, 0.0,
                          0.1, 1.5, 0.3, 0.0,
                          0.0, 0.4, 0.8, 0.0,
                          x, y, 0.5, 1.0);
        matrix inverse = 1.0 / m;
        matrix product = m * transpose(inverse) * 0.5;
        product[1][2] = y;
        r = transform(m, point(p)) * 0.2 + transform(inverse, vector(q)) * 0.2 +
            transform(m, normal(q)) * 0.1 +
            vector(determinant(m), product[1][2], product[2][1]);
    }
    else if (test == 20) {
        r = transform("camera", "world", point(p)) * 0.5 +
            transform("world", "camera", vector(q)) * 0.25 +
            transform(space, "common", normal(p)) * 0.125;
    }
    else if (test == 21) {
        point ndc = transform("camera", "NDC", point(x * 0.2, y * 0.2, 3.0));
        point pixel = transform("camera", "raster", point(x * 0.2, y * 0.2, 3.0));
        point screen = transform("camera", "screen", point(x * 0.2, y * 0.2, 3.0));
        /* Not `matrix("camera", "world")` or `getmatrix()`: the OSL runtime of the CPU device
         * crashes when it folds those constants for a camera shader. */
        matrix scaled = matrix("camera", 2.0);
        r = vector(ndc[0] + ndc[1], (pixel[0] + pixel[1]) * 0.01, screen[0] + screen[1]) +
            vector(scaled[3][0], scaled[0][0], scaled[3][2]) * 0.1;
    }
    else if (test == 22) {
        vector sensor = vector(0.0);
        vector resolution = vector(0.0);
        float ratio = 0.0;
        int found = getattribute("cam:sensor_size", sensor);
        found += getattribute("cam:image_resolution", resolution);
        found += getattribute("cam:aperture_aspect_ratio", ratio);
        found += getattribute("cam:does_not_exist", ratio);
        r = vector(sensor[0] * 0.1 + sensor[1] * 0.01 + sensor[2],
                   resolution[0] * 0.01 + resolution[1] * 0.001 + resolution[2],
                   ratio + found);
    }
    else if (test == 23) {
        float size = 0.0;
        float focus = 0.0;
        vector aperture = vector(0.0);
        float averaged = 0.0;
        getattribute("cam:aperture_size", size);
        getattribute("cam:focal_distance", focus);
        getattribute("cam:aperture_position", aperture);
        getattribute("cam:sensor_size", averaged);
        vector lens_sample = camera_shader_random_sample();
        r = vector(size * 100.0 + focus, aperture[0] * 100.0 + aperture[1] * 10.0,
                   lens_sample[0] + lens_sample[1] * 0.1 + averaged * 0.01);
    }
    else if (test == 24) {
        vector h = hashnoise(point(xi, yi, 3.0));
        r = vector(hashnoise(xi, yi), hashnoise(xi * 0.25), hashnoise(point(xi, yi, 1.0), 2.0)) + h;
    }
    else if (test == 25) {
        vector c = cellnoise(p * 2.3 + 0.01);
        r = vector(cellnoise(x * 5.0 + 0.01), noise("cell", p * 3.1 + 0.01, y),
                   noise("hash", xi, yi)) + c;
    }
    else if (test == 26) {
        r = vector(noise(p * 2.5), snoise(x * 2.0, y * 2.0), noise("perlin", p * 3.0, x));
    }
    else if (test == 27) {
        vector n = snoise(p * 2.0);
        vector m = noise("uperlin", p * 1.5, y);
        r = n + m * 0.5 + vector(noise(x * 3.0), snoise(x * 3.0), 0.0);
    }
    else if (test == 28) {
        color hsv = transformc("rgb", "hsv", color(0.2 + raster[0] * 0.6, raster[1], 0.4));
        color back = color("hsv", raster[0], 0.8, 0.9);
        color hsl = transformc("hsl", "rgb", color(raster[1], 0.7, 0.3 + raster[0] * 0.4));
        r = vector(hsv) + vector(back) * 0.5 + vector(hsl) * 0.25;
    }
    else if (test == 29) {
        color c = wavelength_color(380.0 + raster[0] * 420.0);
        r = vector(c) + vector(luminance(c), 0.0, luminance(color(raster[0], raster[1], 0.3)));
    }
    else if (test == 30) {
        color c = color(0.1 + raster[0], 0.2 + raster[1] * 0.5, 0.3);
        r = vector(transformc("rgb", "YIQ", c)) + vector(transformc("rgb", "XYZ", c)) * 0.5 +
            vector(transformc("xyY", "rgb", color(0.3 + raster[0] * 0.1, 0.33, 0.5))) * 0.25 +
            vector(transformc("sRGB", "rgb", c)) * 0.125;
    }
    else if (test == 31) {
        float coefficients[5] = {0.5, -1.0, 0.25, 0.125, -0.0625};
        float values[8] = {-1.0, 0.1, 0.3, 0.5, 0.9, 1.3, 1.7, 2.5};
        float table[8] = {-2.0, -1.5, -0.7, 0.0, 0.4, 0.9, 1.5, 2.0};
        float copy[8];
        copy = table;
        copy[3] = x;
        /* An array of colors is initialized one element at a time. */
        color palette[4] = {color(1.0, 0.0, 0.0), color(0.0, 1.0, 0.0), color(0.0, 0.0, 1.0),
                            color(0.5, 0.25, 0.125)};
        int cell = bisect(x, table);
        r = vector(polynomial(x, coefficients), first_above(y, values),
                   cell + copy[3] * 0.01 + copy[5] * 0.001) + vector(palette[cell % 4]) * 0.1;
    }
    else if (test == 32) {
        Surface a;
        a.radius = x;
        a.tint = vector(y, 0.5, 0.25);
        a.weights[0] = 0.5;
        a.weights[1] = x * y;
        a.weights[2] = 2.0;
        Surface b = a;
        b.weights[2] = b.weights[0] + b.weights[1];
        int count = 0;
        float sum = 0.0;
        do {
            sum += b.weights[count] * (count + 1);
            count++;
        } while (count < 3);
        r = vector(b.radius + b.tint[0], sum, count + b.tint[2]);
    }
    else if (test == 33) {
        /* Parameters: numbers are read at render time, strings are constants. */
        r = tint * scale + vector(flag, derived, 0.0);
        if (space == "world") {
            r[2] = 1.0;
        }
        else if (startswith(space, "cam")) {
            r[2] = 2.0 + strlen(space) * 0.1;
        }
    }
    else if (test == 34) {
        float s = 0.0;
        float c = 0.0;
        sincos(x * 2.0, s, c);
        r = vector(s + c * 0.5, degrees(x) * 0.01 + radians(y * 50.0), hypot(x, y) + log(fabs(x) + 0.5, 3.0));
    }
    else if (test == 35) {
        float bad = sqrt(-1.0) + log(0.0) * 0.0;
        r = vector(isnan(x / (y - y)) + isinf(1.0 / (fabs(x) + 1e-30)) * 2.0 + isfinite(x) * 4.0,
                   raytype("camera") + raytype("shadow") * 2.0, bad);
    }
    else if (test == 36) {
        /* Leave the shader early: the remaining pixels keep the value written so far. */
        throughput = color(5.0, 4.5, 4.0);
        if (x > 0.5) {
            return;
        }
        if (y > 0.5) {
            exit();
        }
        r = vector(x, y, 0.0);
    }
    else if (test == 37) {
        /* Zero throughput: no ray. */
        if (x * x + y * y > 3.0) {
            throughput = color(0.0);
            return;
        }
        r = vector(1.0, x, y);
    }
    else if (test == 38) {
        /* Emission in W/m^2 is large: compare the color and the order of magnitude. */
        color c = blackbody(600.0 + raster[0] * 13400.0);
        float sum = c[0] + c[1] + c[2];
        r = vector(c[0] / sum, c[1] / sum, log10(sum) * 0.25);
    }
    else if (test == 39) {
        r = vector(hash(xi * 0.5) & 255, hash(point(xi, yi, 0.25)) & 255, hash(xi, yi) & 255) * 0.01 +
            vector(hash((int)xi) & 15, hash(point(xi, yi, 2.0), yi) & 15, 0.0) * 0.1;
    }
    /* Derivatives. The raster position changes by one pixel in x and y. */
    else if (test == 40) {
        r = vector(Dx(sin(x * 3.0)), Dy(cos(y * 3.0) * x), filterwidth(x * y)) * 16.0;
    }
    else if (test == 41) {
        r = vector(Dx(pow(fabs(x) + 0.5, y)), Dy(pow(fabs(x) + 0.5, y)), Dx(sqrt(x + 2.5))) * 16.0;
    }
    else if (test == 42) {
        r = vector(Dx(log(x + 2.5) * exp(y * 0.5)), Dy(atan2(y, x + 2.5)), Dx(x / (y + 3.0))) * 16.0;
    }
    else if (test == 43) {
        r = (Dx(normalize(p)) + Dy(cross(p, q)) * 0.25 +
             vector(Dx(length(p)), Dy(dot(p, q)), Dx(distance(p, q)))) * 16.0;
    }
    else if (test == 44) {
        r = vector(Dx(noise(p * 2.5)), Dy(snoise(x * 2.0, y * 2.0)), Dx(noise("perlin", p * 3.0, x))) *
            4.0;
    }
    else if (test == 45) {
        matrix m = matrix(1.0, 0.2, 0.0, 0.1,
                          0.1, 1.5, 0.3, 0.0,
                          0.0, 0.4, 0.8, 0.2,
                          0.3, 0.1, 0.5, 1.0);
        r = (Dx(transform(m, point(p))) + Dy(transform(m, vector(p))) * 0.5 +
             Dx(rotate(p, 0.7, point(0.0), point(0.3, 0.5, 1.0))) * 0.25) * 16.0;
    }
    else if (test == 46) {
        r = vector(Dx(mix(x, y * x, 0.3 + x * 0.1)), Dy(min(x * 2.0, y * y)),
                   Dx(smoothstep(-1.0, 1.5, x * y))) * 16.0;
    }
    else if (test == 47) {
        r = vector(Dx(fmod(x * 3.0, 0.7)), Dy(fabs(y) * x + max(x, y)), Dx(asin(x * 0.4) + acos(y * 0.4))) *
            16.0;
    }
    else if (test == 48) {
        float s = 0.0;
        float c = 0.0;
        sincos(x * y, s, c);
        r = vector(Dx(s * tanh(x)), Dy(c + sinh(y * 0.5)), Dx(inversesqrt(x + 2.5) + cbrt(x + 2.5))) * 16.0;
    }
    else if (test == 49) {
        /* Derivatives through arrays, components and control flow. */
        float values[3] = {x, y * x, 1.0};
        vector v = vector(values[1], values[0] * values[0], 0.0);
        v[2] = v[0] * v[1];
        float w = 0.0;
        for (int i = 0; i < 3; i++) {
            w += values[i] * (i + 1);
        }
        if (x > 0.0) {
            w = w * w;
        }
        r = vector(Dx(v[2]), Dy(v[2]), Dx(w) * 0.25) * 16.0;
    }

    throughput = color(r * 0.25 + 4.0);
}
"""

# The parts of the language beyond what lens shaders need: splines, all kinds of noise, strings
# computed at render time, messages, closures, arrays of every type and the remaining functions.
PROBE2_SHADER = r"""
struct Lens {
    float radius;
    vector tint;
    float weights[3];
    string kind;
};

struct Glass {
    float ior;
    vector tint;
    string kind;
};

float pick(string which, float a, float b)
{
    if (which == "first") {
        return a;
    }
    if (startswith(which, "sec")) {
        return b;
    }
    return a + b;
}

/* A loop with a function call in its condition, and returns from nested loops. */
int count_below(float limit, float values[6])
{
    int n = 0;
    while (n < 6 && pick("first", values[n], 0.0) < limit) {
        n++;
    }
    return n;
}

float search(float target, float values[6], output int where)
{
    where = -1;
    for (int outer = 0; outer < 3; outer++) {
        int i = 0;
        do {
            if (values[i] * (outer + 1) > target) {
                where = i + outer * 10;
                return values[i];
            }
            i++;
            if (i == 5) {
                continue;
            }
        } while (i < 6);
        if (outer == 1 && target > 100.0) {
            break;
        }
    }
    return -2.0;
}

shader probe2(int test = 0,
              float scale = 1.0,
              string name = "camera",
              string other = "second",
              matrix xform = matrix(1.0, 0.0, 0.0, 0.0,
                                    0.0, 2.0, 0.0, 0.0,
                                    0.5, 0.0, 1.0, 0.0,
                                    0.1, 0.2, 0.3, 1.0),
              float table[4] = {0.25, 0.5, 1.0, 2.0},
              int count = 3,
              string derived = concat(name, "_", other),
              output point position = point(0.0),
              output vector direction = vector(0.0, 0.0, 1.0),
              output color throughput = color(0.0),
              output float extra[2] = {0.0, 0.0},
              output string label = "none",
              output closure color surface = 0)
{
    point raster = camera_shader_raster_position();
    float x = raster[0] * 4.0 - 2.0;
    float y = raster[1] * 4.0 - 2.0;
    float u01 = raster[0];
    float v01 = raster[1];
    float xi = floor(raster[0] * 64.0);
    float yi = floor(raster[1] * 64.0);
    vector p = vector(x, y, x * y * 0.5 + 0.25);
    vector q = vector(y - 0.3, 0.7 - x, 1.5);
    vector r = vector(0.0);

    if (test == 0) {
        float knots[7] = {0.0, 0.1, 0.5, 0.2, 0.9, 1.0, 0.3};
        r = vector(spline("catmull-rom", u01, knots), spline("bezier", v01, knots),
                   spline("linear", u01 * 1.2 - 0.1, knots));
    }
    else if (test == 1) {
        float knots[6] = {0.0, 0.1, 0.5, 0.2, 0.9, 1.0};
        color colors[5] = {color(0.0), color(1.0, 0.0, 0.0), color(0.0, 1.0, 0.0),
                           color(0.0, 0.0, 1.0), color(1.0)};
        r = vector(spline("bspline", u01, knots), spline("hermite", v01, 6, knots), 0.0) +
            spline("constant", u01, colors);
    }
    else if (test == 2) {
        color colors[6] = {color(0.0), color(1.0, 0.0, 0.0), color(0.0, 1.0, 0.0),
                           color(0.0, 0.0, 1.0), color(1.0), color(0.5, 0.25, 0.125)};
        float knots[8] = {0.0, 0.0, 0.2, 0.9, 0.4, 1.0, 1.0, 5.0};
        r = spline("catmull-rom", u01, colors) + spline("bspline", v01, 5, colors) * 0.5 +
            vector(spline("unknown", u01, knots), spline("linear", v01, 5, knots),
                   spline("bezier", u01, 7, knots));
    }
    else if (test == 3) {
        float rising[6] = {0.0, 0.0, 0.3, 0.6, 1.0, 1.0};
        float falling[6] = {1.0, 1.0, 0.7, 0.2, 0.0, 0.0};
        float bezier[7] = {0.0, 0.1, 0.2, 0.5, 0.7, 0.8, 1.0};
        r = vector(splineinverse("linear", u01 * 1.2 - 0.1, rising),
                   splineinverse("catmull-rom", v01, falling),
                   splineinverse("bezier", u01, bezier) + splineinverse("bspline", v01, 5, rising));
    }
    else if (test == 4) {
        float knots[5] = {x, y, x * y, 1.0, 0.0};
        float fixed[6] = {0.0, 0.1, 0.5, 0.2, 0.9, 1.0};
        color colors[4] = {color(x, 0.0, y), color(1.0, y, 0.0), color(0.0, 1.0, x), color(x * y)};
        r = vector(Dx(spline("bspline", u01, knots)), Dy(spline("catmull-rom", v01, fixed)),
                   Dx(spline("linear", u01 * 1.5 - 0.2, fixed))) * 16.0 +
            Dy(spline("catmull-rom", v01, colors)) * 4.0;
    }
    else if (test == 5) {
        r = vector(pnoise(x * 3.0, 4.0), psnoise(x * 3.0, y * 3.0, 4.0, 5.0),
                   pnoise(x * 2.0, y * 2.0, 3.0, 0.2));
    }
    else if (test == 6) {
        vector a = pnoise(p * 2.0, x, point(3.0, 4.0, 5.0), 2.0);
        vector b = psnoise(p * 2.0, point(2.0, 2.5, -3.0));
        r = vector(pnoise(p * 2.0, point(3.0, 4.0, 5.0)), psnoise(p * 3.0, y, point(3.0), 4.0),
                   0.0) + a * 0.5 + b * 0.25 + (vector)psnoise(x * 2.0, 3.0) * 0.125;
    }
    else if (test == 7) {
        vector c = pnoise("cell", p * 3.0 + 0.01, point(4.0, 4.0, 4.0));
        r = vector(pnoise("perlin", x * 3.0, 4.0), pnoise("uperlin", p * 2.0, point(3.0)),
                   pnoise("hash", xi, yi, 16.0, 8.0)) + c +
            vector(pnoise("cell", x * 3.0 + 0.01, 4.0), pnoise("snoise", x, y, 2.0, 2.0),
                   pnoise("noise", p, y, point(2.0), 3.0));
    }
    else if (test == 8) {
        r = vector(noise("simplex", x * 3.0), noise("usimplex", x * 3.0, y * 3.0),
                   noise("simplex", p * 2.0)) + vector(noise("simplex", p * 2.0, x), 0.0, 0.0);
    }
    else if (test == 9) {
        vector a = noise("usimplex", p * 2.0);
        vector b = noise("simplex", x * 2.0, y * 2.0);
        vector c = noise("simplex", p, y * 3.0);
        vector d = noise("usimplex", x * 4.0);
        r = a + b * 0.5 + c * 0.25 + d * 0.125;
    }
    else if (test == 10) {
        r = vector(Dx(noise("simplex", p * 2.0)), Dy(noise("usimplex", x * 2.0, y * 2.0)),
                   Dx(pnoise(p * 2.0, point(3.0, 4.0, 5.0)))) * 4.0 +
            vector(Dy(psnoise(x * 3.0, y * 3.0, 4.0, 5.0)), Dx(noise("simplex", x * 3.0)),
                   Dx(noise("simplex", p, x))) * 4.0;
    }
    else if (test == 11) {
        /* The kind of noise is chosen at render time. */
        string kind = (x > 0.0) ? "perlin" : "cell";
        if (y > 1.0) {
            kind = "usimplex";
        }
        /* Names that the OSL runtime only knows when they are not constants. */
        string flat = (x > 1.0) ? "null" : ((x > 0.0) ? "unull" : "simplexnoise");
        vector a = noise(flat, p);
        r = vector(noise(kind, p * 3.0 + 0.01), noise(flat, x * 2.0), noise(flat, x, y)) + a * 0.5;
    }
    else if (test == 12) {
        string s = (x > 0.0) ? "abc" : "wxyz1";
        string t = concat(s, "_", name);
        r = vector(strlen(t), strlen(s) + (s == "abc") * 10 + (s != t) * 20,
                   (t == "abc_camera") + startswith(t, "wx") * 2 + endswith(s, "c") * 4 +
                   endswith(t, name) * 8);
    }
    else if (test == 13) {
        string s = format("%d_%s_%4.2f|%g", 3, name, 1.5, 0.25);
        string sub = substr(name, 1, 3);
        string tail = substr(name, -2, 10);
        r = vector(strlen(s) + (s == "3_camera_1.50|0.25") * 100, stoi("42") + stof("1.5") +
                   stoi("x7") + stoi(" -12abc"),
                   strlen(sub) + (sub == "ame") * 10 + (tail == "ra") * 20 + getchar(name, 2) * 0.01 +
                   getchar(name, 20));
    }
    else if (test == 14) {
        string names[3] = {"camera", "world", "NDC"};
        int i = ((int)(x + 2.0)) % 3;
        string space = names[i];
        names[1] = "raster";
        r = transform(space, "world", p) * 0.2 +
            vector(strlen(names[(int)(y + 2.0) % 3]), space == "NDC", arraylength(names));
    }
    else if (test == 15) {
        int found[6] = {-1, -1, -1, -1, -1, -1};
        int none[2] = {7, 7};
        int a = regex_search("foobar42", found, "([a-z]+)([0-9]+)");
        int b = regex_match("foobar", "foo.*") + regex_search("abc", none, "x") * 2 +
                regex_match("foobar", "oba") * 4 + regex_search("foobar", "oba") * 8;
        r = vector(a + found[0] * 0.1 + found[1] * 0.01, b + none[0] * 0.01,
                   found[2] + found[3] * 0.1 + found[4] * 0.01 + found[5] * 0.001);
    }
    else if (test == 16) {
        string parts[4] = {"k", "k", "k", "k"};
        string words[3];
        string limited[4];
        int n = split("a,bb,,ccc", parts, ",");
        int m = split("  hello big  world again ", words);
        int k = split("1-2-3-4", limited, "-", (int)(x + 2.5));
        r = vector(n + strlen(parts[1]) * 0.1 + strlen(parts[3]) * 0.01,
                   m + strlen(words[0]) * 0.1 + strlen(words[2]) * 0.01,
                   k + strlen(limited[1]) * 0.1 + strlen(limited[3]) * 0.01 +
                   (limited[2] == "3") * 0.001);
    }
    else if (test == 17) {
        float early = 5.0;
        int had_early = getmessage("late", early);
        setmessage("foo", x * 2.0);
        setmessage("foo", 100.0);
        setmessage("late", 3.0);
        setmessage("color", color(x, y, 1.0));
        string names[2] = {"a", "bcd"};
        setmessage("names", names);
        setmessage("text", (x > 0.0) ? "right" : "lefty");
        float value = 0.0;
        int ok = getmessage("foo", value);
        float late = 7.0;
        ok += getmessage("late", late) * 2;
        vector wrong = vector(9.0);
        ok += getmessage("color", wrong) * 4;
        color right = color(0.0);
        ok += getmessage("color", right) * 8;
        string got[2];
        ok += getmessage("names", got) * 16;
        string text = "";
        ok += getmessage("text", text) * 32;
        float missing = 2.0;
        ok += getmessage("missing", missing) * 64 + had_early * 128;
        r = vector(value + late * 0.1 + early * 0.01, ok * 0.1,
                   right[0] + wrong[1] + strlen(got[1]) * 0.1 + strlen(text) * 0.01 + missing);
    }
    else if (test == 18) {
        /* Closures have no meaning for a camera. */
        closure color c = diffuse(N) * 0.5 + emission() * color(1.0, 0.0, 0.0);
        closure color d = 0;
        surface = c + d * x;
        r = vector(x, y, 1.0);
    }
    else if (test == 19) {
        r = transform(xform, point(p)) + vector(table[0], table[1], table[2] + table[3]) * count +
            vector(strlen(derived), derived == "camera_second", 0.0);
    }
    else if (test == 20) {
        matrix ms[3] = {matrix(1.0), matrix(2.0), matrix(1.0, 0.0, 0.0, 0.0,
                                                         0.0, 1.0, 0.0, 0.0,
                                                         0.0, 0.0, 1.0, 0.0,
                                                         1.0, 2.0, 3.0, 1.0)};
        ms[1][3][0] = x;
        matrix copy[3];
        copy = ms;
        int i = (x > 0.0) + (y > 0.0);
        r = transform(copy[i], point(p)) + vector(determinant(ms[0]), ms[1][3][0], ms[2][3][2]);
    }
    else if (test == 21) {
        r = vector(u + v + time + dtime, I[0] + Ng[1] + dPdu[0] + dPdv[1],
                   Ps[2] + dPdtime[0]) +
            vector(surfacearea() + isconnected(label) * 0.1,
                   backfacing() + isconnected(throughput) * 0.1,
                   isconnected(scale) + isconnected(position) * 2);
    }
    else if (test == 22) {
        color c = color(x, y, x * y);
        color scaled = c * color(0.5, 2.0, 1.0) + color(y);
        r = vector(Dx(luminance(c)), Dy(c[2]), Dx(scaled[1])) * 16.0 +
            vector(Dy(luminance(scaled)), 0.0, 0.0) * 16.0;
    }
    else if (test == 23) {
        color c = color(0.3 + u01 * 0.5, 0.2 + v01 * 0.6, 0.4 + u01 * v01 * 0.3);
        color hsv = transformc("rgb", "hsv", c);
        color yiq = transformc("rgb", "YIQ", c);
        color xyz = transformc("rgb", "XYZ", c);
        color hsl = transformc("rgb", "hsl", c);
        color back = transformc("hsv", "rgb", hsv);
        color xyy = transformc("rgb", "xyY", c);
        r = vector(Dx(hsv[1]) + Dy(hsv[2]), Dx(yiq[0]) + Dy(yiq[1]), Dx(xyz[2]) + Dy(xyz[0])) * 64.0 +
            vector(Dx(hsl[2]) + Dy(hsl[1]), Dx(back[0]) + Dy(back[1]), Dx(xyy[2]) + Dy(xyy[0])) * 64.0;
    }
    else if (test == 24) {
        r = smoothstep(vector(-1.0), vector(1.0), p) + mix(p, vector(1.0), vector(0.2, 0.4, 0.6)) +
            mix(p, q, 0.25) + select(p, vector(1.0), q) + select(p, q, x > 0.0) +
            step(vector(0.1), p) + clamp(p, vector(-0.5), vector(0.5)) + min(p, q) + max(p, 0.2);
    }
    else if (test == 25) {
        float pair[2] = {0.0, 0.0};
        float four[4] = {0.0, 0.0, 0.0, 0.0};
        float three[3] = {0.0, 0.0, 0.0};
        int number = 0;
        string text = "unchanged";
        float frame = 0.0;
        point resolution = point(0.0);
        color size = color(0.0);
        int ok = getattribute("cam:sensor_size", pair);
        ok += 2 * getattribute("cam:aperture_size", number);
        ok += 4 * getattribute("cam:image_resolution", four);
        ok += 8 * getattribute("cam:focal_distance", text);
        ok += 16 * getattribute("scene:frame", frame);
        ok += 32 * getattribute("object", "cam:aperture_size", frame);
        ok += 64 * getattribute("cam:sensor_size", three);
        ok += 128 * getattribute("cam:image_resolution", resolution);
        ok += 256 * getattribute("cam:aperture_aspect_ratio", size);
        r = vector(pair[0] + pair[1] * 0.1 + three[0] * 0.01 + three[2],
                   ok * 0.01 + (text == "unchanged"),
                   four[0] * 0.01 + four[1] * 0.001 + four[2] + four[3] + frame * 0.1 +
                   resolution[1] * 0.01 + size[2]);
    }
    else if (test == 26) {
        int a = (int)(x * 10.0);
        int b = (int)(y * 3.0);
        int big = (int)(x * 3.0e9);
        float nan = sqrt(-1.0) - sqrt(-1.0) + (x - x) / (y - y);
        r = vector(a / b + a % b * 0.1, abs(a) + min(a, b) * 0.1 + max(a, b) * 0.01,
                   (a && b) + (a || b) * 2 + (!a) * 4 + (-a) * 0.01) +
            vector((a << (b & 3)) * 0.01 + (a >> 1) * 0.001, (big > 0) + (big < 0) * 2,
                   (int)nan + (a == b) + (a != b) * 2 + (a <= b) * 4 + (a >= b) * 8);
    }
    else if (test == 27) {
        float values[6] = {0.1, 0.4, 0.7, 1.1, 1.5, 1.9};
        int where = 0;
        float found = search(x + 1.0, values, where);
        int total = 0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (j > i) {
                    break;
                }
                if ((i + j) % 2) {
                    continue;
                }
                total += i * j + 1;
            }
        }
        int steps = 0;
        while (1) {
            steps++;
            if (steps > 3 + (int)(y + 2.0)) {
                break;
            }
        }
        r = vector(found + where * 0.1, count_below(x + 1.0, values) + total * 0.1, steps);
    }
    else if (test == 28) {
        vector a = (x > 0.0) ? p : -p;
        color c = (y > 0.0) ? color(1.0, 0.0, 0.0) : color(0.0, 1.0, 0.0);
        point in_camera = point("camera", 1.0, 2.0, 3.0);
        normal up = normal("world", 0.0, 1.0, 0.0);
        vector ndc = vector("NDC", x * 0.1, y * 0.1, 0.0);
        r = a + c + (vector)c[1] + vector(in_camera) * 0.1 + vector(up) * 0.1 + ndc * 0.01;
    }
    else if (test == 29) {
        string kind = (x > 0.0) ? "camera" : "glossy";
        r = vector(raytype(kind), raytype("diffuse") + raytype("camera") * 2,
                   (hash(3) & 255) * 0.01 + (hash(kind) & 15) * 0.1) +
            vector((hash("abc") & 255) * 0.01, (hash(name) & 255) * 0.01, (hash("") & 255) * 0.01);
    }
    else if (test == 30) {
        matrix m = matrix(x + 3.0);
        matrix m2 = m * m;
        matrix m3 = -m;
        matrix m4 = m / 2.0;
        matrix m5 = 2.0 / m;
        matrix m6 = m4 * 3.0;
        matrix placed = matrix("camera", 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0,
                               0.0, 0.0, 1.0, 0.0,
                               1.0, 2.0, 3.0, 1.0);
        r = vector(m2[0][0] * 0.1, m3[1][1] + (m == m2) + (m != m3) * 2, m4[2][2] + m5[3][3]) +
            transform(placed, point(p)) * 0.1 + transform(m / m4, vector(p)) * 0.1 +
            transform(m4, normal(p)) * 0.1 + vector(m6[0][0], determinant(m5), 0.0) * 0.1;
    }
    else if (test == 31) {
        float source[4] = {0.0, 0.0, 0.0, 0.0};
        for (int i = 0; i < 4; i++) {
            source[i] = x * i;
        }
        float copy[4];
        copy = source;
        copy[2] += 1.0;
        vector vectors[2];
        vectors[0] = p;
        vectors[1] = vectors[0] * 2.0;
        int ints[3] = {1, 2, 3};
        ints[(int)(x + 2.0) % 3] = 7;
        extra[0] = copy[2];
        extra[1] = extra[0] * 2.0;
        label = "set";
        r = vector(copy[2] + copy[3] * 0.1, vectors[1][1] + ints[0] * 0.1 + ints[2] * 0.01,
                   extra[1] + (label == "set")) + Dx(vectors[1]) * 16.0;
    }
    else if (test == 32) {
        /* Strings from several places reach one function, and positions computed at render
         * time index into them. */
        string s = (x > 0.0) ? "first" : "second";
        int at = (int)(y + 2.0);
        string part = substr(s, at, 2);
        string joined = format("%s-%s/%d", s, other, 5);
        r = vector(pick(s, 1.0, 2.0) + pick(other, 0.1, 0.2) + pick("third", 0.01, 0.02),
                   strlen(part) + getchar(s, at) * 0.01 + (part == "rs") * 10,
                   strlen(joined) + (joined == "first-second/5") * 100 + stoi(substr(joined, -1, 1)));
    }
    else if (test == 33) {
        int indices[4] = {0, 0, 0, 0};
        float distances[4] = {1.0, 1.0, 1.0, 1.0};
        float widths[4] = {2.0, 2.0, 2.0, 2.0};
        int n = pointcloud_search("cloud.ptc", p, 1.0, 4, "index", indices, "distance", distances);
        int ok = pointcloud_get("cloud.ptc", indices, n, "width", widths);
        int written = pointcloud_write("out.ptc", p, "width", x);
        r = vector(n + distances[0], ok + widths[1], written);
    }
    else if (test == 34) {
        point pp = point(p);
        r = vector(area(pp), Dz(x), filterwidth(x)) * 16.0 + calculatenormal(pp) * 256.0 +
            filterwidth(p) * 16.0;
    }
    else if (test == 35) {
        Lens a;
        a.radius = x;
        a.tint = vector(y, 0.5, 0.25);
        a.weights[0] = 0.5;
        a.weights[1] = x * y;
        a.weights[2] = 2.0;
        a.kind = (x > 0.0) ? "convex" : "concave";
        Lens b = a;
        b.weights[1] += 1.0;
        Glass glasses[2];
        glasses[0].ior = 1.5 + x * 0.1;
        glasses[0].tint = a.tint;
        glasses[0].kind = a.kind;
        glasses[1] = glasses[0];
        glasses[1].ior = 3.0;
        glasses[1].kind = "flat";
        int i = y > 0.0;
        r = vector(glasses[i].ior + glasses[i].tint[0], b.weights[1] + a.weights[1] * 0.1 + b.weights[2],
                   strlen(glasses[i].kind) + (a.kind == "convex") * 10 + (b.kind == a.kind) * 20);
    }
    else if (test == 36) {
        /* Conversions of numbers that are not finite or do not fit. */
        /* Not a constant: the OSL runtime cannot load bytecode with an infinite constant. */
        float zero = x - x;
        float inf = 1.0e30 * (1.0e30 + zero);
        int huge = (int)(1.0e12 * (x + 3.0));
        int tiny = (int)(-1.0e12 * (x + 3.0));
        r = vector((huge > 1000) + (tiny < -1000) * 2 + ((int)inf > 0) * 4,
                   5 / (int)zero + 5 % (int)zero + 1.0 / zero + fmod(x, zero),
                   (7 >> 1) + (1 << 4) * 0.1 + (-9 / 2) * 0.01 + (-9 % 4) * 0.001);
    }
    else if (test == 37) {
        /* The same string arrives from a parameter, a constant and a computation. */
        string a = name;
        string b = "camera";
        string c = concat("cam", "era");
        string d = (x > 0.0) ? a : other;
        r = vector((a == b) + (b == c) * 2 + (d == c) * 4 + (d != other) * 8,
                   strlen(d) + (derived == concat(a, "_second")),
                   endswith(derived, other) + startswith(derived, d) * 2);
    }
    else if (test == 38) {
        r = vector(wavelength_color(500.0 + x * 50.0)) +
            vector(Dx(x * x + y), Dy(x * y), Dx(dot(p, q))) * 16.0;
    }
    else if (test == 39) {
        /* Derivatives of everything that reaches the direction of the ray. */
        float knots[6] = {0.0, 0.1, 0.5, 0.2, 0.9, 1.0};
        float bend = spline("catmull-rom", u01, knots) + noise("simplex", p) * 0.1 +
                     pnoise(p * 2.0, point(3.0)) * 0.1;
        r = vector(Dx(bend), Dy(bend), Dx(bend * bend)) * 16.0;
    }
    else if (test == 40) {
        /* A dictionary: an XML document queried with XPath. */
        string xml = "<lens name=\"cooke\"><surface radius=\"22.5\" ior=\"1.62\" id=\"3\">first</surface><surface radius=\"-435.0, 2.5\" ior=\"1.0\" id=\"4\"/><stop size=\"1 2 3\"/></lens>";
        int surface = dict_find(xml, "//surface");
        float radius = 0.0;
        point pair = point(0.0);
        int id = 0;
        string lens_name = "none";
        vector size = vector(0.0);
        int ok = dict_value(surface, "radius", radius);
        int second = dict_next(surface);
        ok += 2 * dict_value(second, "radius", pair);
        ok += 4 * dict_value(second, "id", id);
        int none = dict_next(second);
        int lens = dict_find(xml, "/lens");
        ok += 8 * dict_value(lens, "name", lens_name);
        int stop = dict_find(lens, "stop");
        ok += 16 * dict_value(stop, "size", size);
        ok += 32 * dict_value(stop, "missing", id);
        int nothing = dict_find(xml, "//nothing");
        int invalid = dict_find("<a><b></a>", "//b");
        int count = 0;
        for (int n = dict_find(xml, "//surface"); n; n = dict_next(n)) {
            count++;
        }
        /* The node is chosen at render time. */
        float ior = 0.0;
        dict_value((x > 0.0) ? surface : second, "ior", ior);
        r = vector(radius + pair[0] * 0.001 + pair[1] + ior, ok + id * 0.01 + (none != 0) * 0.001,
                   strlen(lens_name) + size[0] + size[1] * 0.1 + size[2] * 0.01 + count * 10 +
                   (nothing != 0) + (invalid < 0) * 3);
    }
    else if (test == 41) {
        /* Gabor noise, filtered with the derivatives of the position. */
        r = vector(noise("gabor", p * 3.0), noise("gabor", x * 3.0, y * 3.0),
                   noise("gabor", x * 5.0));
    }
    else if (test == 42) {
        r = vector(noise("gabor", p * 4.0, "anisotropic", 1, "direction", vector(1.0, 0.5, 0.0),
                         "bandwidth", 2.0, "impulses", 8.0, "do_filter", 0),
                   noise("gabor", p * 4.0, "anisotropic", 2, "direction", vector(0.5, 2.0, 0.0),
                         "do_filter", 0),
                   noise("gabor", p * 4.0, "do_filter", 0, "impulses", 30.0, "bandwidth", 0.5));
    }
    else if (test == 43) {
        vector a = noise("gabor", p * 3.0);
        vector b = noise("gabor", x * 3.0, y * 2.0, "do_filter", 0);
        r = a + b * 0.5 + vector(noise("gabor", p * 2.0, x), 0.0, noise("gabor", x * 4.0, "do_filter", 0));
    }
    else if (test == 44) {
        vector a = pnoise("gabor", p * 3.0, point(4.0, 3.0, 2.0), "do_filter", 0);
        r = vector(pnoise("gabor", p * 3.0, point(4.0, 4.0, 4.0)), pnoise("gabor", x * 3.0, 2.0),
                   pnoise("gabor", x * 3.0, y * 3.0, 3.0, 2.0, "anisotropic", 1)) + a * 0.5;
    }
    else if (test == 45) {
        r = vector(Dx(noise("gabor", p * 3.0)), Dy(noise("gabor", x * 3.0, y * 3.0, "do_filter", 0)),
                   Dx(noise("gabor", p * 2.0, "anisotropic", 1, "do_filter", 0))) * 4.0;
    }
    else if (test == 46) {
        /* A position that changes fast from pixel to pixel: the filter removes detail. */
        r = vector(noise("gabor", p * 40.0), noise("gabor", p * 150.0, "bandwidth", 3.0),
                   noise("gabor", x * 60.0, y * 90.0, "anisotropic", 1, "direction",
                         vector(0.0, 1.0, 0.0)));
    }

    throughput = color(r * 0.25 + 4.0);
}
"""

# Image lookups. The probe image has as many pixels as the render, so that lookups at the pixel
# centers return pixel values without interpolation.
TEXTURE_PROBE_SHADER = r"""
shader texture_probe(int test = 0,
                     int dynamic_names = 0,
                     string file = "",
                     string second = "",
                     string missing = "",
                     output point position = point(0.0),
                     output vector direction = vector(0.0, 0.0, 1.0),
                     output color throughput = color(0.0))
{
    point raster = camera_shader_raster_position();
    float s = raster[0];
    float t = raster[1];
    vector r = vector(0.0);

    if (test == 0) {
        r = vector(texture(file, s, t));
    }
    else if (test == 1) {
        float alpha = -1.0;
        float gray = texture(file, s, t, "alpha", alpha);
        float alpha3 = -1.0;
        color c = texture(file, s, t, "alpha", alpha3, "interp", "closest", "wrap", "clamp",
                          "blur", 0.0);
        r = vector(gray, alpha, alpha3 + c[1]);
    }
    else if (test == 2) {
        /* Derivatives given by the shader. */
        r = vector(texture(file, s, t, 1.0 / 64.0, 0.0, 0.0, 1.0 / 64.0));
    }
    else if (test == 3) {
        float alpha = -1.0;
        string error = "unset";
        color c = texture(missing, s, t, "missingcolor", color(0.25, 0.5, 0.75), "missingalpha", 0.125,
                          "alpha", alpha, "errormessage", error);
        color d = texture(missing, s, t);
        r = vector(c) + vector(d) * 0.1 + vector(alpha, strlen(error), 0.0);
    }
    else if (test == 4) {
        int resolution[2] = {0, 0};
        float size[2] = {0.0, 0.0};
        vector size3 = vector(0.0);
        int channels = 0;
        int exists = -1;
        color average = color(0.0);
        float wrong = 3.0;
        int ok = gettextureinfo(file, "resolution", resolution);
        ok += 2 * gettextureinfo(file, "resolution", size);
        ok += 4 * gettextureinfo(file, "resolution", size3);
        ok += 8 * gettextureinfo(file, "channels", channels);
        ok += 16 * gettextureinfo(file, "exists", exists);
        ok += 32 * gettextureinfo(file, "averagecolor", average);
        ok += 64 * gettextureinfo(file, "channels", wrong);
        ok += 128 * gettextureinfo(file, "unknown", channels);
        ok += 256 * gettextureinfo(file, s, t, "resolution", resolution);
        r = vector(resolution[0] * 0.01 + resolution[1] * 0.0001 + size[0] * 0.1 + size3[2],
                   ok * 0.01 + channels + exists * 0.1, wrong);
    }
    else if (test == 5) {
        /* Several lookups for one ray, and lookups in a loop. */
        color sum = texture(file, s, t) + texture(second, s, t) * 0.5;
        for (int i = 0; i < 5; i++) {
            sum += texture(file, 1.0 - s, (i + 0.5) / 5.0) * (0.125 * (i + 1));
            if (sum[0] > 2.0) {
                break;
            }
        }
        r = vector(sum);
    }
    else if (test == 6) {
        vector dir = normalize(vector(s - 0.5, 0.31, t - 0.5));
        r = vector(environment(file, dir));
    }
    else if (test == 7) {
        /* The image is chosen at render time. The OSL runtime only looks up images with a
         * constant name, so the reference comes from looking up both. */
        float alpha = 0.0;
        color c = color(0.0);
        if (dynamic_names) {
            string which = (s > 0.5) ? file : second;
            c = texture(which, s, t, "alpha", alpha);
        }
        else {
            float alpha_file = 0.0;
            float alpha_second = 0.0;
            color c_file = texture(file, s, t, "alpha", alpha_file);
            color c_second = texture(second, s, t, "alpha", alpha_second);
            c = (s > 0.5) ? c_file : c_second;
            alpha = (s > 0.5) ? alpha_file : alpha_second;
        }
        r = vector(c) * alpha;
    }
    else if (test == 8) {
        /* A missing image: information and the default color. */
        int exists = 5;
        int resolution[2] = {7, 7};
        int ok = gettextureinfo(missing, "exists", exists);
        ok += 2 * gettextureinfo(missing, "resolution", resolution);
        r = vector(exists + ok * 0.1, resolution[0], 0.0) + vector(texture(missing, s, t));
    }

    throughput = color(r * 0.25 + 4.0);
}
"""

# A camera that uses images the way lens shaders do: one image distorts the picture and
# another shapes the aperture.
TEXTURE_LENS_SHADER = r"""
shader texture_lens(float focal_length = 50.0,
                    float distortion = 0.02,
                    string distortion_map = "",
                    string aperture_image = "",
                    output point position = point(0.0),
                    output vector direction = vector(0.0, 0.0, 1.0),
                    output color throughput = color(1.0))
{
    vector sensor_size;
    getattribute("cam:sensor_size", sensor_size);
    float focal_distance;
    getattribute("cam:focal_distance", focal_distance);
    float aperture_size;
    getattribute("cam:aperture_size", aperture_size);

    point raster = camera_shader_raster_position();
    color offset = texture(distortion_map, raster[0], raster[1]);
    point Pcam = raster - point(0.5) + (point(offset[0], offset[1], 0.0) - point(0.5, 0.5, 0.0)) *
                 distortion;
    Pcam *= sensor_size / focal_length;
    direction = normalize(vector(Pcam[0], Pcam[1], 1.0));

    /* The aperture image is the transmission of each point of the lens. */
    vector lens = camera_shader_random_sample();
    throughput = texture(aperture_image, lens[0], lens[1]) * 2.0;
    point Pfocus = direction * focal_distance / direction[2];
    position = point((lens[0] - 0.5) * 4.0 * aperture_size, (lens[1] - 0.5) * 4.0 * aperture_size,
                     0.0);
    direction = normalize(Pfocus - position);
}
"""

# Explicit ray differentials: the shader outputs dPdx, dPdy, dDdx and dDdy itself.
EXPLICIT_DERIVATIVES_SHADER = r"""
shader explicit_derivatives(float focal_length = 35.0,
                            output point position = point(0.0),
                            output vector direction = vector(0.0, 0.0, 1.0),
                            output color throughput = color(1.0),
                            output vector dPdx = vector(0.0),
                            output vector dPdy = vector(0.0),
                            output vector dDdx = vector(0.0),
                            output vector dDdy = vector(0.0))
{
    vector sensor_size;
    getattribute("cam:sensor_size", sensor_size);
    vector resolution;
    getattribute("cam:image_resolution", resolution);

    point Pcam = camera_shader_raster_position() - point(0.5);
    Pcam *= sensor_size / focal_length;
    direction = normalize(vector(Pcam[0], Pcam[1], 1.0));
    dDdx = vector(sensor_size[0] / (focal_length * resolution[0]), 0.0, 0.0);
    dDdy = vector(0.0, sensor_size[1] / (focal_length * resolution[1]), 0.0);
}
"""

# Traces rays from the sensor through the spherical surfaces of a Cooke triplet, a classic
# three element photographic lens. Distances are in millimeters, the optical axis is z, and
# light travels from the scene at negative z to the sensor at positive z.
LENS_STACK_SHADER = r"""
#define NUM_SURFACES 6

/* Distance along the ray to the sphere with the center on the optical axis. `t` is the
 * intersection nearest to the vertex of the surface. */
int intersect_surface(point origin, vector dir, float vertex_z, float radius, output float t)
{
    point center = point(0.0, 0.0, vertex_z + radius);
    vector oc = origin - center;
    float b = dot(oc, dir);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - c;
    if (discriminant < 0.0) {
        return 0;
    }
    float root = sqrt(discriminant);
    /* Rays travel towards negative z. The vertex is on the far side of a sphere with its
     * center behind the vertex, and on the near side otherwise. */
    t = (radius > 0.0) ? -b + root : -b - root;
    return t > 0.0;
}

shader lens_stack(float fstop = 4.0,
                  float sensor_shift = 0.0,
                  float dispersion = 0.0,
                  int flip = 1 [[ string widget = "checkBox" ]],
                  output point position = point(0.0),
                  output vector direction = vector(0.0, 0.0, 1.0),
                  output color throughput = color(0.0))
{
    /* Radius of curvature, distance to the next surface, refractive index and semi-diameter of
     * each surface, from the scene to the sensor. */
    float radius[NUM_SURFACES] = {22.01359, -435.76044, -22.21328, 20.29192, 79.68360, -18.39533};
    float thickness[NUM_SURFACES] = {3.25896, 6.00755, 0.99997, 4.75041, 2.95208, 42.20778};
    float ior[NUM_SURFACES] = {1.62041, 1.0, 1.60342, 1.0, 1.62041, 1.0};
    float abbe[NUM_SURFACES] = {60.3, 0.0, 38.0, 0.0, 60.3, 0.0};
    float semi_diameter[NUM_SURFACES] = {9.5, 9.5, 5.0, 5.0, 7.5, 7.5};
    float focal_length = 50.0;

    /* Position of each surface on the axis. */
    float vertex[NUM_SURFACES];
    float z = 0.0;
    for (int i = 0; i < NUM_SURFACES; i++) {
        vertex[i] = z;
        z += thickness[i];
    }

    /* Move the sensor to focus at the focal distance of the camera. */
    float focus_distance = 0.0;
    getattribute("cam:focal_distance", focus_distance);
    focus_distance *= 1000.0;
    float sensor_z = z + sensor_shift;
    if (focus_distance > focal_length * 2.0) {
        sensor_z += focal_length * focal_length / (focus_distance - focal_length);
    }

    vector sensor_size;
    getattribute("cam:sensor_size", sensor_size);
    point raster = camera_shader_raster_position() - point(0.5);
    /* A lens projects the image upside down. */
    float orientation = flip ? -1.0 : 1.0;
    point origin = point(raster[0] * sensor_size[0] * orientation,
                         raster[1] * sensor_size[1] * orientation, sensor_z);

    /* Aim at a point of the rear element. */
    vector lens_sample = camera_shader_random_sample();
    float aim_radius = semi_diameter[NUM_SURFACES - 1] * sqrt(lens_sample[0]);
    float aim_angle = M_2PI * lens_sample[1];
    point aim = point(aim_radius * cos(aim_angle), aim_radius * sin(aim_angle),
                      vertex[NUM_SURFACES - 1]);
    vector dir = normalize(aim - origin);

    /* One wavelength per ray. */
    float wavelength = 587.6;
    color weight = color(1.0);
    if (dispersion > 0.0) {
        wavelength = 420.0 + 240.0 * hashnoise(origin, lens_sample[0]);
        weight = wavelength_color(wavelength) * 2.6;
    }

    /* The aperture stop is in the gap behind the middle element. The lens is about f/5 wide
     * open. */
    float stop_z = vertex[3] + thickness[3] * 0.5;
    float stop_radius = min(focal_length / (2.0 * max(fstop, 0.5)), 4.9);

    for (int i = NUM_SURFACES - 1; i >= 0; i--) {
        float t = 0.0;
        if (!intersect_surface(origin, dir, vertex[i], radius[i], t)) {
            return;
        }
        point hit = origin + dir * t;
        if (hypot(hit[0], hit[1]) > semi_diameter[i]) {
            /* Blocked by the lens barrel. */
            return;
        }

        if (i == 3) {
            /* The ray crossed the stop on its way to this surface. */
            float stop_t = (stop_z - origin[2]) / dir[2];
            point stop_hit = origin + dir * stop_t;
            if (hypot(stop_hit[0], stop_hit[1]) > stop_radius) {
                return;
            }
        }

        /* Refraction, from the medium behind the surface to the one in front of it. */
        float n_behind = ior[i];
        float n_front = (i > 0) ? ior[i - 1] : 1.0;
        if (dispersion > 0.0) {
            /* Indices are given for 587.6 nm. The Abbe number gives their change with the
             * wavelength, approximated as linear in 1 / wavelength^2. */
            float lambda_term = (1.0 / (wavelength * wavelength) - 1.0 / (587.6 * 587.6)) /
                                (1.0 / (486.1 * 486.1) - 1.0 / (656.3 * 656.3));
            if (abbe[i] > 0.0) {
                n_behind += dispersion * (n_behind - 1.0) / abbe[i] * lambda_term;
            }
            if (i > 0 && abbe[i - 1] > 0.0) {
                n_front += dispersion * (n_front - 1.0) / abbe[i - 1] * lambda_term;
            }
        }

        vector n = normalize(hit - point(0.0, 0.0, vertex[i] + radius[i]));
        if (dot(n, dir) > 0.0) {
            n = -n;
        }
        vector refracted = refract(dir, n, n_behind / n_front);
        if (length(refracted) == 0.0) {
            /* Total internal reflection. */
            return;
        }
        origin = hit;
        dir = normalize(refracted);
    }

    /* To camera space: meters, looking along positive z, with the front vertex at the origin. */
    position = point(origin[0], origin[1], -origin[2]) * 0.001;
    direction = vector(dir[0], dir[1], -dir[2]);
    throughput = weight;
}
"""


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="CPU", choices=("METAL", "CPU", "METAL_CPU"))
    parser.add_argument("--session", action="store_true")
    parser.add_argument("--scenes", default="all")
    parser.add_argument("--resolution", type=int, default=192)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--threads", type=int, default=0)
    return parser.parse_args(argv)


def new_scene(args, resolution=None, samples=None):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = resolution or args.resolution
    scene.render.resolution_y = resolution or (args.resolution * 3 // 4)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    if args.threads:
        scene.render.threads_mode = "FIXED"
        scene.render.threads = args.threads
    cycles = scene.cycles
    cycles.device = "CPU" if args.device == "CPU" else "GPU"
    cycles.samples = samples or args.samples
    cycles.use_adaptive_sampling = False
    cycles.use_denoising = False
    cycles.seed = 7
    cycles.max_bounces = 6
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    scene.world = world
    bpy.ops.object.camera_add(location=(0.0, -6.0, 1.6), rotation=(math.radians(84), 0.0, 0.0))
    scene.camera = bpy.context.object
    return scene


def set_camera_shader(scene, source, params=None):
    """Make the scene camera a custom camera running the OSL `source`."""
    from cycles import osl

    cam = scene.camera.data
    text = bpy.data.texts.new("camera_shader.osl")
    text.write(source)
    cam.type = 'CUSTOM'
    cam.custom_mode = 'INTERNAL'
    cam.custom_shader = text

    messages = []
    osl.update_custom_camera_shader(cam, lambda kind, message: messages.append((kind, message)))
    for kind, message in messages:
        if 'ERROR' in kind:
            raise RuntimeError(f"Camera shader failed to compile: {message}")
    if not cam.custom_bytecode:
        raise RuntimeError("Camera shader has no bytecode")
    for name, value in (params or {}).items():
        cam.cycles_custom[name] = value
    return cam


def template_source(name):
    path = Path(bpy.utils.system_resource('SCRIPTS')) / "templates_osl" / name
    return path.read_text()


def material(name, color, roughness=0.5, metallic=0.0, emission=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if emission:
        bsdf.inputs["Emission Color"].default_value = (*color, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission
    return mat


def build_set(scene):
    """Objects from near to far, a checker floor and small lights that show the bokeh."""
    world_nodes = scene.world.node_tree.nodes
    world_nodes["Background"].inputs["Color"].default_value = (0.35, 0.45, 0.6, 1.0)
    world_nodes["Background"].inputs["Strength"].default_value = 0.6

    bpy.ops.mesh.primitive_plane_add(size=60.0)
    floor = bpy.context.object
    floor_mat = bpy.data.materials.new("Floor")
    floor_mat.use_nodes = True
    nodes = floor_mat.node_tree.nodes
    checker = nodes.new("ShaderNodeTexChecker")
    checker.inputs["Scale"].default_value = 60.0
    checker.inputs["Color1"].default_value = (0.8, 0.8, 0.8, 1.0)
    checker.inputs["Color2"].default_value = (0.15, 0.15, 0.18, 1.0)
    floor_mat.node_tree.links.new(checker.outputs["Color"],
                                  nodes["Principled BSDF"].inputs["Base Color"])
    floor.data.materials.append(floor_mat)

    colors = [(0.8, 0.15, 0.1), (0.1, 0.6, 0.2), (0.15, 0.25, 0.8), (0.8, 0.7, 0.1),
              (0.7, 0.2, 0.7)]
    for i, color in enumerate(colors):
        distance = -3.5 + i * 2.2
        side = -1.6 + i * 0.8
        if i % 2:
            bpy.ops.mesh.primitive_cube_add(size=0.9, location=(side, distance, 0.45))
        else:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(side, distance, 0.5),
                                                 segments=48, ring_count=24)
            bpy.ops.object.shade_smooth()
        bpy.context.object.data.materials.append(
            material(f"Object{i}", color, roughness=0.3, metallic=(i == 2)))

    # Small bright lights far away.
    for i in range(7):
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.06, location=(-3.0 + i, 9.0 + (i % 3) * 2.0, 1.2 + (i % 2) * 0.8),
            segments=16, ring_count=8)
        hue = [(1.0, 0.6, 0.3), (0.4, 0.7, 1.0), (1.0, 1.0, 1.0)][i % 3]
        bpy.context.object.data.materials.append(material(f"Light{i}", hue, emission=60.0))

    bpy.ops.object.light_add(type='SUN', location=(0.0, 0.0, 10.0),
                             rotation=(math.radians(50), math.radians(10), math.radians(30)))
    bpy.context.object.data.energy = 3.0
    bpy.context.object.data.angle = math.radians(2.0)


def probe_scene(test, source=None, **params):
    def build(args):
        # One sample in the pixel center, and a camera that is not at the origin.
        scene = new_scene(args, resolution=64, samples=1)
        scene.render.resolution_y = 64
        scene.cycles.pixel_filter_type = 'BOX'
        scene.cycles.filter_width = 0.01
        scene.camera.location = (0.5, -2.0, 1.0)
        scene.camera.rotation_euler = (math.radians(70), math.radians(5), math.radians(20))
        scene.camera.data.dof.use_dof = True
        scene.camera.data.dof.focus_distance = 3.0
        scene.camera.data.dof.aperture_fstop = 2.0
        scene.camera.data.dof.aperture_ratio = 1.5
        scene.camera.data.dof.aperture_blades = 5
        scene.camera.data.sensor_width = 24.0
        nodes = scene.world.node_tree.nodes
        nodes["Background"].inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        nodes["Background"].inputs["Strength"].default_value = 1.0
        set_camera_shader(scene, source or PROBE_SHADER, {"test": test, **params})
    return build


def write_image(path, size, function, channels=4):
    """Save an image of `function(x, y)` colors, with x and y from 0 to 1 at the pixel centers."""
    import numpy as np

    y, x = np.meshgrid((np.arange(size) + 0.5) / size, (np.arange(size) + 0.5) / size,
                       indexing='ij')
    pixels = np.ones((size, size, 4), dtype=np.float32)
    values = function(x, y)
    for channel, value in enumerate(values):
        pixels[:, :, channel] = value
    # Written directly, so that no color management changes the values. Files start at the top.
    import OpenImageIO as oiio
    output = oiio.ImageOutput.create(str(path))
    output.open(str(path), oiio.ImageSpec(size, size, 4, "float"))
    output.write_image(np.ascontiguousarray(pixels[::-1]))
    output.close()
    return str(path)


def texture_files(args):
    """The images of the texture scenes, written next to the renders."""
    import numpy as np

    directory = args.output / "textures"
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "file": write_image(directory / "probe_a.exr", 64,
                            lambda x, y: (x, y * 0.5 + 0.25, x * y, 0.25 + 0.5 * x)),
        "second": write_image(directory / "probe_b.exr", 64,
                              lambda x, y: (1.0 - y, 0.5 * x * x, 0.75 - 0.5 * y, 1.0 - 0.5 * y)),
        "missing": str(directory / "does_not_exist.exr"),
        "distortion_map": write_image(
            directory / "distortion.exr", 256,
            lambda x, y: (0.5 + 0.5 * np.sin(y * 9.0), 0.5 + 0.5 * np.sin(x * 7.0), 0.0 * x, 1.0)),
        "aperture_image": write_image(
            directory / "aperture.exr", 128,
            lambda x, y: (np.where(np.hypot(x - 0.5, y - 0.5) < 0.45, 1.0, 0.0),
                          np.where(np.hypot(x - 0.4, y - 0.5) < 0.35, 1.0, 0.0),
                          np.where(np.abs(x - 0.5) + np.abs(y - 0.5) < 0.4, 1.0, 0.0), 1.0)),
    }


def texture_probe_scene(test):
    def build(args):
        files = texture_files(args)
        probe_scene(test, TEXTURE_PROBE_SHADER)(args)
        cam = bpy.context.scene.camera.data
        for name in ("file", "second", "missing"):
            cam.cycles_custom[name] = files[name]
        cam.cycles_custom["dynamic_names"] = (args.device == "METAL")
    return build


def scene_texture_lens(args):
    files = texture_files(args)
    scene = new_scene(args)
    build_set(scene)
    scene.camera.data.dof.use_dof = True
    scene.camera.data.dof.focus_distance = 5.0
    scene.camera.data.dof.aperture_fstop = 1.4
    set_camera_shader(scene, TEXTURE_LENS_SHADER,
                      {"focal_length": 40.0, "distortion_map": files["distortion_map"],
                       "aperture_image": files["aperture_image"]})


def scene_texture_lens_cache(args):
    # With the texture cache images are loaded in tiles while rendering: a lookup whose tile is
    # not loaded yet makes the kernel generate the camera ray again later.
    scene_texture_lens(args)
    scene = bpy.context.scene
    cache = args.output / "textures" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.texture_cache_directory = str(cache)
    scene.render.use_texture_cache = True
    scene.render.use_auto_generate_texture_cache = True


def scene_explicit_derivatives(args):
    scene = new_scene(args)
    build_set(scene)
    set_camera_shader(scene, EXPLICIT_DERIVATIVES_SHADER)


def scene_template_basic(args):
    scene = new_scene(args)
    build_set(scene)
    set_camera_shader(scene, template_source("basic_camera.osl"), {"focal_length": 40.0})


def scene_template_advanced_distortion(args):
    scene = new_scene(args)
    build_set(scene)
    set_camera_shader(scene, template_source("advanced_camera.osl"),
                      {"focal_length": 30.0, "do_distortion": True, "distortion_k1": -0.25,
                       "distortion_k2": 0.05})


def scene_template_advanced_swirl_dof(args):
    scene = new_scene(args)
    build_set(scene)
    scene.camera.data.dof.use_dof = True
    scene.camera.data.dof.focus_distance = 5.0
    scene.camera.data.dof.aperture_fstop = 1.4
    set_camera_shader(scene, template_source("advanced_camera.osl"),
                      {"focal_length": 50.0, "do_swirl": True, "do_dof": True,
                       "swirl_scale": 40.0, "swirl_amplitude": 0.02, "swirl_w": 1.5})


def scene_template_cubemap(args):
    scene = new_scene(args)
    build_set(scene)
    set_camera_shader(scene, template_source("cubemap_camera.osl"))


def lens_stack_scene(**params):
    def build(args):
        scene = new_scene(args)
        build_set(scene)
        scene.camera.data.dof.use_dof = True
        scene.camera.data.dof.focus_distance = 6.0
        scene.camera.data.sensor_width = 36.0
        set_camera_shader(scene, LENS_STACK_SHADER, params)
    return build


def scene_lens_stack_motion(args):
    # Camera and object motion blur with a custom camera.
    lens_stack_scene(fstop=8.0)(args)
    scene = bpy.context.scene
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 1.0
    camera = scene.camera
    camera.keyframe_insert("location", frame=1)
    camera.location.x += 0.4
    camera.keyframe_insert("location", frame=2)
    scene.frame_set(1)


def scene_lens_stack_passes(args):
    # Adaptive sampling, denoising data and vector passes.
    lens_stack_scene(fstop=5.6)(args)
    scene = bpy.context.scene
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.02
    layer = scene.view_layers[0]
    layer.use_pass_z = True
    layer.use_pass_normal = True
    layer.use_pass_vector = True
    layer.cycles.denoising_store_passes = True


def scene_no_shader(args):
    # A custom camera without shader renders black on every device.
    scene = new_scene(args, samples=4)
    build_set(scene)
    scene.camera.data.type = 'CUSTOM'


SCENES = {}
for _test in range(50):
    SCENES[f"probe_{_test:02d}"] = probe_scene(_test)
# The same shader with other parameter values: only the parameter array changes.
SCENES["probe_params_a"] = probe_scene(33, scale=2.5, tint=(0.25, 0.5, 2.0), flag=True)
SCENES["probe_params_b"] = probe_scene(33, scale=0.5, tint=(3.0, 1.0, 0.125), space="world")
for _test in range(47):
    SCENES[f"probe2_{_test:02d}"] = probe_scene(_test, PROBE2_SHADER)
# Other values of the parameters, including the strings that select code paths.
SCENES["probe2_params_a"] = probe_scene(19, PROBE2_SHADER, name="world", other="first", count=5,
                                        xform=(2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0,
                                               0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 1.0))
SCENES["probe2_params_b"] = probe_scene(32, PROBE2_SHADER, name="NDC", other="first")
SCENES["probe2_params_c"] = probe_scene(37, PROBE2_SHADER, name="world", other="camera")
for _test in range(9):
    SCENES[f"probe_texture_{_test:02d}"] = texture_probe_scene(_test)
SCENES.update({
    "texture_lens": scene_texture_lens,
    "texture_lens_cache": scene_texture_lens_cache,
    "explicit_derivatives": scene_explicit_derivatives,
    "template_basic": scene_template_basic,
    "template_advanced_distortion": scene_template_advanced_distortion,
    "template_advanced_swirl_dof": scene_template_advanced_swirl_dof,
    "template_cubemap": scene_template_cubemap,
    "lens_stack_f8": lens_stack_scene(fstop=8.0),
    "lens_stack_open": lens_stack_scene(fstop=2.0, sensor_shift=0.3),
    "lens_stack_dispersion": lens_stack_scene(fstop=5.6, dispersion=3.0),
    "lens_stack_unflipped": lens_stack_scene(fstop=11.0, flip=False),
    "lens_stack_motion": scene_lens_stack_motion,
    "lens_stack_passes": scene_lens_stack_passes,
    "no_shader": scene_no_shader,
})


# The OSL runtime of the CPU device crashes when it folds gettextureinfo() with constant
# arguments in a camera shader, so there is no reference for these scenes. They are checked
# against the values that the shader must compute: the probe image has 64 x 64 pixels and four
# channels, and a missing image has no information and the color for missing images.
NO_REFERENCE = {
    "probe_texture_04": (0.64 + 0.0064 + 6.4 + 1.0, 3.19 + 4.0 + 0.1, 3.0),
    "probe_texture_08": (5.0 + 1.0, 7.0, 1.0),
}


def check_expected(path, expected):
    import numpy as np

    image = bpy.data.images.load(str(path), check_existing=False)
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(-1, image.channels)[:, :3]
    bpy.data.images.remove(image)
    values = (pixels.astype(np.float64) - 4.0) * 4.0
    difference = np.abs(values - np.array(expected)).max()
    if difference > 2e-4:
        raise RuntimeError(f"{path.name}: expected {expected}, rendered {values[0].tolist()} "
                           f"(largest difference {difference})")


def enable_device(args):
    if args.device == "CPU":
        return
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == "METAL" or (args.device == "METAL_CPU" and device.type == "CPU")


# Strings only exist while a shader is translated for Metal. A string that is made from a number
# at render time and then looked at is the one thing that cannot be translated.
UNSUPPORTED_SHADER = r"""
shader unsupported(output point position = point(0.0),
                   output vector direction = vector(0.0, 0.0, 1.0),
                   output color throughput = color(1.0))
{
    point Pcam = camera_shader_raster_position() - point(0.5);
    string column = format("%d", (int)(Pcam[0] * 8.0));
    if (column == "2" || endswith(column, "3")) {
        throughput = color(1.0, 0.5, 0.25);
    }
    direction = normalize(vector(Pcam[0], Pcam[1], 1.0));
}
"""


def run_session(args):
    """Edits that a render session has to follow, each rendered to its own image."""
    scene = new_scene(args)
    build_set(scene)
    enable_device(args)
    scene.render.use_persistent_data = True
    camera = scene.camera
    camera.data.dof.use_dof = True
    camera.data.dof.focus_distance = 6.0
    cam = set_camera_shader(scene, LENS_STACK_SHADER, {"fstop": 11.0})
    report = {}

    def render(name, compared=True):
        directory = args.output if compared else args.output / "not_compared"
        scene.render.filepath = str(directory / f"{name}.exr")
        start = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        report[name] = {"render_seconds": time.perf_counter() - start}
        print(f"SESSION {name} {json.dumps(report[name])}", flush=True)
        return scene.render.filepath

    def image_maximum(path):
        # Of the color channels: alpha is opaque.
        image = bpy.data.images.load(path, check_existing=False)
        pixels = image.pixels[:]
        bpy.data.images.remove(image)
        return max(max(pixels[channel::4]) for channel in range(3))

    render("step_01_lens_f11")
    # Numeric parameters change without a new translation.
    cam.cycles_custom["fstop"] = 5.6
    cam.update_tag()
    render("step_02_lens_f5_6")
    cam.cycles_custom["dispersion"] = 3.0
    cam.cycles_custom["sensor_shift"] = 0.4
    cam.update_tag()
    render("step_03_lens_dispersion")
    # Another shader.
    set_camera_shader(scene, template_source("advanced_camera.osl"),
                      {"focal_length": 30.0, "do_distortion": True, "distortion_k1": -0.25})
    render("step_04_template")
    # A shader that Metal cannot translate renders black there, with a warning. The CPU renders
    # it, so the devices are not compared.
    set_camera_shader(scene, UNSUPPORTED_SHADER)
    path = render("step_05_unsupported", compared=False)
    if args.device == "METAL" and image_maximum(path) > 1e-6:
        raise RuntimeError("An untranslated camera shader must render black")
    # The session recovers with a supported shader.
    set_camera_shader(scene, LENS_STACK_SHADER, {"fstop": 11.0})
    render("step_06_lens_f11_again")
    # A regular camera, then the custom camera again.
    cam.type = 'PERSP'
    render("step_07_perspective")
    cam.type = 'CUSTOM'
    render("step_08_lens_f11_third")
    # Custom camera without shader.
    cam.custom_shader = None
    cam.custom_bytecode = ""
    cam.custom_bytecode_hash = ""
    cam.update_tag()
    render("step_09_no_shader")
    # Animated parameter and camera.
    set_camera_shader(scene, LENS_STACK_SHADER, {"fstop": 11.0})
    for frame, fstop in ((1, 16.0), (2, 8.0), (3, 5.6)):
        cam.cycles_custom["fstop"] = fstop
        cam.keyframe_insert('cycles_custom["fstop"]', frame=frame)
        camera.location.x = (frame - 1) * 0.3
        camera.keyframe_insert("location", frame=frame)
    for frame in (1, 2, 3):
        scene.frame_set(frame)
        render(f"step_1{frame}_frame_{frame}")

    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.session:
        run_session(args)
        return
    if args.scenes == "all":
        names = list(SCENES)
    else:
        names = [name for name in SCENES
                 if any(name == pattern or (pattern.endswith("*") and name.startswith(pattern[:-1]))
                        for pattern in args.scenes.split(","))]
    report = {}
    for name in names:
        if name in NO_REFERENCE and args.device != "METAL":
            print(f"SCENE {name} skipped: the OSL runtime of the CPU device cannot run it",
                  flush=True)
            continue
        SCENES[name](args)
        enable_device(args)
        scene = bpy.context.scene
        directory = args.output / "not_compared" if name in NO_REFERENCE else args.output
        scene.render.filepath = str(directory / f"{name}.exr")
        start = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        seconds = time.perf_counter() - start
        if name in NO_REFERENCE:
            check_expected(Path(scene.render.filepath), NO_REFERENCE[name])
        report[name] = {"render_seconds": seconds}
        print(f"SCENE {name} {json.dumps(report[name])}", flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


main()
