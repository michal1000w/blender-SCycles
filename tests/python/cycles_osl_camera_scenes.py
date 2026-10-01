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


def probe_scene(test, **params):
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
        set_camera_shader(scene, PROBE_SHADER, {"test": test, **params})
    return build


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
SCENES.update({
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


def enable_device(args):
    if args.device == "CPU":
        return
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == "METAL" or (args.device == "METAL_CPU" and device.type == "CPU")


# Gabor noise is not translated for Metal.
UNSUPPORTED_SHADER = r"""
shader unsupported(output point position = point(0.0),
                   output vector direction = vector(0.0, 0.0, 1.0),
                   output color throughput = color(1.0))
{
    point Pcam = camera_shader_raster_position() - point(0.5);
    Pcam += 0.02 * noise("gabor", Pcam * 20.0);
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
        SCENES[name](args)
        enable_device(args)
        scene = bpy.context.scene
        scene.render.filepath = str(args.output / f"{name}.exr")
        start = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        seconds = time.perf_counter() - start
        report[name] = {"render_seconds": seconds}
        print(f"SCENE {name} {json.dumps(report[name])}", flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


main()
