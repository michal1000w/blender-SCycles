#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent oracle for streamed coherent facets with closed convex Glass.

Pure NumPy/SciPy; no Blender, no renderer code. The formulation intentionally
differs from the Cycles kernel:

* fields are 3D complex E-vectors, not Jones pairs in a transported basis;
* ideal mirrors apply the declared zero-phase Householder map E -> (I - 2nn^T)E;
* dielectric interfaces use Born-Wolf Fresnel coefficients with p = s x k for
  every wave, and flux-normalized transmission amplitudes;
* entry/exit (TT) routes are found by numerically minimizing the optical path
  length over the two facet planes (Fermat), then checked against Snell's law;
* reflection routes use exact double-precision image sources;
* the source solid angle per detector area is a central finite difference of
  the solved launch direction over the detector plane;
* visibility is a brute-force double-precision segment/triangle test.

The detector is a white Lambertian receiver of the full 3D field power for three
independent world-axis dipole modes, radiance = albedo/pi * irradiance. Paths of
one coherence group interfere with Gaussian mutual coherence exp(-dL^2/2Lc^2).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

EPS_SEGMENT = 1e-9


def unit(v):
    return v / np.linalg.norm(v)


class Facet:
    def __init__(self, obj, index, points, kind, ior, extinction=(0.0, 0.0, 0.0)):
        self.obj = obj
        self.index = index
        self.p = np.asarray(points, dtype=float)
        self.kind = kind  # 'mirror', 'glass', 'detector', 'opaque'
        self.ior = ior
        self.extinction = np.asarray(extinction, dtype=float)  # Interior, per metre (RGB).
        e0 = self.p[1] - self.p[0]
        e1 = self.p[2] - self.p[0]
        self.normal = unit(np.cross(e0, e1))  # Outward for validated Glass.
        self.u = unit(e0)
        self.v = np.cross(self.normal, self.u)

    def plane_distance(self, x):
        return float(np.dot(x - self.p[0], self.normal))

    def barycentric(self, x):
        e0 = self.p[1] - self.p[0]
        e1 = self.p[2] - self.p[0]
        n = np.cross(e0, e1)
        d = np.dot(n, n)
        w = x - self.p[0]
        b1 = np.dot(np.cross(w, e1), n) / d
        b2 = np.dot(np.cross(e0, w), n) / d
        return 1.0 - b1 - b2, b1, b2

    def contains(self, x, tolerance=0.0):
        return min(self.barycentric(x)) >= -tolerance

    def reflect_point(self, x):
        return x - 2.0 * self.plane_distance(x) * self.normal

    def intersect_segment(self, a, b):
        """Parameter t in (0,1) where segment a->b crosses the facet, else None."""
        d = b - a
        denom = np.dot(d, self.normal)
        if abs(denom) < 1e-300:
            return None
        t = np.dot(self.p[0] - a, self.normal) / denom
        if not (0.0 < t < 1.0):
            return None
        return t if self.contains(a + t * d, 1e-12) else None


class Scene:
    def __init__(self, data):
        self.facets = []
        for obj in data['objects']:
            for i, tri in enumerate(obj['triangles']):
                self.facets.append(Facet(obj['name'], i, tri, obj['kind'], obj.get('ior', 1.0),
                                         obj.get('extinction', (0.0, 0.0, 0.0))))
        self.sources = data['sources']
        self.albedo = float(data.get('detector_albedo', 1.0))
        self.detector_normal = unit(np.asarray(data['detector_normal'], dtype=float))
        self.detector_object = data['detector_object']
        self.route_cache = None
        camera = data.get('camera')
        if camera is not None:
            # Footprint corners (with a 10% halo) for the conservative prefilter.
            center = np.asarray(camera['detector_center'], dtype=float)
            right = np.asarray(camera['right'], dtype=float)
            up = np.asarray(camera['up'], dtype=float)
            half = 0.55 * float(camera['ortho_scale'])
            corners = [center] + [center + a * half * right + b * half * up
                                  for a in (-1, 1) for b in (-1, 1)]
            self.route_cache = {}
            events = int(data.get('max_events', 2))
            lookup = {(f.obj, f.index): f for f in self.facets}
            for i, source in enumerate(self.sources):
                position = np.asarray(source['position'], dtype=float)
                keys = data.get('route_keys')
                if keys is not None:
                    # Routes kept by an earlier prefilter of this exact scene.
                    routes = [Route(family, tuple(lookup[tuple(f)] for f in facets), tuple(ev))
                              for family, facets, ev in keys[i]]
                else:
                    routes = prefilter_routes(self, position, corners, events)
                self.route_cache[(tuple(position), events)] = routes

    def route_keys(self):
        """Serializable kept routes per source, for worker processes."""
        result = []
        for source in self.sources:
            position = tuple(np.asarray(source['position'], dtype=float))
            routes = next(v for (p, _), v in self.route_cache.items() if p == position)
            result.append([[r.family, [[f.obj, f.index] for f in r.facets], list(r.events)] for r in routes])
        return result

    def glass_objects(self):
        return sorted({f.obj for f in self.facets if f.kind == 'glass'})

    def visible(self, a, b, exclude):
        """No facet other than the listed endpoint facets crosses the open segment."""
        length = np.linalg.norm(b - a)
        for f in self.facets:
            if (f.obj, f.index) in exclude:
                continue
            t = f.intersect_segment(a, b)
            if t is not None and EPS_SEGMENT < t * length < length - EPS_SEGMENT:
                return False
        return True


# --- Route solvers: return list of interface points, or None -------------------


def solve_reflection(source, receiver, facets):
    """Ordered planar reflections by exact images (double precision)."""
    images = [source]
    for f in facets:
        images.append(f.reflect_point(images[-1]))
    points = [None] * len(facets)
    end = receiver
    for i in range(len(facets) - 1, -1, -1):
        f = facets[i]
        a, b = images[i + 1], end
        d = b - a
        denom = np.dot(d, f.normal)
        if abs(denom) < 1e-300:
            return None
        t = np.dot(f.p[0] - a, f.normal) / denom
        if not (0.0 < t < 1.0):
            return None
        points[i] = a + t * d
        end = points[i]
    chain = [source] + points + [receiver]
    for i, f in enumerate(facets):
        before = f.plane_distance(chain[i])
        after = f.plane_distance(chain[i + 2])
        if before * after <= 0.0:
            return None
        if f.kind == 'glass' and before <= 0.0:
            return None  # Exterior reflection only.
    return points


_WARM = {}  # Previous Fermat solution per facet pair, only as a starting point.


def solve_transmission(source, receiver, entry, exit_):
    """Fermat stationary entry/exit chord through one Glass volume."""
    n = entry.ior
    if entry.plane_distance(source) <= 0.0 or exit_.plane_distance(receiver) <= 0.0:
        return None
    scale = max(np.linalg.norm(receiver - source), 1e-12)
    c0 = entry.p.mean(axis=0)
    c1 = exit_.p.mean(axis=0)

    def points(q):
        q = q * scale
        return (c0 + q[0] * entry.u + q[1] * entry.v, c1 + q[2] * exit_.u + q[3] * exit_.v)

    def opl(q):
        p0, p1 = points(q)
        return (np.linalg.norm(p0 - source) + n * np.linalg.norm(p1 - p0) +
                np.linalg.norm(receiver - p1)) / scale

    def grad(q):
        p0, p1 = points(q)
        a = unit(p0 - source)
        b = unit(p1 - p0)
        c = unit(receiver - p1)
        g0 = a - n * b
        g1 = n * b - c
        return np.array([g0 @ entry.u, g0 @ entry.v, g1 @ exit_.u, g1 @ exit_.v])

    def polish(q):
        # Newton with a finite-difference Hessian of the analytic gradient.
        for _ in range(8):
            g = grad(q)
            h = 1e-7
            H = np.zeros((4, 4))
            for j in range(4):
                dq = np.zeros(4)
                dq[j] = h
                H[:, j] = (grad(q + dq) - grad(q - dq)) / (2 * h)
            try:
                q = q - np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                return None
            if not np.all(np.isfinite(q)):
                return None
        return q if np.max(np.abs(grad(q))) <= 1e-11 else None

    # A previous solution for the same source and facet pair is only a
    # starting point; any failure falls back to a cold global BFGS solve.
    key = (id(entry), id(exit_), tuple(source))
    q = None
    if key in _WARM:
        q = polish(_WARM[key])
    if q is None:
        cold = minimize(opl, np.zeros(4), jac=grad, method='BFGS',
                        options={'gtol': 1e-15, 'maxiter': 10000}).x
        q = polish(cold)
    if q is None:
        _WARM.pop(key, None)
        return None
    _WARM[key] = q
    p0, p1 = points(q)
    # Finite triangle membership is checked by connect(); the plane solution
    # is also used by the spreading stencil across internal triangle edges.
    # Chord runs inside: behind the entry plane and arriving from behind the exit.
    if entry.plane_distance(p1) >= 0.0 or exit_.plane_distance(p0) >= 0.0:
        return None
    return [p0, p1]


def solve_general(source, receiver, facets, events, iors, media=None):
    """Any planar route: the optical length is a sum of norms of affine maps of
    the plane coordinates, hence convex, so its stationary point is the unique
    minimum. Per-event side conditions then decide physical admissibility."""
    k = len(facets)
    if media is None:
        media = [None] + [None if n == 1.0 else True for n in iors[1:]]
    inside_leg = [m is not None for m in media]
    # Necessary endpoint conditions (exterior roles need their air endpoint outside).
    if facets[0].kind == 'glass' and facets[0].plane_distance(source) <= 0.0:
        return None
    if facets[-1].kind == 'glass' and facets[-1].plane_distance(receiver) <= 0.0:
        return None
    # Necessary half-space conditions for every leg. A Glass event fixes the
    # side a leg leaves from and arrives on (outside +, inside -); a mirror
    # keeps both legs on one side. Endpoint sets are the facet vertices.
    def depart_side(i):
        f = facets[i]
        if f.kind != 'glass':
            return 0
        return -1 if inside_leg[i + 1] else 1

    def arrive_side(i):
        f = facets[i]
        if f.kind != 'glass':
            return 0
        return -1 if inside_leg[i] else 1

    sets = [[source]] + [list(f.p) for f in facets] + [[receiver]]
    for i in range(k):
        f = facets[i]
        prev_set, next_set = sets[i], sets[i + 2]
        a_side, d_side = arrive_side(i), depart_side(i)
        if a_side and not any(a_side * f.plane_distance(x) > 0.0 for x in prev_set):
            return None
        if d_side and not any(d_side * f.plane_distance(x) > 0.0 for x in next_set):
            return None
        if f.kind == 'mirror':
            prev_d = [f.plane_distance(x) for x in prev_set]
            next_d = [f.plane_distance(x) for x in next_set]
            if (min(prev_d) > 0 and max(next_d) <= 0) or (max(prev_d) < 0 and min(next_d) >= 0):
                return None
    scale = max(np.linalg.norm(receiver - source), 1e-12)
    centers = [f.p.mean(axis=0) for f in facets]

    def points(q):
        q = q * scale
        return [c + q[2 * i] * f.u + q[2 * i + 1] * f.v for i, (c, f) in enumerate(zip(centers, facets))]

    def opl(q):
        chain = [source] + points(q) + [receiver]
        return sum(n * np.linalg.norm(b - a) for a, b, n in zip(chain[:-1], chain[1:], iors)) / scale

    def grad(q):
        chain = [source] + points(q) + [receiver]
        d = [unit(b - a) for a, b in zip(chain[:-1], chain[1:])]
        g = np.zeros(2 * k)
        for i, f in enumerate(facets):
            v = iors[i] * d[i] - iors[i + 1] * d[i + 1]
            g[2 * i] = v @ f.u
            g[2 * i + 1] = v @ f.v
        return g

    key = (tuple(id(f) for f in facets), tuple(events), tuple(source))
    q = _WARM.get(key)
    if q is None or np.max(np.abs(grad(q))) > 1e-3:
        q = minimize(opl, np.zeros(2 * k), jac=grad, method='BFGS',
                     options={'gtol': 1e-15, 'maxiter': 20000}).x
    for _ in range(10):
        g = grad(q)
        H = np.zeros((2 * k, 2 * k))
        for j in range(2 * k):
            dq = np.zeros(2 * k)
            dq[j] = 1e-7
            H[:, j] = (grad(q + dq) - grad(q - dq)) / 2e-7
        try:
            q = q - np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return None
    if not np.all(np.isfinite(q)) or np.max(np.abs(grad(q))) > 1e-11:
        _WARM.pop(key, None)
        return None
    _WARM[key] = q
    pts = points(q)
    chain = [source] + pts + [receiver]
    for i, (f, ev) in enumerate(zip(facets, events)):
        before = f.plane_distance(chain[i])
        after = f.plane_distance(chain[i + 2])
        inside = inside_leg[i]
        if ev == 'R':
            if before * after <= 0.0:
                return None
            if f.kind == 'glass' and (before < 0.0) != inside:
                return None
        else:
            if (before > 0.0) == inside or (after > 0.0) == (not inside):
                return None
    return pts


# --- Field transport ---------------------------------------------------------


def fresnel(n1, n2, cos_i, reflection=False):
    """Born-Wolf coefficients (p = s x k). Beyond the critical angle only the
    reflection exists: cos_t is imaginary, r is complex with |r| = 1."""
    sin_t2 = (n1 / n2) ** 2 * (1.0 - cos_i * cos_i)
    if sin_t2 > 1.0:
        if not reflection:
            return None
        cos_t = 1j * math.sqrt(sin_t2 - 1.0)
        rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
        rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
        return rs, rp, 0.0, 0.0, cos_t
    cos_t = math.sqrt(1.0 - sin_t2)
    rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
    ts = 2 * n1 * cos_i / (n1 * cos_i + n2 * cos_t)
    tp = 2 * n1 * cos_i / (n2 * cos_i + n1 * cos_t)
    flux = math.sqrt(n2 * cos_t / (n1 * cos_i))
    return rs, rp, ts * flux, tp * flux, cos_t


def interface(field, k_in, facet, event, n1, n2):
    """Returns outgoing (field, direction)."""
    normal = facet.normal
    n_inc = normal if np.dot(k_in, normal) < 0 else -normal  # Points into incident medium.
    cos_i = -float(np.dot(k_in, n_inc))
    if facet.kind == 'mirror':
        k_out = k_in - 2 * np.dot(k_in, normal) * normal
        return field - 2 * np.dot(field, normal) * normal, k_out
    s = np.cross(k_in, n_inc)
    if np.linalg.norm(s) < 1e-14:
        s = facet.u
    s = unit(s)
    p_in = np.cross(s, k_in)
    coefficients = fresnel(n1, n2, cos_i, reflection=event == 'R')
    if coefficients is None:
        return None
    rs, rp, ts, tp, cos_t = coefficients
    es = np.dot(field, s)
    ep = np.dot(field, p_in)
    if event == 'R':
        k_out = k_in + 2 * cos_i * n_inc
        return rs * es * s + rp * ep * np.cross(s, k_out), k_out
    eta = n1 / n2
    k_out = unit(eta * k_in + (eta * cos_i - cos_t) * n_inc)
    return ts * es * s + tp * ep * np.cross(s, k_out), k_out


def route_field(source, receiver, facets, events, points, iors):
    """Mode fields at the receiver for three world-axis dipoles (1/sqrt(2) each)."""
    chain = [source] + list(points) + [receiver]
    k0 = unit(chain[1] - chain[0])
    fields = []
    for axis in np.eye(3):
        e = (axis - np.dot(axis, k0) * k0) / math.sqrt(2.0)
        k = k0
        for i, (f, ev) in enumerate(zip(facets, events)):
            # Exterior Glass reflection: Fresnel against the Glass IOR.
            # Glass reflection: against the Glass IOR from air, against air from inside.
            if ev == 'R' and f.kind == 'glass':
                opposite = f.ior if iors[i] == 1.0 else 1.0
            else:
                opposite = iors[i + 1]
            out = interface(e, k, f, ev, iors[i], opposite)
            if out is None:
                return None
            e, k = out
            # Transport direction must agree with the solved geometry.
            assert np.linalg.norm(k - unit(chain[i + 2] - chain[i + 1])) < 1e-6, 'direction mismatch'
        fields.append(e)
    return fields


def optical_length(chain, iors):
    return sum(n * np.linalg.norm(b - a) for a, b, n in zip(chain[:-1], chain[1:], iors))


class Route:
    def __init__(self, family, facets, events):
        self.family = family
        self.facets = facets
        self.events = events

    def iors(self):
        values = [1.0]
        for f, ev in zip(self.facets, self.events):
            if ev == 'T':
                values.append(f.ior if values[-1] == 1.0 else 1.0)
            else:
                values.append(values[-1])
        return values

    def media(self):
        """Glass facet whose volume encloses each leg (None for air)."""
        inside = None
        result = [None]
        for f, ev in zip(self.facets, self.events):
            if ev == 'T' and f.kind == 'glass':
                inside = f if inside is None else None
            result.append(inside)
        return result

    def solve(self, source, receiver):
        if not self.facets:
            return []
        if len(self.facets) >= 3:
            return solve_general(source, receiver, self.facets, self.events, self.iors(), self.media())
        if self.events == ('T', 'T'):
            return solve_transmission(source, receiver, *self.facets)
        return solve_reflection(source, receiver, self.facets)


def enumerate_routes(scene, max_events):
    interfaces = [f for f in scene.facets if f.kind in ('mirror', 'glass')]
    routes = [Route('direct', (), ())]
    if max_events >= 1:
        for f in interfaces:
            routes.append(Route('glass_R' if f.kind == 'glass' else 'mirror_R', (f,), ('R',)))
    if max_events >= 2:
        for a in interfaces:
            for b in interfaces:
                if a is b:
                    continue
                kinds = ''.join('G' if f.kind == 'glass' else 'M' for f in (a, b))
                routes.append(Route('RR_' + kinds, (a, b), ('R', 'R')))
                if a.kind == 'glass' and b.kind == 'glass' and a.obj == b.obj:
                    routes.append(Route('TT', (a, b), ('T', 'T')))
    if max_events >= 3:
        def extend(facets, events, medium):
            depth = len(facets)
            if depth >= 3 and medium is None:
                name = ''.join(('M' if f.kind == 'mirror' else 'G') + e for f, e in zip(facets, events))
                routes.append(Route('long_' + name, tuple(facets), tuple(events)))
            if depth == max_events or (medium is not None and depth + 1 > max_events):
                return
            for f in interfaces:
                if facets and f is facets[-1]:
                    continue
                if medium is None:
                    options = [('R', None)] + ([('T', f.obj)] if f.kind == 'glass' else [])
                elif f.obj == medium:
                    options = [('R', medium), ('T', None)]
                else:
                    continue
                for event, next_medium in options:
                    extend(facets + [f], events + [event], next_medium)
        extend([], [], None)
    return routes


def connect(scene, route, source, receiver, detector_index):
    points = route.solve(source, receiver)
    if points is None:
        return None
    if not all(f.contains(x) for f, x in zip(route.facets, points)):
        return None
    # A hit exactly on an edge shared with a coplanar neighbour of the same
    # object (measure zero) belongs only to the lowest-index such triangle.
    for f, x in zip(route.facets, points):
        if min(f.barycentric(x)) <= 1e-15:
            for g in scene.facets:
                if (g.obj == f.obj and g.index < f.index and np.dot(g.normal, f.normal) > 1 - 1e-12
                        and abs(g.plane_distance(x)) < 1e-15 and g.contains(x)):
                    return None
    chain = [source] + list(points) + [receiver]
    endpoint = [None] + [(f.obj, f.index) for f in route.facets] + [(scene.detector_object, detector_index)]
    for i in range(len(chain) - 1):
        exclude = {x for x in (endpoint[i], endpoint[i + 1]) if x is not None}
        if i == len(chain) - 2:
            # The receiver lies on the detector plane; exclude its own object.
            exclude |= {(f.obj, f.index) for f in scene.facets if f.obj == scene.detector_object}
        if not scene.visible(chain[i], chain[i + 1], exclude):
            return None
    if np.dot(scene.detector_normal, chain[-2] - receiver) <= 0.0:
        return None
    return points


def launch_direction(scene, route, source, receiver, detector_index):
    """Launch direction of the stationary route on the facet planes.

    Local spreading depends only on the plane geometry, so the difference
    stencil neither requires the perturbed hit to stay inside the same finite
    triangle nor retests visibility (the unperturbed route was fully checked)."""
    points = route.solve(source, receiver)
    if points is None:
        return None, None
    first = points[0] if points else receiver
    return unit(first - source), points


def spreading(scene, route, source, receiver, detector_index, h):
    """dOmega_source / dA_detector by central differences on the detector plane."""
    n = scene.detector_normal
    t1 = unit(np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0]))
    t2 = np.cross(n, t1)
    derivatives = []
    for t in (t1, t2):
        plus, _ = launch_direction(scene, route, source, receiver + h * t, detector_index)
        minus, _ = launch_direction(scene, route, source, receiver - h * t, detector_index)
        if plus is None or minus is None:
            return None
        derivatives.append((plus - minus) / (2 * h))
    d0, _ = launch_direction(scene, route, source, receiver, detector_index)
    return abs(float(np.dot(np.cross(derivatives[0], derivatives[1]), d0)))


def route_margin(route, source, receiver):
    """Smallest barycentric coordinate of the solved hit points (no visibility)."""
    points = route.solve(source, receiver)
    if points is None:
        return None
    return min(min(f.barycentric(x)) for f, x in zip(route.facets, points)) if points else 1.0


def prefilter_routes(scene, source, corners, max_events, margin=-0.2):
    """Routes worth testing over a small detector footprint.

    Exact per-point tests still decide every contribution. A route is dropped
    only if at every footprint corner its unconstrained stationary points lie
    beyond a generous barycentric margin outside a facet (or do not exist on
    the correct sides). For centimetre facets and sub-millimetre footprints the
    hit points move far less than that margin, so no connectable route is lost."""
    kept = []
    for route in enumerate_routes(scene, max_events):
        values = [route_margin(route, source, c) for c in corners]
        if any(v is not None and v > margin for v in values):
            kept.append(route)
    return kept


def detector_paths(scene, source_data, receiver, max_events, detector_index=0, h=1e-7,
                   omit=()):
    source = np.asarray(source_data['position'], dtype=float)
    paths = []
    key = (tuple(source), max_events)
    routes = scene.route_cache.get(key) if scene.route_cache is not None else None
    if routes is None:
        routes = enumerate_routes(scene, max_events)
    for route in routes:
        if route.family in omit or route.family.split('_')[0] in omit:
            continue
        points = connect(scene, route, source, receiver, detector_index)
        if points is None:
            continue
        S = spreading(scene, route, source, receiver, detector_index, h)
        if S is None or not (S > 0):
            continue
        iors = route.iors()
        fields = route_field(source, receiver, route.facets, route.events, points, iors)
        if fields is None:
            continue
        chain = [source] + list(points) + [receiver]
        L = optical_length(chain, iors)
        amplitude = math.sqrt(source_data['power_W'] / (4 * math.pi) * S)
        # Ballistic attenuation on legs inside a Glass medium (Beer-Lambert on power).
        channel = np.ones(3)
        for leg, medium in enumerate(route.media()):
            if medium is not None:
                channel *= np.exp(-0.5 * medium.extinction * np.linalg.norm(chain[leg + 1] - chain[leg]))
        paths.append({'family': route.family, 'L': L, 'amplitude': amplitude,
                      'fields': fields, 'phase': source_data['phase_rad'], 'channel': channel,
                      'group': source_data['group'], 'wavelength': source_data['wavelength_m'],
                      'coherence_length': source_data['coherence_length_m']})
    return paths


def radiance(scene, receiver, max_events, omit=(), incoherent=False):
    paths = []
    for index, source in enumerate(scene.sources):
        for p in detector_paths(scene, source, receiver, max_events, omit=omit):
            p['source'] = index
            paths.append(p)
    total = 0.0
    for a in paths:
        for b in paths:
            if a['group'] != b['group']:
                continue
            if incoherent and a is not b:
                continue
            dL = a['L'] - b['L']
            gamma = math.exp(-0.5 * (dL / a['coherence_length']) ** 2)
            phase = 2 * math.pi * dL / a['wavelength'] + a['phase'] - b['phase']
            # Complex (e.g. total internal reflection) fields: E_a . conj(E_b).
            overlap = sum(np.dot(ea, np.conj(eb)) for ea, eb in zip(a['fields'], b['fields']))
            # RGB channels differ only by interior extinction; report their mean.
            weight = float(np.mean(a['channel'] * b['channel']))
            total += gamma * a['amplitude'] * b['amplitude'] * weight * (overlap * np.exp(1j * phase)).real
    return scene.albedo / math.pi * float(total), paths


def pixel_points(camera, row, col, order):
    """World detector points for an order x order midpoint rule in one pixel."""
    offsets = (np.arange(order) + 0.5) / order
    origin = np.asarray(camera['detector_center'], dtype=float)
    right = np.asarray(camera['right'], dtype=float)
    up = np.asarray(camera['up'], dtype=float)
    width = float(camera['ortho_scale'])
    res = int(camera['resolution'])
    for oy in offsets:
        for ox in offsets:
            yield origin + ((col + ox) / res - 0.5) * width * right + \
                ((row + oy) / res - 0.5) * width * up


def image(scene, camera, max_events, order=2, omit=(), incoherent=False, statistics=False):
    res = int(camera['resolution'])
    result = np.zeros((res, res))
    variance = np.zeros((res, res))
    for row in range(res):
        for col in range(res):
            values = [radiance(scene, x, max_events, omit, incoherent)[0]
                      for x in pixel_points(camera, row, col, order)]
            result[row, col] = np.mean(values)
            variance[row, col] = np.var(values)
    return (result, variance) if statistics else result


def main():
    data = json.loads(Path(sys.argv[1]).read_text())
    scene = Scene(data)
    camera = data['camera']
    receiver = np.asarray(camera['detector_center'], dtype=float)
    value, paths = radiance(scene, receiver, data['max_events'])
    print(json.dumps({'center_radiance': value,
                      'families': sorted({p['family'] for p in paths}),
                      'path_count': len(paths)}, indent=1))


if __name__ == '__main__':
    main()
