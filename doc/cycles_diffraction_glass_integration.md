# Glass integration requirements

The rough diffraction closure is implemented and tested locally on CPU/Metal,
but it is not a drop-in replacement for the current Glass shader. This file
records concrete integration work, not supported-feature claims.

Current entry points:

- `scene/shader_nodes.cpp`: `GlassBsdfNode` declares and compiles color,
  roughness, IOR, thin-film thickness and IOR; distribution selects GGX,
  Beckmann, or multi-GGX.
- `kernel/svm/node_types.h`: `SVMNodeGlassBsdfData` defines the packed inputs.
- `kernel/svm/closure.h`: Glass setup orients indices for backfaces, honors
  reflective/refractive caustics independently, converts reflection and
  transmission colors separately, and installs film and multiscattering data.
- `kernel/osl/shaders/node_glass_bsdf.osl`: existing Glass uses the generalized
  Schlick closure with distribution and film parameters.
- `scene/shader_nodes.h`: Glass currently has no diffraction-specific extra
  closure reservation. The new matched-index model can require two slots.

Required implementation before claiming complete Glass support:

1. Carry linked grating inputs, tangent and wavelength through Blender node
   declarations, Cycles node sockets, packed SVM data, OSL parameters and closure
   setup. Reserve all required slots, including any ordinary/diffraction blend.
2. Preserve separate reflection/transmission tint and caustics selection without
   treating a disabled lobe as energy that should be reassigned to another lobe.
3. Support the selected distribution; do not silently reinterpret Beckmann or
   multi-GGX as single-scattering GGX.
4. Integrate coating response consistently in evaluation and sampling. A coating
   makes matched-index straight-through energy angle-dependent; the existing
   constant-atom split is not valid for that combination without a new derivation.
5. Validate front/back interfaces, linked parameters, zero diffraction, smooth
   and rough limits, shader mixtures, allocation limits, SVM/OSL parity, guiding
   and BDPT camera/light connections in actual saved Blender scenes.

The tested standalone film helper does not yet satisfy item 4. The current
world-frame numerical regression does not replace item 5. Preserve the working
Fast installed build while implementing these paths in validation builds.

Coated continuous sampling now has a separate implementation. For a visible
facet `h`, let `q(h)=1-T0(h)` be the power excluding the matched-index straight
order. Sample the usual VNDF, then choose an order with probability
`power(order,h)/q(h)`. Evaluation sums `VNDF(h)*power(order,h)*Jacobian/q(h)`
over every inverse root. Its physical BSDF sum has no `1/q(h)` factor; only
the proposal density does. Thus the continuous estimator needs no rejection
loop or averaged-atom integral. Off-hemisphere proposals still terminate as
in the ordinary rough model. Zero-budget facets have no continuous energy.

The missing discrete part remains important: the straight-through BSDF mass
requires the visible-facet integral of `T0(h)`. It cannot reuse the uncoated
constant atom, and a sampled facet's mass is not that integral. Any bounded
Fast quadrature or GPU lookup for it must be tested for approximation error,
reversal symmetry and energy, with a separately measured higher-quality option.
The eventual mixture must apply its discrete/continuous proposal probabilities
explicitly; the coated continuous helper returns the physical BSDF value and
its per-facet-conditioned PDF, with no global mixture factor folded in.

A combined local mixture reference now verifies those factors with bounded
proposal support and separate physical atom mass. It is not a replacement for
the renderer's separated closures: guiding RIS relies on that separation.
Coated atom storage must retain enough material/frame information to reevaluate
its direction-dependent mass when the incident direction or roughness changes.
Freezing the atom in a closure weight at shader setup would be incorrect for
such later evaluations. The continuous closure can retain its per-facet proposal
and physical BSDF value, with mixture selection handled by the renderer.

## Glass material integration (delivery pass)

Glass now exposes Diffraction Weight, Pitch (nm), Depth (nm), Duty Cycle and
Tangent. Zero coverage retains the existing glass implementation. Nonzero
coverage uses the tested joint dielectric closure, with independently filtered
reflection/transmission, spectral transmission tint, back-face index orientation
and the node's thin film. GGX and Beckmann share the SVM/OSL physical-input
conversion. The tangent points across grooves. A degenerate tangent receives
a deterministic surface frame.

This Fast model is single-scattering. Multi-scattering GGX with diffraction
currently produces an explicit compiler error; it is not silently substituted.
The matched-index coated atom uses 64 quadrature samples, a bounded approximate
choice (previous measured GGX pilot max absolute atom error 0.00771); it is not
a certified high-accuracy default. Existing zero-coverage materials retain
their chosen distribution. Eevee does not implement diffraction.

Build passes including OSL and Metal generic kernels, recorded in
`/tmp/diffraction_glass_delivery_build_v2.log`. The new presentation fixture is
`tests/python/cycles_diffraction_glass_delivery.py`: smooth GGX, rough coated GGX
and rough Beckmann spheres, fixed samples, adaptive sampling and denoising off.
Rendered verification is tracked separately; compilation is not render proof.
