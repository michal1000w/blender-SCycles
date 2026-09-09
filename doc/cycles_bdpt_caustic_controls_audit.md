# Diffuse continuation under caustic switches

The BDPT light generator tested the reflection/transmission label directly when
honoring the caustic switches. Consequently, disabling reflective caustics also
stopped ordinary diffuse reflection, shortening light paths in an entirely
diffuse scene. The shared `bdpt_caustic_event_enabled()` helper now limits that
test to glossy/singular events and explicitly preserves null transmission.
This fixes current-event classification; full future light-side caustic-path
classification is a separate concern.

The independent host event cases pass alongside the existing transport tests
(9 transport tests; 43 guiding host tests). The actual renderer regression
`tests/python/cycles_bdpt_caustic_controls.py` uses a purely diffuse room and
all four switch combinations. Its optional transparent sheet adds null-event
coverage. Each case renders exactly one sample with guiding disabled, comparing
the physical image and canonical cached light vertices without a learning phase.

Both suites were run on the unfixed installed renderer before installing the
correction. Disabling reflective caustics changed the image by up to 15.40/15.23
and changed cached vertices. Disabling only refractive caustics already passed;
these fixtures do not demonstrate an old null-transmission failure.

After the fix, both suites pass all four combinations at the original
`rtol=1e-6, atol=1e-6`. Each cache is 1,517,856 bytes and byte-identical across
switches within the suite. Images differ only within that existing tolerance.
Logs confirm all 16 before/after cases completed exactly one sample. Reports,
EXRs, caches, and consolidated comparisons remain under
`build/metal-guiding-tests/caustic-controls*`.

This is a verified regression fix, not a claim that BDPT guiding meets CPU
quality or that the complete transport feature set is finished. No benchmark
renderer changed during the preceding 18-render matched-work study.
