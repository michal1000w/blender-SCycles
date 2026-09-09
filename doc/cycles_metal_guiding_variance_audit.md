# Guiding-specific variance audit

The physical spherical point-light fixture uses a black world, opaque objects,
no denoising/adaptive sampling, 96x96 resolution and seed 101. All comparisons
below use verified 512 SPP against the same CPU-guided 8192-SPP seed991 reference.
These are single-seed diagnostics, not final multi-seed acceptance or timing results.

| Estimator | MSE |
| --- | ---: |
| CPU guided RIS | 9.232944063459344e-7 |
| CPU unguided PT | 1.055528391697603e-6 |
| Metal unguided PT | 1.055085909151645e-6 |
| Metal guided RIS PT | 6.340225916768196e-6 |
| Metal guided RIS BDPT | 7.098273914329991e-6 |

The near-identical unguided CPU/GPU errors, followed by a roughly sixfold
increase when GPU guiding is enabled, put the guiding sampler/proposal/MIS
interaction ahead of BDPT-only weighting as the next investigation. They do
not prove that every CPU/GPU baseline is equivalent. CPU guiding improves this
fixture modestly; its high-sample image remains the reference.

The moving mesh and spot fixtures also execute successfully, with close global
means but respectively about 8.9x and 4.5x CPU-guided error for Metal BDPT at512SPP.
Raw EXRs/NPYs, verified counts and metrics are retained in
`build/metal-guiding-tests/emitter-nee-validation/comparison.json`.

Next checks: compare GPU MIS and ROUGHNESS against existing RIS/default at the
same512SPP; inspect the proposal/PDF used by each; then correct the sampler or
integrate the learned mixture as evidence requires. The existing retained-history
and learned-mixture integration plan remains part of the full guiding objective.
Do not alter defaults or normalize brightness simply to hide this regression.
The separate material-scene block brightness offset is still unresolved.

## Related primary research, not yet adopted

[Variance-aware MIS (2019)](https://www.iliyan.com/publications/VarianceAwareMIS/)
explains why balance and power weights can squander stratification benefits.
Its supplemental implementation multiplies strategy density by a factor
`1 + mean^2 / variance`, estimated in a preliminary stage. This may be relevant
to the additional BDPT variance, but cannot by itself explain the measured
ordinary-PT guiding regression. Any implementation here must freeze learned
weights before their application and preserve unbiased treatment of training
samples; no source code has been imported or executed.

[Correlation-aware MIS (2021)](https://onlinelibrary.wiley.com/doi/10.1111/cgf.142628)
addresses shared-prefix correlation using subpath-density information. This is
relevant to cached light-path reuse, but is not yet an implementation decision.

Primary2019 PDF and supplemental archive are stored in `/tmp/cycles-guiding-variance-mis-2019.pdf`
and `/tmp/cycles-guiding-variance-mis-2019-code.zip`; extracted text and selected
BDPT source files are also in `/tmp`. The supplemental top-level license was
inspected (permissive PBRT notice); nothing has been vendored.

Both additional mode controls completed at verified512SPP: Metal MIS MSE
5.170733847941404e-6 and ROUGHNESS MSE 4.843319131126769e-6. All three modes
therefore degrade this fixture relative to unguided Metal (roughly4.6-6x).
Defaults remain RIS. The next investigation is the shared learned proposal,
its PDF/sampling and training, including the pending learned-mixture integration;
the failure is not isolated to RIS. No render jobs remain active at this checkpoint.

## MIS-aware NEE training correction under validation

The field dump `point-field-audit/field_128` rules out a missing point-light lobe:
near the floor origin its strongest component has mass 0.6961, concentration
1549.99 and axis (-0.3392, -0.2351, 0.9108), close to the actual light direction.
The inspection output is retained in `point-field-audit/inspection.json`.

The CPU recorder adds ordinary NEE as scattered outgoing radiance at its path
segment. OpenPGL's `PathSegmentDataStorage::prepareSamples` starts propagation
at segment `i + 1`: it does not train vertex i with its own NEE direction.
Immediate emitter hits retain their MIS weight when `useNEEMiWeights` is true.
This was checked against the already downloaded OpenPGL source at commit
b4e4b86 in `/tmp/cycles-guiding-openpgl-distance-research`, and the local CPU
recording and PrepareSamples call sites. Upstream source:
https://github.com/OpenPathGuidingLibrary/openpgl/blob/b4e4b86/openpgl/data/PathSegmentDataStorage.h

The GPU had additionally appended standalone NEE observations at the current
vertex, teaching direct illumination already handled by NEE. With MIS-aware
training enabled, it now omits those additional observations while retaining
shadow contribution propagation through all earlier vertices, emitter-hit
observations, and indirect BDPT connection observations. This changes training,
not the rendered light contribution. Non-MIS training retains the previous
standalone behavior and needs its own audit. Build and install succeeded;
actual Metal validation is in `nee-training-validation`. No quality improvement
is claimed until its renders complete and raw-image metrics are checked.

The first corrected Metal RIS PT render completed at verified 512 SPP:
MSE 1.451943485547699e-6 versus the unchanged CPU-guided reference, down from
6.340225916768196e-6 (77.1% reduction). It remains 1.5726 times the CPU-guided
512-SPP error and therefore fails the requested quality target. Mean is
0.1602446726, relative reference difference -3.7422e-5. This is a single-seed
training correction result, not final convergence evidence. Additional MIS,
ROUGHNESS and BDPT controls are running with the same sample count.

Corrected MIS and ROUGHNESS PT also completed at verified 512 SPP, with MSE
1.369639587226453e-6 and 1.352667394424330e-6 respectively. Their CPU-guided
error ratios are 1.4834 and 1.4650: both still fail parity. The additional BDPT
control retains its explicit light-path setting of 16384; equal camera SPP does
not imply identical auxiliary light-path work. Final benchmark reporting must
include that work and training settings, with time as a separate measurement.

The BDPT control also completed at verified 512 SPP: MSE
5.739774101743112e-6 (6.2166 times CPU guided), versus the previous
7.098273914329991e-6. The improvement is much smaller than PT's; BDPT-specific
variance remains. Its mean is 0.1602808681, relative reference difference
+1.8845e-4. All four corrected controls are finite and complete, and all fail
CPU-guided quality parity. No render remains active at this checkpoint.

## Material and memory-scheduling follow-up

The installed build now avoids reserving nonexistent standalone NEE history
records in MIS-aware training. All19 persistent-session checks pass, and the
mixed-transparent one-SPP retention test preserves images across16/128MiB:
PT MSE2.4818065354551e-18, BDPT MSE1.7317793747406084e-16; BDPT canonical
cache bytes are identical. Reports are in `nee-training-material/persistent`
and `nee-training-material/retention`. No speedup is inferred from reservation
arithmetic alone.

At verified512SPP, seed101,256x256, the corrected material images have MSE:
rough glass PT0.014274127305782041, BDPT0.010122376970181372;
transmission PT0.015184199933909572, BDPT0.011595066138370082.
Their BDPT block means remain respectively1.0487% and1.1191% above the CPU-guided
two-seed8192 reference. Lower whole-image BDPT MSE is not a correctness pass.
Fresh CPU-guided512 controls are being rendered on the same installed binary;
the earlier CPU controls predate shared transport corrections and are retained
separately in the comparison report. High-SPP references are also earlier-build
images, so final convergence acceptance still requires current-build references.

Both fresh CPU-guided controls completed at verified512SPP and match the prior
controls' MSE exactly: rough glass0.011657485911544141 and
transmission0.011722157330414491. Thus the current same-build PT error ratios
remain1.2245 and1.2953; BDPT ratios0.8683 and0.9892 do not resolve the local
brightness discrepancy. All processes are terminal at this checkpoint.


## Matched light-work cache-refresh experiment

The unsuccessful confidence regularization is removed from the active GPU
publisher. Its tested primitives and 19-float/41-stat storage remain, so this is
an unregularized behavior restoration, not a bitwise restoration of the older
layout. Rebuild/install succeeded. Optional `CYCLES_BDPT_DIAGNOSTICS` now reports
all batches (previously only the first five sample indices) and camera sample
counts, enabling measured emitted-work totals without changing normal logging.

A serial experiment under `build/metal-guiding-tests/cache-reuse-work` compares
seeds 101, 211, 307. Every render completes 512 camera SPP at 96-square resolution.
CPU-guided PT and Metal-guided PT are controls. BDPT uses refresh intervals
1/2/4/8 with 4096/8192/16384/32768 emitted paths per full update. The existing
scheduler prorates short batches; the driver verifies all camera samples are
covered exactly once and exactly 2,097,152 light paths were emitted per render.
It rejects cache-retry runs because the current count records successful batches,
not extra failed generation work. All configurations retain full BDPT transport
as currently implemented; sensor connections are not disabled for this test.

The three-seed analysis uses the same high-SPP CPU-guided reference, records each
seed's MSE and the sample standard deviation, and keeps instrumented render time
separate. It makes no final confidence or warm performance claim. This experiment
will determine whether changing reuse at equal emitted work changes the observed
point-light variance; no result is asserted while the driver is running.


### Matched-work results

All 18 renders completed. Every image has 512 SPP; each of the 12 BDPT renders
has exactly 2,097,152 emitted light paths and complete camera-sample coverage.
Actual cache updates are 512/257/130/67 for configured intervals 1/2/4/8 because
initial/render-scheduler batches can be shorter than a full update.

| Estimator | Three-seed mean MSE | Ratio to CPU-guided mean MSE |
| --- | ---: | ---: |
| CPU guided PT | 9.58725725e-7 | 1 |
| Metal guided PT | 1.29833000e-6 | 1.354225 |
| BDPT update 1 | 5.60777852e-6 | 5.849200 |
| BDPT update 2 | 5.59225687e-6 | 5.833010 |
| BDPT update 4 | 5.46305962e-6 | 5.698251 |
| BDPT update 8 | 5.56194581e-6 | 5.801394 |

Changing reuse at the same emitted work does not close the gap. Retain the
existing schedule; these three seeds do not justify selecting a new default.
`comparison.json` retains individual-seed results and sample standard deviations.

The average cross-seed residual products are only 0.02%–1.17% of BDPT MSE,
versus 4.4% for CPU PT and 2.9% for GPU PT. This suggests predominantly random
error in this particular fixture, rather than a stable spatial error. The
calculation assumes independent seed realizations and includes shared noise
from the finite reference; it does not prove general unbiasedness. Script and
arrays are preserved as `decompose_error.py`, `error_decomposition.json`, and
`*_mean_error.npy`. The separate material-scene brightness issue remains open.

Next: investigate technique weighting and the guiding proposals rather than
increasing cache refresh frequency or giving BDPT extra light work.

## Power candidate: three-seed material controls

`tests/python/cycles_guiding_material_benchmark.py` now runs serial CPU-guided,
Metal-guided PT, and Metal-guided BDPT controls for rough glass and transmission.
It records installed MIS-header provenance, rejects incomplete camera coverage,
checks exact emitted BDPT work, and analyzes raw EXRs against CPU-guided references.
It requires an empty output directory and preserves failed artifacts. Default
settings are 512 SPP, seeds 101/211/307, 256x256, 128 training samples, no time
budget, denoising, adaptive sampling, or contribution clamping. Runtime is separate.

The installed power candidate completed all 18 renders in
`build/metal-guiding-tests/mis-power-material-multiseed`. Every render actually
used 512 SPP; every BDPT render emitted exactly 2,097,152 light paths. Primary
references remain two CPU-guided 8192-SPP seeds. Mean MSE results:

| Scene | CPU guided | Metal guided PT | Metal guided power BDPT |
| --- | ---: | ---: | ---: |
| Rough glass | .01159134053 | .01480465685 | .01062971431 |
| Transmission | .01193497104 | .01602568960 | .01144360851 |

Ratios to CPU are 1.27722/1.34275 for PT, and .91704/.95883 for BDPT.
This confirms a remaining ordinary-GPU-guiding deficit, despite the measured
BDPT material gains. Three seeds are not a proof of convergence or unbiasedness.
Transmission BDPT tall-block relative mean offsets are +.42485%, +1.18909%,
and +.61712%; all positive, requiring higher-SPP regional checks. No brightness
normalization was applied. The prior single-seed 10–12% material regression
versus balance has not yet received a matched multi-seed balance rerun.
CPU seed-101 MSE exactly reproduces the prior controls. Repeated GPU seed-101
results vary, consistent with the previously documented learned-field/atomic
repeatability limitation. Full raw data, per-seed/region metrics, sample standard
deviations, and work counts remain in the report directory.

## Training duration and publication cadence audit

The stored streaming point field has 383 of its 402 nonempty distributions using
all 16 components. Component extinction therefore lacks evidence as the dominant
quality explanation; no revival heuristic was added from that hypothesis.

Four equal-512-SPP PT renders increased the requested training limit from 128 to
512 for CPU and Metal (`training-duration-audit`). GPU rough/transmission MSE
became .01595779694/.01680816282, 6.45%/4.29% worse than the respective recent
128-limit controls. CPU images did not change. The reason is a unit mismatch,
not proof that additional CPU training has no effect: CPU checks OpenPGL field
iteration count, while Metal checks trained camera samples.

New debug diagnostics in `PathTrace::guiding_update_structures()` record actual
camera batches, before/after CPU field iterations, limit, and observation count.
A fresh CPU 512-SPP rough-glass control with limit128 confirms 128 field updates
covering camera samples0–508 (509 samples); the final three samples are frozen.
Its image is bit-for-bit identical to the prior control. Metal limit128 publishes
at samples1,2,4,8,16,32,64,128 and then freezes. Thus prior comparisons remain
valid equal-render-SPP measurements, but identical numeric training settings
must not be described as identical training work.

The panel now says `Training Updates` on CPU and `Training Camera Samples` on
active Metal, with an explanatory tooltip. CPU sampling/stopping behavior and
saved setting values are unchanged. Runtime UI drawing checks pass for CPU and
an enabled Metal device; the first fixture had no enabled Metal device and
correctly followed the CPU fallback, then the fixture was corrected. Evidence:
`training-units/report.json`, `/tmp/cycles-guiding-training-units-ui2.log`, and
`/tmp/cycles-guiding-training-units-cpu128.log`. The reusable material benchmark
now records training units and extracts actual counts when diagnostics exist.

A host-scheduling candidate then published every four camera samples after
startup. All four 512-SPP GPU trials passed exact publication checks. At training
limit128, rough-glass MSE worsened7.27% and transmission improved3.09%. At limit512,
rough improved2.55% and transmission worsened0.897%, relative to the corresponding
coarse-cadence runs. None reached CPU quality. This is a mixed single-seed result,
so the candidate was rejected and the original geometric/64-sample cadence
restored. Raw evidence is in `frequent-update-candidate`. The restored build and
install logs are `/tmp/cycles-guiding-frequent-update-restore-{build,install}.log`.
The UI/diagnostic improvements remain; no sampling-speed or quality gain is claimed.
# 4096-SPP power-MIS material convergence check

The installed pre-signed-volume baseline completed all 12 serial material renders
at exactly 4096 camera samples, seeds 101 and 211. Each BDPT render emitted exactly
16,777,216 auxiliary light paths with contiguous camera-sample coverage. The
primary references remain the existing two independent 8192-SPP CPU-guided
renders per scene. No image normalization, reference change, or time matching
was applied. Concurrent component verification makes the recorded times unsuitable
for warmed performance acceptance.

| Scene | CPU-guided mean MSE | Metal PT / CPU MSE | Metal BDPT / CPU MSE |
| --- | ---: | ---: | ---: |
| Rough glass | .001479267981 | 1.43194065 | 1.02674525 |
| Transmission | .001535597422 | 1.44271275 | 1.08761958 |

The CPU-guided quality requirement remains unmet. The earlier BDPT advantage at
512 SPP does not persist in these 4096-SPP means. BDPT block-region mean offsets
relative to the CPU-guided reference are +.4758% and +.6856% for rough glass,
and +.8638% and +.6658% for transmission. These persistent positive offsets need
transport investigation; two seeds alone are not proof of a particular bias.
Raw images, commands, training counts, work counts, reference provenance, and
all regional statistics are in `build/metal-guiding-tests/mis-power-material-4096/`.

The separate contribution-moment spatial-split candidate was rejected after six
512-SPP renders. Its three-seed mean changes versus the matching Metal PT baseline
were about -.42% MSE for rough glass and +.38% for transmission, with substantial
seed variation. It did not close the CPU gap and is no longer active. See
`doc/cycles_metal_guiding_spatial_split_plan.md` for preserved results and the
restoration/verification record.
