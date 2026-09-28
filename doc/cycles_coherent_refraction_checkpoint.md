# Refraction work stopped at the authorized usage ceiling

**Superseded working-tree status:** at the user's subsequent request, the unfinished edits described below were archived and removed from the active build. The restored repository now passes the complete CPU/OSL/Metal build and launcher/demo-loading smoke. See [the current handoff](cycles_diffraction_handoff.md) and [restoration archive](../build/quarantine/streamed_convex_refraction_unvalidated/README.md). The text below preserves the earlier interruption context; it no longer describes the active source tree.

The account usage tool reported 99% weekly usage on 2026-09-28, reaching the user's explicit ceiling. The implementation agent was interrupted. No further build or GPU validation was started.

The immutable v45 package remains the validated delivery. The working tree now contains **unfinished, unvalidated refraction implementation edits**, including `scene/coherent_convex_mesh.h`, `scene/scene.cpp`, `scene/CMakeLists.txt`, `kernel/light/coherent_facet_integrator.h`, `kernel/light/coherent_facet_stream.h`, `kernel/light/coherent_history.h`, `kernel/light/coherent_history_kernel.h`, and `kernel/integrator/bidirectional.h` under `intern/cycles`. Do not package or describe this worktree as an accepted successor to v45.

Independent staged tests are in `tests/output/diffraction/coherent_closed_slab_pending_v46`; production acceptance has not run. Their scene budgets, saved-camera quadrature, omission controls and predeclared thresholds are recorded with the fixtures. The earlier exact unsupported-host probe in `coherent_closed_slab_pending_v45_exact` is evidence of v45's support boundary, not evidence for the new edits.

Resume only with renewed usage authorization or restored allowance within the user's ceiling. Inspect the interrupted code first, complete volume validation and typed entry/exit ownership consistently, then build and run independent CPU/Metal numerical and PT/BDPT/guiding checks. Preserve the existing v45 package, failed experiments and prospective thresholds. The full original goal remains incomplete.
