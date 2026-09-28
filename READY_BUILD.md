# Ready repository build

Double-click **Launch Blender.command**, or run:

```sh
./Launch\ Blender.command
```

This launches `install/Blender.app` with its matching installed add-on, OSL shaders and Metal runtime headers. To rebuild and install:

```sh
./compile.sh -j4 --no-fetch-libraries
```

The current installation passed `./compile.sh -j4 --no-fetch-libraries`, including installed diffraction/coherent/polarization header checks, add-on synchronization, Glass OSL synchronization and background startup. The launcher starts the installed binary; an older running Blender instance must be restarted to use the new UI.

The unfinished closed-Glass refraction experiment was removed from the active build and archived, not completed. Ready diffraction, polarizer and bounded/streamed mirror features remain available. Use Fast diffraction for the validated practical workflow; Realistic remains experimental. Existing unsupported configurations retain explicit guards.

- [Extensive handoff, evidence, limitations and continuation guidelines](doc/cycles_diffraction_handoff.md)
- [Versioned v45 delivery report and demo galleries](doc/cycles_diffraction_delivery_v45.md)
- [Archived unfinished work and restoration proof](build/quarantine/streamed_convex_refraction_unvalidated/README.md)

The immutable v45 application remains available in `build/diffraction_delivery_20260928_coherent_facets_2r_v45/` as a separate validated fallback. The repository launcher uses the current installation and does not depend on that archive. The whole original research/transport goal remains incomplete; this checkout has been restored to a buildable, usable validated scope.
