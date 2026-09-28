# Cycles Render → Diffraction

The Render properties contain one Diffraction panel with independent switches:

- **Enable Diffraction Effects**: master gate for material diffraction, source coherence, coherent connections and Glass polarization.
- **Material Diffraction**: disables material Diffraction Weight inputs, including linked weights, while retaining each native carrier.
- **Coherent Interference**: disables authored light coherence groups and coherent specular connections together; lighting becomes incoherent.
- **Glass Polarization**: disables Glass Polarizer inputs while retaining the native Glass substrate.

The Coherent Specular Connections settings are a child panel. Pitch, relief depth, duty cycle, tangent and filter angle remain material-node settings. Source group, wavelength, phase and coherence length remain light-data settings. The panel does not invent a global quality override.

The new switches default to enabled to preserve existing authored scenes. They only permit features: ordinary scenes with zero diffraction weights, zero coherence groups and unchecked polarizers retain native behavior. The existing coherent-connections checkbox still defaults to disabled.

The switches act on the imported Cycles shader graph before either SVM or OSL compilation. Disabled linked feature sockets are disconnected on that transient graph and set to their native defaults. The original Blender nodes, socket links and light settings are unchanged, saved normally, and restored by re-import when re-enabled. A setting change forces shader and light synchronization, including in persistent sessions.

The dedicated Smooth Diffraction node has no pre-existing native counterpart. When material diffraction is disabled, it becomes a smooth native Glass interface for a lossless substrate or a physical native conductor for an absorbing substrate, using its authored reference substrate index relative to Upper IOR. This removes its grating cache and wavelength-dependent diffraction. A spectral optical-constant table is not claimed to have an equivalent broadband native substrate; the OFF carrier uses the stored reference index.

These are Blender scene controls. Direct Cycles API users continue to control the material nodes and source groups explicitly. The switches do not expand the supported geometry or material domains of enabled coherent transport.
