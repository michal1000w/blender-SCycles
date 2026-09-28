#!/usr/bin/env python3
"""Independent scalar two-source coherence references, NOT a Cycles renderer."""
import argparse
import json
from pathlib import Path
import numpy as np


def fields(x, separation, distance):
    """Stable spherical-wave path difference; suppress the irrelevant common phase."""
    left = np.hypot(x + separation / 2, distance)
    right = np.hypot(x - separation / 2, distance)
    # Difference of squared distances, factored to avoid subtracting nearly equal lengths.
    opd = 2 * x * separation / (left + right)
    return 1 / left, 1 / right, opd


def intensity(x, wavelength=532e-9, separation=100e-6, distance=1,
              phase=0, coherence_length=np.inf, correlation=1):
    """Gaussian wavenumber spectrum: coherence_length means 1 / sigma_k, in metres."""
    if wavelength <= 0 or separation <= 0 or distance <= 0 or coherence_length <= 0:
        raise ValueError('Positive physical lengths are required')
    if not 0 <= correlation <= 1:
        raise ValueError('Correlation must lie in [0, 1]')
    a, b, opd = fields(np.asarray(x, dtype=float), separation, distance)
    gamma = correlation * np.exp(-0.5 * (opd / coherence_length) ** 2)
    return a*a + b*b + 2*a*b*gamma*np.cos(2*np.pi*opd/wavelength - phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    x = np.linspace(-.02, .02, 2001)
    wavelength, separation, distance = 532e-9, 100e-6, 1.0
    parameters = dict(wavelength=wavelength, separation=separation, distance=distance)
    profiles = {
        'coherent_phase_0': intensity(x, **parameters),
        'coherent_phase_pi': intensity(x, phase=np.pi, **parameters),
        'incoherent': intensity(x, correlation=0, **parameters),
        'partial_1um': intensity(x, coherence_length=1e-6, **parameters),
        'partial_2um': intensity(x, coherence_length=2e-6, **parameters),
        'red_633nm': intensity(x, wavelength=633e-9),
    }
    profiles['two_wavelengths'] = .5*(profiles['coherent_phase_0'] + profiles['red_633nm'])
    checks = {}
    def check(name, value, limit):
        checks[name] = {'error': float(value), 'limit': limit, 'passed': bool(value <= limit)}
        if not checks[name]['passed']:
            raise AssertionError((name, value, limit))
    a, b, opd = fields(x, separation, distance)
    check('constructive_on_axis', abs(profiles['coherent_phase_0'][1000] - 4*a[1000]**2), 1e-12)
    check('destructive_on_axis', abs(profiles['coherent_phase_pi'][1000]), 1e-12)
    check('incoherent_sum', np.max(abs(profiles['incoherent']-(a*a+b*b))), 1e-12)
    check('phase_shift_complement', np.max(abs(profiles['coherent_phase_0']+
          profiles['coherent_phase_pi']-2*profiles['incoherent'])), 1e-12)
    check('nonnegative', max(0, -min(float(v.min()) for v in profiles.values())), 1e-12)
    # Independent complex-field quadrature of a Gaussian wavenumber spectrum.
    nodes, weights = np.polynomial.hermite.hermgauss(64)
    for length in [1e-6, 2e-6, 10e-6]:
        k = 2*np.pi/wavelength + np.sqrt(2)*nodes/length
        u = a[None, :]*np.exp(1j*k[:, None]*opd[None, :]) + b[None, :]
        numeric = np.sum(weights[:, None]*abs(u)**2, axis=0)/np.sqrt(np.pi)
        expected = intensity(x, coherence_length=length, **parameters)
        check(f'complex_spectrum_quadrature_{length:g}', np.max(abs(numeric-expected)), 2e-12)
    # Uniform phase integration independently recovers the incoherent limit.
    phases = 2*np.pi*np.arange(128)/128
    u = a[None, :]*np.exp(1j*2*np.pi*opd[None, :]/wavelength) + b[None, :]*np.exp(1j*phases[:, None])
    check('random_phase_incoherent_limit', np.max(abs(np.mean(abs(u)**2, axis=0)-profiles['incoherent'])), 2e-12)
    for scale in [.01, 10, 100]:
        scaled = intensity(x*scale, wavelength=wavelength*scale,
                           separation=separation*scale, distance=distance*scale)*scale**2
        check(f'unit_rescaling_{scale:g}', np.max(abs(scaled-profiles['coherent_phase_0'])), 1e-12)
    # Far-field fringe period is a separate paraxial approximation, not the exact oracle.
    farfield = 2/distance**2*(1+np.cos(2*np.pi*separation*x/(wavelength*distance)))
    check('paraxial_fixture_agreement', np.max(abs(farfield-profiles['coherent_phase_0'])), .012)
    # Solve the spherical path-difference condition using bisection, independently
    # of intensity evaluation. Envelope variation does not define the phase fringe.
    lo, hi = 0., .02
    for _ in range(60):
        mid=(lo+hi)/2
        delta=np.hypot(mid+separation/2,distance)-np.hypot(mid-separation/2,distance)
        if delta < wavelength: lo=mid
        else: hi=mid
    measured=(lo+hi)/2
    check('first_phase_fringe_position', abs(measured-wavelength*distance/separation), 1e-7)
    # Orthogonal polarizations eliminate the cross term in a vector-field sum.
    vector_power=abs(a*np.exp(1j*2*np.pi*opd/wavelength))**2 + abs(b)**2
    check('orthogonal_polarization_limit', np.max(abs(vector_power-profiles['incoherent'])), 1e-12)
    cases=[]
    for name, phase, length, correlation, wl in [
        ('coherent_phase_0',0,None,1,wavelength), ('coherent_phase_pi',float(np.pi),None,1,wavelength),
        ('incoherent',0,None,0,wavelength), ('partial_1um',0,1e-6,1,wavelength),
        ('partial_2um',0,2e-6,1,wavelength), ('red_633nm',0,None,1,633e-9),
    ]:
        cases.append(dict(name=name, wavelength_m=wl, separation_m=separation,
                          detector_distance_m=distance, phase_radians=phase,
                          coherence_length_m=length, correlation=correlation))
    report={'scope':'Independent scalar two-point-source references, NOT Cycles feature output',
            'renderer_support':'unimplemented', 'coherence_length_definition':'1/sigma_k; Gaussian wavenumber spectrum',
            'field_model':'Equal scalar spherical source amplitudes 1/r; no surfaces, polarization coupling or detector BSDF',
            'first_phase_fringe_m':measured, 'paraxial_fringe_period_m':wavelength*distance/separation,
            'checks':checks, 'cases':cases}
    (args.output/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.output/'profiles.npz',x_m=x,**profiles)
    np.savetxt(args.output/'profiles.csv',np.column_stack([x,*profiles.values()]),delimiter=',',
               header=','.join(['x_m',*profiles]),comments='')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    groups=[('Source phase',['coherent_phase_0','coherent_phase_pi']),
            ('Temporal coherence',['coherent_phase_0','partial_1um','partial_2um','incoherent']),
            ('Wavelength',['coherent_phase_0','red_633nm','two_wavelengths']),
            ('Incoherent / orthogonal-polarization limit',['coherent_phase_0','incoherent'])]
    for ax,(title,names) in zip(axes.flat,groups):
        for name in names: ax.plot(x*1000,profiles[name],label=name.replace('_',' '),lw=1.2)
        ax.set(title=title,xlabel='Detector position (mm)',ylabel='Scalar irradiance (relative units)',ylim=(-.05,4.15))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Independent coherence references — NOT Cycles renders',fontsize=16)
    fig.savefig(args.output/'reference_profiles.png',dpi=160)
    plt.close(fig)
    print(json.dumps({'checks':len(checks),'passed':all(c['passed'] for c in checks.values()),
                      'fringe_spacing_mm':measured*1000}))


if __name__ == '__main__':
    main()
