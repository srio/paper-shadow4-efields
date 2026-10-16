"""
shadow4 phase plate (transmitted beam of a perfect crystal in Bragg geometry), compared with crystalpy.

Reproduces the Bragg-transmission case of main11 (Fig. hirano_intensity (b) and hirano_phase (b), after
Hirano et al., NIM A 336 (1993) 343): Si 220, 50 um, 10 keV.

1) scan: a vertically divergent beam is traced through the plate set at the corrected Bragg angle (f_central=1,
   no detuning). Each ray probes a different deviation from the Bragg angle, obtained from the ray direction in
   the crystal frame (footprint). The ray intensities |t_s|^2, |t_p|^2 and the phase difference phi_p - phi_s are
   compared with crystalpy (calculation_method=1, Guigay, as used by shadow4, and calculation_method=0,
   Zachariasen, as used for the figures in main11).
2) phase plate: a pencil beam (sigma polarized) through the plate detuned by -5 Darwin half widths (f_detune=2),
   with the plate rotated around the beam by alpha=+-45 deg: output Stokes parameters.

Abscissas are in units of the Darwin HALF-width (crystalpy darwinHalfwidthS), as in main11.
"""
import numpy

from syned.beamline.element_coordinates import ElementCoordinates
from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
from shadow4.beamline.optical_elements.phase_plate_crystals.s4_plane_phase_plate_crystal import \
    S4PlanePhasePlateCrystal, S4PlanePhasePlateCrystalElement

from crystalpy.diffraction.DiffractionSetupXraylib import DiffractionSetupXraylib
from crystalpy.diffraction.GeometryType import BraggTransmission
from crystalpy.diffraction.Diffraction import Diffraction
from crystalpy.util.ComplexAmplitudePhoton import ComplexAmplitudePhoton

ENERGY = 10000.0
THICKNESS = 50e-6
DETUNE_HDW = -5.0    # detuning for the phase plate, in Darwin half widths


def get_source(polarization_degree=1.0, vdiv=0.0, nrays=2000):
    s = SourceGeometrical(name='source', nrays=nrays, seed=5676561)
    s.set_spatial_type_point()
    s.set_depth_distribution_off()
    if vdiv == 0:
        s.set_angular_distribution_flat(hdiv1=0, hdiv2=0, vdiv1=0, vdiv2=0)
    else:
        s.set_angular_distribution_uniform(hdiv1=-1e-12, hdiv2=1e-12, vdiv1=-vdiv, vdiv2=vdiv)
    s.set_energy_distribution_singleline(ENERGY)
    s.set_polarization(polarization_degree=polarization_degree, phase_diff=0.0, coherent_beam=1)
    return s.get_beam()


def trace_plate(beam, alpha=0.0, f_detune=0, detune_half_darwin_widths=0.0):
    oe = S4PlanePhasePlateCrystal(name='Si 220 phase plate', material='Si',
                                  miller_index_h=2, miller_index_k=2, miller_index_l=0,
                                  thickness=THICKNESS,
                                  f_central=1, f_phot_cent=0, phot_cent=ENERGY,
                                  f_detune=f_detune, detune_half_darwin_widths=detune_half_darwin_widths,
                                  material_constants_library_flag=0)
    coordinates = ElementCoordinates(p=1.0, q=1.0, angle_azimuthal=alpha)
    el = S4PlanePhasePlateCrystalElement(optical_element=oe, coordinates=coordinates, input_beam=beam)
    return el.trace_beam()


def crystalpy_scan(setup, deviations, calculation_method):
    ts = numpy.zeros(deviations.size, dtype=complex)
    tp = numpy.zeros_like(ts)
    for i, d in enumerate(deviations):
        photon = ComplexAmplitudePhoton(energy_in_ev=ENERGY,
                                        direction_vector=setup.vectorIncomingPhotonDirection(ENERGY, d, angle_center_flag=1),
                                        Esigma=1.0, Epi=1.0)
        c = Diffraction.calculateDiffractedComplexAmplitudes(setup, photon, calculation_method=calculation_method, is_thick=0)
        ts[i] = complex(numpy.ravel(c["S"])[0])
        tp[i] = complex(numpy.ravel(c["P"])[0])
    return ts, tp


def wrap(phase_deg):
    return (phase_deg + 180.0) % 360.0 - 180.0


if __name__ == "__main__":
    import os
    SHOW_PLOTS = 1

    setup = DiffractionSetupXraylib(geometry_type=BraggTransmission(), crystal_name="Si", thickness=THICKNESS,
                                    miller_h=2, miller_k=2, miller_l=0, asymmetry_angle=0.0, azimuthal_angle=0.0)
    theta_bc = setup.angleBraggCorrected(ENERGY)
    half_dw = setup.darwinHalfwidthS(ENERGY)
    print("Si 220, %g keV, %g um: corrected Bragg angle %.5f deg, Darwin half-width (sigma) %.3f urad" %
          (1e-3 * ENERGY, 1e6 * THICKNESS, numpy.degrees(theta_bc), 1e6 * half_dw))

    #
    # 1) scan with a divergent beam (one ray = one deviation)
    #
    beam_s, foot_s = trace_plate(get_source(1.0, vdiv=10 * half_dw))
    beam_p, foot_p = trace_plate(get_source(0.0, vdiv=10 * half_dw))
    vz = foot_s.get_column(6)                       # ray direction in the crystal frame
    x_s4 = (numpy.arcsin(numpy.abs(vz)) - theta_bc) / half_dw
    i = numpy.argsort(x_s4)
    x_s4 = x_s4[i]
    IS = beam_s.get_column(23)[i]
    IP = beam_p.get_column(23)[i]
    # phi_p - phi_s; column 39 is phi_s - phi_p. Use a 45-deg linear input to get both phases on each ray.
    beam_45, _ = trace_plate(get_source(0.5, vdiv=10 * half_dw))
    DPHI = wrap(-numpy.degrees(beam_45.get_column(39)[i]))

    x = numpy.linspace(-10, 10, 2001)
    ts1, tp1 = crystalpy_scan(setup, x * half_dw, calculation_method=1)
    ts0, tp0 = crystalpy_scan(setup, x * half_dw, calculation_method=0)
    # crystalpy evaluated at the exact deviation of each ray
    ts_r, tp_r = crystalpy_scan(setup, x_s4 * half_dw, calculation_method=1)
    print("rays: %d, max |I_s(shadow4) - I_s(crystalpy)| = %.2e, max |I_p(shadow4) - I_p(crystalpy)| = %.2e, "
          "max |dphi(shadow4) - dphi(crystalpy)| = %.2e deg" % (x_s4.size,
          numpy.abs(IS - numpy.abs(ts_r) ** 2).max(), numpy.abs(IP - numpy.abs(tp_r) ** 2).max(),
          numpy.abs(wrap(DPHI - numpy.degrees(numpy.angle(tp_r / ts_r)))).max()))

    #
    # 2) phase plate detuned by DETUNE_HDW Darwin half widths, rotated by alpha = +-45 deg
    #
    ts, tp = crystalpy_scan(setup, numpy.array([DETUNE_HDW * half_dw]), calculation_method=1)
    ts, tp = ts[0], tp[0]
    print("\nPhase plate at %g Darwin half widths from Bragg:" % DETUNE_HDW)
    print("   crystalpy (Guigay):  |t_s|^2 = %.4f, |t_p|^2 = %.4f, phi_p - phi_s = %.2f deg" %
          (abs(ts) ** 2, abs(tp) ** 2, numpy.degrees(numpy.angle(tp / ts))))
    b_s, _ = trace_plate(get_source(1.0), f_detune=2, detune_half_darwin_widths=DETUNE_HDW)
    b_p, _ = trace_plate(get_source(0.0), f_detune=2, detune_half_darwin_widths=DETUNE_HDW)
    b_45, _ = trace_plate(get_source(0.5), f_detune=2, detune_half_darwin_widths=DETUNE_HDW)
    print("   shadow4 (alpha=0):   |t_s|^2 = %.4f, |t_p|^2 = %.4f, phi_p - phi_s = %.2f deg" %
          (b_s.get_column(23)[0], b_p.get_column(23)[0], -numpy.degrees(b_45.get_column(39)[0])))
    for a in [45, -45]:
        b, _ = trace_plate(get_source(1.0), alpha=numpy.radians(a), f_detune=2, detune_half_darwin_widths=DETUNE_HDW)
        print("   shadow4 (alpha=%+d deg, sigma input): transmission = %.4f, P1 = %+.4f, P2 = %+.4f, P3 = %+.4f" %
              (a, b.get_column(30)[0] / get_source(1.0).get_column(30)[0],
               b.get_column(42)[0], b.get_column(43)[0], b.get_column(44)[0]))

    #
    # plots
    #
    import matplotlib
    if not SHOW_PLOTS:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(x, numpy.abs(ts1) ** 2, '-', color='C0', lw=1, label=r'$\sigma$ crystalpy')
    ax.plot(x, numpy.abs(tp1) ** 2, '--', color='C1', lw=1, label=r'$\pi$ crystalpy')
    ax.plot(x_s4[::10], IS[::10], '.', color='C0', ms=4, label=r'$\sigma$ shadow4')
    ax.plot(x_s4[::10], IP[::10], '.', color='C1', ms=4, label=r'$\pi$ shadow4')
    ax.axvline(DETUNE_HDW, color='gray', ls=':', lw=1)
    ax.set_xlabel('deviation [Darwin half-widths]')
    ax.set_ylabel('Transmittance')
    ax.set_title('(a)', loc='left')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "phase_plate_intensity.png"), dpi=200)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(x, wrap(numpy.degrees(numpy.angle(tp1 / ts1))), '-', color='k', lw=1, label='crystalpy (Guigay)')
    ax.plot(x, wrap(numpy.degrees(numpy.angle(tp0 / ts0))), ':', color='gray', lw=1, label='crystalpy (Zachariasen)')
    ax.plot(x_s4[::10], DPHI[::10], '.', color='C2', ms=4, label='shadow4')
    ax.axvline(DETUNE_HDW, color='gray', ls=':', lw=1)
    ax.set_xlabel('deviation [Darwin half-widths]')
    ax.set_ylabel(r'$\phi_\pi - \phi_\sigma$ [deg]')
    ax.set_title('(b)', loc='left')
    ax.set_ylim(-190, 190)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "phase_plate_phase.png"), dpi=200)
    print("\nFiles written to disk: %s/phase_plate_intensity.png, %s/phase_plate_phase.png" % (figures_dir, figures_dir))
    if SHOW_PLOTS:
        plt.show()
