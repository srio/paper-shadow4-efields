"""
shadow4 phase plate: scan of the Stokes parameters of the transmitted beam versus the detuning angle, compared with
the measurements at the DESY PETRA III P09 beamline (main6, Section 5, "Experimental benchmarking").

System: Si 111 plate in symmetric Bragg geometry, thickness 10 um, photon energy 2838 eV (theta_B ~ 44 deg),
transmitted (forward-diffracted) beam. The incident beam is linearly polarized in the horizontal plane,
S_in = (1, 1, 0, 0), and the plate is rotated around the beam by alpha = 45 deg (azimuthal angle).

For each detuning angle, a pencil beam is traced through the plate (S4PlanePhasePlateCrystal, f_detune=1). An empty
element rotated by -alpha brings the reference frame back to the laboratory frame, and the Stokes parameters are
calculated in the right-handed laboratory basis (x, -z) (beam along y), as the measured P1 (H/V) and P2 (+-45 deg).
Note that the Stokes parameters in the SHADOW4 columns 30-33, 42-44 refer to the sigma direction of each ray, which
after the plate is the sigma direction of the diffraction plane.

Notes on the comparison:
- the crystal data are from DABAX with the Windt f1f2 file, as in the crystalpy comparison of main6;
- the angle axis of the measurements is reversed (detuning = -angle_exp), and alpha = +45 deg gives the measured
  signs of P1 and P2;
- the experimental P3 is reconstructed as +sqrt(1 - P1^2 - P2^2), so its sign is not measured, and it is compared
  with |P3|;
- the calculation is not convolved with the angular and energy distributions of the beam.
"""
import numpy

from syned.beamline.element_coordinates import ElementCoordinates
from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
from shadow4.beamline.optical_elements.phase_plate_crystals.s4_plane_phase_plate_crystal import \
    S4PlanePhasePlateCrystal, S4PlanePhasePlateCrystalElement
from shadow4.beamline.optical_elements.ideal_elements.s4_empty import S4Empty, S4EmptyElement
from dabax.dabax_xraylib import DabaxXraylib

ENERGY = 2838.0         # eV
THICKNESS = 10e-6       # m
ALPHA = 45.0            # deg, rotation of the plate around the beam
ARCSEC = numpy.radians(1.0 / 3600)
DABAX = DabaxXraylib(file_f1f2="f1f2_Windt.dat")


def get_source():
    # pencil beam (a single ray), linear horizontal polarization
    s = SourceGeometrical(name='pencil', nrays=1, seed=5676561)
    s.set_spatial_type_point()
    s.set_depth_distribution_off()
    s.set_angular_distribution_flat(hdiv1=0, hdiv2=0, vdiv1=0, vdiv2=0)
    s.set_energy_distribution_singleline(ENERGY)
    s.set_polarization(polarization_degree=1.0, phase_diff=0.0, coherent_beam=1)
    return s.get_beam()


def trace(beam, detune_angle, alpha_deg=ALPHA):
    oe = S4PlanePhasePlateCrystal(name='Si 111 phase plate', material='Si',
                                  miller_index_h=1, miller_index_k=1, miller_index_l=1,
                                  thickness=THICKNESS,
                                  f_central=1, f_phot_cent=0, phot_cent=ENERGY,
                                  f_detune=1, detune_angle=detune_angle,
                                  material_constants_library_flag=1, dabax=DABAX)
    el = S4PlanePhasePlateCrystalElement(optical_element=oe,
                                         coordinates=ElementCoordinates(p=1.0, q=0.0, angle_azimuthal=numpy.radians(alpha_deg)),
                                         input_beam=beam)
    out, _ = el.trace_beam()
    # back to the laboratory frame
    el = S4EmptyElement(optical_element=S4Empty(),
                        coordinates=ElementCoordinates(p=0.0, q=1.0, angle_radial=0.0, angle_radial_out=numpy.pi,
                                                       angle_azimuthal=-numpy.radians(alpha_deg)),
                        input_beam=out)
    out, _ = el.trace_beam()
    return out


def stokes_lab(beam):
    """Normalized Stokes parameters (S0, P1, P2, P3) of the first ray, in the laboratory basis (x, -z)."""
    E = beam.get_columns([7, 8, 9]) * numpy.exp(1j * beam.get_column(14)) + \
        beam.get_columns([16, 17, 18]) * numpy.exp(1j * beam.get_column(15))
    E1, E2 = E[0, 0], -E[2, 0]
    S0 = numpy.abs(E1) ** 2 + numpy.abs(E2) ** 2
    return S0, \
           (numpy.abs(E1) ** 2 - numpy.abs(E2) ** 2) / S0, \
           2 * numpy.real(E1 * numpy.conj(E2)) / S0, \
           2 * numpy.imag(numpy.conj(E1) * E2) / S0


def get_experiment(fname="2838_10microns_Si111_111SB_Stokes.dat"):
    a = numpy.loadtxt(fname, skiprows=5)
    angle_arcsec, P1, P2 = a[:, 0], a[:, 3], a[:, 5]
    P3_abs = numpy.sqrt(numpy.clip(1.0 - P1 ** 2 - P2 ** 2, 0.0, None))  # sign not measured
    return angle_arcsec, P1, P2, P3_abs


if __name__ == "__main__":
    import os
    SHOW_PLOTS = 1
    here = os.path.dirname(os.path.abspath(__file__))

    beam = get_source()

    angle_arcsec = numpy.linspace(-300, 300, 601)    # in the experimental axis
    detune = -angle_arcsec * ARCSEC                  # detuning angle of the crystal (reversed axis)

    stokes = numpy.array([stokes_lab(trace(beam, d)) for d in detune])
    S0, P1, P2, P3 = stokes.T

    xe, P1e, P2e, P3e = get_experiment(os.path.join(here, "2838_10microns_Si111_111SB_Stokes.dat"))
    se = numpy.array([stokes_lab(trace(beam, -x * ARCSEC)) for x in xe])
    print("Si 111, %g eV, %g um, alpha=%g deg, %d experimental points:" % (ENERGY, 1e6 * THICKNESS, ALPHA, xe.size))
    print("   rms(P1 - P1exp) = %.3f, rms(P2 - P2exp) = %.3f, rms(|P3| - |P3|exp) = %.3f" % (
          numpy.sqrt(numpy.mean((se[:, 1] - P1e) ** 2)), numpy.sqrt(numpy.mean((se[:, 2] - P2e) ** 2)),
          numpy.sqrt(numpy.mean((numpy.abs(se[:, 3]) - P3e) ** 2))))
    i = numpy.argmax(numpy.abs(P3))
    print("   max |P3| = %.3f (P3 = %+.3f) at %+.1f arcsec (detuning %+.1f urad), transmission S0 = %.3f" % (
          numpy.abs(P3[i]), P3[i], angle_arcsec[i], 1e6 * detune[i], S0[i]))

    import matplotlib
    if not SHOW_PLOTS:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(angle_arcsec, P1, '-', color='k', lw=1, label=r'$P_1$ shadow4')
    ax.plot(angle_arcsec, P2, '-', color='C0', lw=1, label=r'$P_2$ shadow4')
    ax.plot(angle_arcsec, P3, '-', color='C3', lw=1, label=r'$P_3$ shadow4')
    ax.plot(angle_arcsec, numpy.abs(P3), ':', color='C3', lw=1, label=r'$|P_3|$ shadow4')
    ax.plot(xe, P1e, 'o', color='k', ms=4, mfc='none', label=r'$P_1$ exp')
    ax.plot(xe, P2e, 's', color='C0', ms=4, mfc='none', label=r'$P_2$ exp')
    ax.plot(xe, P3e, '^', color='C3', ms=4, mfc='none', label=r'$\sqrt{1-P_1^2-P_2^2}$ exp')
    ax.set_xlabel('angle [arcsec] (detuning = -angle)')
    ax.set_ylabel('normalized Stokes parameters')
    ax.set_title(r'Si 111, %d $\mu$m, %g eV, $\alpha$=%g$^\circ$' % (round(1e6 * THICKNESS), ENERGY, ALPHA))
    ax.set_xlim(-300, 300)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(here, "..", "figures", "phase_plate_stokes_scan.png"), dpi=200)
    print("File written to disk: %s" % os.path.join(here, "..", "figures", "phase_plate_stokes_scan.png"))
    if SHOW_PLOTS:
        plt.show()
