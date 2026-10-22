"""
shadow4 phase plate: scan of the Stokes parameters of the transmitted beam versus the detuning angle, compared with
the measurements at the DESY PETRA III P09 beamline (main6, Section 5, "Experimental benchmarking").

System: Si 111 plate in symmetric Bragg geometry, thickness 10 um, photon energy 2838 eV (theta_B ~ 44 deg),
transmitted (forward-diffracted) beam. The incident beam is linearly polarized in the horizontal plane,
S_in = (1, 1, 0, 0), and the plate is rotated around the beam by alpha = 45 deg (azimuthal angle).

For each detuning angle, the beam is traced through the plate (S4PlanePhasePlateCrystal, f_detune=1). The beam is
either a pencil beam (a single ray) or the beam of the P09 undulator (U32, 2 m, monochromatic at 2838 eV, PETRA III
high-beta electron beam). An empty element rotated by -alpha brings the reference frame back to the laboratory frame,
and the Stokes parameters, summed over the rays, are calculated in the right-handed laboratory basis (x, -z) (beam
along y), as the measured P1 (H/V) and P2 (+-45 deg).
Note that the Stokes parameters in the SHADOW4 columns 30-33, 42-44 refer to the sigma direction of each ray, which
after the plate is the sigma direction of the diffraction plane.

Notes on the comparison:
- the crystal data are from DABAX with the Windt f1f2 file, as in the crystalpy comparison of main6;
- the angle axis of the measurements is reversed (detuning = -angle_exp), and alpha = +45 deg gives the measured
  signs of P1 and P2;
- the experimental P3 is reconstructed as +sqrt(1 - P1^2 - P2^2), so its sign is not measured, and it is compared
  with |P3|;
- with the pencil beam, the calculation is not convolved with the angular and energy distributions of the beam;
  the undulator source includes the angular distribution, but not the energy bandwidth of the monochromator.
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

def get_source_undulator(nrays=5000):
    # electron beam
    from shadow4.sources.s4_electron_beam import S4ElectronBeam
    electron_beam = S4ElectronBeam(energy_in_GeV=6.0, energy_spread=0.001, current=0.1)
    electron_beam.set_sigmas_all(sigma_x=0.0001415, sigma_y=4.9e-06, sigma_xp=7.1e-06, sigma_yp=2.1e-06)
    electron_beam.set_dispersion_all(0, 0, 0, 0)

    # magnetic structure
    from shadow4.sources.undulator.s4_undulator import S4Undulator
    source = S4Undulator(
        K_vertical=2.381738,  # syned Undulator parameter
        period_length=0.0314,  # syned Undulator parameter
        number_of_periods=64,  # syned Undulator parameter
        emin=2837.9998557749605,  # Photon energy scan from energy (in eV)
        emax=2837.9998557749605,  # Photon energy scan to energy (in eV)
        ng_e=1,  # Photon energy scan number of points
        maxangle=3.0520561496869934e-05,  # Maximum radiation semiaperture in RADIANS
        ng_t=100,  # Number of points in angle theta
        ng_p=11,  # Number of points in angle phi
        ng_j=20,  # Number of points in electron trajectory (per period) for internal calculation only
        code_undul_phot='internal',  # internal, pysru, srw
        flag_emittance=1,  # when sampling rays: Use emittance (0=No, 1=Yes)
        flag_size=0,  # when sampling rays: 0=point,1=Gaussian,2=FT(Divergences)
        distance=100.0,  # distance to far field plane
        srw_range=0.05,  # for SRW backpropagation, the range factor
        srw_resolution=50,  # for SRW backpropagation, the resolution factor
        srw_semianalytical=0,  # for SRW backpropagation, use semianalytical treatement of phase
        magnification=0.05,  # for internal/wofry backpropagation, the magnification factor
        flag_backprop_recalculate_source=0,
        # for internal or pysru/wofry backpropagation: source reused (0) or recalculated (1)
        flag_backprop_weight=0,  # for internal or pysru/wofry backpropagation: apply Gaussian weight to amplitudes
        weight_ratio=0.5,  # for flag_backprop_recalculate_source=1, the ratio value sigma/window halfwidth
        flag_energy_spread=0,  # for monochromatod sources, apply (1) or not (0) electron energy spread correction
    )

    # light source
    from shadow4.sources.undulator.s4_undulator_light_source import S4UndulatorLightSource
    light_source = S4UndulatorLightSource(name='Undulator Light Source', electron_beam=electron_beam,
                                          magnetic_structure=source, nrays=nrays, seed=5676561)
    beam = light_source.get_beam()

    # test plot
    # from srxraylib.plot.gol import plot_scatter
    # rays = beam.get_rays()
    # plot_scatter(1e6 * rays[:, 0], 1e6 * rays[:, 2], title='(X,Z) in microns')
    return beam

def get_source_undulator_with_monochromator(nrays=5000):
    import numpy as np
    from dabax.dabax_xraylib import DabaxXraylib
    from shadow4.beamline.s4_beamline import S4Beamline

    beamline = S4Beamline()

    # electron beam
    from shadow4.sources.s4_electron_beam import S4ElectronBeam
    electron_beam = S4ElectronBeam(energy_in_GeV=6.0, energy_spread=0.001, current=0.1)
    electron_beam.set_sigmas_all(sigma_x=0.0001415, sigma_y=4.9e-06, sigma_xp=7.1e-06, sigma_yp=2.1e-06)
    electron_beam.set_dispersion_all(0, 0, 0, 0)

    # magnetic structure
    from shadow4.sources.undulator.s4_undulator import S4Undulator
    source = S4Undulator(
        K_vertical=2.381738,  # syned Undulator parameter
        period_length=0.0314,  # syned Undulator parameter
        number_of_periods=64,  # syned Undulator parameter
        emin=2836.9998557749605,  # Photon energy scan from energy (in eV)
        emax=2838.9998557749605,  # Photon energy scan to energy (in eV)
        ng_e=3,  # Photon energy scan number of points
        maxangle=3.0520561496869934e-05,  # Maximum radiation semiaperture in RADIANS
        ng_t=100,  # Number of points in angle theta
        ng_p=11,  # Number of points in angle phi
        ng_j=20,  # Number of points in electron trajectory (per period) for internal calculation only
        code_undul_phot='internal',  # internal, pysru, srw
        flag_emittance=1,  # when sampling rays: Use emittance (0=No, 1=Yes)
        flag_size=0,  # when sampling rays: 0=point,1=Gaussian,2=FT(Divergences)
        distance=100.0,  # distance to far field plane
        srw_range=0.05,  # for SRW backpropagation, the range factor
        srw_resolution=50,  # for SRW backpropagation, the resolution factor
        srw_semianalytical=0,  # for SRW backpropagation, use semianalytical treatement of phase
        magnification=0.05,  # for internal/wofry backpropagation, the magnification factor
        flag_backprop_recalculate_source=0,
        # for internal or pysru/wofry backpropagation: source reused (0) or recalculated (1)
        flag_backprop_weight=0,  # for internal or pysru/wofry backpropagation: apply Gaussian weight to amplitudes
        weight_ratio=0.5,  # for flag_backprop_recalculate_source=1, the ratio value sigma/window halfwidth
        flag_energy_spread=0,  # for monochromatod sources, apply (1) or not (0) electron energy spread correction
    )

    # light source
    from shadow4.sources.undulator.s4_undulator_light_source import S4UndulatorLightSource
    light_source = S4UndulatorLightSource(name='Undulator Light Source', electron_beam=electron_beam,
                                          magnetic_structure=source, nrays=nrays, seed=5676561)
    beam = light_source.get_beam()

    beamline.set_light_source(light_source)

    # optical element number XX
    boundary_shape = None

    from shadow4.beamline.optical_elements.crystals.s4_plane_crystal import S4PlaneCrystal
    optical_element = S4PlaneCrystal(name='Generic Crystal',
                                     boundary_shape=boundary_shape, material='Si',
                                     miller_index_h=1, miller_index_k=1, miller_index_l=1,
                                     f_bragg_a=False, asymmetry_angle=0.0,
                                     is_thick=1, thickness=0.001,
                                     f_central=1, f_phot_cent=0, phot_cent=2838.0,
                                     file_refl='HOPG2_20_v2.002',
                                     f_ext=0,
                                     material_constants_library_flag=0,
                                     # 0=xraylib,1=dabax,2=preprocessor v1,3=preprocessor v2
                                     method_efields_management=0,  # 0=new in S4; 1=like in S3; 2=S4 standard Jones
                                     dabax=None,  # used when material_constants_library_flag=1,
                                     )
    from syned.beamline.element_coordinates import ElementCoordinates
    coordinates = ElementCoordinates(p=1.0, q=0.1, angle_radial=0.7999271483081181, angle_azimuthal=0.0,
                                     angle_radial_out=0.7999271483081182)
    movements = None
    from shadow4.beamline.optical_elements.crystals.s4_plane_crystal import S4PlaneCrystalElement
    beamline_element = S4PlaneCrystalElement(optical_element=optical_element, coordinates=coordinates,
                                             movements=movements, input_beam=beam)

    beam, footprint = beamline_element.trace_beam()

    beamline.append_beamline_element(beamline_element)

    # optical element number XX
    boundary_shape = None

    from shadow4.beamline.optical_elements.crystals.s4_plane_crystal import S4PlaneCrystal
    optical_element = S4PlaneCrystal(name='Generic Crystal',
                                     boundary_shape=boundary_shape, material='Si',
                                     miller_index_h=1, miller_index_k=1, miller_index_l=1,
                                     f_bragg_a=False, asymmetry_angle=0.0,
                                     is_thick=1, thickness=0.001,
                                     f_central=1, f_phot_cent=0, phot_cent=2838.0,
                                     file_refl='HOPG2_20_v2.002',
                                     f_ext=0,
                                     material_constants_library_flag=0,
                                     # 0=xraylib,1=dabax,2=preprocessor v1,3=preprocessor v2
                                     method_efields_management=0,  # 0=new in S4; 1=like in S3; 2=S4 standard Jones
                                     dabax=None,  # used when material_constants_library_flag=1,
                                     )
    from syned.beamline.element_coordinates import ElementCoordinates
    coordinates = ElementCoordinates(p=0.1, q=1.0, angle_radial=0.7999271483081181, angle_azimuthal=3.141592653589793,
                                     angle_radial_out=0.7999271483081182)
    movements = None
    from shadow4.beamline.optical_elements.crystals.s4_plane_crystal import S4PlaneCrystalElement
    beamline_element = S4PlaneCrystalElement(optical_element=optical_element, coordinates=coordinates,
                                             movements=movements, input_beam=beam)

    beam, footprint = beamline_element.trace_beam()

    beamline.append_beamline_element(beamline_element)

    # optical element number XX

    from shadow4.beamline.optical_elements.ideal_elements.s4_empty import S4Empty
    optical_element = S4Empty(name='Empty Element')

    from syned.beamline.element_coordinates import ElementCoordinates
    coordinates = ElementCoordinates(p=0.0, q=0.0, angle_radial=0.0, angle_azimuthal=3.141592653589793,
                                     angle_radial_out=3.141592653589793)
    from shadow4.beamline.optical_elements.ideal_elements.s4_empty import S4EmptyElement
    beamline_element = S4EmptyElement(optical_element=optical_element, coordinates=coordinates, input_beam=beam)

    beam, footprint = beamline_element.trace_beam()

    beamline.append_beamline_element(beamline_element)

    # test plot
    if 0:
        from srxraylib.plot.gol import plot_scatter
        plot_scatter(beam.get_photon_energy_eV(nolost=1), beam.get_column(23, nolost=1),
                     title='(Intensity,Photon Energy)', plot_histograms=0)
        plot_scatter(1e6 * beam.get_column(1, nolost=1), 1e6 * beam.get_column(3, nolost=1), title='(X,Z) in microns')

    return beam
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
    """
    Stokes parameters of the beam in the laboratory basis (x, -z), summed over the good rays (the rays are assumed to
    be nearly parallel to y). Returns S0 (intensity per incident ray, i.e., the transmission for unit-intensity rays)
    and the normalized P1, P2, P3 (sum of S_k over sum of S0).
    """
    good = beam.get_column(10) > 0
    E = beam.get_columns([7, 8, 9]) * numpy.exp(1j * beam.get_column(14)) + \
        beam.get_columns([16, 17, 18]) * numpy.exp(1j * beam.get_column(15))
    E1, E2 = E[0, good], -E[2, good]
    S0 = (numpy.abs(E1) ** 2 + numpy.abs(E2) ** 2).sum()
    S1 = (numpy.abs(E1) ** 2 - numpy.abs(E2) ** 2).sum()
    S2 = (2 * numpy.real(E1 * numpy.conj(E2))).sum()
    S3 = (2 * numpy.imag(numpy.conj(E1) * E2)).sum()
    return S0 / beam.get_number_of_rays(), S1 / S0, S2 / S0, S3 / S0


def get_experiment(fname="2838_10microns_Si111_111SB_Stokes.dat"):
    a = numpy.loadtxt(fname, skiprows=5)
    angle_arcsec, P1, P2 = a[:, 0], a[:, 3], a[:, 5]
    P3_abs = numpy.sqrt(numpy.clip(1.0 - P1 ** 2 - P2 ** 2, 0.0, None))  # sign not measured
    return angle_arcsec, P1, P2, P3_abs


if __name__ == "__main__":
    import os
    SHOW_PLOTS = 1
    USE_UNDULATOR = 1
    # 0: pencil beam (a single ray), 1: P09 undulator (5000 rays), 2: 1: P09 undulator polychromatic + monochromator
    import sys
    if len(sys.argv) > 1: USE_UNDULATOR = int(sys.argv[1])
    here = os.path.dirname(os.path.abspath(__file__))

    if USE_UNDULATOR == 1:
        beam = get_source_undulator(nrays=5000)
        angle_arcsec = numpy.linspace(-300, 300, 241)    # in the experimental axis
        figure_file = "desy_p09_scan_undulator.png"
    elif USE_UNDULATOR == 2:
        beam = get_source_undulator_with_monochromator(nrays=5000)
        angle_arcsec = numpy.linspace(-300, 300, 241)    # in the experimental axis
        figure_file = "desy_p09_scan_undulator_polychromatic.png"
    else:
        beam = get_source()
        angle_arcsec = numpy.linspace(-300, 300, 601)
        figure_file = "desy_p09_scan_pencil_beam.png"
    print("Source beam calculated: %d rays" % beam.get_number_of_rays())
    S0_in = stokes_lab(beam)[0]   # intensity incident on the phase plate (per ray)

    detune = -angle_arcsec * ARCSEC                  # detuning angle of the crystal (reversed axis)

    stokes = numpy.array([stokes_lab(trace(beam, d)) for d in detune])
    S0, P1, P2, P3 = stokes.T
    DOP = numpy.sqrt(P1 ** 2 + P2 ** 2 + P3 ** 2)
    P3_as_exp = numpy.sqrt(numpy.clip(1.0 - P1 ** 2 - P2 ** 2, 0.0, None))   # as reconstructed in the experiment

    xe, P1e, P2e, P3e = get_experiment(os.path.join(here, "2838_10microns_Si111_111SB_Stokes.dat"))
    se = numpy.array([stokes_lab(trace(beam, -x * ARCSEC)) for x in xe])
    print("Si 111, %g eV, %g um, alpha=%g deg, %d experimental points:" % (ENERGY, 1e6 * THICKNESS, ALPHA, xe.size))
    print("   rms(P1 - P1exp) = %.3f, rms(P2 - P2exp) = %.3f, rms(|P3| - |P3|exp) = %.3f" % (
          numpy.sqrt(numpy.mean((se[:, 1] - P1e) ** 2)), numpy.sqrt(numpy.mean((se[:, 2] - P2e) ** 2)),
          numpy.sqrt(numpy.mean((numpy.abs(se[:, 3]) - P3e) ** 2))))
    print("   rms(sqrt(1-P1^2-P2^2) - |P3|exp) = %.3f" % (
          numpy.sqrt(numpy.mean((numpy.sqrt(numpy.clip(1 - se[:, 1] ** 2 - se[:, 2] ** 2, 0, None)) - P3e) ** 2))))
    print("   min DOP = %.3f at %+.1f arcsec" % (DOP.min(), angle_arcsec[numpy.argmin(DOP)]))
    i = numpy.argmax(numpy.abs(P3))
    print("   max |P3| = %.3f (P3 = %+.3f, DOP = %.3f) at %+.1f arcsec (detuning %+.1f urad), plate transmission = %.4f" % (
          numpy.abs(P3[i]), P3[i], DOP[i], angle_arcsec[i], 1e6 * detune[i], S0[i] / S0_in))
    i = numpy.argmax(numpy.where(angle_arcsec < 0, numpy.abs(P3), 0))
    print("   max |P3| (negative angles) = %.3f (P3 = %+.3f) at %+.1f arcsec, plate transmission = %.4f" % (
          numpy.abs(P3[i]), P3[i], angle_arcsec[i], S0[i] / S0_in))
    print("   plate transmission: min %.4f, max %.4f, at -300 arcsec %.4f" % ((S0 / S0_in).min(), (S0 / S0_in).max(), S0[0] / S0_in))
    i = numpy.argmax(P2); print("   max P2 = %.3f at %+.1f arcsec; min P1 = %.3f at %+.1f arcsec" % (P2[i], angle_arcsec[i], P1.min(), angle_arcsec[numpy.argmin(P1)]))

    import matplotlib
    if not SHOW_PLOTS:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(angle_arcsec, P1, '-', color='k', lw=1, label=r'$P_1$ shadow4')
    ax.plot(angle_arcsec, P2, '-', color='C0', lw=1, label=r'$P_2$ shadow4')
    ax.plot(angle_arcsec, P3, '-', color='C3', lw=1, label=r'$P_3$ shadow4')
    ax.plot(angle_arcsec, numpy.abs(P3), ':', color='C3', lw=1, label=r'$|P_3|$ shadow4')
    if USE_UNDULATOR == 2:
        ax.plot(angle_arcsec, P3_as_exp, '--', color='C3', lw=1, label=r'$\sqrt{1-P_1^2-P_2^2}$ shadow4')
    ax.plot(xe, P1e, 'o', color='k', ms=4, mfc='none', label=r'$P_1$ exp')
    ax.plot(xe, P2e, 's', color='C0', ms=4, mfc='none', label=r'$P_2$ exp')
    ax.plot(xe, P3e, '^', color='C3', ms=4, mfc='none', label=r'$\sqrt{1-P_1^2-P_2^2}$ exp')
    ax.set_xlabel('angle [arcsec] (detuning = -angle)')
    ax.set_ylabel('normalized Stokes parameters')
    ax.set_title(r'Si 111, %d $\mu$m, %g eV, $\alpha$=%g$^\circ$%s' % (round(1e6 * THICKNESS), ENERGY, ALPHA,
                 ["", ", undulator (monochromatic)", ", undulator + Si 111 DCM"][USE_UNDULATOR]))
    ax.set_xlim(-300, 300)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(here, "..", "figures", figure_file), dpi=200)
    print("File written to disk: %s" % os.path.join(here, "..", "figures", figure_file))
    if SHOW_PLOTS:
        plt.show()
