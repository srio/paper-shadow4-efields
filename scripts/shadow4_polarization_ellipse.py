#
# Polarization ellipse of a SHADOW4 beam, with the conventions of C. Detlefs et al., Eur. Phys. J. Special Topics
# 208, 359 (2012), Fig. 1: right-handed frame (x, y, k) with x X y = k, time dependence exp(-i w t), ellipse seen
# by an observer looking toward the source.
#
# In SHADOW4 this frame is x = u_sigma, y = -u_pi, k = v (the SHADOW frame (u_sigma, u_pi, v) is left-handed), i.e.
# the standard Jones vector V = (Ex, Ey) = beam.get_jones_components(use_standard=1). The Stokes parameters are the
# beam columns 30-33 (S0..S3) and the Poincare-Stokes parameters P1, P2, P3 are the columns 42-44.
#
# A geometrical source is created with an elliptical polarization defined by the ellipse azimuth psi and ellipticity
# chi (default: the example of Detlefs Fig. 1, psi = chi = 30 deg, a left-handed ellipse with P3 > 0). The source
# inputs (P, Phi) are obtained from the Stokes parameters, and the ellipse is then recovered from the beam in two ways:
# with S4Beam.get_polarization_ellipse (from the Stokes parameters): averaged over the rays (flag_return=0), for all
# the rays (flag_return=1) or for one ray (flag_return=2). The curves are obtained with
# S4Beam.get_polarization_ellipse_curve and plotted with plot_ellipse. The ellipses are checked against the Jones
# vectors of the rays in shadow4tests (test_library/test_beam_polarization_ellipse.py).
# A second geometrical source with right circular polarization (chi = -45 deg, i.e. P = 0.5, Phi = 90 deg) is also run,
# and the synchrotron sources: ESRF1 bending magnet (8 keV), a wiggler (6.084 GeV, K = 20.49, 6 periods of 15 cm,
# 50 keV) and an undulator (6.084 GeV, K = 1.656, 111 periods of 18 mm, at resonance 8.234 keV), all without electron
# beam emittance.
#
# Usage:
#   PYTHONPATH=~/OASYS2.0/shadow4 ~/venv/oasys2/bin/python shadow4_polarization_ellipse.py [psi_deg chi_deg]
#   (runs the geometrical source with psi_deg, chi_deg [default 30 30], the RCP geometrical source, and the bending
#   magnet, wiggler and undulator sources; the figures are written to ../figures/polarization_ellipse_<source>.png,
#   with <source> = geometrical, geometrical_rcp, bending_magnet, wiggler, undulator)
#
import os
import sys
import numpy
from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
from shadow4.beam.s4_beam import S4Beam

SHOW_PLOTS = 1

def stokes_from_ellipse(psi, chi):
    """Normalized Stokes parameters (S1, S2, S3) of an ellipse with azimuth psi and ellipticity chi (rad)."""
    return (numpy.cos(2 * chi) * numpy.cos(2 * psi),
            numpy.cos(2 * chi) * numpy.sin(2 * psi),
            numpy.sin(2 * chi))


def shadow_inputs_from_stokes(S1, S2, S3, S0=1.0):
    """SHADOW geometrical-source inputs: polarization degree P and phase difference Phi (rad)."""
    P = numpy.sqrt(S0 + S1) / (numpy.sqrt(S0 + S1) + numpy.sqrt(S0 - S1))
    Phi = numpy.arctan2(-S3, -S2)
    return P, Phi


def plot_ellipse(ellS, ellJ, iray=0, nsample=20, file_png=None, ray_label='', title=''):
    """Plots the ellipse of the polarized part of the beam (ellS, flag_return=0) with its axes, and the ellipses of
    the ray iray and of a sample of nsample rays (ellJ, all the rays, flag_return=1)."""
    import matplotlib
    if not SHOW_PLOTS: matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, b, psi, chi = ellS["a"], ellS["b"], ellS["psi"], ellS["chi"]
    X, Y = S4Beam.get_polarization_ellipse_curve(ellJ)            # curves of all the rays
    E1, E2 = X[iray], Y[iray]
    nrays = X.shape[0]
    sample = numpy.unique(numpy.linspace(0, nrays - 1, nsample).astype(int))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for j, i in enumerate(sample):
        ax.plot(X[i], Y[i], '-', color='C1', lw=0.6, alpha=0.6,
                label='E(t) of %d rays' % sample.size if j == 0 else None)
    ax.plot(E1, E2, '-', color='red', lw=2, label='E(t) of ray %d%s' % (iray, ray_label))
    ax.plot(*S4Beam.get_polarization_ellipse_curve(ellS), '--', color='k', lw=1, label='from Stokes parameters')
    ax.plot([-a * numpy.cos(psi), a * numpy.cos(psi)], [-a * numpy.sin(psi), a * numpy.sin(psi)],
            '-.', color='C2', label='major axis, a')
    ax.plot([b * numpy.sin(psi), -b * numpy.sin(psi)], [-b * numpy.cos(psi), b * numpy.cos(psi)],
            '-.', color='C0', label='minor axis, b')
    ax1, ay1 = numpy.abs(E1).max(), numpy.abs(E2).max()   # rectangle 2|Ex| x 2|Ey| of the ray
    ax.plot([-ax1, ax1, ax1, -ax1, -ax1], [-ay1, -ay1, ay1, ay1, -ay1], ':', color='gray')
    k = 40                                                 # arrow: sense of rotation for increasing time
    ax.annotate('', xy=(E1[k + 5], E2[k + 5]), xytext=(E1[k], E2[k]),
                arrowprops=dict(arrowstyle='-|>', color='red', lw=2, mutation_scale=20))
    lim = 1.15 * max(numpy.abs(X[sample]).max(), numpy.abs(Y[sample]).max(), ax1, ay1, a)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    # inverted hat with a combining caron (U+030C): matplotlib mathtext misplaces \check
    ax.set_xlabel(r'$E_x$ along $x$' + '\u030c' + r'$=\hat{u}_\sigma$')
    ax.set_ylabel(r'$E_y$ along $y$' + '\u030c' + r'$=-\hat{u}_\pi$')
    P1, P2, P3 = ellS["P"]
    ax.set_title(title + r'$\psi$=%.1f$^\circ$, $\chi$=%.1f$^\circ$, P$_1$=%.3f, P$_2$=%.3f, P$_3$=%.3f (%s)' %
                 (numpy.degrees(psi), numpy.degrees(chi), P1, P2, P3,
                  'left-handed' if P3 > 0.05 else ('right-handed' if P3 < -0.05 else 'nearly linear')), fontsize=9)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    if file_png is not None:
        fig.savefig(file_png, dpi=200)
        print("File written to disk: %s" % file_png)

    if SHOW_PLOTS: plt.show()


def get_beam_geometrical(psi_deg=30.0, chi_deg=30.0, nrays=1000):
    """Geometrical source: all rays with the same elliptical polarization (azimuth psi, ellipticity chi)."""
    S1, S2, S3 = stokes_from_ellipse(numpy.radians(psi_deg), numpy.radians(chi_deg))
    P, Phi = shadow_inputs_from_stokes(S1, S2, S3)
    print("Requested ellipse: psi = %.2f deg, chi = %.2f deg -> P1=%.4f, P2=%.4f, P3=%.4f" %
          (psi_deg, chi_deg, S1, S2, S3))
    print("SHADOW source inputs: polarization_degree P = %.6f, phase_diff Phi = %.4f deg" % (P, numpy.degrees(Phi)))

    source = SourceGeometrical(name='Geometrical Source', nrays=nrays, seed=5676561)
    source.set_spatial_type_point()
    source.set_depth_distribution_off()
    source.set_angular_distribution_flat(hdiv1=-1e-6, hdiv2=1e-6, vdiv1=-1e-6, vdiv2=1e-6)
    source.set_energy_distribution_singleline(10000.0, unit='eV')
    # P, Phi = 0.5, numpy.radians(90)
    source.set_polarization(polarization_degree=P, phase_diff=Phi, coherent_beam=0)
    return source.get_beam()


def get_beam_bending_magnet(photon_energy=8000.0, nrays=5000):
    """ESRF-1 bending magnet (6.04 GeV, 0.8 T), monochromatic. The polarization of each ray depends on its vertical
    emission angle: sigma and pi amplitudes from the synchrotron emission functions, phase difference +/- 90 deg with
    the sign of the vertical angle (elliptical polarization of opposite handedness above and below the orbit plane)."""
    from shadow4.sources.s4_electron_beam import S4ElectronBeam
    from shadow4.sources.bending_magnet.s4_bending_magnet import S4BendingMagnet
    from shadow4.sources.bending_magnet.s4_bending_magnet_light_source import S4BendingMagnetLightSource

    electron_beam = S4ElectronBeam(energy_in_GeV=6.04, energy_spread=0.001, current=0.2)
    electron_beam.set_sigmas_all(sigma_x=7.8e-05, sigma_y=4.87179e-05, sigma_xp=3.6e-05, sigma_yp=1.05556e-06)
    radius = numpy.abs(S4BendingMagnet.calculate_magnetic_radius(-0.8, electron_beam.energy()))
    magnetic_structure = S4BendingMagnet(radius=radius, magnetic_field=-0.8, length=1e-3 * radius,
                                         emin=photon_energy, emax=photon_energy, ng_e=100, flag_emittance=0)
    light_source = S4BendingMagnetLightSource(name='BendingMagnet', electron_beam=electron_beam,
                                              magnetic_structure=magnetic_structure, nrays=nrays, seed=5676561)
    return light_source.get_beam()

def get_beam_wiggler(return_light_source=False):
    """Wiggler (6.084 GeV, K = 20.49, 6 periods of 15 cm) at 50 keV, without emittance. With return_light_source=True
    it returns (beam, light_source), the light source giving access to the electron trajectory."""
    # electron beam
    from shadow4.sources.s4_electron_beam import S4ElectronBeam
    electron_beam = S4ElectronBeam(energy_in_GeV=6.084, energy_spread=0.001, current=0.1)
    electron_beam.set_sigmas_all(sigma_x=0, sigma_y=0, sigma_xp=0, sigma_yp=0)
    electron_beam.set_dispersion_all(0, 0, 0, 0)

    # magnetic structure
    from shadow4.sources.wiggler.s4_wiggler import S4Wiggler
    source = S4Wiggler(
        magnetic_field_periodic=1,  # 0=external, 1=periodic
        file_with_magnetic_field="",  # used only if magnetic_field_periodic=0
        K_vertical=20.4911,  # syned Wiggler pars: used only if magnetic_field_periodic=1
        period_length=0.15,  # syned Wiggler pars: used only if magnetic_field_periodic=1
        number_of_periods=6,  # syned Wiggler pars: used only if magnetic_field_periodic=1
        emin=50000.0,  # Photon energy scan from energy (in eV)
        emax=50000.0,  # Photon energy scan to energy (in eV)
        ng_e=1,  # Photon energy scan number of points for spectrum and internal calculation
        ng_j=501,  # Number of points in electron trajectory (per period) for internal calculation only
        psi_interval_number_of_points=101,  # the number psi (vertical angle) points for internal calculation only
        flag_interpolation=1,  # Use interpolation to sample psi (0=No, 1=Yes)
        flag_emittance=0,  # Use emittance (0=No, 1=Yes)
        shift_x_flag=4,  # 0="No shift", 1="Half excursion", 2="Minimum", 3="Maximum", 4="Value at zero", 5="User value"
        shift_x_value=0.0,  # used only if shift_x_flag=5
        shift_betax_flag=4,
        # 0="No shift", 1="Half excursion", 2="Minimum", 3="Maximum", 4="Value at zero", 5="User value"
        shift_betax_value=0.0,  # used only if shift_betax_flag=5
    )

    # light source
    from shadow4.sources.wiggler.s4_wiggler_light_source import S4WigglerLightSource
    light_source = S4WigglerLightSource(name='Wiggler Light Source', electron_beam=electron_beam,
                                        magnetic_structure=source, nrays=20000, seed=5676561)
    beam = light_source.get_beam()
    return (beam, light_source) if return_light_source else beam

def get_beam_undulator():
    # electron beam
    from shadow4.sources.s4_electron_beam import S4ElectronBeam
    electron_beam = S4ElectronBeam(energy_in_GeV=6.084, energy_spread=0.001, current=0.1)
    electron_beam.set_sigmas_all(sigma_x=0, sigma_y=0, sigma_xp=0, sigma_yp=0)
    electron_beam.set_dispersion_all(0, 0, 0, 0)

    # magnetic structure
    from shadow4.sources.undulator.s4_undulator import S4Undulator
    source = S4Undulator(
        K_vertical=1.6563,  # syned Undulator parameter
        period_length=0.018,  # syned Undulator parameter
        number_of_periods=111,  # syned Undulator parameter
        emin=8233.962924699139,  # Photon energy scan from energy (in eV)
        emax=8233.962924699139,  # Photon energy scan to energy (in eV)
        ng_e=1,  # Photon energy scan number of points
        maxangle=1.7970134151911963e-05,  # Maximum radiation semiaperture in RADIANS
        ng_t=100,  # Number of points in angle theta
        ng_p=11,  # Number of points in angle phi
        ng_j=20,  # Number of points in electron trajectory (per period) for internal calculation only
        code_undul_phot='internal',  # internal, pysru, srw
        flag_emittance=0,  # when sampling rays: Use emittance (0=No, 1=Yes)
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
                                          magnetic_structure=source, nrays=50000, seed=5676561)
    return light_source.get_beam()


if __name__ == "__main__":
    # both sources; the figures are written to ../figures/polarization_ellipse_<source>.png (used in the paper)
    figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    psi_deg, chi_deg = (30.0, 30.0) if len(sys.argv) < 3 else (float(sys.argv[1]), float(sys.argv[2]))

    for SOURCE in ["geometrical", "geometrical_rcp", "bending_magnet", "wiggler", "undulator"]:
        print("\n===== %s source =====" % SOURCE)
        if SOURCE == "geometrical":
            beam = get_beam_geometrical(psi_deg, chi_deg)
        elif SOURCE == "geometrical_rcp":
            beam = get_beam_geometrical(0.0, -45.0)                # right circular: P = 0.5, Phi = 90 deg
        elif SOURCE == "bending_magnet":
            beam = get_beam_bending_magnet()
        elif SOURCE == "wiggler":
            beam = get_beam_wiggler()
        elif SOURCE == "undulator":
            beam = get_beam_undulator()


        iray = 0
        ellS = beam.get_polarization_ellipse(flag_return=0)               # beam (averaged Stokes parameters)
        ellJ = beam.get_polarization_ellipse(flag_return=1)               # all the rays
        ellJ1 = beam.get_polarization_ellipse(flag_return=2, iray=iray)   # a single ray
        for name, ell in [("Stokes of the beam", ellS), ("ray %d" % iray, ellJ1)]:
            print("%-18s: psi = %6.2f deg, chi = %6.2f deg, a = %.4f, b = %.4f, P1=%.4f, P2=%.4f, P3=%.4f, DOP=%.6f" %
                  (name, numpy.degrees(ell["psi"]), numpy.degrees(ell["chi"]), ell["a"], ell["b"], *ell["P"],
                   ell["dop"]))

        print("all rays          : chi in [%.2f, %.2f] deg, mean dop = %.6f" %
              (numpy.degrees(numpy.nanmin(ellJ["chi"])), numpy.degrees(numpy.nanmax(ellJ["chi"])),
               numpy.nanmean(ellJ["dop"])))

        ray_label = ''
        if SOURCE in ["bending_magnet", "wiggler", "undulator"]:                     # position of the ray with respect to the orbit plane
            zp = beam.get_column(6, nolost=1)[iray]
            ray_label = " (Z'%s0, %s)" % ((">", "above") if zp > 0 else ("<", "below"))
            print("ray %d: Z' = %.3e rad%s" % (iray, zp, ray_label))
        title = {"bending_magnet": "Bending magnet: ", "wiggler": "Wiggler: ", "undulator": "Undulator: "}.get(SOURCE, "")
        plot_ellipse(ellS, ellJ, iray=iray, nsample=20, ray_label=ray_label, title=title,
                     file_png=os.path.join(figures_dir, "polarization_ellipse_%s.png" % SOURCE))
