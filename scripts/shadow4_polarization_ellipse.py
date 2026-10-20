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
# A second geometrical source with right circular polarization (chi = -45 deg, i.e. P = 0.5, Phi = 90 deg) is also run.
#
# Usage:
#   PYTHONPATH=~/OASYS2.0/shadow4 ~/venv/oasys2/bin/python shadow4_polarization_ellipse.py [psi_deg chi_deg]
#   (runs the geometrical source with psi_deg, chi_deg [default 30 30], the RCP geometrical source and the bending
#   magnet source; the figures are written to ../figures/polarization_ellipse_geometrical.png,
#   ../figures/polarization_ellipse_geometrical_rcp.png and ../figures/polarization_ellipse_bending_magnet.png)
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


def plot_ellipse(ellS, ellJ, iray=0, nsample=20, file_png=None):
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
    ax.plot(E1, E2, '-', color='red', lw=2, label='E(t) of ray %d' % iray)
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
    ax.set_title(r'$\psi$=%.1f$^\circ$, $\chi$=%.1f$^\circ$, P$_1$=%.3f, P$_2$=%.3f, P$_3$=%.3f (%s)' %
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


if __name__ == "__main__":
    # both sources; the figures are written to ../figures/polarization_ellipse_<source>.png (used in the paper)
    figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    psi_deg, chi_deg = (30.0, 30.0) if len(sys.argv) < 3 else (float(sys.argv[1]), float(sys.argv[2]))

    for SOURCE in ["geometrical", "geometrical_rcp", "bending_magnet"]:
        print("\n===== %s source =====" % SOURCE)
        if SOURCE == "geometrical":
            beam = get_beam_geometrical(psi_deg, chi_deg)
        elif SOURCE == "geometrical_rcp":
            beam = get_beam_geometrical(0.0, -45.0)                # right circular: P = 0.5, Phi = 90 deg
        else:
            beam = get_beam_bending_magnet()

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

        plot_ellipse(ellS, ellJ, iray=iray, nsample=20,
                     file_png=os.path.join(figures_dir, "polarization_ellipse_%s.png" % SOURCE))