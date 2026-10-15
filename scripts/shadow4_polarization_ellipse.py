#
# Polarization ellipse of a SHADOW4 beam, with the conventions of C. Detlefs et al., Eur. Phys. J. Special Topics
# 208, 359 (2012), Fig. 1: right-handed frame (e1, e2, k) with e1 x e2 = k, time dependence exp(-i w t), ellipse seen
# by an observer looking into the source.
#
# In SHADOW4 this frame is e1 = u_sigma, e2 = -u_pi, k = v (the SHADOW frame (u_sigma, u_pi, v) is left-handed), i.e.
# the standard Jones vector V = (V1, V2) = beam.get_jones_components(use_standard=1). The Stokes parameters are the
# beam columns 30-33 (S0..S3) and the Poincare-Stokes parameters P1, P2, P3 are the columns 42-44.
#
# A geometrical source is created with an elliptical polarization defined by the ellipse azimuth psi and ellipticity
# chi (default: the example of Detlefs Fig. 1, psi = chi = 30 deg, a left-handed ellipse with P3 > 0). The source
# inputs (P, Phi) are obtained from the Stokes parameters, and the ellipse is then recovered from the beam.
#
# Usage:
#   PYTHONPATH=~/OASYS2.0/shadow4 ~/venv/oasys2/bin/python shadow4_polarization_ellipse.py [psi_deg chi_deg]
#   (runs the geometrical source with psi_deg, chi_deg [default 30 30] and the bending magnet source; the figures
#   are written to ../figures/polarization_ellipse_geometrical.png and ../figures/polarization_ellipse_bending_magnet.png)
#
import os
import sys
import numpy
from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical


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


def ellipse_from_beam(beam):
    """Stokes parameters summed over the good rays, and the ellipse of the polarized part."""
    S0, S1, S2, S3 = [beam.get_column(col, nolost=1).sum() for col in [30, 31, 32, 33]]
    dop = numpy.sqrt(S1 ** 2 + S2 ** 2 + S3 ** 2) / S0
    psi = numpy.mod(0.5 * numpy.arctan2(S2, S1), numpy.pi)     # azimuth of the major axis from e1, [0, pi)
    if numpy.isclose(psi, numpy.pi): psi = 0.0                  # psi = pi - epsilon is the same direction as 0
    chi = 0.5 * numpy.arcsin(S3 / (dop * S0))            # ellipticity angle, > 0 for left-handed
    intensity = dop * S0                                  # polarized intensity
    a = numpy.sqrt(intensity) * numpy.cos(chi)            # semi-axes
    b = numpy.sqrt(intensity) * numpy.abs(numpy.sin(chi))
    return dict(S=(S0, S1, S2, S3), P=(S1 / S0, S2 / S0, S3 / S0), dop=dop, psi=psi, chi=chi, a=a, b=b)


def plot_ellipse(beam, ell, file_png=None, iray=0, nsample=20):
    """Plots the field of ray iray, the ellipses of a sample of nsample rays (every ray is fully polarized and has
    its own ellipse) and the ellipse of the polarized part of the beam, obtained from the summed Stokes parameters."""
    import matplotlib
    if file_png is not None: matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrays = beam.get_number_of_rays(nolost=1)
    tau = numpy.linspace(0, 2 * numpy.pi, 361)

    # ellipse from the Stokes parameters of the beam (normalized to one ray)
    a, b, psi, chi = ell["a"] / numpy.sqrt(nrays), ell["b"] / numpy.sqrt(nrays), ell["psi"], ell["chi"]
    x = a * numpy.cos(tau)
    y = numpy.sign(chi) * b * numpy.sin(tau)               # counter-clockwise for chi > 0 (left-handed)
    e1 = x * numpy.cos(psi) - y * numpy.sin(psi)
    e2 = x * numpy.sin(psi) + y * numpy.cos(psi)

    # electric field of one ray over one period: E_i(tau) = Re[V_i exp(-i tau)]  (Detlefs, Eq. 2)
    V1, V2 = beam.get_jones_components(nolost=1, use_standard=1)
    E1 = numpy.real(V1[iray] * numpy.exp(-1j * tau))
    E2 = numpy.real(V2[iray] * numpy.exp(-1j * tau))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    sample = numpy.unique(numpy.linspace(0, nrays - 1, nsample).astype(int))
    for j, i in enumerate(sample):
        ax.plot(numpy.real(V1[i] * numpy.exp(-1j * tau)), numpy.real(V2[i] * numpy.exp(-1j * tau)), '-',
                color='C1', lw=0.6, alpha=0.6, label='E(t) of %d rays' % sample.size if j == 0 else None)
    ax.plot(E1, E2, '-', color='red', lw=2, label='E(t) of ray %d' % iray)
    ax.plot(e1, e2, '--', color='k', lw=1, label='from Stokes parameters')
    ax.plot([-a * numpy.cos(psi), a * numpy.cos(psi)], [-a * numpy.sin(psi), a * numpy.sin(psi)],
            '-.', color='C2', label='major axis, a')
    ax.plot([b * numpy.sin(psi), -b * numpy.sin(psi)], [-b * numpy.cos(psi), b * numpy.cos(psi)],
            '-.', color='C0', label='minor axis, b')
    ax.plot([-abs(V1[iray]), abs(V1[iray]), abs(V1[iray]), -abs(V1[iray]), -abs(V1[iray])],
            [-abs(V2[iray]), -abs(V2[iray]), abs(V2[iray]), abs(V2[iray]), -abs(V2[iray])], ':', color='gray')
    k = 40                                                 # arrow: sense of rotation for increasing time
    ax.annotate('', xy=(E1[k + 5], E2[k + 5]), xytext=(E1[k], E2[k]),
                arrowprops=dict(arrowstyle='-|>', color='red', lw=2, mutation_scale=20))
    lim = 1.15 * max(numpy.abs(V1[sample]).max(), numpy.abs(V2[sample]).max(), abs(V1[iray]), abs(V2[iray]), a)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel(r'$E_1$ along $\hat{\epsilon}_1=\hat{u}_\sigma$')
    ax.set_ylabel(r'$E_2$ along $\hat{\epsilon}_2=-\hat{u}_\pi$')
    P1, P2, P3 = ell["P"]
    ax.set_title(r'$\psi$=%.1f$^\circ$, $\chi$=%.1f$^\circ$, P$_1$=%.3f, P$_2$=%.3f, P$_3$=%.3f (%s)' %
                 (numpy.degrees(psi), numpy.degrees(chi), P1, P2, P3,
                  'left-handed' if P3 > 0.05 else ('right-handed' if P3 < -0.05 else 'nearly linear')), fontsize=9)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    if file_png is None:
        plt.show()
    else:
        fig.savefig(file_png, dpi=200)
        print("File written to disk: %s" % file_png)


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

    for SOURCE in ["geometrical", "bending_magnet"]:
        print("\n===== %s source =====" % SOURCE)
        if SOURCE == "geometrical":
            beam = get_beam_geometrical(psi_deg, chi_deg)
        else:
            beam = get_beam_bending_magnet()

        ell = ellipse_from_beam(beam)
        print("From the beam: psi = %.2f deg, chi = %.2f deg, P1=%.4f, P2=%.4f, P3=%.4f, DOP=%.6f" %
              (numpy.degrees(ell["psi"]), numpy.degrees(ell["chi"]), *ell["P"], ell["dop"]))

        plot_ellipse(beam, ell, file_png=os.path.join(figures_dir, "polarization_ellipse_%s.png" % SOURCE))
        # plot_ellipse(beam, ell, file_png=None)