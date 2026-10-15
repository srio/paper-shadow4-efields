#
# Section 7.2: practical case of an X-ray polarizer/analyzer with Si 400 channel-cut crystals at 6457 eV
# (theta_B ~ 45 deg), as in the experiment of Schulze et al. (2022) at the European XFEL.
#
# Source: point, Gaussian horizontal divergence sigma_h = 273 nrad, sigma_v = 0, 0.01% bandwidth,
# linearly polarized along x (horizontal, P=1).
# Polarizer: N_REFLECTIONS reflections (+,-,+,-,...), crystal 1 rotated by beta around the beam
#            (beta=0: scattering plane vertical, i.e. sigma along the source electric field).
# Analyzer:  N_REFLECTIONS reflections, rotated by eta around the beam with respect to the polarizer
#            (eta=0: parallel, eta=90 deg: crossed).
# Linear purity: P = I(eta=90) / I(eta=0), compared with sigma_h^2 + sigma_v^2 (Schulze, 2018) and, for a
# rotated polarizer, with sigma_h^2 (1 +/- sin(2 beta)). The sign of the asymmetric term depends on the sense of
# the crossing (eta = +90 or -90 deg) and on the parity of the number of reflections; it is fitted from the data.
#
# Results are written to shadow4_practical_polarizer_*.dat and the figure to ../figures/polarizer.png
#
# Usage:
#   PYTHONPATH=~/OASYS2.0/shadow4 ~/venv/oasys2/bin/python shadow4_practical_polarizer.py          (run and plot)
#   PYTHONPATH=~/OASYS2.0/shadow4 ~/venv/oasys2/bin/python shadow4_practical_polarizer.py plot     (plot only)
#
import os
import numpy
from syned.beamline.element_coordinates import ElementCoordinates
from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
from shadow4.beamline.optical_elements.crystals.s4_plane_crystal import S4PlaneCrystal, S4PlaneCrystalElement

PHOTON_ENERGY = 6457.0          # eV
BANDWIDTH = 1e-4                # 0.01 %
SIGMA_H = 273e-9                # rad
SIGMA_V = 0.0                   # rad
N_REFLECTIONS = 6               # per channel-cut crystal (polarizer and analyzer)
NRAYS = 100000

HERE = os.path.dirname(os.path.abspath(__file__))


def get_source_beam(nrays=NRAYS, seed=5676561):
    source = SourceGeometrical(name='Geometrical Source', nrays=nrays, seed=seed)
    source.set_spatial_type_point()
    source.set_depth_distribution_off()
    source.set_angular_distribution_gaussian(sigdix=SIGMA_H, sigdiz=SIGMA_V)
    source.set_energy_distribution_uniform(value_min=PHOTON_ENERGY * (1 - BANDWIDTH / 2),
                                           value_max=PHOTON_ENERGY * (1 + BANDWIDTH / 2), unit='eV')
    source.set_polarization(polarization_degree=1.0, phase_diff=0.0, coherent_beam=0)
    return source.get_beam()


def trace_crystal(beam, angle_azimuthal):
    # Si 400, thick crystal, xraylib; automatically aligned (f_central=1) to the corrected Bragg angle
    optical_element = S4PlaneCrystal(name='Si 400', boundary_shape=None, material='Si',
                                     miller_index_h=4, miller_index_k=0, miller_index_l=0,
                                     f_bragg_a=False, asymmetry_angle=0.0,
                                     is_thick=1, thickness=0.001,
                                     f_central=1, f_phot_cent=0, phot_cent=PHOTON_ENERGY,
                                     f_ext=0, material_constants_library_flag=0,  # 0=xraylib
                                     method_efields_management=0)
    coordinates = ElementCoordinates(p=0.1, q=0.1, angle_radial=0.0, angle_azimuthal=angle_azimuthal,
                                     angle_radial_out=0.0)
    element = S4PlaneCrystalElement(optical_element=optical_element, coordinates=coordinates, input_beam=beam)
    beam_out, _ = element.trace_beam()
    return beam_out


def run_polarimeter(beam, beta=0.0, eta=0.0, n=N_REFLECTIONS):
    """beta: polarizer rotation around the beam; eta: analyzer rotation with respect to the polarizer (rad).
    The orientation angles are relative to the previous element (they are not reset in SHADOW): the crystals of
    each channel cut are at 180 deg (+,-), and the first analyzer crystal at 180 deg + eta."""
    angles = [beta] + [numpy.pi] * (n - 1) + [numpy.pi + eta] + [numpy.pi] * (n - 1)
    for angle in angles:
        beam = trace_crystal(beam, angle)
    return beam.get_intensity(nolost=1)


def purity(source_beam, beta):
    i_parallel = run_polarimeter(source_beam.duplicate(), beta=beta, eta=0.0)
    i_cross = run_polarimeter(source_beam.duplicate(), beta=beta, eta=numpy.pi / 2)
    return i_parallel, i_cross


def asymmetry_sign(beta_data):
    """+1 if the purity grows for positive beta, -1 otherwise."""
    beta_deg, pur = beta_data[:, 0], beta_data[:, 3]
    return 1 if pur[beta_deg > 0].mean() > pur[beta_deg < 0].mean() else -1


def plot(eta_data, beta_data, file_png, plot_beta=False):
    """Figure of the paper: analyzer rotation (eta). With plot_beta=True, also the polarizer rotation (beta)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p0 = SIGMA_H ** 2 + SIGMA_V ** 2
    if plot_beta:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(5.5, 4))

    eta_deg, ratio = eta_data[:, 0], eta_data[:, 1]
    eta_fine = numpy.linspace(0, 90, 2001)
    ax1.semilogy(eta_fine, numpy.cos(numpy.radians(eta_fine)) ** 2 + p0, '-', color='C0',
                 label=r'$\cos^2\eta + \sigma_h^2+\sigma_v^2$')
    ax1.semilogy(eta_deg, ratio, 'x', color='C3', label='SHADOW4')
    ax1.set_xlabel(r'analyzer angle $\eta$ [deg]')
    ax1.set_ylabel(r'$I(\eta)/I(0)$')
    ax1.set_ylim(1e-15, 2)
    if plot_beta: ax1.set_title('(a) analyzer rotation')
    ax1.legend(fontsize=8)

    if not plot_beta:
        fig.tight_layout()
        fig.savefig(file_png, dpi=200)
        print("File written to disk: %s" % file_png)
        return

    beta_deg, pur = beta_data[:, 0], beta_data[:, 3]
    sign = asymmetry_sign(beta_data)
    beta_fine = numpy.linspace(beta_deg.min(), beta_deg.max(), 501)
    ax2.plot(beta_fine, p0 * (1 + sign * numpy.sin(numpy.radians(2 * beta_fine))), '-', color='C0',
             label=r'$\sigma_h^2\,(1%s\sin 2\beta)$' % ('+' if sign > 0 else '-'))
    ax2.axhline(p0, linestyle='--', color='gray', label=r'$\sigma_h^2+\sigma_v^2$')
    ax2.plot(beta_deg, pur, 'x', color='C3', label='SHADOW4')
    ax2.set_xlabel(r'polarizer angle $\beta$ [deg]')
    ax2.set_ylabel(r'linear purity $\mathcal{P}$')
    ax2.set_title('(b) polarizer rotation')
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(file_png, dpi=200)
    print("File written to disk: %s" % file_png)


if __name__ == "__main__":
    import sys
    file_eta = os.path.join(HERE, "shadow4_practical_polarizer_eta.dat")
    file_beta = os.path.join(HERE, "shadow4_practical_polarizer_beta.dat")
    file_png = os.path.join(HERE, "..", "figures", "polarizer.png")
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot(numpy.loadtxt(file_eta), numpy.loadtxt(file_beta), file_png)
        sys.exit(0)

    source_beam = get_source_beam()
    print("Si 400 polarimeter at %g eV, %d+%d reflections, %d rays, sigma_h = %g nrad, sigma_v = %g nrad" %
          (PHOTON_ENERGY, N_REFLECTIONS, N_REFLECTIONS, NRAYS, 1e9 * SIGMA_H, 1e9 * SIGMA_V))
    print("Prediction sigma_h^2 + sigma_v^2 = %.3e\n" % (SIGMA_H ** 2 + SIGMA_V ** 2))

    # (a) analyzer rotation, polarizer at beta=0
    i_parallel = run_polarimeter(source_beam.duplicate(), beta=0.0, eta=0.0)
    eta_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 85, 89, 89.9, 89.99, 89.995, 90.0]
    eta_data = []
    print("%10s %14s %14s" % ("eta[deg]", "I(eta)/I(0)", "cos^2(eta)"))
    for eta_deg in eta_list:
        i = run_polarimeter(source_beam.duplicate(), beta=0.0, eta=numpy.radians(eta_deg))
        eta_data.append([eta_deg, i / i_parallel])
        print("%10.3f %14.6e %14.6e" % (eta_deg, i / i_parallel, numpy.cos(numpy.radians(eta_deg)) ** 2))
    eta_data = numpy.array(eta_data)
    print("\nbeta=0: I_parallel = %.6g, I_cross = %.6g, purity = %.4e (%d rays)\n" %
          (i_parallel, eta_data[-1, 1] * i_parallel, eta_data[-1, 1], NRAYS))

    # (b) polarizer rotation
    beta_data = []
    print("%10s %14s %14s %14s" % ("beta[deg]", "I_parallel", "I_cross", "purity"))
    for beta_deg in numpy.arange(-60, 61, 7.5):
        i_par, i_cross = purity(source_beam, numpy.radians(beta_deg))
        beta_data.append([beta_deg, i_par, i_cross, i_cross / i_par])
        print("%10.1f %14.6g %14.6g %14.4e" % (beta_deg, i_par, i_cross, i_cross / i_par))
    beta_data = numpy.array(beta_data)
    sign = asymmetry_sign(beta_data)
    print("\npurity compared with sigma_h^2 (1 %s sin 2 beta):" % ('+' if sign > 0 else '-'))
    for b, pur in zip(beta_data[:, 0], beta_data[:, 3]):
        print("%10.1f %14.4e %14.4e" % (b, pur, (SIGMA_H ** 2 + SIGMA_V ** 2) * (1 + sign * numpy.sin(numpy.radians(2 * b)))))

    numpy.savetxt(file_eta, eta_data, header="eta[deg]  I(eta)/I(0)")
    numpy.savetxt(file_beta, beta_data, header="beta[deg]  I_parallel  I_cross  purity")
    plot(eta_data, beta_data, file_png)
