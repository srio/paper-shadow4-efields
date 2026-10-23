#
# Approximated calculation of the Stokes parameters of the beam transmitted by the Si 111 phase plate of P09
# (2838 eV, 10 um, plate rotated by 45 deg around the beam), by convolution of the plane-wave (pencil beam)
# result of crystalpy with a Gaussian.
#
# The Stokes parameters S0...S3 of a plane wave versus the detuning angle are calculated with crystalpy
# (calc_xcrystal_angular_scan, Guigay method, transmitted beam), for a horizontally polarized incident beam
# (standard Jones vector (1, 0)) and the crystal rotated by chi=-45 deg around the beam (crystalpy chi is
# counter-clockwise for an observer looking toward the source; it corresponds to alpha=+45 deg in SHADOW4).
# As in the SHADOW4 simulations, the detuning is minus the measured angle.
#
# The angular spread of the incident beam (from its divergence and, mainly, from its energy spread, which is
# equivalent to an angular spread (Delta E / E) tan(theta_B)) is approximated by a Gaussian of rms width sigma.
# The (non normalized) Stokes parameters S0...S3 are convolved with the Gaussian, as they are additive for an
# incoherent superposition, and the normalized ones are P_k = S_k / S0 after the convolution.
#
# Outputs:
#   - figures/crystalpy_desy_p09_convolution.png: P1, P2, P3 versus the measured angle for the value of sigma
#     estimated from the beam, compared with the measurements.
#   - figures/crystalpy_desy_p09_convolution_rms.png: rms differences with the measurements versus sigma.
#
# Usage: python crystalpy_desy_p09_convolution.py [thickness_in_microns]
#   10 (default): 10 um plate (measurements 2838_10microns_Si111_111SB_Stokes.dat), figures as above.
#   5: 5 um plate (measurements 2838_5microns_Si111_111SB_Stokes.dat), figures with the suffix "_5um".
#
import os
import contextlib, io
import numpy

from crystalpy.util.calc_xcrystal import calc_xcrystal_angular_scan, apply_convolution_on_bunch_out_dict
from dabax.dabax_xraylib import DabaxXraylib

ENERGY = 2838.0                              # eV
THICKNESS = 10e-6                            # m
ARCSEC = numpy.radians(1.0 / 3600)
CHI_DEG = -45.0                              # crystalpy chi (= -alpha of SHADOW4)

# rms angular spread at the phase plate (arcsec), from the SHADOW4 simulations (Sec. 4): energy spread after the
# Si 111 DCM (0.128 eV rms, equivalent to 9.0 arcsec rms) and divergence of the undulator beam projected on the
# diffraction plane of the plate (2.1 arcsec rms)
SIGMA_ENERGY_ARCSEC = 9.0
SIGMA_DIVERGENCE_ARCSEC = 2.1
SIGMA_ARCSEC = numpy.sqrt(SIGMA_ENERGY_ARCSEC ** 2 + SIGMA_DIVERGENCE_ARCSEC ** 2)


def get_data(angle_max_arcsec=600.0, npoints=2401, fwhm_arcsec=0.0):
    """Stokes parameters S0...S3 (non normalized) of the transmitted plane wave versus the detuning (rad), convolved
    with a Gaussian of FWHM fwhm_arcsec (no convolution if zero)."""
    detuning_max = angle_max_arcsec * ARCSEC
    with contextlib.redirect_stdout(io.StringIO()): # crystalpy prints the inputs
        bunch_out_dict, diffraction_setup, deviations = calc_xcrystal_angular_scan(
            crystal_name='Si',
            thickness=THICKNESS,
            miller_h=1, miller_k=1, miller_l=1,
            asymmetry_angle=0.0,
            energy=ENERGY,
            angle_deviation_min=-detuning_max,
            angle_deviation_max=detuning_max,
            angle_deviation_points=npoints,
            angle_center_flag=1,            # deviations from the Bragg angle corrected for refraction
            flag_calculate_stokes=1,        # non normalized Stokes parameters
            jones_in=[1, 0],                # horizontal linear polarization
            chi_deg=CHI_DEG,
            flag_calculate_stokes_convolution=1 if fwhm_arcsec > 0 else 0,
            fwhm_calculate_stokes_convolution=fwhm_arcsec * ARCSEC,
            calculation_method=1,           # Guigay
            is_thick=0,
            use_transfer_matrix=0,
            geometry_type_index=2,          # Bragg transmission
            calculation_strategy_flag=0,
            material_constants_library_flag=1,
            dabax=DabaxXraylib(file_f1f2="f1f2_Windt.dat"),
        )
    return bunch_out_dict, deviations


def convolved(bunch_out_dict, deviations, sigma_arcsec):
    """Measured angle (arcsec, increasing) and S0, P1, P2, P3 after the convolution with a Gaussian of rms sigma."""
    d = apply_convolution_on_bunch_out_dict(bunch_out_dict, sigma=sigma_arcsec * ARCSEC, deviations=deviations)
    angle = -deviations / ARCSEC                     # detuning = -measured angle
    S0, S1, S2, S3 = [numpy.asarray(d[k]) for k in ("P0", "P1", "P2", "P3")]
    i = numpy.argsort(angle)
    return angle[i], S0[i], S1[i] / S0[i], S2[i] / S0[i], S3[i] / S0[i]


def get_experiment(fname):
    a = numpy.loadtxt(fname, skiprows=5)
    angle_arcsec, P1, P2 = a[:, 0], a[:, 3], a[:, 5]
    P3_abs = numpy.sqrt(numpy.clip(1.0 - P1 ** 2 - P2 ** 2, 0.0, None))  # sign not measured
    return angle_arcsec, P1, P2, P3_abs


def rms_differences(angle, P1, P2, xe, P1e, P2e, P3e):
    """rms differences with the measurements of P1, P2 and sqrt(1-P1^2-P2^2)."""
    P1i, P2i = numpy.interp(xe, angle, P1), numpy.interp(xe, angle, P2)
    P3_as_exp = numpy.sqrt(numpy.clip(1 - P1i ** 2 - P2i ** 2, 0, None))
    rms = lambda a, b: numpy.sqrt(numpy.mean((a - b) ** 2))
    return rms(P1i, P1e), rms(P2i, P2e), rms(P3_as_exp, P3e)


if __name__ == "__main__":
    import matplotlib
    SHOW_PLOTS = 1
    if not SHOW_PLOTS: matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import sys
    thickness_um = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    THICKNESS = thickness_um * 1e-6             # used by get_data()
    suffix = "" if thickness_um == 10 else "_%dum" % thickness_um

    here = os.path.dirname(os.path.abspath(__file__))
    xe, P1e, P2e, P3e = get_experiment(os.path.join(here, "2838_%dmicrons_Si111_111SB_Stokes.dat" % thickness_um))
    bunch_out_dict, deviations = get_data()

    # rms differences versus sigma
    sigmas = numpy.arange(0.0, 20.01, 0.5)
    rms = []
    for s in sigmas:
        angle, S0, P1, P2, P3 = convolved(bunch_out_dict, deviations, s)
        rms.append(rms_differences(angle, P1, P2, xe, P1e, P2e, P3e))
    rms = numpy.array(rms)
    for label, k in [("P1", 0), ("P2", 1), ("sqrt(1-P1^2-P2^2)", 2)]:
        i = numpy.argmin(rms[:, k])
        print("rms(%s): %.3f at sigma=0, %.3f at sigma=%.1f arcsec (estimated), minimum %.3f at sigma=%.1f arcsec" % (
            label, rms[0, k], numpy.interp(SIGMA_ARCSEC, sigmas, rms[:, k]), SIGMA_ARCSEC, rms[i, k], sigmas[i]))

    # curves for the estimated sigma, calculated directly by crystalpy (flag_calculate_stokes_convolution=1)
    FWHM_ARCSEC = 2 * numpy.sqrt(2 * numpy.log(2)) * SIGMA_ARCSEC
    bunch_out_dict_conv, deviations_conv = get_data(fwhm_arcsec=FWHM_ARCSEC)
    angle, S0, P1, P2, P3 = convolved(bunch_out_dict_conv, deviations_conv, 0.0)
    print("Gaussian: sigma = %.2f arcsec, FWHM = %.2f arcsec" % (SIGMA_ARCSEC, FWHM_ARCSEC))
    r1, r2, r3 = rms_differences(angle, P1, P2, xe, P1e, P2e, P3e)
    DOP = numpy.sqrt(P1 ** 2 + P2 ** 2 + P3 ** 2)
    win = numpy.abs(angle) <= 300
    print("sigma = %.2f arcsec rms: rms(P1) = %.3f, rms(P2) = %.3f, rms(sqrt(1-P1^2-P2^2)) = %.3f" % (SIGMA_ARCSEC, r1, r2, r3))
    print("   min P1 = %.3f at %+.1f arcsec, max P2 = %.3f at %+.1f arcsec, min DOP = %.3f at %+.1f arcsec" % (
        P1[win].min(), angle[win][numpy.argmin(P1[win])], P2[win].max(), angle[win][numpy.argmax(P2[win])],
        DOP[win].min(), angle[win][numpy.argmin(DOP[win])]))
    i = numpy.argmax(numpy.abs(P3) * win)
    print("   max |P3| = %.3f (P3 = %+.3f, DOP = %.3f) at %+.1f arcsec" % (abs(P3[i]), P3[i], DOP[i], angle[i]))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(angle, P1, '-', color='k', lw=1, label=r'$P_1$ conv.')
    ax.plot(angle, P2, '-', color='C0', lw=1, label=r'$P_2$ conv.')
    ax.plot(angle, P3, '-', color='C3', lw=1, label=r'$P_3$ conv.')
    ax.plot(angle, numpy.sqrt(numpy.clip(1 - P1 ** 2 - P2 ** 2, 0, None)), '--', color='C3', lw=1,
            label=r'$\sqrt{1-P_1^2-P_2^2}$ conv.')
    ax.plot(xe, P1e, 'o', color='k', ms=4, mfc='none', label=r'$P_1$ exp')
    ax.plot(xe, P2e, 's', color='C0', ms=4, mfc='none', label=r'$P_2$ exp')
    ax.plot(xe, P3e, '^', color='C3', ms=4, mfc='none', label=r'$\sqrt{1-P_1^2-P_2^2}$ exp')
    ax.set_xlabel('angle [arcsec] (detuning = -angle)')
    ax.set_ylabel('normalized Stokes parameters')
    ax.set_title(r'Si 111, %d $\mu$m, %g eV, crystalpy $\ast$ Gaussian ($\sigma$=%.1f arcsec)' % (thickness_um, ENERGY, SIGMA_ARCSEC))
    ax.set_xlim(-300, 300)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(here, "..", "figures", "crystalpy_desy_p09_convolution%s.png" % suffix), dpi=200)

    fig2, ax2 = plt.subplots(figsize=(6.5, 3.5))
    ax2.plot(sigmas, rms[:, 0], '-', color='k', label=r'$P_1$')
    ax2.plot(sigmas, rms[:, 1], '-', color='C0', label=r'$P_2$')
    ax2.plot(sigmas, rms[:, 2], '-', color='C3', label=r'$\sqrt{1-P_1^2-P_2^2}$')
    ax2.axvline(SIGMA_ARCSEC, color='gray', ls=':')
    ax2.set_xlabel(r'rms width $\sigma$ of the Gaussian [arcsec]')
    ax2.set_ylabel('rms difference with the measurements')
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(os.path.join(here, "..", "figures", "crystalpy_desy_p09_convolution%s_rms.png" % suffix), dpi=200)
    print("Files written to disk: figures/crystalpy_desy_p09_convolution%s.png, figures/crystalpy_desy_p09_convolution%s_rms.png" % (suffix, suffix))
    if SHOW_PLOTS: plt.show()
