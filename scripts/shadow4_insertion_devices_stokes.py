#
# Stokes parameters of the rays of synchrotron sources versus the vertical direction Z' (column 6): a bending magnet,
# a wiggler and an undulator, all without electron beam emittance, so that Z' is the vertical emission angle psi.
#
# Bending magnet: ESRF-1 (6.04 GeV, B = -0.8 T, critical energy 19.4 keV) at 8 keV. The SHADOW4 rays are compared with
# the synchrotron radiation formulas (e.g., Jackson, Classical Electrodynamics, Sec. 14.6; Kim, X-ray Data Booklet,
# Sec. 2.1):
#     E_sigma ~ (1 + X^2) K_2/3(xi),    |E_pi| ~ X (1 + X^2)^1/2 K_1/3(xi),    X = gamma psi,
#     xi = (E / 2 E_c) (1 + X^2)^3/2,  phase difference +/- 90 deg with the sign of psi.
# The magnetic field B < 0 makes the electrons turn clockwise seen from above (toward +x, see
# S4BendingMagnetLightSource): the radiation above the orbit plane (psi > 0) is right-handed (S3 < 0) and below it is
# left-handed (S3 > 0), as in F. Schafers, in Modern Developments in X-Ray and Neutron Optics (Springer, 2008), p. 30.
# This is verified with the Lienard-Wiechert field of an electron on a circular orbit (handedness_lienard_wiechert).
#
# Wiggler: 6.0 GeV, K = 20.49, 6 periods of 15 cm, at 8 keV. At each pole the radiation is like that of a bending
# magnet: the handedness above (below) the orbit plane depends on the sense of rotation of the electron at the pole
# (sign of the trajectory curvature d2x/dy2). The rays are separated by the sign of the curvature at their emission
# point; the two families have opposite handedness, and S3 cancels for the whole beam.
#
# Undulator: 6.0 GeV, K = 1.656, 111 periods of 18 mm, at the resonance 8.008 keV, within +/-18 urad. The amplitudes
# are from the undulator emission, with a phase difference of +/- 90 deg with the sign of psi (as for the bending magnet).
#
# The theoretical values are plotted only for the bending magnet.
#
# Usage:
#   PYTHONPATH=~/OASYS2.0/shadow4 ~/venv/oasys2/bin/python shadow4_insertion_devices_stokes.py
#   (the figures are written to ../figures/stokes_<source>.png, <source> = bending_magnet, wiggler, undulator, and
#   ../figures/stokes_synchrotron_sources.png with the three sources, used in the paper)
#
import os
import numpy
import scipy.special

SHOW_PLOTS = 1

PHOTON_ENERGY = 8000.0
ELECTRON_ENERGY = 6.04       # GeV
CRITICAL_ENERGY = 0.665 * ELECTRON_ENERGY ** 2 * 0.8 * 1e3   # eV


def stokes_analytical(psi, photon_energy=PHOTON_ENERGY, electron_energy=ELECTRON_ENERGY,
                      critical_energy=CRITICAL_ENERGY):
    """Normalized Stokes parameters (P1, P2, P3) and angular distribution (arbitrary units) of the bending magnet
    radiation versus the vertical angle psi (rad), for electrons turning clockwise seen from above."""
    gamma = electron_energy / 0.51099895e-3
    X = gamma * psi
    xi = 0.5 * photon_energy / critical_energy * (1 + X ** 2) ** 1.5
    a_s = (1 + X ** 2) * scipy.special.kv(2 / 3, xi)
    a_p = numpy.abs(X) * numpy.sqrt(1 + X ** 2) * scipy.special.kv(1 / 3, xi)
    S0 = a_s ** 2 + a_p ** 2
    return (a_s ** 2 - a_p ** 2) / S0, 0 * psi, -numpy.sign(psi) * 2 * a_s * a_p / S0, S0


def handedness_lienard_wiechert(psi_gamma=0.5, gamma=30.0, clockwise=True):
    """Independent check of the handedness: far field of an electron on a circular orbit in the (x, y) plane (Lienard-
    Wiechert, E ~ n x ((n - beta) x beta_dot) / (1 - n.beta)^3, observer time t = t' - n.r), moving along +y at
    t' = 0 and turning toward +x (clockwise seen from above) or -x, for an observer at the vertical angle
    psi = psi_gamma / gamma. Returns the sense of rotation of the field in time, in the standard frame
    (u_sigma, -u_pi, v), for an observer looking toward the source: +1 counter-clockwise (LCP), -1 clockwise (RCP).
    Units c = R = 1 (the handedness does not depend on gamma)."""
    beta = numpy.sqrt(1 - 1 / gamma ** 2)
    w = beta
    sx = 1 if clockwise else -1
    tp = numpy.linspace(-3 / gamma, 3 / gamma, 200001) / w
    r = numpy.stack([sx * (1 - numpy.cos(w * tp)), numpy.sin(w * tp), 0 * tp], -1)
    b = beta * numpy.stack([sx * numpy.sin(w * tp), numpy.cos(w * tp), 0 * tp], -1)
    bd = beta * w * numpy.stack([sx * numpy.cos(w * tp), -numpy.sin(w * tp), 0 * tp], -1)
    psi = psi_gamma / gamma
    n = numpy.array([0, numpy.cos(psi), numpy.sin(psi)])
    E = numpy.cross(n, numpy.cross(n - b, bd)) / ((1 - b @ n) ** 3)[:, None]
    t = tp - r @ n
    u_s = numpy.array([1.0, 0, 0])
    u_p = numpy.cross(u_s, n)                       # SHADOW convention u_pi = u_sigma x v
    Ex, Ey = E @ u_s, -(E @ u_p)                    # standard frame (u_sigma, -u_pi, v)
    L = numpy.sum(Ex[:-1] * numpy.diff(Ey) - Ey[:-1] * numpy.diff(Ex))
    return int(numpy.sign(L))


def draw_stokes(ax1, ax2, zp, S, title="", analytical=None, groups=None, group_component=3,
                annotate_handedness=False, p_range=None, fs=1.0, legends=True):
    """Draws (ax1) the Stokes parameters S (shape (4, nrays)) summed in bins of zp, and (ax2) the normalized Stokes
    parameters of the rays, versus zp (rad). analytical: function of psi returning (P1, P2, P3, S0), plotted as lines.
    groups: list of (mask, label, marker, color) to plot the Stokes parameter group_component (2 or 3) separately for
    families of rays (e.g., wiggler poles, undulator quadrants). p_range: (min, max) of the axis of ax2 (if not None,
    the parameters outside it, e.g., P1 ~ 1, are not plotted). fs: scale of the font sizes."""
    P = S[1:] / S[0]
    names = ['$S_0$', '$S_1$', '$S_2$', '$S_3$']
    colors = ['k', 'C0', 'C2', 'C3']

    # (a) Stokes parameters summed in bins of Z' (angular distributions)
    if ax1 is None: ax1 = _NullAxes()
    bins = numpy.linspace(zp.min(), zp.max(), 81)
    centers = 0.5 * (bins[1:] + bins[:-1])
    hS = [numpy.histogram(zp, bins=bins, weights=S[k])[0] for k in range(4)]
    norm = hS[0].max()
    for k in range(4):
        ax1.plot(1e6 * centers, hS[k] / norm, 'o', ms=(5 if k == 0 else 3) * fs, color=colors[k],
                 mfc='none' if k == 0 else colors[k], label=names[k], zorder=3 if k == 0 else 2)
    if groups is not None:
        for mask, label, marker, gcolor in groups:
            hg = numpy.histogram(zp[mask], bins=bins, weights=S[group_component][mask])[0]
            ax1.plot(1e6 * centers, hg / norm, marker, ms=3 * fs, color=gcolor, label=names[group_component] + ', ' + label)
    if analytical is not None:
        psi = numpy.linspace(zp.min(), zp.max(), 501)
        P1a, P2a, P3a, S0a = analytical(psi)
        ax1.plot(1e6 * psi, S0a / S0a.max(), '-', color='k', lw=0.8, label='analytical')
        ax1.plot(1e6 * psi, P1a * S0a / S0a.max(), '-', color='C0', lw=0.8)
        ax1.plot(1e6 * psi, P3a * S0a / S0a.max(), '-', color='C3', lw=0.8)
    ax1.axhline(0, color='gray', lw=0.5); ax1.axvline(0, color='gray', lw=0.5)
    ax1.set_ylabel("Stokes param. (binned)", fontsize=10 * fs)
    if legends: ax1.legend(fontsize=7 * fs, loc='upper right')
    ax1.set_title(title, fontsize=10 * fs)

    # (b) normalized Stokes parameters of the rays
    step = max(1, zp.size // 1000)
    pnames = ['$P_1$', '$P_2$', '$P_3$']
    for k, color in enumerate(['C0', 'C2', 'C3']):
        if p_range is not None and (numpy.abs(P[k]).min() > max(numpy.abs(p_range))):        # out of range
            ax2.text(0.98, 0.95, "%s > %.4f (not shown)" % (pnames[k], P[k].min()), transform=ax2.transAxes,
                     ha='right', va='top', fontsize=8 * fs, color=color)
            continue
        if groups is not None and k == group_component - 1:
            for mask, glabel, marker, gcolor in groups:
                idx = numpy.argwhere(mask).ravel()[::step]
                ax2.plot(1e6 * zp[idx], P[k][idx], marker, ms=2 * fs, color=gcolor, label=pnames[k] + ', ' + glabel)
        else:
            ax2.plot(1e6 * zp[::step], P[k][::step], '.', ms=2 * fs, color=color, label=pnames[k] + ' (rays)')
    if analytical is not None:
        ax2.plot(1e6 * psi, P1a, '-', color='C0', lw=0.8, label='analytical')
        ax2.plot(1e6 * psi, P3a, '-', color='C3', lw=0.8)
    ax2.axhline(0, color='gray', lw=0.5); ax2.axvline(0, color='gray', lw=0.5)
    if annotate_handedness:
        ax2.text(0.97 * 1e6 * zp.max(), -0.75, 'RCP\n(above)', ha='right', fontsize=8 * fs, color='C3')
        ax2.text(0.97 * 1e6 * zp.min(), 0.45, 'LCP\n(below)', ha='left', fontsize=8 * fs, color='C3')
    ax2.set_xlabel(r"Z' ($\mu$rad)", fontsize=10 * fs)
    ax2.set_ylabel(r"$P_k$ of the rays", fontsize=10 * fs)
    ax2.set_ylim(*(p_range if p_range is not None else (-1.05, 1.05)))
    if legends: ax2.legend(fontsize=7 * fs, loc='lower left')
    for ax in (ax1, ax2): ax.tick_params(labelsize=9 * fs)


class _NullAxes(object):
    """Axes that ignore all the drawing calls (to draw only the right column)."""
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def plot_stokes(zp, S, file_png=None, **kwargs):
    """Figure with (a) the Stokes parameters summed in bins of zp and (b) the normalized Stokes parameters of the rays
    (see draw_stokes)."""
    import matplotlib
    if not SHOW_PLOTS: matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 7.0), sharex=True)
    draw_stokes(ax1, ax2, zp, S, **kwargs)
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, va='top')
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, va='top')
    fig.tight_layout()
    if file_png is not None:
        fig.savefig(file_png, dpi=200)
        print("File written to disk: %s" % file_png)
    if SHOW_PLOTS: plt.show()


def plot_stokes_all(results, file_png=None):
    """Figure for the paper: one row per source, with (left) the polarization ellipses of a sample of rays and of the
    beam, and (right) the normalized Stokes parameters of the rays versus Z'.
    results: list of (zp, S, kwargs of draw_stokes, (ellS, ellJ, kwargs of draw_ellipse))."""
    import matplotlib
    if not SHOW_PLOTS: matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shadow4_polarization_ellipse import draw_ellipse
    fig, axs = plt.subplots(len(results), 2, figsize=(11.0, 4.6 * len(results)),
                            gridspec_kw=dict(width_ratios=[1, 1.25]))
    for i, (zp, S, kwargs, (ellS, ellJ, ekwargs)) in enumerate(results):
        kw = dict(kwargs); title = kw.pop("title", "")
        legend = dict(fontsize=8.5, loc='upper center', ncol=2, framealpha=0.9) if i == 0 else False
        ek = dict(ekwargs); ek["ray_label"] = ""                 # ray details in the caption of the figure
        draw_ellipse(axs[i, 0], ellS, ellJ, fs=1.25, title_stokes=False, legend=legend, **ek)
        draw_stokes(None, axs[i, 1], zp, S, fs=1.25, **kw)
        axs[i, 0].set_title(title, fontsize=13, loc='left')
        axs[i, 0].text(0.02, 0.03, '(%s)' % "abcdefgh"[2 * i], transform=axs[i, 0].transAxes, va='bottom', fontsize=13)
        axs[i, 1].text(0.02, 0.95, '(%s)' % "abcdefgh"[2 * i + 1], transform=axs[i, 1].transAxes, va='top', fontsize=13)
    fig.tight_layout()
    if file_png is not None:
        fig.savefig(file_png, dpi=200)
        print("File written to disk: %s" % file_png)
    if SHOW_PLOTS: plt.show()


def print_stokes(zp, S, groups=None):
    """Averaged normalized Stokes parameters above and below the orbit plane (and for each group of rays)."""
    sets = [("all rays", numpy.ones_like(zp, dtype=bool))]
    if groups is not None: sets += [(label, mask) for mask, label, marker, gcolor in groups]
    for gname, gmask in sets:
        for name, mask in [("above (Z'>0)", zp > 0), ("below (Z'<0)", zp < 0), ("all", zp == zp)]:
            m = gmask & mask
            print("  %-34s %-13s: %6d rays, <P1> = %+.4f, <P2> = %+.4f, <P3> = %+.4f" %
                  (gname, name, m.sum(), *(S[1:, m].sum(axis=1) / S[0, m].sum())))


if __name__ == "__main__":
    from shadow4_synchrotron_sources import get_beam_bending_magnet, get_beam_wiggler, get_beam_undulator

    for clockwise in [True, False]:
        for psi_gamma in [0.5, -0.5]:
            h = handedness_lienard_wiechert(psi_gamma=psi_gamma, clockwise=clockwise)
            print("Lienard-Wiechert: electrons %-17s, psi %s 0: %s" % (
                "clockwise" if clockwise else "counter-clockwise", ">" if psi_gamma > 0 else "<",
                "LCP (S3>0)" if h > 0 else "RCP (S3<0)"))

    figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    results = []
    for SOURCE in ["bending_magnet", "wiggler", "undulator"]:
        print("\n===== %s =====" % SOURCE)
        groups, analytical, title, kwargs = None, None, "", {}
        if SOURCE == "bending_magnet":
            beam = get_beam_bending_magnet(photon_energy=PHOTON_ENERGY, nrays=20000)
            analytical = stokes_analytical
            title = "Bending magnet (ESRF1, B=-0.8 T, 8 keV)"
            kwargs = dict(analytical=analytical, annotate_handedness=True)
        elif SOURCE == "wiggler":
            beam, light_source = get_beam_wiggler(return_light_source=True)
            # sign of the trajectory curvature (d2x/dy2) at the emission point of each ray
            traj = light_source._S4WigglerLightSource__result_trajectory
            curv = numpy.interp(beam.get_column(2, nolost=1), traj[1], traj[6])
            groups = [(curv > 0, "e- turning to +x", 'v', 'C3'), (curv < 0, "e- turning to -x", '^', 'C4')]
            title = "Wiggler (K=20.49, 6 periods, 8 keV)"
            kwargs = dict(groups=groups, group_component=3)
        else:
            beam = get_beam_undulator()
            xp = beam.get_column(4, nolost=1)
            zp_ = beam.get_column(6, nolost=1)
            # quadrants: P2 changes sign with X' Z'
            groups = [(xp * zp_ > 0, "X'Z'>0", 'v', 'C2'), (xp * zp_ < 0, "X'Z'<0", '^', 'C8')]
            title = "Undulator (K=1.656, 111 periods, 8.008 keV)"
            kwargs = dict(groups=groups, group_component=2, p_range=(-0.06, 0.06))
        kwargs["title"] = title

        zp = beam.get_column(6, nolost=1)                      # Z' (direction cosine ~ vertical angle)
        S = numpy.array([beam.get_column(col, nolost=1) for col in [30, 31, 32, 33]])
        print_stokes(zp, S, groups=groups)
        if analytical is not None:
            P = S[1:] / S[0]
            P1a, P2a, P3a, S0a = analytical(zp)
            print("  max |P - P_analytical| over the rays: P1 %.2e, P2 %.2e, P3 %.2e" %
                  (numpy.abs(P[0] - P1a).max(), numpy.abs(P[1] - P2a).max(), numpy.abs(P[2] - P3a).max()))
        plot_stokes(zp, S, file_png=os.path.join(figures_dir, "stokes_%s.png" % SOURCE), **kwargs)

        # polarization ellipses (beam and rays) for the combined figure
        iray = 0
        ellS = beam.get_polarization_ellipse(flag_return=0)
        ellJ = beam.get_polarization_ellipse(flag_return=1)
        ray_label = " (Z'%s0, %s" % ((">", "above") if zp[iray] > 0 else ("<", "below"))
        if SOURCE == "wiggler":
            ray_label += ", e- turning to %sx" % ("+" if curv[iray] > 0 else "-")
        ray_label += ")"
        ekwargs = dict(iray=iray, nsample=20, ray_label=ray_label, equal_aspect=(SOURCE != "undulator"))
        results.append((zp, S, kwargs, (ellS, ellJ, ekwargs)))

    plot_stokes_all(results, file_png=os.path.join(figures_dir, "stokes_synchrotron_sources.png"))
