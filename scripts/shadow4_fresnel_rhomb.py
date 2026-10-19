


# Fresnel rhomb: two total internal reflections glass -> air.
# The reflecting surfaces are defined with the relative refraction index n2/n1 = 1/N_GLASS
# (n1 = N_GLASS glass, n2 = 1 air), with Re(N_GLASS) = 1.51 as in Born & Wolf (Fig. 1.16) and section 8.
# USE_ABSORPTION = 1 includes the absorption of the glass, N_GLASS = 1.51 + i K_GLASS (equivalent beta of the
# relative index -K_GLASS / 1.51^2); K_GLASS = 1.08e-8 is the extinction coefficient of N-BK7 at 446 nm
# (refractiveindex.info).
USE_ABSORPTION = 1
K_GLASS = 1.08e-8
N_GLASS_REAL = 1.51
N_GLASS = N_GLASS_REAL + 1j * K_GLASS if USE_ABSORPTION else N_GLASS_REAL

import numpy


def run_beamline(phase_diff=0.0, polarization_degree=0.5, alpha=0.0):
    # alpha: rotation of the rhomb around the beam (azimuthal angle of the first surface, rad)
    import numpy as np
    from dabax.dabax_xraylib import DabaxXraylib
    from shadow4.beamline.s4_beamline import S4Beamline

    beamline = S4Beamline()

    #
    #
    #
    from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
    light_source = SourceGeometrical(name='Geometrical Source linear at 45deg', nrays=5000, seed=5676561)
    light_source.set_spatial_type_point()
    light_source.set_depth_distribution_off()
    light_source.set_angular_distribution_uniform(hdiv1=-1e-10, hdiv2=1e-10, vdiv1=-0.8, vdiv2=0.8)
    light_source.set_energy_distribution_singleline(4460, unit='A')
    light_source.set_polarization(polarization_degree=polarization_degree, phase_diff=phase_diff, coherent_beam=1)
    beam = light_source.get_beam()

    beamline.set_light_source(light_source)

    # optical element number XX
    boundary_shape = None

    from shadow4.beamline.optical_elements.mirrors.s4_plane_mirror import S4PlaneMirror
    optical_element = S4PlaneMirror(name='Glass-air internal reflection: n2/n1 = 1/N_GLASS',
                                    boundary_shape=boundary_shape,
                                    f_reflec=1,  # reflectivity of surface: 0=no reflectivity, 1=full polarization
                                    f_refl=1,
                                    # for f_reflec=1: file: 0=prerefl, 2=(mrad, refl), 3=(eV, refl), 4=(eV, mrad, refl); 1=refr index, 5=xraylib, 6=dabax
                                    file_refl='prerefl_glass.dat',  # for f_refl=0,2,3,4
                                    refraction_index=1 / N_GLASS + 0j,  # for f_refl=1
                                    coating_material='Ni', coating_density=8.902,  # for f_refl=5,6
                                    coating_roughness=0,  # for f_refl=0,1,5,6
                                    dabax=None,
                                    # if using dabax (f_reflec=1,f_refl=6), instance of DabaxXraylib() (use None for default)
                                    )

    from syned.beamline.element_coordinates import ElementCoordinates
    coordinates = ElementCoordinates(p=0.1, q=0.1, angle_radial=0.7853981633974483, angle_azimuthal=alpha,
                                     angle_radial_out=0.7853981633974483)
    movements = None
    from shadow4.beamline.optical_elements.mirrors.s4_plane_mirror import S4PlaneMirrorElement
    beamline_element = S4PlaneMirrorElement(optical_element=optical_element, coordinates=coordinates,
                                            movements=movements, input_beam=beam)

    beam, footprint = beamline_element.trace_beam()

    beamline.append_beamline_element(beamline_element)

    beam_one_reflection, footprint_one_reflection = beam.duplicate(), footprint.duplicate()

    # optical element number XX
    boundary_shape = None

    from shadow4.beamline.optical_elements.mirrors.s4_plane_mirror import S4PlaneMirror
    optical_element = S4PlaneMirror(name='Glass-air internal reflection: n2/n1 = 1/N_GLASS',
                                    boundary_shape=boundary_shape,
                                    f_reflec=1,  # reflectivity of surface: 0=no reflectivity, 1=full polarization
                                    f_refl=1,
                                    # for f_reflec=1: file: 0=prerefl, 2=(mrad, refl), 3=(eV, refl), 4=(eV, mrad, refl); 1=refr index, 5=xraylib, 6=dabax
                                    file_refl='prerefl_glass.dat',  # for f_refl=0,2,3,4
                                    refraction_index=1 / N_GLASS + 0j,  # for f_refl=1
                                    coating_material='Ni', coating_density=8.902,  # for f_refl=5,6
                                    coating_roughness=0,  # for f_refl=0,1,5,6
                                    dabax=None,
                                    # if using dabax (f_reflec=1,f_refl=6), instance of DabaxXraylib() (use None for default)
                                    )

    from syned.beamline.element_coordinates import ElementCoordinates
    coordinates = ElementCoordinates(p=0.1, q=0.1, angle_radial=0.7853981633974483, angle_azimuthal=3.141592653589793,  # relative to the first surface
                                     angle_radial_out=0.7853981633974483)
    movements = None
    from shadow4.beamline.optical_elements.mirrors.s4_plane_mirror import S4PlaneMirrorElement
    beamline_element = S4PlaneMirrorElement(optical_element=optical_element, coordinates=coordinates,
                                            movements=movements, input_beam=beam)

    beam, footprint = beamline_element.trace_beam()

    beamline.append_beamline_element(beamline_element)

    # test plot
    if 0:
        from srxraylib.plot.gol import plot_scatter
        plot_scatter(beam.get_photon_energy_eV(nolost=1), beam.get_column(23, nolost=1),
                     title='(Intensity,Photon Energy)', plot_histograms=0)
        plot_scatter(1e6 * beam.get_column(1, nolost=1), 1e6 * beam.get_column(3, nolost=1), title='(X,Z) in microns')
    return beam, footprint, beam_one_reflection, footprint_one_reflection

def run_rhomb_pencil(theta, polarization_degree=0.5, phase_diff=0.0, alpha=0.0, n_glass=None):
    """
    Two internal reflections (Fresnel rhomb) of a pencil beam (a single ray) at the incidence angle theta (rad, to the
    normal). The rhomb is rotated by alpha (rad) around the beam; the second surface is at 180 deg from the first one.
    n_glass is the (complex) refraction index of the glass, n + i k (default N_GLASS).
    Returns the Stokes parameters S0, P1, P2, P3 of the ray after the rhomb (in the frame of the rhomb).
    """
    if n_glass is None: n_glass = N_GLASS
    from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
    from shadow4.beamline.optical_elements.mirrors.s4_plane_mirror import S4PlaneMirror, S4PlaneMirrorElement
    from syned.beamline.element_coordinates import ElementCoordinates
    light_source = SourceGeometrical(name='pencil', nrays=1, seed=5676561)
    light_source.set_spatial_type_point()
    light_source.set_depth_distribution_off()
    light_source.set_angular_distribution_flat(hdiv1=0, hdiv2=0, vdiv1=0, vdiv2=0)
    light_source.set_energy_distribution_singleline(4460, unit='A')
    light_source.set_polarization(polarization_degree=polarization_degree, phase_diff=phase_diff, coherent_beam=1)
    beam = light_source.get_beam()
    for angle_azimuthal in [alpha, numpy.pi]:
        optical_element = S4PlaneMirror(name='Glass-air internal reflection: n2/n1 = 1/N_GLASS',
                                        f_reflec=1, f_refl=1, refraction_index=1 / n_glass + 0j)
        coordinates = ElementCoordinates(p=0.1, q=0.1, angle_radial=theta, angle_radial_out=theta,
                                         angle_azimuthal=angle_azimuthal)
        beam, _ = S4PlaneMirrorElement(optical_element=optical_element, coordinates=coordinates,
                                       input_beam=beam).trace_beam()
    return numpy.array([beam.get_column(col)[0] for col in (30, 42, 43, 44)])


def scan_incident_angle(angles_deg, polarization_degree=0.5, phase_diff=0.0, alpha=0.0, n_glass=None):
    """Scan of the incidence angle (deg) with pencil beams; returns an array (n_angles, 4) with S0, P1, P2, P3."""
    return numpy.array([run_rhomb_pencil(numpy.radians(a), polarization_degree=polarization_degree,
                                         phase_diff=phase_diff, alpha=alpha, n_glass=n_glass) for a in angles_deg])


if __name__ == '__main__':
    import os
    import numpy as np
    import matplotlib
    SHOW_PLOTS = 1
    if not SHOW_PLOTS: matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # incident beam: linear polarization at 45 deg (P=0.5, Phi=0), Jones vector (1,1)/sqrt(2) in the SHADOW basis.
    # (use phase_diff=numpy.pi/2 for a circularly polarized input, converted to linear polarization by the rhomb)
    PHASE_DIFF = 0.0
    beam, footprint, beam_one_reflection, footprint_one_reflection = run_beamline(phase_diff=PHASE_DIFF)

    angle = footprint_one_reflection.get_column(21) # to Y axis, i.e., grazing
    angle1 = 90 - np.degrees(angle)                 # incident (= reflected) angle to the normal
    isort = np.argsort(angle1)
    angle1 = angle1[isort]
    int_s = footprint_one_reflection.get_column(24)[isort]
    int_p = footprint_one_reflection.get_column(25)[isort]
    # phase change (s minus p) introduced by one reflection: phase difference after it minus that of the incident ray
    s_minus_p = footprint_one_reflection.get_column(39)[isort] + PHASE_DIFF
    s_minus_p = np.degrees(np.angle(np.exp(1j * s_minus_p)))
    P3 = footprint.get_column(44)[isort]            # S3/S0 after two reflections

    n = N_GLASS_REAL
    print("n = %s (absorption %s), critical angle %.3f deg" % (N_GLASS, "ON" if USE_ABSORPTION else "OFF",
                                                                np.degrees(np.arcsin(1 / n))))
    for a in [48.62, 51.34, 54.62, 60.0]:
        j = np.argmin(np.abs(angle1 - a))
        print("   angle %.2f deg: phase s-p (one reflection) %.2f deg, P3 (two reflections) %+.4f" % (angle1[j], s_minus_p[j], P3[j]))
    ok = angle1[np.abs(P3) > 0.99]
    print("   |P3| > 0.99 for angles %.1f - %.1f deg" % (ok.min(), ok.max()))

    # Equivalent systems (scan of the incidence angle with pencil beams): light linearly polarized at 45 deg with the
    # rhomb at alpha=0, and light with horizontal linear polarization with the rhomb rotated by alpha=-45 deg.
    # (With the fan beam of run_beamline() the two systems cannot be compared ray by ray, because the vertical fan is
    # not in the plane of incidence of the rotated rhomb.)
    angles_fine = np.linspace(0.1, 89.9, 899)   # 0.1 deg step
    angles_scan = angles_fine[9::10]              # 1 deg step
    scan_45 = scan_incident_angle(angles_fine, polarization_degree=0.5, phase_diff=PHASE_DIFF, alpha=0.0)
    scan_H = scan_incident_angle(angles_scan, polarization_degree=1.0, phase_diff=0.0, alpha=np.radians(-45))
    print("   pencil-beam scan, 45-deg linear (alpha=0) vs horizontal linear (alpha=-45 deg): "
          "max|dS0|=%.1e, max|dP1|=%.1e, max|dP2|=%.1e, max|dP3|=%.1e" % tuple(np.abs(scan_H - scan_45[9::10]).max(axis=0)))
    ok = angles_fine[scan_45[:, 3] > 0.99]

    # Effect of the absorption of the glass, n = N_GLASS_REAL + i k, compared with the lossless glass. The relative
    # index of the glass-air interface is 1 / (N_GLASS_REAL + i k) ~ 1/N_GLASS_REAL - i k / N_GLASS_REAL^2, i.e., an
    # equivalent beta = -k / N_GLASS_REAL^2 (negative, because the absorbing medium is the incident one).
    # k = K_GLASS is the actual glass; larger values are used to show the trend.
    scan_lossless = scan_incident_angle(angles_fine, polarization_degree=0.5, phase_diff=PHASE_DIFF, alpha=0.0,
                                        n_glass=N_GLASS_REAL)
    for k_glass in [K_GLASS, 1e-4, 1e-3, 1e-2]:
        scan_k = scan_incident_angle(angles_fine, polarization_degree=0.5, phase_diff=PHASE_DIFF, alpha=0.0,
                                     n_glass=N_GLASS_REAL + 1j * k_glass)
        beta_eq = numpy.imag(1 / (N_GLASS_REAL + 1j * k_glass))
        print("   absorption k=%.2e (equivalent beta=%.2e): max|dS0/S0|=%.1e, max|dP1|=%.1e, max|dP2|=%.1e, max|dP3|=%.1e, "
              "P3 > 0.99 for %.1f - %.1f deg" % ((k_glass, beta_eq, (np.abs(scan_k[:, 0] - scan_lossless[:, 0]) / scan_lossless[:, 0]).max()) +
              tuple(np.abs(scan_k[:, 1:] - scan_lossless[:, 1:]).max(axis=0)) +
              (angles_fine[scan_k[:, 3] > 0.99].min(), angles_fine[scan_k[:, 3] > 0.99].max())))
    print("   pencil-beam scan, 45-deg linear: P3 > 0.99 for angles %.1f - %.1f deg; max P3 - fan beam P3 at the same angles: %.1e" % (
          ok.min(), ok.max(), np.abs(np.interp(angle1, angles_fine, scan_45[:, 3]) - P3)[(angle1 > 0.1) & (angle1 < 89.9) & (np.abs(angle1 - 41.47) > 0.2)].max()))

    figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    theta_c = np.degrees(np.arcsin(1 / n))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(angle1, int_s, '.', color='C0', ms=2, label=r'$\sigma$')
    ax.plot(angle1, int_p, '.', color='C1', ms=2, label=r'$\pi$')
    ax.axvline(theta_c, color='gray', ls=':', lw=1)
    ax.set_xlabel('Incident (= reflected) angle to the normal [deg]')
    ax.set_ylabel('Ray intensity (one reflection)')
    ax.set_title('(a)', loc='left')
    ax.set_xlim(0, 90); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(figures_dir, "internalR.png"), dpi=200)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(angle1, s_minus_p, '.', color='C2', ms=2)
    ax.axvline(theta_c, color='gray', ls=':', lw=1)
    ax.axhline(45, color='gray', ls='--', lw=1)
    ax.set_xlabel('Incident (= reflected) angle to the normal [deg]')
    ax.set_ylabel(r'Phase change $\phi_\sigma-\phi_\pi$ (one reflection) [deg]')
    ax.set_title('(b)', loc='left')
    ax.set_xlim(0, 90); ax.set_ylim(-190, 190); ax.set_yticks([-180, -90, -45, 0, 45, 90, 180]); ax.grid(alpha=0.3)
    # inset: the two incidence angles with a phase change of 45 deg (Fresnel rhomb), from the ray-traced data
    tir = angle1 > theta_c + 0.5
    x_tir, y_tir = angle1[tir], s_minus_p[tir] - 45
    icross = np.where(np.diff(np.sign(y_tir)) != 0)[0]
    crossings = [x_tir[i] - y_tir[i] * (x_tir[i + 1] - x_tir[i]) / (y_tir[i + 1] - y_tir[i]) for i in icross]
    print("   phase change = 45 deg (one reflection) at %s deg" % ", ".join(["%.2f" % c for c in crossings]))
    axin = ax.inset_axes([0.58, 0.08, 0.39, 0.40])
    axin.plot(angle1, s_minus_p, '.', color='C2', ms=2)
    axin.axhline(45, color='gray', ls='--', lw=1)
    for ic, c in enumerate(crossings):
        axin.axvline(c, color='k', ls=':', lw=0.8)
        axin.plot([c], [45], 'o', color='k', ms=4, mfc='none')
        axin.text(c + (-0.4 if ic == 0 else 0.4), 43.8, r'%.2f$^\circ$' % c, ha='right' if ic == 0 else 'left',
                  fontsize=7)
    axin.set_xlim(45, 59); axin.set_ylim(43.5, 46.5)
    axin.tick_params(labelsize=7); axin.grid(alpha=0.3)
    ax.indicate_inset_zoom(axin, edgecolor='gray')
    fig.tight_layout(); fig.savefig(os.path.join(figures_dir, "internalPhase.png"), dpi=200)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(angles_fine, scan_45[:, 3], '-', color='C3', lw=1, label=r'45$^\circ$-linear, $\alpha$=0')
    ax.plot(angles_scan, scan_H[:, 3], 'o', color='k', ms=3, mfc='none',
            label=r'H-linear, $\alpha$=-45$^\circ$')
    ax.legend(fontsize=8)
    ax.axvline(theta_c, color='gray', ls=':', lw=1)
    ax.set_xlabel('Incident (= reflected) angle to the normal [deg]')
    ax.set_ylabel(r'$P_3=S_3/S_0$ (two reflections)')
    ax.set_xlim(0, 90); ax.set_ylim(-1.05, 1.05); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(figures_dir, "rtrhomb.png"), dpi=200)
    print("Files written to disk: %s/internalR.png, internalPhase.png, rtrhomb.png" % figures_dir)

    if SHOW_PLOTS: plt.show()
