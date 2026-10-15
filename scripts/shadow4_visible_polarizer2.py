def run_beamline1():
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
    light_source.set_energy_distribution_singleline(7.5, unit='eV')
    light_source.set_polarization(polarization_degree=0.5, phase_diff=0, coherent_beam=1)
    beam = light_source.get_beam()

    beamline.set_light_source(light_source)

    # optical element number XX
    boundary_shape = None

    from shadow4.beamline.optical_elements.mirrors.s4_plane_mirror import S4PlaneMirror
    optical_element = S4PlaneMirror(name=': use delta=1-n2/n1=1-1.51', boundary_shape=boundary_shape,
                                    f_reflec=1,  # reflectivity of surface: 0=no reflectivity, 1=full polarization
                                    f_refl=1,
                                    # for f_reflec=1: file: 0=prerefl, 2=(mrad, refl), 3=(eV, refl), 4=(eV, mrad, refl); 1=refr index, 5=xraylib, 6=dabax
                                    file_refl='prerefl_glass.dat',  # for f_refl=0,2,3,4
                                    refraction_index=1.51 + 0j,  # for f_refl=1
                                    coating_material='Ni', coating_density=8.902,  # for f_refl=5,6
                                    coating_roughness=0,  # for f_refl=0,1,5,6
                                    dabax=None,
                                    # if using dabax (f_reflec=1,f_refl=6), instance of DabaxXraylib() (use None for default)
                                    )

    from syned.beamline.element_coordinates import ElementCoordinates
    coordinates = ElementCoordinates(p=0.1, q=0.1, angle_radial=0.7853981633974483, angle_azimuthal=0.0,
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

    return beam, footprint

def run_beamline2():
    import numpy as np
    from dabax.dabax_xraylib import DabaxXraylib
    from shadow4.beamline.s4_beamline import S4Beamline

    beamline = S4Beamline()

    #
    #
    #
    from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
    # source linearly polarized at 45 deg (P=0.5), like in run_beamline1; the beam movement below rotates the whole
    # source by +45 deg around the beam axis, so that the field becomes horizontal and the divergence plane is rotated
    # by 45 deg (in the scattering plane of the mirror at alpha=-45 deg)
    light_source = SourceGeometrical(name='Geometrical Source linear at 45 deg (rotated to H)', nrays=5000, seed=5676561)
    light_source.set_spatial_type_point()
    light_source.set_depth_distribution_off()
    light_source.set_angular_distribution_uniform(hdiv1=-1e-10, hdiv2=1e-10, vdiv1=-0.8, vdiv2=0.8)
    light_source.set_energy_distribution_singleline(7.5, unit='eV')
    light_source.set_polarization(polarization_degree=0.5, phase_diff=0, coherent_beam=1)
    beam = light_source.get_beam()

    beamline.set_light_source(light_source)

    # optical element number XX

    from shadow4.beamline.optical_elements.ideal_elements.s4_beam_movement import S4BeamMovement
    optical_element = S4BeamMovement(name='Undefined', apply_flag=1,
                                     translation_x=0, translation_y=0, translation_z=0,
                                     rotation_x=0, rotation_y=0.7853981633974483, rotation_z=0)  # +45 deg (rad)

    from syned.beamline.element_coordinates import ElementCoordinates
    coordinates = ElementCoordinates(p=0.0, q=0.0, angle_radial=0.0, angle_azimuthal=0.0, angle_radial_out=0.0)
    from shadow4.beamline.optical_elements.ideal_elements.s4_beam_movement import S4BeamMovementElement
    beamline_element = S4BeamMovementElement(optical_element=optical_element, coordinates=coordinates, input_beam=beam)

    beam, footprint = beamline_element.trace_beam()

    beamline.append_beamline_element(beamline_element)

    # optical element number XX
    boundary_shape = None

    from shadow4.beamline.optical_elements.mirrors.s4_plane_mirror import S4PlaneMirror
    optical_element = S4PlaneMirror(name='Mirror at alpha=45 - use delta=1-n2/n1=1-1.51', boundary_shape=boundary_shape,
                                    f_reflec=1,  # reflectivity of surface: 0=no reflectivity, 1=full polarization
                                    f_refl=1,
                                    # for f_reflec=1: file: 0=prerefl, 2=(mrad, refl), 3=(eV, refl), 4=(eV, mrad, refl); 1=refr index, 5=xraylib, 6=dabax
                                    file_refl='prerefl_glass.dat',  # for f_refl=0,2,3,4
                                    refraction_index=1.51 + 0j,  # for f_refl=1
                                    coating_material='Ni', coating_density=8.902,  # for f_refl=5,6
                                    coating_roughness=0,  # for f_refl=0,1,5,6
                                    dabax=None,
                                    # if using dabax (f_reflec=1,f_refl=6), instance of DabaxXraylib() (use None for default)
                                    )

    from syned.beamline.element_coordinates import ElementCoordinates
    coordinates = ElementCoordinates(p=0.1, q=0.1, angle_radial=0.7853981633974483, angle_azimuthal=-0.7853981633974483,
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

    return beam, footprint

if __name__ == '__main__':
    import numpy as np
    from srxraylib.plot.gol import plot, plot_image, plot_image_with_histograms, plot_show

    beam1, footprint1 = run_beamline1()
    beam2, footprint2 = run_beamline2()

    titles=["45 deg linear - mirror at alpha=0", "H linear - mirror at alpha=-45"]
    for i, footprint in enumerate([footprint1, footprint2]):
        angle = footprint.get_column(21)  # to Y axis, i.e., grazing
        int_s = footprint.get_column(24)
        int_p = footprint.get_column(25)
        s_minus_p = footprint.get_column(39)

        angle1 = 90 - np.degrees(angle)
        plot(angle1, int_s, angle1, int_p, marker=['.', '.', '.'], linestyle=['', '', ''], legend=['s-pol', 'p-pol'],
             ytitle="Ray intensity", xtitle="Reflected angle (to normal) [deg]", grid=1, title=titles[i], show=0)

        plot(angle1, np.degrees(s_minus_p), marker='.', linestyle='', xtitle="Reflected angle (to normal) [deg]",
             ytitle="Phase s-p [deg]", grid=1, title=titles[i], show=0)

    plot_show()