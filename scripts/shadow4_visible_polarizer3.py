import numpy

def run_beamline1(polarization_degree=0.5, angle_azimuthal=0.0, angle_radial=numpy.pi/4):
    import numpy as np
    from dabax.dabax_xraylib import DabaxXraylib
    from shadow4.beamline.s4_beamline import S4Beamline

    beamline = S4Beamline()

    #
    #
    #
    from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
    light_source = SourceGeometrical(name='Geometrical Source linear at 45deg', nrays=10, seed=5676561)
    light_source.set_spatial_type_point()
    light_source.set_depth_distribution_off()
    light_source.set_angular_distribution_flat(hdiv1=0, hdiv2=0, vdiv1=0, vdiv2=0)
    light_source.set_energy_distribution_singleline(7.5, unit='eV')
    light_source.set_polarization(polarization_degree=polarization_degree, phase_diff=0, coherent_beam=1)
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
    coordinates = ElementCoordinates(p=0.1, q=0.1, angle_radial=angle_radial, angle_azimuthal=angle_azimuthal,
                                     angle_radial_out=angle_radial)
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
    from srxraylib.plot.gol import plot, plot_image, plot_image_with_histograms, plot_show

    # two cases
    polarization_degree = [0.5, 1]
    angle_azimuthal = [0, numpy.radians(-45)]
    titles = ["45 deg-linear source, mirror at alpha=0", "H-linear source, mirror at alpha=-45 deg"]



    for jj in range(len(angle_azimuthal)):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>> jj: ", jj)
        angles = numpy.linspace(1, 89, 89)
        IS = numpy.zeros_like(angles)
        IP = numpy.zeros_like(angles)
        S_minus_P = numpy.zeros_like(angles)

        for i, angle in enumerate(angles):
            beam, footprint = run_beamline1(polarization_degree=polarization_degree[jj], angle_azimuthal=angle_azimuthal[jj], angle_radial=numpy.radians(angle))
            Is, Ip = beam.get_column(24, nolost=1)[0], beam.get_column(25, nolost=1)[0]
            S_minus_P[i] = beam.get_column(39, nolost=1)[0]
            IS[i] = Is
            IP[i] = Ip
            # print(angle, Is, Ip)

        plot(angles, IS, angles, IP, marker=['.','.','.'], linestyle=['','',''], legend=['s-pol','p-pol'], ytitle="Ray intensity", xtitle="Reflected angle (to normal) [deg]", grid=1, title=titles[jj], show=0)
        plot(angles, numpy.degrees(S_minus_P),marker='.', linestyle='', xtitle="Reflected angle (to normal) [deg]", ytitle="Phase s-p [deg]", grid=1, title=titles[jj], show=0)

    plot_show()