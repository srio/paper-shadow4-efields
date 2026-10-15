#
# Glass mirror (n=1.51) at 7.5 eV: s and p reflectivities and phase difference versus incidence angle (Figs. 7, 8).
#
# The calculation is done in two physically equivalent ways:
#   case 1: source linearly polarized at 45 deg (P=0.5, field along u_sigma + u_pi), divergence in the vertical
#           plane, mirror facing up (alpha=0). The divergence scans the incidence angle on the mirror.
#   case 2: the same source beam rotated by +45 deg around the beam axis (the field becomes horizontal, and the
#           divergence plane is rotated by 45 deg), with the field stored in the default (horizontal) sigma basis,
#           i.e., a horizontal linearly polarized beam, and the mirror rotated by alpha=-45 deg around the beam.
# Both cases must give the same footprint (s and p intensities and phase difference, in the mirror frame).
# Note: with alpha=+45 deg the intensities are the same but the phase s-p differs by 180 deg, and if the
# divergence plane is not rotated with the mirror, only the central ray is equivalent.
#
# Usage:
#   PYTHONPATH=~/OASYS2.0/shadow4 ~/venv/oasys2/bin/python shadow4_visible_polarizer.py
#
import numpy


def get_source_beam(polarization_degree=0.5, phase_diff=0.0):
    from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
    light_source = SourceGeometrical(name='Geometrical Source', nrays=5000, seed=5676561)
    light_source.set_spatial_type_point()
    light_source.set_depth_distribution_off()
    light_source.set_angular_distribution_uniform(hdiv1=-1e-10, hdiv2=1e-10, vdiv1=-0.8, vdiv2=0.8)
    light_source.set_energy_distribution_singleline(7.5, unit='eV')
    light_source.set_polarization(polarization_degree=polarization_degree, phase_diff=phase_diff, coherent_beam=1)
    return light_source.get_beam()


def rotate_source_beam(beam, angle):
    """Rotates the whole beam (positions, directions, electric fields) by angle (rad) around the beam axis (y), and
    re-expresses the field in the default basis (sigma = projection of x, pi = sigma x v)."""
    from shadow4.tools.arrayofvectors import vector_cross, vector_norm, vector_dot
    beam = beam.duplicate()
    beam.rotate(angle, axis=2)
    j0, j1 = beam.get_jones_components()
    e_S, e_P = beam.get_efield_directions()
    E = j0[:, None] * e_S + j1[:, None] * e_P                  # complex electric field (3D)
    v = beam.get_columns([4, 5, 6]).T
    s = numpy.zeros_like(v)
    s[:, 0] = 1.0
    s = vector_norm(s - vector_dot(s, v)[:, None] * v)
    p = vector_norm(vector_cross(s, v))
    beam.set_jones_components(vector_dot(E, s), vector_dot(E, p), e_S=s, e_P=p)
    return beam


def run_beamline(beam, angle_azimuthal=0.0):
    """Glass mirror at 45 deg incidence, rotated by angle_azimuthal (rad) around the beam."""
    from shadow4.beamline.optical_elements.mirrors.s4_plane_mirror import S4PlaneMirror, S4PlaneMirrorElement
    from syned.beamline.element_coordinates import ElementCoordinates

    optical_element = S4PlaneMirror(name='glass mirror', boundary_shape=None,
                                    f_reflec=1,  # reflectivity of surface: 0=no reflectivity, 1=full polarization
                                    f_refl=1,    # 1=refraction index
                                    refraction_index=1.51 + 0j,
                                    coating_roughness=0,
                                    )
    coordinates = ElementCoordinates(p=0.1, q=0.1, angle_radial=numpy.pi / 4, angle_azimuthal=angle_azimuthal,
                                     angle_radial_out=numpy.pi / 4)
    beamline_element = S4PlaneMirrorElement(optical_element=optical_element, coordinates=coordinates,
                                            movements=None, input_beam=beam)
    beam, footprint = beamline_element.trace_beam()
    return beam, footprint


if __name__ == '__main__':

    from srxraylib.plot.gol import plot, plot_show

    source_beam = get_source_beam(polarization_degree=0.5, phase_diff=0.0)
    beam1, footprint1 = run_beamline(source_beam, angle_azimuthal=0.0)
    beam2, footprint2 = run_beamline(rotate_source_beam(source_beam, numpy.pi / 4),
                                     angle_azimuthal=-numpy.pi / 4)

    titles = ["source: linear at 45 deg (P=0.5), mirror at alpha=0",
              "source: horizontal linear (rotated), mirror at alpha=-45 deg", ]

    # comparison of the two cases
    d_s = numpy.abs(footprint1.get_column(24) - footprint2.get_column(24)).max()
    d_p = numpy.abs(footprint1.get_column(25) - footprint2.get_column(25)).max()
    d_phase = numpy.abs(numpy.angle(numpy.exp(1j * (footprint1.get_column(39) - footprint2.get_column(39))))).max()
    print("case 1 vs case 2: max |dI_s| = %.2e, max |dI_p| = %.2e, max |d(phase s-p)| = %.2e rad" %
          (d_s, d_p, d_phase))

    for i, footprint in enumerate([footprint1, footprint2]):
        angle = footprint.get_column(21)  # to Y axis, i.e., grazing
        int_s = footprint.get_column(24)
        int_p = footprint.get_column(25)
        s_minus_p = footprint.get_column(39)

        angle1 = 90 - numpy.degrees(angle)
        plot(angle1, int_s, angle1, int_p, marker=['.', '.', '.'], linestyle=['', '', ''], legend=['s-pol', 'p-pol'],
             ytitle="Ray intensity", xtitle="Reflected angle (to normal) [deg]", title=titles[i], grid=1, show=0)

        plot(angle1, numpy.degrees(s_minus_p), marker='.', linestyle='', xtitle="Reflected angle (to normal) [deg]",
             ytitle="Phase s-p [deg]", title=titles[i], grid=1, show=0)

    plot_show()
