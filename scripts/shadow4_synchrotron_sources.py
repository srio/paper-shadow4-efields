import numpy

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

def get_beam_bending_magnet_as_wiggler(photon_energy=8000.0, nrays=5000):
    # electron beam
    from shadow4.sources.s4_electron_beam import S4ElectronBeam
    electron_beam = S4ElectronBeam(energy_in_GeV=6.04, energy_spread=0.001, current=0.2)
    electron_beam.set_sigmas_all(sigma_x=0, sigma_y=0, sigma_xp=0, sigma_yp=0)
    electron_beam.set_dispersion_all(0, 0, 0, 0)

    # magnetic structure
    from shadow4.sources.wiggler.s4_wiggler import S4Wiggler
    source = S4Wiggler(
        magnetic_field_periodic=0,  # 0=external, 1=periodic
        file_with_magnetic_field="Bcte.dat",  # used only if magnetic_field_periodic=0
        K_vertical=10.0,  # syned Wiggler pars: used only if magnetic_field_periodic=1
        period_length=0.1,  # syned Wiggler pars: used only if magnetic_field_periodic=1
        number_of_periods=10,  # syned Wiggler pars: used only if magnetic_field_periodic=1
        emin=photon_energy,  # Photon energy scan from energy (in eV)
        emax=photon_energy,  # Photon energy scan to energy (in eV)
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
    light_source = S4WigglerLightSource(name='BM using Wiggler Light Source', electron_beam=electron_beam,
                                        magnetic_structure=source, nrays=nrays, seed=5676561)
    return light_source.get_beam()


def get_beam_wiggler(return_light_source=False):
    """Wiggler (6.0 GeV, K = 20.49, 6 periods of 15 cm) at 50 keV, without emittance. With return_light_source=True
    it returns (beam, light_source), the light source giving access to the electron trajectory."""
    # electron beam
    from shadow4.sources.s4_electron_beam import S4ElectronBeam
    electron_beam = S4ElectronBeam(energy_in_GeV=6.0, energy_spread=0.001, current=0.2)
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
        emin=8000.0,  # Photon energy scan from energy (in eV)
        emax=8000.0,  # Photon energy scan to energy (in eV)
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
    import numpy as np
    from srxraylib.plot.gol import plot, plot_image, plot_image_with_histograms, plot_show



    ff = get_beam_bending_magnet_as_wiggler

    titles = ["bending magnet", "BM as wiggler"]
    functions = [get_beam_bending_magnet, get_beam_bending_magnet_as_wiggler]

    for i in range(len(functions)):
        title = titles[i]
        beam = functions[i]()

        #
        # P3
        #
        plot(beam.get_column(6, nolost=1),
             beam.get_column(44, nolost=1),
                                   title=titles[i]+" P3", xtitle="Z' column 6", ytitle="P3 column 44",
             marker='.', linestyle="", grid=1, show=0)

        #
        # trajectory
        #
        plot(beam.get_column(2, nolost=1),
             beam.get_column(1, nolost=1),
                                   title=titles[i]+" trajectory", xtitle="Y column 2", ytitle="X column 1",
             marker='.', linestyle="", grid=1, show=0)

        #
        # velocity
        #
        plot(beam.get_column(2, nolost=1),
             beam.get_column(4, nolost=1),
                                   title=titles[i]+" velocity", xtitle="Y column 2", ytitle="X' column 4",
             marker='.', linestyle="", grid=1, show=0)

    plot_show()