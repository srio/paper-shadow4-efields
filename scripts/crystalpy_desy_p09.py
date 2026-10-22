def get_data(RorT):

    if RorT == "R":
        geometry_type_index = 0
    elif RorT == "T":
        geometry_type_index = 2

    import numpy
    from crystalpy.util.calc_xcrystal import calc_xcrystal_angular_scan, calc_xcrystal_energy_scan, \
        calc_xcrystal_alphazachariasen_scan
    from dabax.dabax_xraylib import DabaxXraylib

    bunch_out_dict, diffraction_setup, deviations = calc_xcrystal_angular_scan(
        crystal_name='Si',
        thickness=1e-05,  # in m
        miller_h=1,
        miller_k=1,
        miller_l=1,
        asymmetry_angle=0.0,
        energy=2838.0,  # in eV
        angle_deviation_min=-0.001,  # in rad
        angle_deviation_max=0.001,  # in rad
        angle_deviation_points=2000,
        angle_center_flag=1,  # 0=Absolute angle, 1=Theta Bragg Corrected, 2=Theta Bragg
        flag_calculate_stokes=1,  # 0=No, 1=Yes, 2=Yes normalized
        jones_in=[1, 0],  # for flag_calculate_stokes > 0
        chi_deg=45,  # for flag_calculate_stokes > 0
        calculation_method=1,  # 0=Zachariasen, 1=Guigay
        is_thick=0,
        use_transfer_matrix=1,
        geometry_type_index=geometry_type_index,
        calculation_strategy_flag=0,  # 0=mpmath 1=numpy 2=numpy-truncated
        material_constants_library_flag=1,  # 0=xraylib, 1=dabax
        dabax=DabaxXraylib(file_f1f2="f1f2_Windt.dat"),
        # for material_constants_library_flag=1, the pointer to DabaxXraylib()
    )

    tmp = numpy.zeros((bunch_out_dict["energies"].size, 7 + 4))
    tmp[:, 0] = deviations / 1e-06
    tmp[:, 1] = 2838.0
    tmp[:, 2] = bunch_out_dict["phaseP"]
    tmp[:, 3] = bunch_out_dict["phaseS"]
    # tmp[:, 4] = circular polarization
    tmp[:, 5] = bunch_out_dict["intensityP"]
    tmp[:, 6] = bunch_out_dict["intensityS"]

    # from srxraylib.plot.gol import plot
    # plot(tmp[:, 0], tmp[:, 6], tmp[:, 0], tmp[:, 5], xtitle="angle", legend=["S-pol", "P-pol"])

    return     tmp[:, 0], tmp[:, 6], tmp[:, 5], diffraction_setup

if __name__ == "__main__":
    import numpy
    from srxraylib.plot.gol import plot, plot_show
    import matplotlib.pylab as plt

    x, rs, rp, DsetupR = get_data("R")
    plot(x, rs, x, rp, xtitle=r"$\theta-\theta_B$ [$\mu$rad]", ytitle="Reflectivity", legend=[r"$\sigma$", r"$\pi$"], grid=1, show=0)
    plt.savefig("../figures/crystalpyR.png")

    x, ts, tp, DsetupT = get_data("T")
    plot(x, ts, x, tp, xtitle=r"$\theta-\theta_B$ [$\mu$rad]", ytitle="Transmittance", legend=[r"$\sigma$", r"$\pi$"], grid=1, show=0)
    plt.savefig("../figures/crystalpyT.png")

    print("Angle Bragg corrected [deg]: ", numpy.degrees(DsetupR.angleBraggCorrected(2838.0)))
    print("Darwin half-width S [urad]: ", 1e6 * DsetupR.darwinHalfwidthS(2838.0))

    plot_show()