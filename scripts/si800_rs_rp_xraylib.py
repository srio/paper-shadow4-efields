#
# Si 800 reflection at theta_B ~ 45 deg (section 7, X-ray polarizers):
# Bragg angle, corrected Bragg angle, complex amplitudes r_s, r_p and reflectances R_s, R_p,
# calculated with crystalpy using xraylib for the structure factors.
#
# Usage:
#   PYTHONPATH=~/OASYS2.0/shadow4 ~/venv/oasys2/bin/python si800_rs_rp_xraylib.py
#
import numpy
import scipy.constants as codata

from crystalpy.diffraction.DiffractionSetupXraylib import DiffractionSetupXraylib
from crystalpy.diffraction.GeometryType import BraggDiffraction
from crystalpy.diffraction.Diffraction import Diffraction
from crystalpy.util.ComplexAmplitudePhoton import ComplexAmplitudePhoton


def amplitudes(setup, energy, deviation=0.0, calculation_method=1, is_thick=1):
    """r_s, r_p at a deviation (rad) from the corrected Bragg angle."""
    direction = setup.vectorIncomingPhotonDirection(energy, deviation, angle_center_flag=1)
    photon = ComplexAmplitudePhoton(energy_in_ev=energy, direction_vector=direction, Esigma=1.0, Epi=1.0)
    c = Diffraction.calculateDiffractedComplexAmplitudes(setup, photon,
                                                         calculation_method=calculation_method,
                                                         is_thick=is_thick)
    return complex(numpy.ravel(c["S"])[0]), complex(numpy.ravel(c["P"])[0])


if __name__ == "__main__":
    setup = DiffractionSetupXraylib(geometry_type=BraggDiffraction(), crystal_name="Si", thickness=1e-2,
                                    miller_h=8, miller_k=0, miller_l=0,
                                    asymmetry_angle=0.0, azimuthal_angle=0.0)

    energy_45 = codata.h * codata.c / codata.e / (2 * setup.dSpacingSI() * numpy.sin(numpy.pi / 4))
    print("Si 800 (xraylib): d-spacing = %.6f A, energy for theta_B = 45 deg: %.3f eV\n" %
          (1e10 * setup.dSpacingSI(), energy_45))

    for energy in [12914.735, 12914.0]:
        theta_b = setup.angleBragg(energy)
        theta_bc = setup.angleBraggCorrected(energy)
        print("E = %.3f eV: theta_B = %.6f deg, theta_Bc = %.6f deg, cos(2 theta_B) = %.3e" %
              (energy, numpy.degrees(theta_b), numpy.degrees(theta_bc), numpy.cos(2 * theta_b)))
        for method, name in [(1, "Guigay"), (0, "Zachariasen")]:
            r_s, r_p = amplitudes(setup, energy, calculation_method=method)
            print("   %-11s (thick crystal, at theta_Bc): r_s = %+.6f %+.6fi, r_p = %+.6e %+.6ei, "
                  "R_s = %.6f, R_p = %.6e" %
                  (name, r_s.real, r_s.imag, r_p.real, r_p.imag, abs(r_s) ** 2, abs(r_p) ** 2))
        print()
