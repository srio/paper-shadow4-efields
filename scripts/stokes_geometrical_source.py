#
# Stokes parameters and standard Jones vectors of the geometrical source for several inputs (P, Phi)
# (Table "source_polarization" of the paper).
#
# The Stokes parameters are computed with equation (stokes_from_P) and from the beam columns 30-33. The standard Jones
# vector (Ex, Ey) = (E_sigma, -E_pi) is obtained with beam.get_jones_components(use_standard=1) for a coherent beam
# (phi_sigma = 0), and compared with the analytical value Ex = P / sqrt(D), Ey = -(1 - P) exp(i Phi) / sqrt(D),
# with D = 1 - 2P + 2P^2.
#
# Usage:
#   PYTHONPATH=~/OASYS2.0/shadow4 ~/venv/oasys2/bin/python stokes_geometrical_source.py
#
import numpy
from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical

def fmt_complex(z):
    re, im = numpy.real(z), numpy.imag(z)
    if abs(re) < 1e-10: re = 0.0
    if abs(im) < 1e-10: im = 0.0
    if im == 0: return "%+.4f" % re
    if re == 0: return "%+.4fi" % im
    return "%+.4f%+.4fi" % (re, im)

cases = [(1, 0), (0, 0), (0.5, 0), (0.5, 180), (0.5, 90), (0.5, -90), (0.75, -90)]
# elliptical polarization of Fig. 1 in Detlefs et al. (2012), psi = chi = 30 deg (Fig. ellipse_geometrical a)
psi, chi = numpy.radians(30.0), numpy.radians(30.0)
S1, S2, S3 = numpy.cos(2 * chi) * numpy.cos(2 * psi), numpy.cos(2 * chi) * numpy.sin(2 * psi), numpy.sin(2 * chi)
cases.append((numpy.sqrt(1 + S1) / (numpy.sqrt(1 + S1) + numpy.sqrt(1 - S1)), numpy.degrees(numpy.arctan2(-S3, -S2))))
print("   P    Phi |  eq: S0     S1     S2     S3 | code: S0     S1     S2     S3 | Jones (Ex, Ey) eq | code")
for P, Phi in cases:
    D = 1 - 2*P + 2*P**2; f = 2*P*(1-P)/D; ph = numpy.radians(Phi)
    eq = [1, (2*P-1)/D, -f*numpy.cos(ph), -f*numpy.sin(ph)]
    jones_eq = (P / numpy.sqrt(D), -(1 - P) * numpy.exp(1j * ph) / numpy.sqrt(D))
    s = SourceGeometrical(nrays=1000, spatial_type="Point", angular_distribution="Flat")
    s.set_angular_distribution_flat(hdiv1=0, hdiv2=0, vdiv1=0, vdiv2=0)
    s.set_polarization(polarization_degree=P, phase_diff=ph, coherent_beam=1)
    b = s.get_beam()
    code = [b.get_column(c).mean() for c in [30, 31, 32, 33]]
    Ex, Ey = b.get_jones_components(use_standard=1)
    assert numpy.allclose(Ex, jones_eq[0]) and numpy.allclose(Ey, jones_eq[1])
    print("%6.4f %7.2f | %s | %s | (%s, %s) | (%s, %s)" % (P, Phi,
          " ".join("%+.3f" % v for v in eq), " ".join("%+.3f" % v for v in code),
          fmt_complex(jones_eq[0]), fmt_complex(jones_eq[1]), fmt_complex(Ex[0]), fmt_complex(Ey[0])))
