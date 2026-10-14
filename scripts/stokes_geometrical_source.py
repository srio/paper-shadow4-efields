import numpy
from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
cases = [(1, 0), (0, 0), (0.5, 0), (0.5, 180), (0.5, 90), (0.75, -90)]
print("   P    Phi |  eq: S0     S1     S2     S3 | code: S0     S1     S2     S3")
for P, Phi in cases:
    D = 1 - 2*P + 2*P**2; f = 2*P*(1-P)/D; ph = numpy.radians(Phi)
    eq = [1, (2*P-1)/D, -f*numpy.cos(ph), -f*numpy.sin(ph)]
    s = SourceGeometrical(nrays=1000, spatial_type="Point", angular_distribution="Flat")
    s.set_angular_distribution_flat(hdiv1=0, hdiv2=0, vdiv1=0, vdiv2=0)
    s.set_polarization(polarization_degree=P, phase_diff=ph, coherent_beam=0)
    b = s.get_beam()
    code = [b.get_column(c).mean() for c in [30, 31, 32, 33]]
    print("%5.2f %5d | %s | %s" % (P, Phi, " ".join("%+.3f" % v for v in eq), " ".join("%+.3f" % v for v in code)))
