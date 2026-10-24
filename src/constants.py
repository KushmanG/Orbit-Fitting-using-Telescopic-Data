"""
Numbers we use everywhere. Unless a name says otherwise, the units are
astronomical units (au), days, and radians.
"""
import numpy as np

AU_KM = 149_597_870.7                 # IAU 2012 definition of the au
GM_SUN = 2.9591220828559115e-04       # Sun's GM in au^3/day^2 (= k^2, Gaussian constant squared)
C_AU_DAY = 299_792.458 * 86_400 / AU_KM   # speed of light, ~173.14 au/day

R_EARTH_KM = 6378.137                 # MPC parallax constants are in units of this radius
EPS_J2000 = np.radians(84381.448 / 3600.0)   # obliquity of the ecliptic at J2000 (value JPL uses)

DEG = np.pi / 180.0
ARCSEC = DEG / 3600.0
