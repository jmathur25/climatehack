import pyproj

# OSGB is also called "OSGB 1936 / British National Grid -- United
# Kingdom Ordnance Survey".  OSGB is used in many UK electricity
# system maps, and is used by the UK Met Office UKV model.  OSGB is a
# Transverse Mercator projection, using 'easting' and 'northing'
# coordinates which are in meters.  See https://epsg.io/27700
OSGB = 27700

# WGS84 is short for "World Geodetic System 1984", used in GPS. Uses
# latitude and longitude.
WGS84 = 4326
WGS84_CRS = f"EPSG:{WGS84}"


class Transformers:
    """
    Class to store transformation from one Grid to another.
    Its good to make this only once, but need the
    option of updating them, due to out of data grids.
    """

    def __init__(self):
        """Init"""
        self._osgb_to_lat_lon = None
        self._lat_lon_to_osgb = None
        self.make_transformers()

    def make_transformers(self):
        """
        Make transformers
         Nice to only make these once, as it makes calling the functions below quicker
        """
        self._osgb_to_lat_lon = pyproj.Transformer.from_crs(crs_from=OSGB, crs_to=WGS84)
        self._lat_lon_to_osgb = pyproj.Transformer.from_crs(crs_from=WGS84, crs_to=OSGB)

    @property
    def osgb_to_lat_lon(self):
        """OSGB to lat-lon property"""
        return self._osgb_to_lat_lon

    @property
    def lat_lon_to_osgb(self):
        """lat-lon to OSGB property"""
        return self._lat_lon_to_osgb


transformers = Transformers()


def osgb_to_lat_lon(x, y):
    """
    Change OSGB coordinates to lat, lon
    Args:
        x: osgb east-west
        y: osgb north-south
    Return: 2-tuple of latitude (north-south), longitude (east-west).
    """
    return transformers.osgb_to_lat_lon.transform(x, y)
