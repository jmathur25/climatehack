# OSGB is also called "OSGB 1936 / British National Grid -- United
# Kingdom Ordnance Survey".  OSGB is used in many UK electricity
# system maps, and is used by the UK Met Office UKV model.  OSGB is a
# Transverse Mercator projection, using 'easting' and 'northing'
# coordinates which are in meters.  See https://epsg.io/27700
OSGB = 27700

# WGS84 is short for "World Geodetic System 1984", used in GPS. Uses
# latitude and longitude.
WGS84 = 4326

lat_lon_to_osgb = pyproj.Transformer.from_crs(crs_from=WGS84, crs_to=OSGB)
osgb_to_lat_lon = pyproj.Transformer.from_crs(crs_from=OSGB, crs_to=WGS84)

# np.random.seed(7)
# # roughly UK
# rand_x = np.random.randint(550, 950 - 128)
# rand_y = np.random.randint(375, 700 - 128)

# x_osgb[rand_y,rand_x], y_osgb[rand_y, rand_x]
# osgb_to_lat_lon.transform(x_osgb[rand_y, rand_x], y_osgb[rand_y, rand_x])
