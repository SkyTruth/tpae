import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import box


# Infile
infile = "/Users/christian/Desktop/TPA/data/TEST_PAS_4087.shp"
gdf = gpd.read_file(infile)
gdf = gdf.to_crs(epsg=4087)

# Make Copies for Buffering, to maintain attributes
gdf_10km_buff = gdf.to_crs(epsg=4087).copy()
gdf_50km_buff = gdf.to_crs(epsg=4087).copy()

gdf_10km_buff["geometry"] = gdf_10km_buff.buffer(distance=10000).to_crs(epsg=4087)
gdf_50km_buff["geometry"] = gdf_50km_buff.buffer(distance=50000).to_crs(epsg=4087)
# gdf_50km = gdf.buffer(distance=50000).to_crs(epsg=4087)

# gdf_10km_buff.to_file("/Users/christian/Desktop/TPA/data/TEST_PAS_10km_4087.shp")
# gdf_50km_buff.to_file("/Users/christian/Desktop/TPA/data/TEST_PAS_50km_4087.shp")
ex_zone = gdf_10km_buff.copy()
wider_landscape = gdf_50km_buff.copy()


ex_zone["geometry"] = ex_zone.difference(gdf)
wider_landscape["geometry"] = wider_landscape.difference(gdf_10km_buff)
# ex_zone.to_file("/Users/christian/Desktop/TPA/data/exclusion_zone.shp")
# wider_landscape.to_file("/Users/christian/Desktop/TPA/data/wider_landscape.shp")



def make_grids(geom, crs, cell_size=1000, buffer_dist=5000):
    """
    Create grids 
    """
    aoi = geom.buffer(buffer_dist)
    minx, miny, maxx, maxy = aoi.bounds

    xs = np.arange(minx, maxx, cell_size)
    ys = np.arange(miny, maxy, cell_size)

    cells = [box(x, y, x + cell_size, y + cell_size) for x in xs for y in ys]
    grid = gpd.GeoDataFrame(geometry=cells, crs=crs)

    return grid[grid.intersects(aoi)].copy()


poly = gdf_50km_buff.to_crs(epsg=4087)#.head(1)  # metric CRS
grids = []

# Create the grids
for idx, row in poly.iterrows():
    grid = make_grids(
        row.geometry,
        crs=poly.crs,
        cell_size=1000,   # 1 km cell sizes
        buffer_dist=1000  # 1 km around each polygon
    )
    grid["WDPA_PID"] = row["WDPA_PID"]   # keep WDPA_PID
    grids.append(grid) 

grid_1km = gpd.GeoDataFrame(pd.concat(grids, ignore_index=True),crs=poly.crs)
grid_1km = grid_1km.drop_duplicates(subset="geometry")
# grid_1km.to_file("/Users/christian/Desktop/TPA/data/final_grid.geojson")


# Aggregate geometries
pa_boundary = gdf.union_all().boundary
ex_geom = ex_zone.union_all()
outside_geom = gdf_50km_buff.union_all()
outside_boundary = outside_geom.boundary

# Stack exclusions
grid_1km["exclude"] = False
grid_1km["exclude"] |= grid_1km.intersects(pa_boundary)
grid_1km["exclude"] |= grid_1km.intersects(ex_geom)
grid_1km["exclude"] |= (grid_1km.intersects(outside_boundary) | grid_1km.disjoint(outside_geom))

# Select the valid grid cells for the PA and it's wider landscape 
in_out_grid_1km = grid_1km[grid_1km["exclude"] == False]

pa_union = gdf.union_all()
in_out_grid_1km = in_out_grid_1km.copy()
in_out_grid_1km["site"] = np.where(
    in_out_grid_1km.intersects(pa_union),
    0,
    1
).astype("int8")

in_out_grid_1km["geometry"] = in_out_grid_1km.geometry.set_precision(1.0)  # 1 meter precision
in_out_grid_1km = in_out_grid_1km.drop("exclude", axis=1)
in_out_grid_1km.to_file("/Users/christian/Desktop/TPA/data/TPA_Valid_Grid.shp")

