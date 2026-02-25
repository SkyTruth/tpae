import ee
from math import sqrt
from utils.variables import (
    ANALYSIS_END_YR,
    CRS,
    MAX_EDGE_DIST,
    MAX_PATCH_SIZE,
    MAX_PIXELS,
    OPENING_RADIUS_EDGE,
    SCALE,
)


class HabitatConditionAnalyzer:
    """Computes habitat extent, intactness, and overall condition metrics."""

    def __init__(
        self,
        analysis_end_yr=ANALYSIS_END_YR,
        max_edge_dist=MAX_EDGE_DIST,
        opening_radius_edge=OPENING_RADIUS_EDGE,
        max_patch_size=MAX_PATCH_SIZE,
        crs=CRS,
        scale=SCALE,
        max_pixels=MAX_PIXELS,
    ):
        self.analysis_end_yr = analysis_end_yr
        self.max_edge_dist = max_edge_dist
        self.opening_radius_edge = opening_radius_edge
        self.max_patch_size = max_patch_size
        self.crs = crs
        self.scale = scale
        self.max_pixels = max_pixels

    def get_habitat_raster(
        self, glc_processed, hgfc_processed, gpw_processed, nfw_processed
    ):
        """Create habitat extent raster for analysis_end_yr."""
        glc_current = glc_processed.select(f"GLC_{self.analysis_end_yr}")
        anthro_classes = ee.List([1, 2, 3, 4, 30])
        anthro_mask = glc_current.remap(
            anthro_classes, ee.List.repeat(1, anthro_classes.size()), defaultValue=0
        )
        forest_loss_mask = hgfc_processed.gt(0)
        gpw_current = gpw_processed.select(f"GPW_{self.analysis_end_yr}")
        pasture_mask = gpw_current.eq(1)
        closed_forest_classes = ee.List([6, 8, 10, 12, 14])
        planted_forest_mask = (
            glc_current.remap(
                closed_forest_classes,
                ee.List.repeat(1, closed_forest_classes.size()),
                defaultValue=0,
            ).updateMask(nfw_processed.eq(0))
        ).unmask(0)
        grassland_mask = gpw_current.eq(2)
        grassland_override = grassland_mask.And(
            anthro_mask.add(planted_forest_mask).gt(0)
        )
        anthro_mask = anthro_mask.where(grassland_override, 0)
        planted_forest_mask = planted_forest_mask.where(grassland_override, 0)

        non_habitat_mask = (
            anthro_mask.add(forest_loss_mask).add(pasture_mask).add(planted_forest_mask)
        )
        habitat_mask = non_habitat_mask.eq(0).Or(grassland_override)
        return glc_current.where(grassland_override, 18).updateMask(habitat_mask)

    def calc_habitat_extent_score(self, habitat_raster, site_geom):
        """Calculate Habitat Extent score for analysis_end_yr within a PA."""
        site_area = site_geom.area().getInfo()
        habitat_area = (
            ee.Image.pixelArea()
            .updateMask(habitat_raster)
            .reduceRegion(
                ee.Reducer.sum(),
                site_geom,
                scale=self.scale,
                crs=self.crs,
                maxPixels=self.max_pixels,
            )
            .get("area")
            .getInfo()
        )
        return min(habitat_area / site_area, 1)

    def get_edge_distance_raster(self, habitat_raster, site_geom):
        """Calculate distance from each habitat pixel to nearest non-habitat pixel."""
        site_buffered = site_geom.buffer(self.max_edge_dist + self.opening_radius_edge)
        projection = ee.Projection("EPSG:3857").atScale(30)
        habitat_reprojected = habitat_raster.clip(site_buffered).reproject(projection)
        habitat_binary = habitat_reprojected.unmask().eq(0)
        habitat_binary_opened = habitat_binary.focalMin(
            radius=self.opening_radius_edge, kernelType="square", units="meters"
        ).focalMax(radius=self.opening_radius_edge, kernelType="square", units="meters")
        edge_distance = habitat_binary_opened.distance(
            ee.Kernel.euclidean(self.max_edge_dist, units="meters")
        )
        return (
            edge_distance.min(self.max_edge_dist)
            .unmask(self.max_edge_dist)
            .clip(site_geom)
            .selfMask()
            .rename("edge_distance")
        )

    def calc_edge_distance_score(self, edge_distance_raster, site_geom):
        """Calculate Edge Distance score from edge distance raster."""
        avg_edge_distance = (
            edge_distance_raster.reduceRegion(
                ee.Reducer.mean(),
                site_geom,
                scale=self.scale,
                crs=self.crs,
                maxPixels=self.max_pixels,
            )
            .get("edge_distance")
            .getInfo()
        )
        return avg_edge_distance / self.max_edge_dist

    def get_patch_size_raster(self, habitat_raster, site_geom):
        """Calculate size of connected habitat patches."""
        return (
            habitat_raster.gt(0)
            .connectedPixelCount(maxSize=1024, eightConnected=False)
            .rename("patch_size")
            .clip(site_geom)
        )

    def calc_patch_size_score(self, patch_size_raster, site_geom):
        """Calculate Patch Size score from patch size raster."""
        patch_scale = sqrt((self.max_patch_size * 1000000) / 1024)
        avg_patch_size = (
            patch_size_raster.reduceRegion(
                ee.Reducer.mean(),
                site_geom,
                scale=patch_scale,
                crs=self.crs,
                maxPixels=self.max_pixels,
            )
            .get("patch_size")
            .getInfo()
        )
        return avg_patch_size / 1024

    def calc_intactness_score(self, edge_distance_score, patch_size_score):
        """Calculate Habitat Intactness score from edge distance and patch size."""
        return sqrt(edge_distance_score * patch_size_score)

    def calc_habitat_condition_score(
        self, habitat_extent_score, habitat_intactness_score
    ):
        """Calculate overall Habitat Condition score."""
        return habitat_extent_score * habitat_intactness_score
