import ee
import math
from utils.variables import (
    ANALYSIS_END_YR,
    CRS,
    MAX_PIXELS,
    SCALE,
    INTERACTION_DISTANCE,
    BETA,
    KERNEL_RADIUS_METERS,
    PIXEL_SIZE,
    KERNEL_RADIUS_PIXELS,
    KERNEL_SIZE,
)


class HabitatConditionAnalyzer:
    """Computes habitat extent, intactness, and overall condition metrics."""

    def __init__(
        self,
        analysis_end_yr=ANALYSIS_END_YR,
        crs=CRS,
        scale=SCALE,
        max_pixels=MAX_PIXELS,
        interaction_distance=INTERACTION_DISTANCE,
        beta=BETA,
        kernel_radius_meters=KERNEL_RADIUS_METERS,
        pixel_size=PIXEL_SIZE,
        kernel_radius_pixels=KERNEL_RADIUS_PIXELS,
        kernel_size=KERNEL_SIZE,
    ):
        self.analysis_end_yr = analysis_end_yr
        self.crs = crs
        self.scale = scale
        self.max_pixels = max_pixels
        self.interaction_distance = interaction_distance
        self.beta = beta
        self.kernel_radius_meters = kernel_radius_meters
        self.pixel_size = pixel_size
        self.kernel_radius_pixels = kernel_radius_pixels
        self.kernel_size = kernel_size

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

    def build_kernel(self):
        """Build an exponentially decaying distance-weighted kernel for calculating habitat intactness."""
        # Create 2-D array of weights for kernel
        weights = []
        for row in range(self.kernel_size):
            row_weights = []
            for col in range(self.kernel_size):
                dx = (
                    col - self.kernel_radius_pixels
                ) * self.pixel_size  # horizontal distance from center pixel
                dy = (
                    row - self.kernel_radius_pixels
                ) * self.pixel_size  # vertical distance from center pixel
                d = math.sqrt(
                    dx * dx + dy * dy
                )  # hypotenuse distance from center pixel
                w = (
                    math.exp(-self.beta * d) if d <= self.kernel_radius_meters else 0
                )  # apply exponential decay and clip to kernel radius
                row_weights.append(w)
            weights.append(row_weights)

        # Build custom kernel with weights
        exp_kernel = ee.Kernel.fixed(
            width=self.kernel_size,
            height=self.kernel_size,
            weights=weights,
            normalize=True,  # sum of all weights is 1; this normalizes the intactness values to be between 0 and 1
        )
        return exp_kernel

    def get_intactness_raster(self, habitat_raster, exp_kernel):
        """Create continuous habitat intactness raster."""
        # Get habitat binary
        habitat_binary = habitat_raster.gt(0).unmask(0).toFloat()
        # Apply kernel to habitat binary to get intactness raster
        intactness_raster = (
            habitat_binary.convolve(
                exp_kernel
            )  # sum the weighted habitat binary values in the kernel
            .rename("intactness")
            .updateMask(habitat_binary)
        )  # only habitat pixels get intactness values
        return intactness_raster

    def calc_intactness_score(self, intactness_raster, site_geom):
        """Calculate Habitat Intactness score within a PA from intactness raster."""
        intactness_score = (
            intactness_raster.reduceRegion(
                ee.Reducer.mean(),
                site_geom,
                scale=self.scale,
                crs=self.crs,
                maxPixels=self.max_pixels,
            )
            .get("intactness")
            .getInfo()
        )
        return intactness_score

    def calc_habitat_condition_score(
        self, habitat_extent_score, habitat_intactness_score
    ):
        """Calculate overall Habitat Condition score."""
        return habitat_extent_score * habitat_intactness_score
