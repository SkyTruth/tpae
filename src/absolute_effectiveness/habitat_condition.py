import ee
import math
from utils.variables import (
    EE_CRS_METERS,
    MAX_PIXELS,
    SCALE_M,
    INTERACTION_DISTANCE_M,
    BETA,
    KERNEL_RADIUS_M,
    INTACTNESS_SCALE_M,
    KERNEL_RADIUS_PIXELS,
    KERNEL_SIZE_PIXELS,
    TILE_SCALE,
)
from sites.site_selector import reduce_site


class HabitatConditionAnalyzer:
    """
    Computes habitat extent, intactness, and overall condition metrics.
    """

    def __init__(
        self,
        crs=EE_CRS_METERS,
        scale_m=SCALE_M,
        max_pixels=MAX_PIXELS,
        interaction_distance_m=INTERACTION_DISTANCE_M,
        beta=BETA,
        kernel_radius_m=KERNEL_RADIUS_M,
        intactness_scale_m=INTACTNESS_SCALE_M,
        kernel_radius_pixels=KERNEL_RADIUS_PIXELS,
        kernel_size_pixels=KERNEL_SIZE_PIXELS,
        tile_scale=TILE_SCALE,
    ):
        self.crs = crs
        self.scale_m = scale_m
        self.max_pixels = max_pixels
        self.interaction_distance_m = interaction_distance_m
        self.beta = beta
        self.kernel_radius_m = kernel_radius_m
        self.intactness_scale_m = intactness_scale_m
        self.kernel_radius_pixels = kernel_radius_pixels
        self.kernel_size_pixels = kernel_size_pixels
        self.tile_scale = tile_scale

    def get_land_cover(self, glc_processed, gpw_processed, forty_processed, year):
        """
        Combine land cover products into a single classification for a given year.
        """
        # Start with GLC
        glc = glc_processed.select(f"GLC_{year}")
        # Use GPW to differentiate between natural and anthropogenic grassland
        gpw = gpw_processed.select(f"GPW_{year}")
        grassland_classes = ee.List([15, 16, 17, 18, 20, 21, 22])
        is_grassland = glc.remap(
            grassland_classes,
            ee.List.repeat(1, grassland_classes.size()),
            0,
        ).eq(1)
        # GPW = cultivated grassland (pasture) on GLC grassland, shrubland, or sparse
        lc = glc.where(gpw.eq(1).And(is_grassland), 37)

        # Use ForTy to differentiate between natural and anthropogenic forest
        forty = forty_processed.select(f"FORTY_{year}")
        forty_planted_classes = ee.List([4, 5])
        forest_classes = ee.List([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        is_planted = forty.remap(
            forty_planted_classes,
            ee.List.repeat(1, forty_planted_classes.size()),
            0,
        ).eq(1)
        is_forest = glc.remap(
            forest_classes,
            ee.List.repeat(1, forest_classes.size()),
            0,
        ).eq(1)
        # ForTy = plantation forest or tree crops on GLC forest
        lc = lc.where(is_planted.And(is_forest), 38)
        return lc

    def _remap_habitat_binary(self, lc):
        """Remap anthropogenic classes to 0 and all other classes to 1."""
        anthro_classes = ee.List([1, 2, 3, 4, 30, 37, 38])
        return lc.remap(
            anthro_classes,
            ee.List.repeat(0, anthro_classes.size()),
            defaultValue=1,
        )

    def get_habitat_binary_end(self, lc, hgfc_processed):
        """
        Create a habitat/non-habitat binary raster for the analysis end year.

        1 = natural, 0 = anthropogenic. Forest loss during the analysis period
        is set to anthropogenic.
        """
        habitat_binary_end = self._remap_habitat_binary(lc)
        forest_loss = hgfc_processed.unmask(0).gt(0)
        return habitat_binary_end.where(forest_loss, 0)

    def get_habitat_binary_start(self, lc):
        """
        Create a habitat/non-habitat binary raster for the analysis start year.

        1 = natural, 0 = anthropogenic. Pass land cover built for the start year.
        Forest loss is not applied.
        """
        return self._remap_habitat_binary(lc)

    def calc_habitat_extent_score(self, habitat_raster, site, land_mask=None):
        """
        Calculate Habitat Extent score for analysis_end_yr within a site (from
        site_region).
        Extent score = habitat area / site area
        """
        # Calculate area of habitat within site
        habitat_area = (
            reduce_site(
                ee.Image.pixelArea().updateMask(habitat_raster),
                ee.Reducer.sum(),
                site,
                scale=self.scale_m,
                crs=self.crs,
                maxPixels=self.max_pixels,
            )
            .get("area")
            .getInfo()
        )
        # Calculate total site area (land only when land_mask is provided)
        site_area_img = ee.Image.pixelArea()
        if land_mask is not None:
            site_area_img = site_area_img.updateMask(land_mask)
        site_area = (
            reduce_site(
                site_area_img,
                ee.Reducer.sum(),
                site,
                scale=self.scale_m,
                crs=self.crs,
                maxPixels=self.max_pixels,
            )
            .get("area")
            .getInfo()
        )
        return self.habitat_extent_from_areas(habitat_area, site_area)

    def habitat_extent_from_areas(self, habitat_area, site_area):
        """Habitat Extent score from habitat area and site area (same units)."""
        return min(habitat_area / site_area, 1)

    def build_kernel(self):
        """
        Build an exponentially decaying distance-weighted kernel for calculating habitat intactness.
        """
        # Create 2-D array of weights for kernel
        weights = []
        for row in range(self.kernel_size_pixels):
            row_weights = []
            for col in range(self.kernel_size_pixels):
                dx = (
                    col - self.kernel_radius_pixels
                ) * self.intactness_scale_m  # horizontal distance from center pixel
                dy = (
                    row - self.kernel_radius_pixels
                ) * self.intactness_scale_m  # vertical distance from center pixel
                d = math.sqrt(
                    dx * dx + dy * dy
                )  # hypotenuse distance from center pixel
                w = (
                    math.exp(-self.beta * d) if d <= self.kernel_radius_m else 0
                )  # apply exponential decay and clip to kernel radius
                row_weights.append(w)
            weights.append(row_weights)
        # Build custom kernel using weights
        exp_kernel = ee.Kernel.fixed(
            width=self.kernel_size_pixels,
            height=self.kernel_size_pixels,
            weights=weights,
            normalize=True,  # sum of all weights is 1; this normalizes the intactness values to be between 0 and 1
        )
        return exp_kernel

    def get_intactness_raster(self, habitat_raster, exp_kernel, land_mask):
        """
        Create continuous habitat intactness raster.

        Intactness = kernel-weighted habitat / kernel-weighted land, so water and
        ocean within the kernel are excluded rather than counted as non-habitat:
          I(p) = Σ_land w·habitat / Σ_land w
        """
        # Habitat on land and land, both 1/0 (water and no data are 0 in both)
        kernel_input = (
            ee.Image.cat(
                [
                    habitat_raster.gt(0).And(land_mask).rename("habitat"),
                    land_mask.rename("land"),
                ]
            )
            .unmask(0)
            .toFloat()
        )
        kernel_input_reproj = kernel_input.reproject(
            crs=self.crs,
            scale=self.scale_m,
        )
        weighted = kernel_input_reproj.convolve(exp_kernel)
        habitat_output_mask = habitat_raster.gt(0).updateMask(land_mask)
        intactness_raster = (
            weighted.select("habitat")
            .divide(weighted.select("land"))
            .rename("intactness")
            .updateMask(habitat_output_mask)
        )
        return intactness_raster

    def calc_intactness_score(self, intactness_raster, site):
        """
        Calculate Habitat Intactness score within a site (from site_region) from
        intactness raster.
        """
        # Calculate average intactness of habitat pixels within site
        intactness_score = (
            reduce_site(
                intactness_raster,
                ee.Reducer.mean(),
                site,
                scale=self.intactness_scale_m,
                crs=self.crs,
                maxPixels=self.max_pixels,
                tileScale=self.tile_scale,
            )
            .get("intactness")
            .getInfo()
        )
        # Return 0 if score is invalid
        if intactness_score is None:
            return 0
        try:
            # Check for NaN or non-numeric results
            if isinstance(intactness_score, float) and math.isnan(intactness_score):
                return 0
        except Exception:
            return 0
        return intactness_score

    def calc_habitat_condition_score(
        self, habitat_extent_score, habitat_intactness_score
    ):
        """
        Calculate overall Habitat Condition score.
        Condition score = Extent score * Intactness score
        """
        return habitat_extent_score * habitat_intactness_score
