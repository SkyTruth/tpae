import ee
from utils.variables import (
    EE_CRS_METERS,
    MAX_PIXELS,
    LOSS_OPENING_RADIUS_M,
    SCALE_M,
)
from sites.site_selector import reduce_site

ee.Initialize()

KM2_PER_M2 = 1_000_000


class HabitatLossAnalyzer:
    """
    Computes amount, drivers, and types of habitat loss over the analysis period.
    """

    def __init__(
        self,
        loss_opening_radius_m=LOSS_OPENING_RADIUS_M,
        crs=EE_CRS_METERS,
        scale_m=SCALE_M,
        max_pixels=MAX_PIXELS,
    ):
        self.loss_opening_radius_m = loss_opening_radius_m
        self.crs = crs
        self.scale_m = scale_m
        self.max_pixels = max_pixels

    def get_habitat_loss_raster(self, habitat_binary_start, habitat_binary_end):
        """
        Create habitat loss raster.

        Loss is every pixel that was natural at the start year and anthropogenic
        at the end year, after a morphological opening.
        """
        habitat_loss = habitat_binary_start.eq(1).And(habitat_binary_end.eq(0))
        return self.open_binary_raster(habitat_loss)

    def habitat_loss_from_areas(self, habitat_loss_area, habitat_start_area, site_area):
        """calc_overall_habitat_loss's results from its three areas (km2)."""
        if not habitat_start_area:
            return 0, 0, 0, 0, 0, 0

        habitat_loss_proportion_site = min(habitat_loss_area / site_area, 1)
        habitat_loss_percent_site = habitat_loss_proportion_site * 100
        habitat_loss_percent_start = (
            min(habitat_loss_area / habitat_start_area, 1) * 100
        )
        habitat_loss_score = 1 - habitat_loss_proportion_site

        return (
            habitat_loss_area,
            habitat_start_area,
            site_area,
            habitat_loss_percent_site,
            habitat_loss_percent_start,
            habitat_loss_score,
        )

    def get_driver_class_image(self, lc_end, habitat_loss_raster):
        """
        Create classified image of the drivers of habitat loss:
            1: Cropland
            2: Built-up Land
            3: Pasture
            4: Deforestation without conversion
            5: Planted Forest
        """
        return (
            lc_end.updateMask(habitat_loss_raster)
            .remap(
                [1, 2, 3, 4, 30, 37, 38],
                [1, 1, 1, 1, 2, 3, 5],
                defaultValue=4,
            )
            .rename("driver_class")
        )

    def get_habitat_class_image(self, lc_start, habitat_loss_raster):
        """Create classified image of habitat types that were lost."""
        return lc_start.updateMask(habitat_loss_raster).rename("habitat_class")

    def print_habitat_loss_metrics(
        self,
        label,
        area_km2,
        pct_reference,
        reference_label="of habitat at start year",
        pct_site=None,
    ):
        """
        Print habitat loss metrics: area and a reference percentage, with an
        optional percent of total site area.
        """
        parts = [f"{label}: {area_km2:.2f} km2"]
        if pct_site is not None:
            parts.append(f"{pct_site:.2f}% of total site area")
        parts.append(f"{pct_reference:.2f}% {reference_label}")
        print(", ".join(parts))

    def open_binary_raster(self, binary):
        kwargs = {
            "radius": self.loss_opening_radius_m,
            "kernelType": "square",
            "units": "meters",
        }
        return binary.focalMin(**kwargs).focalMax(**kwargs)

    def sum_areas_by_group_km2(self, site, masks, group_image):
        """Area (km2) within a site (from site_region) under each 0/1 mask in masks
        ({name: mask}) per value of the integer group_image, in one request:
        {group: {name: area}}.

        Masked pixels count as 0 in masks and as group 0.
        """
        names = list(masks)
        area_img = ee.Image.pixelArea().divide(KM2_PER_M2)
        image = ee.Image.cat(
            [area_img.multiply(masks[name].unmask(0)).rename(name) for name in names]
            + [group_image.unmask(0).int().rename("group")]
        )
        groups = (
            reduce_site(
                image,
                ee.Reducer.sum()
                .repeat(len(names))
                .group(groupField=len(names), groupName="group"),
                site,
                scale=self.scale_m,
                crs=self.crs,
                maxPixels=self.max_pixels,
            )
            .get("groups")
            .getInfo()
        )
        return {int(g["group"]): dict(zip(names, g["sum"])) for g in groups}
