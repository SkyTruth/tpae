import ee
from utils.variables import (
    ANALYSIS_END_YR,
    CRS,
    MAX_PIXELS,
    OPENING_RADIUS_LOSS,
    SCALE,
)


class HabitatLossAnalyzer:
    """Computes habitat loss and loss-composition metrics."""

    def __init__(
        self,
        analysis_end_yr=ANALYSIS_END_YR,
        opening_radius_loss=OPENING_RADIUS_LOSS,
        crs=CRS,
        scale=SCALE,
        max_pixels=MAX_PIXELS,
    ):
        self.analysis_end_yr = analysis_end_yr
        self.opening_radius_loss = opening_radius_loss
        self.crs = crs
        self.scale = scale
        self.max_pixels = max_pixels

    def get_habitat_loss_raster(
        self, glc_processed, gpw_processed, hgfc_processed, start_yr
    ):
        """Create habitat loss and start-year habitat rasters."""
        glc_start = glc_processed.select(f"GLC_{start_yr}")
        gpw_start = gpw_processed.select(f"GPW_{start_yr}")
        lc_start = glc_start.where(gpw_start.eq(1), 37)
        anthro_classes = ee.List([1, 2, 3, 4, 30, 37])
        habitat_start = lc_start.remap(
            anthro_classes, ee.List.repeat(0, anthro_classes.size()), defaultValue=1
        )

        forest_loss_binary = hgfc_processed.gt(0)
        glc_end = glc_processed.select(f"GLC_{self.analysis_end_yr}")
        gpw_end = gpw_processed.select(f"GPW_{self.analysis_end_yr}")
        lc_end = glc_end.where(gpw_end.eq(1), 37)
        anthro_end = lc_end.remap(
            anthro_classes, ee.List.repeat(1, anthro_classes.size()), defaultValue=0
        ).where(forest_loss_binary, 1)

        habitat_loss_binary = habitat_start.And(anthro_end)
        habitat_loss_opened = habitat_loss_binary.focalMin(
            radius=self.opening_radius_loss, kernelType="square", units="meters"
        ).focalMax(radius=self.opening_radius_loss, kernelType="square", units="meters")
        return habitat_loss_opened, habitat_start

    def calc_habitat_loss_score(
        self, habitat_loss_raster, habitat_start_raster, site_geom
    ):
        """Calculate habitat loss score where 1 = no habitat loss."""
        habitat_loss_area = (
            ee.Image.pixelArea()
            .updateMask(habitat_loss_raster)
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
        habitat_start_area = (
            ee.Image.pixelArea()
            .updateMask(habitat_start_raster)
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
        if not habitat_start_area:
            return 0
        habitat_loss_proportion = min(habitat_loss_area / habitat_start_area, 1)
        return 1 - habitat_loss_proportion

    def calc_class_area_and_pct(self, class_image, site_geom, top_n=4):
        """Calculate area and percent area for each class in a classified image."""
        class_name = class_image.bandNames().get(0)
        area_by_class = (
            ee.Image.pixelArea()
            .divide(1000000)
            .addBands(class_image)
            .reduceRegion(
                reducer=ee.Reducer.sum().group(
                    groupField=1,
                    groupName=class_name,
                ),
                geometry=site_geom,
                scale=self.scale,
                crs=self.crs,
                maxPixels=self.max_pixels,
            )
        )

        def dict_from_list(item, acc):
            item = ee.Dictionary(item)
            key = item.get(class_name)
            value = item.get("sum")
            return ee.Dictionary(acc).set(key, value)

        class_dict = ee.Dictionary(
            ee.List(area_by_class.get("groups")).iterate(
                dict_from_list, ee.Dictionary({})
            )
        )
        class_dict = class_dict.select(
            class_dict.keys().sort(class_dict.values()).reverse().slice(0, top_n)
        )

        new_keys = class_dict.keys().map(lambda key: ee.String(key).cat("_area"))
        class_dict = ee.Dictionary.fromLists(new_keys, class_dict.values())
        site_area = site_geom.area().divide(1000000).getInfo()

        def add_pct_area(key, value):
            pct_key = ee.String(key).slice(0, -5).cat("_pct")
            pct_value = ee.Number(value).divide(site_area).multiply(100)
            return ee.Dictionary().set(key, value).set(pct_key, pct_value)

        def combine_dicts(item, acc):
            return ee.Dictionary(acc).combine(ee.Dictionary(item), True)

        result_dict = ee.Dictionary(
            class_dict.map(add_pct_area)
            .values()
            .iterate(combine_dicts, ee.Dictionary({}))
        )
        return result_dict.getInfo()

    def get_driver_class_image(self, glc_processed, gpw_processed, habitat_loss_raster):
        """Create classified image of the 4 drivers of habitat loss."""
        glc_end = glc_processed.select(f"GLC_{self.analysis_end_yr}")
        gpw_end = gpw_processed.select(f"GPW_{self.analysis_end_yr}")
        lc_end = glc_end.where(gpw_end.eq(1), 37)
        return (
            lc_end.updateMask(habitat_loss_raster)
            .remap([1, 2, 3, 4, 30, 37], [1, 1, 1, 1, 2, 3], defaultValue=4)
            .rename("driver_class")
        )

    def get_habitat_class_image(self, glc_processed, habitat_loss_raster, start_yr):
        """Create classified image of habitat types that were lost."""
        glc_start = glc_processed.select(f"GLC_{start_yr}")
        return glc_start.updateMask(habitat_loss_raster).rename("habitat_class")

    def translate_results(self, results_dict, labels):
        """Print class metrics with human-readable labels and metrics."""
        normalized = {}
        for key, value in results_dict.items():
            if "_" not in key:
                continue

            raw_class_id, metric = key.rsplit("_", 1)
            if metric not in {"area", "pct"}:
                continue

            try:
                class_id = int(float(raw_class_id))
            except TypeError, ValueError:
                continue

            normalized.setdefault(class_id, {})[metric] = value

        sorted_class_ids = sorted(
            normalized.keys(),
            key=lambda class_id: normalized[class_id].get("area", 0),
            reverse=True,
        )

        for class_id in sorted_class_ids:
            label = labels.get(class_id, f"Class {class_id}")
            area = normalized[class_id].get("area")
            pct = normalized[class_id].get("pct")

            area_text = f"{area:.2f} km2" if area is not None else "N/A km2"
            pct_text = (
                f"{pct:.2f}% of total PA area"
                if pct is not None
                else "N/A of total PA area"
            )
            print(f"{label}: {area_text}, {pct_text}")
