import ee
from utils.variables import (
    ANALYSIS_END_YR,
    EE_CRS_METERS,
    MAX_PIXELS,
    OPENING_RADIUS_LOSS,
    SCALE,
)


class HabitatLossAnalyzer:
    """
    Computes amount, drivers, and types of habitat loss over the analysis period.
    """

    def __init__(
        self,
        analysis_end_yr=ANALYSIS_END_YR,
        opening_radius_loss=OPENING_RADIUS_LOSS,
        crs=EE_CRS_METERS,
        scale=SCALE,
        max_pixels=MAX_PIXELS,
    ):
        self.analysis_end_yr = analysis_end_yr
        self.opening_radius_loss = opening_radius_loss
        self.crs = crs
        self.scale = scale
        self.max_pixels = max_pixels

    def get_habitat_start_raster(self, glc_processed, gpw_processed, start_yr):
        """
        Create start-year habitat raster.
        """
        glc_start = glc_processed.select(f"GLC_{start_yr}")
        gpw_start = gpw_processed.select(f"GPW_{start_yr}")
        lc_start = glc_start.where(gpw_start.eq(1), 37)
        anthro_classes = ee.List([1, 2, 3, 4, 30, 37])
        habitat_start = lc_start.remap(
            anthro_classes, ee.List.repeat(0, anthro_classes.size()), defaultValue=1
        )
        return habitat_start
    
    def get_habitat_loss_raster(
        self, glc_processed, gpw_processed, hgfc_processed, start_yr
    ):
        """
        Create habitat loss raster.
        """
        # Get habitat start raster
        habitat_start = self.get_habitat_start_raster(glc_processed, gpw_processed, start_yr)
        # Get habitat loss raster
        forest_loss_binary = hgfc_processed.gt(0)
        glc_end = glc_processed.select(f"GLC_{self.analysis_end_yr}")
        gpw_end = gpw_processed.select(f"GPW_{self.analysis_end_yr}")
        lc_end = glc_end.where(gpw_end.eq(1), 37)
        anthro_classes = ee.List([1, 2, 3, 4, 30, 37])
        anthro_end = lc_end.remap(
            anthro_classes, ee.List.repeat(1, anthro_classes.size()), defaultValue=0
        ).where(forest_loss_binary, 1)
        habitat_loss_binary = habitat_start.And(anthro_end)
        # Open habitat loss raster to reduce noise
        habitat_loss_opened = habitat_loss_binary.focalMin(
            radius=self.opening_radius_loss, kernelType="square", units="meters"
        ).focalMax(radius=self.opening_radius_loss, kernelType="square", units="meters")
        return habitat_loss_opened, habitat_start

    def calc_overall_habitat_loss(
        self, habitat_loss_raster, habitat_start_raster, site_geom
    ):
        """
        Calculate overall habitat loss metrics and habitat loss score.

        Returns area of habitat loss (km2), area of habitat at start year (km2),
        total PA area (km2), habitat loss as percent of total PA area, habitat
        loss as percent of habitat at start year, and habitat loss score (0-1,
        where 1 = no habitat loss).
        """
        # Calculate area of habitat loss within PA
        habitat_loss_area = (
            ee.Image.pixelArea()
            .divide(1000000)
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
        # Calculate area of habitat in PA at start of analysis period
        habitat_start_area = (
            ee.Image.pixelArea()
            .divide(1000000)
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
        # Calculate total PA area
        site_area = (
            ee.Image.pixelArea()
            .divide(1000000)
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
            return 0, 0, 0, 0, 0, 0

        # Calculate habitat loss as a percent of total site area
        habitat_loss_proportion_site = min(habitat_loss_area / site_area, 1)
        habitat_loss_percent_site = habitat_loss_proportion_site * 100

        # Calculate habitat loss as a percent of habitat at start year
        habitat_loss_percent_start = min(habitat_loss_area / habitat_start_area, 1) * 100

        # Calculate habitat loss score
        habitat_loss_score = 1 - habitat_loss_proportion_site

        return (
            habitat_loss_area,
            habitat_start_area,
            site_area,
            habitat_loss_percent_site,
            habitat_loss_percent_start,
            habitat_loss_score,
        )


    def calc_class_area_and_pct(self, class_image, site_geom, site_area, habitat_start_area, top_n=4):
        """
        Calculate area (km2), percent of total PA area, and percent of habitat
        at start year for each class in a classified image.
        """
        # Calculate area of each class
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
        # Convert class areas to dictionary
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
        # Select top n classes by area
        class_dict = class_dict.select(
            class_dict.keys().sort(class_dict.values()).reverse().slice(0, top_n)
        )
        # Add area suffix to class names
        new_keys = class_dict.keys().map(lambda key: ee.String(key).cat("_area"))
        class_dict = ee.Dictionary.fromLists(new_keys, class_dict.values())
        
        if not habitat_start_area:
            return {}

        def add_pct_metrics(key, value):
            base_key = ee.String(key).slice(0, -5)
            pct_site_key = base_key.cat("_pct_site")
            pct_start_key = base_key.cat("_pct_start")
            pct_site_value = ee.Number(value).divide(site_area).multiply(100)
            pct_start_value = ee.Number(value).divide(habitat_start_area).multiply(100)
            return (
                ee.Dictionary()
                .set(key, value)
                .set(pct_site_key, pct_site_value)
                .set(pct_start_key, pct_start_value)
            )

        def combine_dicts(item, acc):
            return ee.Dictionary(acc).combine(ee.Dictionary(item), True)

        result_dict = ee.Dictionary(
            class_dict.map(add_pct_metrics)
            .values()
            .iterate(combine_dicts, ee.Dictionary({}))
        )
        return result_dict.getInfo()

    def get_driver_class_image(self, glc_processed, gpw_processed, habitat_loss_raster):
        """
        Create classified image of the 4 drivers of habitat loss:
            1: Cropland
            2: Built-up Land
            3: Pasture
            4: Deforestation without conversion
        """
        glc_end = glc_processed.select(f"GLC_{self.analysis_end_yr}")
        gpw_end = gpw_processed.select(f"GPW_{self.analysis_end_yr}")
        lc_end = glc_end.where(gpw_end.eq(1), 37)
        driver_class = (
            lc_end.updateMask(habitat_loss_raster)
            .remap([1, 2, 3, 4, 30, 37], [1, 1, 1, 1, 2, 3], defaultValue=4)
            .rename("driver_class")
        )
        return driver_class

    def get_habitat_class_image(self, glc_processed, habitat_loss_raster, start_yr):
        """
        Create classified image of habitat types that were lost.
        """
        glc_start = glc_processed.select(f"GLC_{start_yr}")
        habitat_class = glc_start.updateMask(habitat_loss_raster).rename("habitat_class")
        return habitat_class

    def print_habitat_loss_metrics(
        self, label, area_km2, pct_site, pct_start
    ):
        """
        Print habitat loss metrics: area, percent of total PA area, and percent
        of habitat at start year.
        """
        print(
            f"{label}: {area_km2:.2f} km2, "
            f"{pct_site:.2f}% of total PA area, "
            f"{pct_start:.2f}% of habitat at start year"
        )

    def translate_results(self, results_dict, labels):
        """
        Print class metrics with human-readable labels.
        """
        normalized = {}
        for key, value in results_dict.items():
            if "_" not in key:
                continue

            if key.endswith("_pct_start"):
                raw_class_id = key[: -len("_pct_start")]
                metric = "pct_start"
            elif key.endswith("_area"):
                raw_class_id = key[: -len("_area")]
                metric = "area"
            elif key.endswith("_pct_site"):
                raw_class_id = key[: -len("_pct_site")]
                metric = "pct_site"
            else:
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
            pct_site = normalized[class_id].get("pct_site")
            pct_start = normalized[class_id].get("pct_start")

            if area is None or pct_site is None or pct_start is None:
                continue

            self.print_habitat_loss_metrics(label, area, pct_site, pct_start)