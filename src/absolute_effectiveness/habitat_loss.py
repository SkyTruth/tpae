import ee
from utils.variables import (
    EE_CRS_METERS,
    MAX_PIXELS,
    LOSS_OPENING_RADIUS_M,
    SCALE_M,
)

ee.Initialize()

KM2_PER_M2 = 1_000_000
METRIC_SUFFIXES = (
    ("_pct_start", "pct_start"),
    ("_pct_class", "pct_class"),
    ("_pct_site", "pct_site"),
    ("_area", "area"),
)


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

    def calc_overall_habitat_loss(
        self, habitat_loss_raster, habitat_start_raster, site_geom, land_mask=None
    ):
        """
        Calculate overall habitat loss metrics and habitat loss score.

        Returns area of habitat loss (km2), area of habitat at start year (km2),
        total site area (km2), habitat loss as percent of total site area, habitat
        loss as percent of habitat at start year, and habitat loss score (0-1,
        where 1 = no habitat loss).
        """
        habitat_loss_area = self.sum_area_km2(site_geom, habitat_loss_raster)
        habitat_start_area = self.sum_area_km2(site_geom, habitat_start_raster)
        site_area = self.sum_area_km2(site_geom, land_mask)
        return self.habitat_loss_from_areas(
            habitat_loss_area, habitat_start_area, site_area
        )

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

    def calc_class_area_and_pct(
        self, class_image, site_geom, site_area, habitat_start_area, top_n=4
    ):
        """
        Calculate area (km2), percent of total site area, and percent of habitat
        at start year for each class in a classified image.
        """
        class_dict = self.top_class_areas(class_image, site_geom, top_n)
        if not habitat_start_area:
            return {}

        def add_metrics(key, value):
            base = ee.String(key).slice(0, -5)
            return (
                ee.Dictionary()
                .set(key, value)
                .set(
                    base.cat("_pct_site"),
                    ee.Number(value).divide(site_area).multiply(100),
                )
                .set(
                    base.cat("_pct_start"),
                    ee.Number(value).divide(habitat_start_area).multiply(100),
                )
            )

        return self.flatten_dict(class_dict.map(add_metrics)).getInfo()

    def calc_habitat_class_area_and_pct(
        self,
        habitat_class_image,
        habitat_start_raster,
        lc_start,
        site_geom,
        top_n=4,
    ):
        """
        Calculate area (km2) and percent of each habitat type's start-year
        extent for lost habitat classes.
        """
        class_dict = self.top_class_areas(habitat_class_image, site_geom, top_n)
        habitat_at_start = lc_start.updateMask(habitat_start_raster).rename(
            "habitat_class"
        )
        start_areas = self.calc_class_area_dict(habitat_at_start, site_geom)

        def add_metrics(key, value):
            base = ee.String(key).slice(0, -5)
            start_area = ee.Number(start_areas.get(base, 0))
            pct_class = ee.Algorithms.If(
                start_area.gt(0),
                ee.Number(value).divide(start_area).multiply(100),
                0,
            )
            return (
                ee.Dictionary().set(key, value).set(base.cat("_pct_class"), pct_class)
            )

        return self.flatten_dict(class_dict.map(add_metrics)).getInfo()

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

    def translate_results(self, results_dict, labels, pct_reference_key="pct_start"):
        """Print class metrics with human-readable labels."""
        normalized = {}
        for key, value in results_dict.items():
            parsed = parse_result_key(key)
            if parsed is None:
                continue
            class_id, metric = parsed
            normalized.setdefault(class_id, {})[metric] = value

        for class_id in sorted(
            normalized,
            key=lambda class_id: normalized[class_id].get("area", 0),
            reverse=True,
        ):
            label = labels.get(class_id, f"Class {class_id}")
            area = normalized[class_id].get("area")
            pct_reference = normalized[class_id].get(pct_reference_key)
            if area is None or pct_reference is None:
                continue

            if pct_reference_key == "pct_class":
                reference_label = f"of site {label.lower()} at start year"
                pct_site = None
            else:
                reference_label = "of habitat at start year"
                pct_site = normalized[class_id].get("pct_site")
                if pct_site is None:
                    continue

            self.print_habitat_loss_metrics(
                label, area, pct_reference, reference_label, pct_site
            )

    def open_binary_raster(self, binary):
        kwargs = {
            "radius": self.loss_opening_radius_m,
            "kernelType": "square",
            "units": "meters",
        }
        return binary.focalMin(**kwargs).focalMax(**kwargs)

    def sum_area_km2(self, geometry, mask=None):
        area_img = ee.Image.pixelArea().divide(KM2_PER_M2)
        if mask is not None:
            area_img = area_img.updateMask(mask)
        return (
            area_img.reduceRegion(
                ee.Reducer.sum(),
                geometry,
                scale=self.scale_m,
                crs=self.crs,
                maxPixels=self.max_pixels,
            )
            .get("area")
            .getInfo()
        )

    def sum_areas_km2(self, geometry, masks):
        """Area (km2) within geometry under each mask in masks ({name: mask}), in one request."""
        area_img = ee.Image.pixelArea().divide(KM2_PER_M2)
        return (
            ee.Image.cat(
                [area_img.updateMask(mask).rename(name) for name, mask in masks.items()]
            )
            .reduceRegion(
                ee.Reducer.sum(),
                geometry,
                scale=self.scale_m,
                crs=self.crs,
                maxPixels=self.max_pixels,
            )
            .getInfo()
        )

    def calc_class_area_dict(self, class_image, site_geom):
        """Calculate area (km2) for each class in a classified image."""
        class_name = class_image.bandNames().get(0)
        grouped = (
            ee.Image.pixelArea()
            .divide(KM2_PER_M2)
            .addBands(class_image)
            .reduceRegion(
                reducer=ee.Reducer.sum().group(
                    groupField=1,
                    groupName=class_name,
                ),
                geometry=site_geom,
                scale=self.scale_m,
                crs=self.crs,
                maxPixels=self.max_pixels,
            )
        )

        def to_entry(item, acc):
            item = ee.Dictionary(item)
            return ee.Dictionary(acc).set(item.get(class_name), item.get("sum"))

        return ee.Dictionary(
            ee.List(grouped.get("groups")).iterate(to_entry, ee.Dictionary({}))
        )

    def top_class_areas(self, class_image, site_geom, top_n):
        areas = self.calc_class_area_dict(class_image, site_geom)
        top = areas.select(areas.keys().sort(areas.values()).reverse().slice(0, top_n))
        keys = top.keys().map(lambda key: ee.String(key).cat("_area"))
        return ee.Dictionary.fromLists(keys, top.values())

    def flatten_dict(self, mapped_dict):
        def combine(item, acc):
            return ee.Dictionary(acc).combine(ee.Dictionary(item), True)

        return ee.Dictionary(mapped_dict.values().iterate(combine, ee.Dictionary({})))


def parse_result_key(key):
    if "_" not in key:
        return None
    for suffix, metric in METRIC_SUFFIXES:
        if key.endswith(suffix):
            try:
                return int(float(key[: -len(suffix)])), metric
            except TypeError, ValueError:
                return None
    return None
