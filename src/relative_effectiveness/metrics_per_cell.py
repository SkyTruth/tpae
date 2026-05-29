"""
Computes relative effectiveness metrics per 1km2 cell.
Reuses absolute effectiveness code, but uses reduceRegions to compute scores
across a series of cells rather than within a single PA.
"""

import ee

from absolute_effectiveness.habitat_condition import HabitatConditionAnalyzer
from absolute_effectiveness.habitat_loss import HabitatLossAnalyzer

__all__ = [
    "RelativeHabitatLossAnalyzer",
    "RelativeHabitatConditionAnalyzer",
]


class RelativeHabitatLossAnalyzer(HabitatLossAnalyzer):
    """Per-cell habitat loss metric.

    Reuses `get_habitat_loss_raster` from `HabitatLossAnalyzer` and scores
    loss within each cell using a single reduceRegions pass.
    """

    def calc_loss_score_per_cell(self, habitat_loss_raster, habitat_start_raster, fc):
        """Score Habitat Loss within each cell as `1 - (loss_area / start_area)`.

        Cells with zero start-year habitat get a score of 0, mirroring the
        absolute pipeline's guard in `calc_habitat_loss_score`.
        """
        loss_and_start = (
            ee.Image.pixelArea()
            .updateMask(habitat_loss_raster)
            .rename("loss_area")
            .addBands(
                ee.Image.pixelArea()
                .updateMask(habitat_start_raster)
                .rename("start_area")
            )
        )

        with_areas = loss_and_start.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.sum(),
            scale=self.scale,
            crs=self.crs,
        )

        def add_score(feature):
            habitat_loss_area = ee.Number(feature.get("loss_area"))
            habitat_start_area = ee.Number(feature.get("start_area"))
            if not habitat_start_area:
                return feature.set("loss_score", 0)
            habitat_loss_proportion = habitat_loss_area.divide(habitat_start_area).min(1)
            return feature.set(
                "loss_score", ee.Number(1).subtract(habitat_loss_proportion)
            )

        return with_areas.map(add_score)


class RelativeHabitatConditionAnalyzer(HabitatConditionAnalyzer):
    """Per-cell habitat extent, intactness, and overall condition metrics.

    Reuses the raster-building methods from `HabitatConditionAnalyzer`
    (`get_habitat_raster`, `build_kernel`, `get_intactness_raster`) and adds
    server-side `reduceRegions`-based scoring for an entire FeatureCollection
    of cells in a single pass per metric.
    """

    def calc_extent_score_per_cell(self, habitat_raster, matched_grids):
        """Score Habitat Extent within each cell of `matched_grids`."""

        # Could just set cell area to 1000000 since that's what it should be,
        # but this might be more accurate given the projection

        cell_area_image = ee.Image.pixelArea().rename("cell_area")
        habitat_area_image = (
            cell_area_image.updateMask(habitat_raster).rename("habitat_area")
        )
        combined = habitat_area_image.addBands(cell_area_image)

        with_area = combined.reduceRegions(
            collection=matched_grids,
            reducer=ee.Reducer.sum(),
            scale=self.scale,
            crs=self.crs,
        )

        def add_score(feature):
            habitat_area = ee.Number(feature.get("habitat_area"))
            cell_area = ee.Number(feature.get("cell_area"))
            extent_score = habitat_area.divide(cell_area).min(1)
            return feature.set("extent_score", extent_score)

        return with_area.map(add_score)

    def calc_intactness_score_per_cell(self, intactness_raster, fc):
        """Score Habitat Intactness within each cell as the mean of intactness."""
        with_intactness = intactness_raster.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean().setOutputs(["intactness_score"]),
            scale=self.intactness_scale,
            crs=self.crs,
            tileScale=self.tile_scale,
        )

        def add_score(feature):
            intactness = feature.get("intactness_score")
            intactness_score = ee.Algorithms.If(intactness, ee.Number(intactness), 0)
            return feature.set("intactness_score", intactness_score)

        return with_intactness.map(add_score)

    def calc_condition_score_per_cell(self, fc):
        """Combine per-cell extent and intactness scores into a Condition score."""

        def compute(feature):
            extent = ee.Number(feature.get("extent_score"))
            intactness = ee.Number(feature.get("intactness_score"))
            return feature.set("condition_score", extent.multiply(intactness))

        return fc.map(compute)

