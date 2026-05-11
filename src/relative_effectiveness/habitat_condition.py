import ee

from absolute_effectiveness.habitat_condition import HabitatConditionAnalyzer


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
        habitat_area_image = cell_area_image.updateMask(habitat_raster).rename("habitat_area")
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

        return with_intactness

    def calc_condition_score_per_cell(self, fc):
        """Combine per-cell extent and intactness scores into a Condition score."""

        def compute(feature):
            extent = ee.Number(feature.get("extent_score"))
            intactness = ee.Number(feature.get("intactness_score"))
            return feature.set("condition_score", extent.multiply(intactness))

        return fc.map(compute)
