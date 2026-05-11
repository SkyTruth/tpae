import ee

from absolute_effectiveness.habitat_loss import HabitatLossAnalyzer


class RelativeHabitatLossAnalyzer(HabitatLossAnalyzer):
    """Per-cell habitat loss metric.

    Reuses `get_habitat_loss_raster` from `HabitatLossAnalyzer` and scores
    loss within each cell using a single reduceRegions pass.
    """

    def calc_loss_score_per_cell(
        self, habitat_loss_raster, habitat_start_raster, fc
    ):
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
            crs=self.crs
        )

        def add_score(feature):
            habitat_loss_area = ee.Number(feature.get("loss_area"))
            habitat_start_area = ee.Number(feature.get("start_area"))
            if not habitat_start_area:
                return feature.set("loss_score", 0)
            habitat_loss_proportion = habitat_loss_area.divide(habitat_start_area).min(1)
            return feature.set("loss_score", ee.Number(1).subtract(habitat_loss_proportion))

        return with_areas.map(add_score)