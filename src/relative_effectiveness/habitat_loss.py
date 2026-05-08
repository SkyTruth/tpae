import ee

from absolute_effectiveness.habitat_loss import HabitatLossAnalyzer


class RelativeHabitatLossAnalyzer(HabitatLossAnalyzer):
    """Per-cell habitat loss metric.

    Reuses `get_habitat_loss_raster` from `HabitatLossAnalyzer` and pins both
    output rasters to `self.crs` at `self.scale`, so the per-cell loss
    `reduceRegions` runs in the same projection as the rasters it samples.
    """

    def get_habitat_loss_raster(
        self, glc_processed, gpw_processed, hgfc_processed, start_yr
    ):
        """Build loss + start rasters, then pin them to `self.crs` at `self.scale`."""
        loss_raster, start_raster = super().get_habitat_loss_raster(
            glc_processed, gpw_processed, hgfc_processed, start_yr
        )
        loss_raster = loss_raster.reproject(crs=self.crs, scale=self.scale)
        start_raster = start_raster.reproject(crs=self.crs, scale=self.scale)
        return loss_raster, start_raster

    def calc_loss_score_per_cell(
        self, habitat_loss_raster, habitat_start_raster, fc
    ):
        """Score Habitat Loss within each cell as `1 - (loss_area / start_area)`.

        Cells with zero start-year habitat get a score of 0, mirroring the
        absolute pipeline's guard in `calc_habitat_loss_score`.
        """
        loss_and_start = (
            ee.Image.pixelArea().updateMask(habitat_loss_raster).rename("loss_area")
        ).addBands(
            ee.Image.pixelArea().updateMask(habitat_start_raster).rename("start_area")
        )
        with_areas = loss_and_start.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.sum(),
            scale=self.scale,
            crs=self.crs,
        )

        def add_score(feature):
            loss = ee.Number(
                ee.Algorithms.If(feature.get("loss_area"), feature.get("loss_area"), 0)
            )
            start = ee.Number(
                ee.Algorithms.If(
                    feature.get("start_area"), feature.get("start_area"), 0
                )
            )
            score = ee.Algorithms.If(
                start.eq(0),
                0,
                ee.Number(1).subtract(loss.divide(start).min(1)),
            )
            return feature.set("loss_score", score)

        return with_areas.map(add_score)
