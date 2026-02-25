import ee
from utils.variables import (
    ANALYSIS_END_YR,
    GLC_ASSET_ID,
    GLC_CLASSES,
    GPW_ASSET_ID,
    HGFC_ASSET_ID,
    NFW_ASSET_ID,
    NFW_THRESHOLD,
)


class DataProcessor:
    """Processor for core source datasets used in the analysis."""

    def __init__(
        self,
        glc_collection,
        gpw_collection,
        nfw_collection,
        hgfc_image,
        analysis_end_yr=ANALYSIS_END_YR,
        glc_classes=GLC_CLASSES,
        nfw_threshold=NFW_THRESHOLD,
    ):
        self.glc_collection = glc_collection
        self.gpw_collection = gpw_collection
        self.nfw_collection = nfw_collection
        self.hgfc_image = hgfc_image
        self.analysis_end_yr = analysis_end_yr
        self.glc_classes = glc_classes
        self.nfw_threshold = nfw_threshold

    @classmethod
    def from_gee_defaults(cls):
        """Build a processor wired to Earth Engine assets from utils.variables."""
        return cls(
            glc_collection=ee.ImageCollection(GLC_ASSET_ID),
            gpw_collection=ee.ImageCollection(GPW_ASSET_ID),
            nfw_collection=ee.ImageCollection(NFW_ASSET_ID),
            hgfc_image=ee.Image(HGFC_ASSET_ID),
        )

    def process_glc(self, test_sites, start_yr):
        """Process Global Land Cover Change data for the analysis period."""
        glc_mosaic = self.glc_collection.filterBounds(test_sites).mosaic()
        analysis_years = list(range(start_yr, self.analysis_end_yr + 1))
        band_names = [f"b{year - 2000 + 1}" for year in analysis_years]
        new_band_names = [f"GLC_{year}" for year in analysis_years]
        glc_selected = glc_mosaic.select(band_names, new_band_names)

        def remap_classes(band):
            return (
                glc_selected.select(band)
                .remap(
                    self.glc_classes,
                    ee.List.sequence(1, len(self.glc_classes)),
                    defaultValue=0,
                )
                .rename([band])
            )

        remapped_bands = [remap_classes(band) for band in new_band_names]
        return ee.Image.cat(remapped_bands)

    def process_gpw(self, start_yr):
        """Process Global Pasture Watch data for the analysis period."""
        year_strings = [str(year) for year in range(start_yr, self.analysis_end_yr + 1)]
        gpw_filtered = self.gpw_collection.filter(
            ee.Filter.inList("system:index", year_strings)
        ).toBands()
        gpw_renamed = gpw_filtered.rename([f"GPW_{year}" for year in year_strings])
        return gpw_renamed.unmask()

    def process_nfw(self, test_sites):
        """Process Natural Forests of the World (2020) data."""
        nfw_mosaic = self.nfw_collection.filterBounds(test_sites).mosaic()
        return nfw_mosaic.gte(self.nfw_threshold)

    def process_hgfc(self, start_yr):
        """Process Hansen Global Forest Change data for the analysis period."""
        hgfc_selected = self.hgfc_image.select("lossyear")
        analysis_mask = hgfc_selected.gte(start_yr - 2000).And(
            hgfc_selected.lte(self.analysis_end_yr - 2000)
        )
        hgfc_masked = hgfc_selected.updateMask(analysis_mask)
        return hgfc_masked.unmask()
