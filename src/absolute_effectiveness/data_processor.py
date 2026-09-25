import ee
from utils.variables import (
    AFCD_2018_ASSET_ID,
    AFCD_2019_ASSET_ID,
    AFCD_2020_ASSET_ID,
    AFCD_2021_ASSET_ID,
    AFCD_2022_ASSET_ID,
    ANALYSIS_END_YR,
    GLC_ASSET_ID,
    GLC_CLASSES,
    GPW_ASSET_ID,
    HGFC_ASSET_ID,
    NFW_ASSET_ID,
    NFW_THRESHOLD,
)


class DataProcessor:
    """
    Imports and pre-processes core source datasets used in the analysis.
    """

    def __init__(
        self,
        glc_collection,
        gpw_collection,
        nfw_collection,
        hgfc_image,
        afcd_image=None,
        analysis_end_yr=ANALYSIS_END_YR,
        glc_classes=GLC_CLASSES,
        nfw_threshold=NFW_THRESHOLD,
    ):
        self.glc_collection = glc_collection
        self.gpw_collection = gpw_collection
        self.nfw_collection = nfw_collection
        self.hgfc_image = hgfc_image
        # Optional region-specific cropland override (African PAs only).
        # None for the global workflow, which leaves the analysis unchanged.
        self.afcd_image = afcd_image
        self.analysis_end_yr = analysis_end_yr
        self.glc_classes = glc_classes
        self.nfw_threshold = nfw_threshold

    @classmethod
    def from_gee_defaults(cls):
        """
        Use Earth Engine asset IDs from utils.variables.
        """
        return cls(
            glc_collection=ee.ImageCollection(GLC_ASSET_ID),
            gpw_collection=ee.ImageCollection(GPW_ASSET_ID),
            nfw_collection=ee.ImageCollection(NFW_ASSET_ID),
            hgfc_image=ee.Image(HGFC_ASSET_ID),
        )

    @classmethod
    def from_africa_defaults(cls):
        """
        Same global sources as from_gee_defaults, plus the African Cropland
        Dataset (AFCD).
        """
        return cls(
            glc_collection=ee.ImageCollection(GLC_ASSET_ID),
            gpw_collection=ee.ImageCollection(GPW_ASSET_ID),
            nfw_collection=ee.ImageCollection(NFW_ASSET_ID),
            hgfc_image=ee.Image(HGFC_ASSET_ID),
            afcd_image = (ee.Image(AFCD_2018_ASSET_ID).rename("AFCD_2018")
                .addBands(ee.Image(AFCD_2019_ASSET_ID).rename("AFCD_2019"))
                .addBands(ee.Image(AFCD_2020_ASSET_ID).rename("AFCD_2020"))
                .addBands(ee.Image(AFCD_2021_ASSET_ID).rename("AFCD_2021"))
                .addBands(ee.Image(AFCD_2022_ASSET_ID).rename("AFCD_2022"))
            ),
        )

    def get_land_mask(self):
        """
        Land mask from Hansen datamask to exclude oceans and permanent water.
        1 = land, 2 = permanent water/ocean, 0 = no data.
        """
        return self.hgfc_image.select("datamask").eq(1)

    def _apply_land_mask(self, image):
        return image.updateMask(self.get_land_mask())

    def process_glc(self, test_sites, start_yr, land_masked=True):
        """
        Process Global Land Cover Change data for the analysis period.
        """
        glc_mosaic = self.glc_collection.filterBounds(test_sites).mosaic()
        analysis_years = list(range(start_yr, self.analysis_end_yr + 1))
        # Rename bands for clarity
        band_names = [f"b{year - 2000 + 1}" for year in analysis_years]
        new_band_names = [f"GLC_{year}" for year in analysis_years]
        glc_selected = glc_mosaic.select(band_names, new_band_names)

        def remap_classes(band):
            """
            Remap GLC classes to 1-36.
            """
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
        glc = ee.Image.cat(remapped_bands)
        return self._apply_land_mask(glc) if land_masked else glc

    def process_gpw(self, start_yr, land_masked=True):
        """
        Process Global Pasture Watch data for the analysis period.
        """
        year_strings = [str(year) for year in range(start_yr, self.analysis_end_yr + 1)]
        gpw_filtered = self.gpw_collection.filter(
            ee.Filter.inList("system:index", year_strings)
        ).toBands()
        gpw_renamed = gpw_filtered.rename([f"GPW_{year}" for year in year_strings])
        gpw = gpw_renamed.unmask()
        return self._apply_land_mask(gpw) if land_masked else gpw

    def process_nfw(self, test_sites, land_masked=True):
        """
        Process Natural Forests of the World (2020) data.
        """
        nfw_mosaic = self.nfw_collection.filterBounds(test_sites).mosaic()
        nfw = nfw_mosaic.gte(self.nfw_threshold)
        return self._apply_land_mask(nfw) if land_masked else nfw

    def process_hgfc(self, start_yr, land_masked=True):
        """
        Process Hansen Global Forest Change data for the analysis period.
        """
        hgfc_selected = self.hgfc_image.select("lossyear")
        analysis_mask = hgfc_selected.gte(start_yr - 2000).And(
            hgfc_selected.lte(self.analysis_end_yr - 2000)
        )
        hgfc_masked = hgfc_selected.updateMask(analysis_mask)
        hgfc = hgfc_masked.unmask()
        return self._apply_land_mask(hgfc) if land_masked else hgfc

    def process_afcd(self, start_yr, land_masked=True):
        """
        Process African Cropland Dataset (AFCD) data for the analysis
        period.

        """
        # Select bands for analysis years
        analysis_years = list(range(start_yr, self.analysis_end_yr + 1))
        band_names = [f"AFCD_{year}" for year in analysis_years]
        afcd_selected = self.afcd_image.select(band_names)
        
        return self._apply_land_mask(afcd_selected) if land_masked else afcd_selected