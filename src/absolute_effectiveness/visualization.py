import ee


class VisualizationService:
    """Builds helper visualization products for map display (not used in analysis)."""

    def __init__(
        self,
        s2_collection_id="COPERNICUS/S2_SR_HARMONIZED",
        max_cloud_pct=20,
    ):
        self.s2_collection_id = s2_collection_id
        self.max_cloud_pct = max_cloud_pct

    def mask_s2_clouds(self, image):
        """Mask clouds in a Sentinel-2 image."""
        qa = image.select("QA60")
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = (
            qa.bitwiseAnd(cloud_bit_mask)
            .eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        )
        return image.updateMask(mask).divide(10000)

    def get_s2_med_composite(self, site_geom, year):
        """Create median Sentinel-2 composite for a given year and site."""
        return (
            ee.ImageCollection(self.s2_collection_id)
            .filterBounds(site_geom)
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", self.max_cloud_pct))
            .map(self.mask_s2_clouds)
            .median()
        )
