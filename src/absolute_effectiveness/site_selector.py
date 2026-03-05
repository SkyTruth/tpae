import logging
import ee
from utils.variables import (
    ANALYSIS_START_YR,
    ANALYSIS_END_YR,
    OECMS_ASSET_ID,
    PAS_ASSET_ID,
    TEST_SITE_IDS,
)

logger = logging.getLogger(__name__)


class SiteSelector:
    """Selects analysis site and derives site-specific context."""

    def __init__(
        self,
        test_site_ids=TEST_SITE_IDS,
        analysis_start_yr=ANALYSIS_START_YR,
        analysis_end_yr=ANALYSIS_END_YR,
        pas_collection_id=PAS_ASSET_ID,
        oecms_collection_id=OECMS_ASSET_ID,
    ):
        self.test_site_ids = test_site_ids
        self.analysis_start_yr = analysis_start_yr
        self.analysis_end_yr = analysis_end_yr
        self.pas = ee.FeatureCollection(pas_collection_id)
        self.oecms = ee.FeatureCollection(oecms_collection_id)

    def get_test_sites(self):
        """Get a set of selected terrestrial PAs and OECMs for testing."""
        all_sites = (
            ee.FeatureCollection([self.pas, self.oecms])
            .flatten()
            .filter(ee.Filter.eq("REALM", "Terrestrial"))
        )
        return all_sites.filter(ee.Filter.inList("SITE_ID", self.test_site_ids))

    def set_start_yr(self, test_sites, site_id):
        """Set analysis start year, constrained by designation and global start year."""
        designation_yr = (
            test_sites.filter(ee.Filter.eq("SITE_ID", site_id))
            .first()
            .get("STATUS_YR")
            .getInfo()
        )
        return max(self.analysis_start_yr, designation_yr)

    def check_start_yr(self, start_yr):
        """Check if start year is valid for analysis."""
        if start_yr > (self.analysis_end_yr - 1):
            msg = (
                f"PA was designated too recently ({start_yr}) to analyze effectiveness."
            )
            logger.error(msg)
            raise ValueError(msg)

    def get_site_geom(self, test_sites, site_id):
        """Get geometry of a specific PA."""
        return test_sites.filter(ee.Filter.eq("SITE_ID", site_id)).geometry()
