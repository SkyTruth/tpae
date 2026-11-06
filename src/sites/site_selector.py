import logging
import ee
from utils.variables import (
    ANALYSIS_START_YR,
    ANALYSIS_END_YR,
    OECMS_ASSET_ID,
    PAS_ASSET_ID,
    SITE_IDS,
)

logger = logging.getLogger(__name__)


class SiteSelector:
    """
    Selects site for analysis and gets site-specific variables.
    """

    def __init__(
        self,
        site_ids=SITE_IDS,
        analysis_start_yr=ANALYSIS_START_YR,
        analysis_end_yr=ANALYSIS_END_YR,
        pas_collection_id=PAS_ASSET_ID,
        oecms_collection_id=OECMS_ASSET_ID,
    ):
        self.site_ids = site_ids
        self.analysis_start_yr = analysis_start_yr
        self.analysis_end_yr = analysis_end_yr
        self.pas = ee.FeatureCollection(pas_collection_id)
        self.oecms = ee.FeatureCollection(oecms_collection_id)

    def get_sites(self):
        """
        Get the selected sites (terrestrial PAs and OECMs).
        """
        all_sites = (
            ee.FeatureCollection([self.pas, self.oecms])
            .flatten()
            .filter(ee.Filter.eq("REALM", "Terrestrial"))
        )
        return all_sites.filter(ee.Filter.inList("SITE_ID", self.site_ids))

    def set_start_yr(self, sites, site_id):
        """
        Set start year to designation year if it is later than the analysis start year.
        """
        designation_yr = (
            sites.filter(ee.Filter.eq("SITE_ID", site_id))
            .first()
            .get("STATUS_YR")
            .getInfo()
        )
        return max(self.analysis_start_yr, designation_yr)

    def check_start_yr(self, start_yr):
        """
        Check if start year is valid for analysis.
        """
        if start_yr > (self.analysis_end_yr - 1):
            msg = f"Site was designated too recently ({start_yr}) to analyze effectiveness."
            logger.error(msg)
            raise ValueError(msg)

    def get_site_geom(self, sites, site_id):
        """
        Get geometry of a specific site.
        """
        return sites.filter(ee.Filter.eq("SITE_ID", site_id)).geometry()
