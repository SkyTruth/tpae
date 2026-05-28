import ee
import geemap
import pandas as pd

from utils.dist_variables import(FOLDERSET)

def calculate_site_disturbance_rate(year, test_site_id, sites):
    """
    Calculate the disturbance rate for a single PA site

    Disturbed pixels = pixels with > 50% disturbance and high confidence

    Parameters
    ----------
    year : str
        Disturbance year ('2023', '2024', or '2025').
    site_id : int
        SITE_ID of the target protected area.
    sites : ee.FeatureCollection
        Feature collection containing the PA sites (test or full list).
    
    Returns
    -------
    dist_rate: number of disturbed pixels / total pixels
    """
    VEGDISTSTATUS = ee.ImageCollection(FOLDERSET[year] + "/VEG-DIST-STATUS").mosaic()

    mask_from = [0, 3, 6, 7, 8, 9, 10]
    mask_to   = [0, 0, 1, 0, 1, 0, 1]
    dist_mask = VEGDISTSTATUS.remap(mask_from, mask_to, 0).eq(1)
    masked_dist = VEGDISTSTATUS.updateMask(dist_mask)

    site = sites.filter(ee.Filter.eq("SITE_ID", test_site_id))
    site_name = site.first().get("NAME").getInfo()
    site_geom = site.geometry()

    dist_pixels = masked_dist.reduceRegion(
        reducer=ee.Reducer.count(), geometry=site_geom, scale=30, maxPixels=1e9
    ).getInfo()['b1']

    total_pixels = VEGDISTSTATUS.reduceRegion(
        reducer=ee.Reducer.count(), geometry=site_geom, scale=30, maxPixels=1e9
    ).getInfo()['b1']

    dist_rate = dist_pixels / total_pixels
    print(f"{site_name} ({year}) disturbance rate: {dist_rate:.4%}")
    return dist_rate


def calculate_FeatureCollection_disturbance_rate(year, features):
    """
    Calculate the disturbance rate for each feature in a Feature Collection

    Disturbed pixels = pixels with > 50% disturbance and high confidence
    Parameters
    ----------
    year : str
        Disturbance year ('2023', '2024', or '2025').
    features : ee.FeatureCollection
        Feature collection containing the PA sites, PSM grid, or other geometries.
    
    Returns
    -------
    dist_rate: number of disturbed pixels / total pixels
    """
    VEGDISTSTATUS = ee.ImageCollection(FOLDERSET[year] + "/VEG-DIST-STATUS").mosaic()

    mask_from = [0, 3, 6, 7, 8, 9, 10]
    mask_to   = [0, 0, 1, 0, 1, 0, 1]
    dist_mask = VEGDISTSTATUS.remap(mask_from, mask_to, 0).eq(1)
    masked_dist = VEGDISTSTATUS.updateMask(dist_mask)

    combined = ee.Image.cat([
        masked_dist.rename('dist'),
        VEGDISTSTATUS.rename('total')
    ])

    stats = combined.reduceRegions(
        collection=features,
        reducer=ee.Reducer.count(),
        scale=30
    )

    def add_rate(feature):
        dist  = ee.Number(feature.get('dist'))
        total = ee.Number(feature.get('total'))
        return feature.set('dist_rate', dist.divide(total))

    stats = stats.map(add_rate)

    stat_features = stats.getInfo()['features']

    df = pd.DataFrame([f['properties'] for f in stat_features])
    df['year'] = year
    return df