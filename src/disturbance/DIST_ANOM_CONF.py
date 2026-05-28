import ee
import geemap
import pandas as pd

from utils.dist_variables import(FOLDERSET)

def calculate_site_AC_dist_rate(year, ANOM_lower, CONF_lower, test_site_id, sites):
    """
    Calculate the Anomaly + Confidence disturbance rate for a single PA site

    Disturbed pixels = pixels 
        > ANOM_lower disturbance (30%)
        > CONF_lower confidence  (400)

    Parameters
    ----------
    year : str
        Disturbance year ('2023', '2024', or '2025').
    site_id : int
        SITE_ID of the target protected area.
    sites : ee.FeatureCollection
        Feature collection containing the PA sites (test or full list).
    anom_lower : int
        Minimum VEGANOMMAX threshold for the disturbance mask.
    conf_lower : int
        Minimum VEGDISTCONF threshold for the disturbance mask.
    
    Returns
    -------
    dist_rate: number of disturbed pixels / total pixels
    """

    # import the appropriate DIST-ANN VEG-DIST-ANOM & CONF layers for the given year
    folder = FOLDERSET[year]

    VEGANOMMAX = ee.ImageCollection(folder+"/VEG-ANOM-MAX").mosaic()
    VEGDISTCONF = ee.ImageCollection(folder+"/VEG-DIST-CONF").mosaic()
    
    combined = ee.Image.cat([
        VEGANOMMAX.rename('anom'),
        VEGDISTCONF.rename('conf')
    ])

    def maskAC(image):
        mask = image.select('anom').gt(ANOM_lower).And(image.select('anom').lt(255)).And(image.select('conf').gt(CONF_lower)).selfMask()
        masked = image.updateMask(mask)
        return masked
    
    ac_masked = maskAC(combined)

    # filter to site of interest and calculate disturbance rate
    site = sites.filter(ee.Filter.eq("SITE_ID", test_site_id))
    site_name = site.first().get("NAME").getInfo()
    site_geom = site.geometry()

    # ee reducer on 'anom' band
    dist_pixels = ac_masked.reduceRegion(
        reducer=ee.Reducer.count(), 
        geometry=site_geom, 
        scale=30, 
        maxPixels=1e9
    ).getInfo()['anom']

    total_pixels = combined.reduceRegion(
        reducer=ee.Reducer.count(), geometry=site_geom, scale=30, maxPixels=1e9
    ).getInfo()['anom']

    # calculate and print disturbance rate
    dist_rate = dist_pixels / total_pixels
    print(f"{site_name} ({year}) disturbance rate: {dist_rate:.4%}")
    return dist_rate

def calculate_FeatureCollection_AC_dist_rate(year, ANOM_lower, CONF_lower, features):
    """
    Calculate the Anomaly + Confidence disturbance rate for a Feature Collection

    Disturbed pixels = pixels 
        > ANOM_lower disturbance (30%)
        > CONF_lower confidence  (400)

    Parameters
    ----------
    year : str
        Disturbance year ('2023', '2024', or '2025').
    features : ee.FeatureCollection
        Feature collection containing the PA sites, PSM grid, or other geometries.
    anom_lower : int
        Minimum VEGANOMMAX threshold for the disturbance mask.
    conf_lower : int
        Minimum VEGDISTCONF threshold for the disturbance mask.
    
    Returns
    -------
    dist_rate: number of disturbed pixels / total pixels
    """
    
    folder = FOLDERSET[year]

    VEGANOMMAX = ee.ImageCollection(folder + "/VEG-ANOM-MAX").mosaic()
    VEGDISTCONF = ee.ImageCollection(folder + "/VEG-DIST-CONF").mosaic()
    combined = ee.Image.cat([
        VEGANOMMAX.rename('anom'),
        VEGDISTCONF.rename('conf')
    ])

    def maskAC(image):
        mask = (
            image.select('anom').gt(ANOM_lower)
            .And(image.select('anom').lt(255))
            .And(image.select('conf').gt(CONF_lower))
            .selfMask()
        )
        return image.updateMask(mask)

    ac_masked = maskAC(combined)

    combined = ee.Image.cat([
        ac_masked.select('anom').rename('dist'),
        VEGANOMMAX.rename('total')
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