import ee
import geemap
import pandas as pd

from utils.dist_variables import(FOLDERSET)

def calculate_site_AC_dist_rate(year, ANOM_lower, CONF_lower, test_site_id, test_sites):
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
    site = test_sites.filter(ee.Filter.eq("SITE_ID", test_site_id))
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

def calculate_all_sites_AC_dist_rate(year, ANOM_lower, CONF_lower, test_sites):
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
        collection=test_sites,
        reducer=ee.Reducer.count(),
        scale=30
    )

    def add_rate(feature):
        dist  = ee.Number(feature.get('dist'))
        total = ee.Number(feature.get('total'))
        return feature.set('dist_rate', dist.divide(total))

    stats = stats.map(add_rate)

    features = stats.select(['NAME', 'SITE_ID', 'dist', 'total', 'dist_rate']).getInfo()['features']

    df = pd.DataFrame([f['properties'] for f in features])
    df['year'] = year
    return df