import ee
import geemap
import pandas as pd

from utils.dist_variables import(FOLDERSET)

# visualization function for the dist_status workflow
def visualize_site_disturbance(year, test_site_id, test_sites, Map=None):
    VEGDISTSTATUS = ee.ImageCollection(FOLDERSET[year] + "/VEG-DIST-STATUS").mosaic()

    dist_from = [0, 3, 6, 7, 8, 9, 10]
    dist_to   = [0, 1, 2, 3, 4, 5, 6]
    VEGDISTSTATUS_REMAP = VEGDISTSTATUS.remap(dist_from, dist_to, defaultValue=0)

    site = test_sites.filter(ee.Filter.eq("SITE_ID", test_site_id))
    site_name = site.first().get("NAME").getInfo()
    site_geom = site.geometry()

    site_DIST = VEGDISTSTATUS_REMAP.clip(site_geom)

    palette = [
        "121212",  # black
        "E48727",  # light orange
        "E01B07",  # red
        "777777",  # grey
        "DDDDDD",  # light grey
        "005555",  # dark slate
        "008888"   # dark cyan
    ]

    legend_dict = {
        "No disturbance":          "#121212",
        "Confirmed <50% ongoing":  "#E48727",
        "Confirmed ≥50% ongoing":  "#E01B07",
        "Confirmed <50% finished": "#777777",
        "Confirmed ≥50% finished": "#DDDDDD",
        "Prev year <50%":          "#005555",
        "Prev year ≥50%":          "#008888",
    }

    if Map is None:
        Map = geemap.Map()
        Map.add_basemap("Esri.WorldImagery")
        Map.addLayer(site_geom, {"color": "red"}, f"{site_name} Boundary")
        Map.centerObject(site)
        Map.add_legend(
            title="DIST-ANN Disturbance Status",
            legend_dict=legend_dict,
            draggable=False,
        )

    Map.addLayer(site_DIST, {"min": 0, "max": 6, "palette": palette}, f"DIST {year} - {site_name}")

    return Map

# visualization function for the anom_conf workflow
def visualize_AC_site_dist(year, test_site_id, test_sites, Map=None, ANOM_lower=30, CONF_lower=400):
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
    ac_vis = ac_masked.where(ac_masked.mask().Not(), -9999)

    site = test_sites.filter(ee.Filter.eq("SITE_ID", test_site_id))
    site_name = site.first().get("NAME").getInfo()
    site_geom = site.geometry()

    site_DIST = ac_vis.clip(site_geom)
    counts = site_DIST.select('anom').reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(), geometry=site, scale=30, maxPixels=1e13
    ).getInfo()
    print('Pixel counts:', counts)

    palette = ['fee5d9', 'fcae91', 'fb6a4a', 'de2d26', '895c5e']

    if Map is None:
        Map = geemap.Map()
        Map.add_basemap("Esri.WorldImagery")
        Map.addLayer(site_geom, {"color": "red"}, f"{site_name} Boundary")
        Map.centerObject(site)

    Map.addLayer(site_DIST.select('anom'), {"min": 0, "max": 100, "palette": palette}, f"DIST {year} - {site_name}")

    return Map

