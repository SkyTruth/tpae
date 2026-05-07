import ee
import geemap
import pandas as pd

FOLDERSET = {
    '2025': "projects/glad/HLSDIST/DIST-ANN_v1_2025",
    '2024': "projects/glad/HLSDIST/DIST-ANN_v1_2024",
    '2023': "projects/glad/HLSDIST/DIST-ANN_v1",
}


def calculate_site_disturbance_rate(year, test_site_id, test_sites):
    VEGDISTSTATUS = ee.ImageCollection(FOLDERSET[year] + "/VEG-DIST-STATUS").mosaic()

    mask_from = [0, 3, 6, 7, 8, 9, 10]
    mask_to   = [0, 0, 1, 0, 1, 0, 1]
    dist_mask = VEGDISTSTATUS.remap(mask_from, mask_to, 0).eq(1)
    masked_dist = VEGDISTSTATUS.updateMask(dist_mask)

    site = test_sites.filter(ee.Filter.eq("SITE_ID", test_site_id))
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


def calculate_all_sites_disturbance_rate(year, test_sites):
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
