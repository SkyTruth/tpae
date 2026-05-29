import json
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from itertools import groupby

with open('data/test_sites_raw.geojson') as f:
    fc = json.load(f)

# Clean geometries
for feature in fc['features']:
    geom = shape(feature['geometry'])
    areal = unary_union([g for g in getattr(geom, 'geoms', [geom]) if g.geom_type in ('Polygon', 'MultiPolygon')])
    feature['geometry'] = mapping(areal)

# Rename SITE_ID column to WDPAID
for feature in fc['features']:
    props = feature['properties']
    if 'SITE_ID' in props:
        props['WDPAID'] = props.pop('SITE_ID')


# Dissolve by WDPAID
features = sorted(fc['features'], key=lambda f: f['properties']['WDPAID'])
dissolved = []
for wdpaid, group in groupby(features, key=lambda f: f['properties']['WDPAID']):
    group = list(group)
    merged_geom = unary_union([shape(f['geometry']) for f in group])
    dissolved.append({
        'type': 'Feature',
        'geometry': mapping(merged_geom),
        'properties': group[0]['properties']  # keep first feature's properties
    })


with open('data/test_sites_cleaned.geojson', 'w') as f:
    json.dump({'type': 'FeatureCollection', 'features': dissolved}, f)