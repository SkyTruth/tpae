import calendar
import json
from datetime import datetime

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DW_CLASSES = {
    0: 'Water', 1: 'Trees', 2: 'Grass', 3: 'Flooded Veg',
    4: 'Crops', 5: 'Shrub/Scrub', 6: 'Built Area', 7: 'Bare Ground', 8: 'Snow/Ice'
}
DW_PALETTE = [
    '#419BDF', '#397D49', '#88B053', '#7A87C6', '#E49635',
    '#DFC35A', '#C4281B', '#A59B8F', '#B39FE1'
]
DW_VIS = {'min': 0, 'max': 8, 'palette': [c.lstrip('#') for c in DW_PALETTE]}

FOLDERSET = {
    '2025': "projects/glad/HLSDIST/DIST-ANN_v1_2025",
    '2024': "projects/glad/HLSDIST/DIST-ANN_v1_2024",
    '2023': "projects/glad/HLSDIST/DIST-ANN_v1",
}


def build_dist_mask(year, anom_lower=30, conf_lower=400):
    """Binary mask (1 = disturbed) from VEGANOMMAX + VEGDISTCONF."""
    folder = FOLDERSET[year]
    veganommax = ee.ImageCollection(folder + "/VEG-ANOM-MAX").mosaic()
    vegdistconf = ee.ImageCollection(folder + "/VEG-DIST-CONF").mosaic()
    return (
        veganommax.gt(anom_lower)
        .And(veganommax.lt(255))
        .And(vegdistconf.gt(conf_lower))
        .rename('dist_mask')
    )


def build_dw_mode_composite(start_date, end_date, geometry):
    """Annual (or arbitrary window) DW mode composite over a geometry."""
    return (
        ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
        .filterDate(start_date, end_date)
        .filterBounds(geometry)
        .select('label')
        .reduce(ee.Reducer.mode())
        .rename('label_mode')
    )


def build_monthly_dw_composites(year, geometry, cross_year=True, cross_year_post_months=2):
    """
    Build monthly DW mode composites; returns dict {position: ee.Image}.

    A fully-masked fallback is merged into each collection before reducing
    so every composite has 1 band named 'label_mode' even when no DW images
    exist for that month and location.

    cross_year=True  → 24 composites: positions 1-12 = prior year,
                        positions 13-24 = disturbance year.
    cross_year=False → 12 composites: positions 1-12 = disturbance year onsly.

    cross_year_post_months > 0 → appends that many months of yr+1 (Jan, Feb, …)
        as positions immediately after the disturbance-year block, extending the
        post window for pixels disturbed late in the year.
    """
    yr = int(year)
    composites = {}
    fallback = ee.Image.constant(0).rename('label').selfMask()

    def make_composite(start, end):
        return (
            ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
            .filterDate(start, end)
            .filterBounds(geometry)
            .select('label')
            .merge(ee.ImageCollection([fallback]))
            .reduce(ee.Reducer.mode())
            .rename('label_mode')
        )

    if cross_year:
        for m in range(1, 13):
            _, days = calendar.monthrange(yr - 1, m)
            composites[m] = make_composite(
                datetime(yr - 1, m, 1).strftime('%Y-%m-%d'),
                datetime(yr - 1, m, days).strftime('%Y-%m-%d'),
            )
        month_offset = 12
    else:
        month_offset = 0

    for m in range(1, 13):
        _, days = calendar.monthrange(yr, m)
        composites[m + month_offset] = make_composite(
            datetime(yr, m, 1).strftime('%Y-%m-%d'),
            datetime(yr, m, days).strftime('%Y-%m-%d'),
        )

    if cross_year_post_months > 0:
        base = 12 + month_offset
        for m in range(1, cross_year_post_months + 1):
            _, days = calendar.monthrange(yr + 1, m)
            composites[base + m] = make_composite(
                datetime(yr + 1, m, 1).strftime('%Y-%m-%d'),
                datetime(yr + 1, m, days).strftime('%Y-%m-%d'),
            )

    return composites


def build_dist_month_image(year, dist_mask, cross_year=True):
    """
    Convert VEGDISTDATE DOY → unified month position per pixel.

    cross_year=True  → positions 13-24 (Jan=13 … Dec=24)
    cross_year=False → positions 1-12  (Jan=1  … Dec=12)
    """
    yr = int(year)
    month_offset = 12 if cross_year else 0

    from_doys, to_months = [], []
    for m in range(1, 13):
        start_doy = datetime(yr, m, 1).timetuple().tm_yday
        _, days = calendar.monthrange(yr, m)
        for doy in range(start_doy, start_doy + days):
            from_doys.append(doy)
            to_months.append(m + month_offset)

    folder = FOLDERSET[year]
    vegdistdate = ee.ImageCollection(folder + "/VEG-DIST-DATE").mosaic()
    return (
        vegdistdate
        .updateMask(dist_mask)
        .updateMask(vegdistdate.gt(0))
        .remap(from_doys, to_months, 0)
        .rename('dist_month')
    )


def build_per_pixel_pre_post(monthly_composites, dist_month_img):
    """
    Per-pixel pre/post composites from a set of monthly composites.

    For position k:
      pre  = composite k where dist_month > k  (observation is before disturbance)
      post = composite k where dist_month < k  (observation is after disturbance)
    """
    positions = sorted(monthly_composites.keys())

    pre_collection = ee.ImageCollection([
        monthly_composites[k].updateMask(dist_month_img.gt(k))
        for k in positions
    ])
    post_collection = ee.ImageCollection([
        monthly_composites[k].updateMask(dist_month_img.lt(k))
        for k in positions
    ])

    dw_pre = pre_collection.reduce(ee.Reducer.mode()).rename('label_mode')
    dw_post = post_collection.reduce(ee.Reducer.mode()).rename('label_mode')
    return dw_pre, dw_post


def get_lc_change(year, test_site_id, test_sites,
                  anom_lower=30, conf_lower=400, min_post_months=2,
                  min_disturbed_pixels=1, cross_year_post_months=3):
    """
    Compare pre vs. post disturbance DynamicWorld land cover at disturbed pixels.

    PRE strategy  — annual composite for the full prior year (yr-1).
    POST strategy — per-pixel monthly composites for the disturbance year,
      optionally extended into the first months of yr+1 for late-year pixels.
      Each pixel's post window starts the month after its VEGDISTDATE, filtered
      to pixels with >= min_post_months of post-disturbance data available.

    Parameters
    ----------
    year : str
        Disturbance year ('2023', '2024', or '2025').
    test_site_id : int
        SITE_ID of the target protected area.
    test_sites : ee.FeatureCollection
        Feature collection containing the test sites (from SiteSelector).
    anom_lower : int
        Minimum VEGANOMMAX threshold for the disturbance mask.
    conf_lower : int
        Minimum VEGDISTCONF threshold for the disturbance mask.
    min_post_months : int
        Minimum months of post-disturbance imagery required per pixel.
    min_disturbed_pixels : int
        Minimum disturbed-pixel count to proceed; returns a skipped sentinel
        dict if the site falls below this threshold.
    cross_year_post_months : int
        Number of months from yr+1 (Jan, Feb, …) to include as post-disturbance
        composites. Allows Nov/Dec pixels to accumulate enough post data.
        Set to 3 by default.
    """
    yr = int(year)
    site = test_sites.filter(ee.Filter.eq("SITE_ID", test_site_id))
    site_name = site.first().get("NAME").getInfo()
    geom = site.geometry()

    dist_mask = build_dist_mask(year, anom_lower, conf_lower)

    # Early exit: skip sites with too few disturbed pixels
    pixel_count = (
        dist_mask.updateMask(dist_mask)
        .reduceRegion(reducer=ee.Reducer.count(), geometry=geom, scale=30, maxPixels=1e9)
        .getInfo()
        .get('dist_mask', 0)
    )
    if pixel_count < min_disturbed_pixels:
        return {
            'site_name': site_name,
            'year': year,
            'skipped': True,
            'reason': f'{pixel_count} disturbed pixels (threshold: {min_disturbed_pixels})',
        }

    # PRE: full prior-year annual composite at disturbed pixels
    dw_pre = (
        build_dw_mode_composite(f'{yr - 1}-01-01', f'{yr - 1}-12-31', geom)
        .updateMask(dist_mask)
    )

    # POST: per-pixel monthly composites, extended into yr+1 for late-year pixels
    dist_month_img = build_dist_month_image(year, dist_mask, cross_year=False)
    monthly_comps = build_monthly_dw_composites(
        year, geom, cross_year=False, cross_year_post_months=cross_year_post_months
    )
    _, dw_post = build_per_pixel_pre_post(monthly_comps, dist_month_img)
    dw_post = dw_post.updateMask(
        dist_month_img.lte(12 + cross_year_post_months - min_post_months)
    )

    def freq_hist(img):
        raw = img.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=geom, scale=30, maxPixels=1e9,
        ).getInfo().get('label_mode', {})
        return {DW_CLASSES[int(float(k))]: int(v) for k, v in raw.items()}

    pre_hist = freq_hist(dw_pre)
    post_hist = freq_hist(dw_post)

    stacked = dw_pre.rename('pre').addBands(dw_post.rename('post'))
    encoded = stacked.expression('b("pre") * 10 + b("post")').rename('transition')
    trans_hist = encoded.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=geom, scale=30, maxPixels=1e9,
    ).getInfo().get('transition', {})

    n = 9
    matrix = np.zeros((n, n), dtype=int)
    for key, count in trans_hist.items():
        code = int(float(key))
        pre_cls, post_cls = code // 10, code % 10
        if 0 <= pre_cls < n and 0 <= post_cls < n:
            matrix[pre_cls, post_cls] = int(count)

    class_labels = [DW_CLASSES[i] for i in range(n)]
    transition_df = pd.DataFrame(matrix, index=class_labels, columns=class_labels)
    transition_df.index.name = 'Pre → Post'

    return {
        'site_name': site_name,
        'year': year,
        'pre_counts': pre_hist,
        'post_counts': post_hist,
        'transition_df': transition_df,
        'dw_pre': dw_pre,
        'dw_post': dw_post,
        'dist_mask': dist_mask,
        'dist_month_img': dist_month_img,
        'geometry': geom,
    }


def _parse_class_hist(raw):
    """Convert a GEE frequencyHistogram dict to {class_name: count}."""
    return {DW_CLASSES[int(float(k))]: int(v) for k, v in raw.items()}


def _parse_transition_df(raw):
    """Convert a GEE transition histogram dict to a 9×9 DW transition DataFrame."""
    n = 9
    matrix = np.zeros((n, n), dtype=int)
    for key, count in raw.items():
        code = int(float(key))
        pre_cls, post_cls = code // 10, code % 10
        if 0 <= pre_cls < n and 0 <= post_cls < n:
            matrix[pre_cls, post_cls] = int(count)
    class_labels = [DW_CLASSES[i] for i in range(n)]
    df = pd.DataFrame(matrix, index=class_labels, columns=class_labels)
    df.index.name = 'Pre → Post'
    return df


def _build_post_dw_serverside(year, geometry, dist_month_img, cross_year_post_months=3):
    """
    Build a post-disturbance DW mode composite via server-side per-image masking.

    Loads the full disturbance-year DW collection once and maps a masking function
    over it so each pixel only retains observations taken after its individual
    disturbance month. Optionally merges in the first N months of yr+1 (all
    observations in those months are post-disturbance by definition).

    This produces a shallower GEE computation graph than the monthly-composite
    approach and is required to avoid compute timeouts in reduceRegions calls
    over many sites simultaneously.
    """
    yr = int(year)

    def _mask_to_post(img):
        return img.updateMask(dist_month_img.lt(img.date().get('month')))

    post_col = (
        ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
        .filterDate(f'{yr}-01-01', f'{yr + 1}-01-01')
        .filterBounds(geometry)
        .select('label')
        .map(_mask_to_post)
    )

    if cross_year_post_months > 0:
        end_date = datetime(yr + 1, cross_year_post_months + 1, 1).strftime('%Y-%m-%d')
        next_year_col = (
            ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
            .filterDate(f'{yr + 1}-01-01', end_date)
            .filterBounds(geometry)
            .select('label')
            .map(lambda img: img.updateMask(dist_month_img.gt(0)))
        )
        post_col = post_col.merge(next_year_col)

    return post_col.reduce(ee.Reducer.mode()).rename('label_mode')


def get_lc_change_all_sites(year, test_sites,
                             anom_lower=30, conf_lower=400,
                             min_post_months=2, min_disturbed_pixels=10,
                             cross_year_post_months=3):
    """
    Run the land-cover-change analysis for every site in a single GEE round-trip.

    Builds dist_mask, dw_pre, and dw_post globally (filtered to the union of all
    site geometries), then uses reduceRegions with a frequencyHistogram reducer to
    pull all per-site histograms in one .getInfo() call. Results are parsed
    client-side into per-site dicts.

    Uses _build_post_dw_serverside (server-side per-image masking) rather than
    the monthly-composite approach used by get_lc_change, to keep the GEE
    computation graph shallow enough for reduceRegions across many sites.

    Result dicts match the tabular outputs of get_lc_change (site_name, year,
    pre_counts, post_counts, transition_df, site_id) but do NOT include GEE image
    objects — use get_lc_change for single-site visualization work.

    Parameters
    ----------
    year : str
        Disturbance year ('2023', '2024', or '2025').
    test_sites : ee.FeatureCollection
        Feature collection containing all test sites (from SiteSelector).
    anom_lower : int
        Minimum VEGANOMMAX threshold for the disturbance mask.
    conf_lower : int
        Minimum VEGDISTCONF threshold for the disturbance mask.
    min_post_months : int
        Minimum months of post-disturbance imagery required per pixel.
    min_disturbed_pixels : int
        Sites whose pre-disturbance histogram totals fewer than this many pixels
        are returned as skipped sentinels.
    cross_year_post_months : int
        Months of yr+1 to append as post-disturbance composites for late-year pixels.

    Returns
    -------
    dict  {site_name: result_dict}
        Keyed by site NAME. Skipped sites have {'skipped': True, 'reason': ...}.
    """
    yr = int(year)
    all_geom = test_sites.geometry()

    dist_mask = build_dist_mask(year, anom_lower, conf_lower)

    dw_pre = (
        build_dw_mode_composite(f'{yr - 1}-01-01', f'{yr - 1}-12-31', all_geom)
        .updateMask(dist_mask)
    )

    dist_month_img = build_dist_month_image(year, dist_mask, cross_year=False)
    dw_post = _build_post_dw_serverside(
        year, all_geom, dist_month_img, cross_year_post_months
    ).updateMask(dist_month_img.lte(12 + cross_year_post_months - min_post_months))

    # Stack pre, post, and encoded transition into one image for a single reduceRegions call.
    # Transition encoding: pre_class * 10 + post_class (values 0-88).
    pre_post = dw_pre.rename('pre').addBands(dw_post.rename('post'))
    encoded = pre_post.expression('b("pre") * 10 + b("post")').rename('transition')
    combined = pre_post.addBands(encoded)

    raw_stats = combined.reduceRegions(
        collection=test_sites,
        reducer=ee.Reducer.frequencyHistogram(),
        scale=30,
        tileScale=4,
    ).getInfo()

    results = {}
    for feature in raw_stats['features']:
        props = feature['properties']
        site_name = props.get('NAME', 'Unknown')
        site_id = props.get('SITE_ID')

        pre_raw = props.get('pre', {})
        post_raw = props.get('post', {})
        trans_raw = props.get('transition', {})

        pre_total = sum(int(v) for v in pre_raw.values())

        if pre_total < min_disturbed_pixels:
            results[site_name] = {
                'site_name': site_name,
                'site_id': site_id,
                'year': year,
                'skipped': True,
                'reason': f'{pre_total} disturbed pixels (threshold: {min_disturbed_pixels})',
            }
            continue

        results[site_name] = {
            'site_name': site_name,
            'site_id': site_id,
            'year': year,
            'pre_counts': _parse_class_hist(pre_raw),
            'post_counts': _parse_class_hist(post_raw),
            'transition_df': _parse_transition_df(trans_raw),
        }

    return results


def get_lc_change_all_sites_sequential(year, test_sites,
                                        anom_lower=30, conf_lower=400,
                                        min_post_months=2, min_disturbed_pixels=10,
                                        cross_year_post_months=3):
    """
    Process every site individually with a tight per-site filterBounds.

    Builds dist_mask and dist_month_img once globally, then loops over each site
    and calls _build_post_dw_serverside with only that site's geometry. Each site
    makes one reduceRegion call on a combined 3-band image (pre, post, transition).

    Slower than the reduceRegions approach (~5-15 min for 30 sites at 10-30s each)
    but never times out, and progress is printed after each site completes.

    Returns the same dict format as get_lc_change_all_sites.
    """
    yr = int(year)

    dist_mask = build_dist_mask(year, anom_lower, conf_lower)
    dist_month_img = build_dist_month_image(year, dist_mask, cross_year=False)

    site_features = test_sites.getInfo()['features']
    total = len(site_features)
    results = {}

    for i, feature in enumerate(site_features, 1):
        props = feature['properties']
        site_id = props['SITE_ID']
        site_name = props.get('NAME', str(site_id))
        geom = ee.Geometry(feature['geometry'])

        print(f'[{i}/{total}] {site_name} ... ', end='', flush=True)

        dw_pre = (
            build_dw_mode_composite(f'{yr - 1}-01-01', f'{yr - 1}-12-31', geom)
            .updateMask(dist_mask)
        )
        dw_post = _build_post_dw_serverside(
            year, geom, dist_month_img, cross_year_post_months
        ).updateMask(dist_month_img.lte(12 + cross_year_post_months - min_post_months))

        pre_post = dw_pre.rename('pre').addBands(dw_post.rename('post'))
        encoded = pre_post.expression('b("pre") * 10 + b("post")').rename('transition')
        combined = pre_post.addBands(encoded)

        site_stats = combined.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=geom, scale=30, maxPixels=1e9,
        ).getInfo()

        pre_raw = site_stats.get('pre', {})
        post_raw = site_stats.get('post', {})
        trans_raw = site_stats.get('transition', {})
        pre_total = sum(int(v) for v in pre_raw.values())

        if pre_total < min_disturbed_pixels:
            results[site_name] = {
                'site_name': site_name,
                'site_id': site_id,
                'year': year,
                'skipped': True,
                'reason': f'{pre_total} disturbed pixels (threshold: {min_disturbed_pixels})',
            }
            print(f'skipped ({pre_total} px)')
            continue

        results[site_name] = {
            'site_name': site_name,
            'site_id': site_id,
            'year': year,
            'pre_counts': _parse_class_hist(pre_raw),
            'post_counts': _parse_class_hist(post_raw),
            'transition_df': _parse_transition_df(trans_raw),
        }
        print(f'done ({pre_total} px)')

    return results


def export_lc_change_all_sites(year, test_sites, description=None, folder='EarthEngine',
                                anom_lower=30, conf_lower=400,
                                min_post_months=2, cross_year_post_months=3):
    """
    Submit an async GEE Export task for multi-site LC change analysis.

    Builds the same combined image as get_lc_change_all_sites but submits the
    reduceRegions result as an Export.table.toDrive task instead of calling
    .getInfo(). Export tasks get a 5-day compute budget vs. the ~5-minute
    .getInfo() timeout, making this the reliable path for the full 30-site run.

    Usage
    -----
    task = export_lc_change_all_sites('2024', test_sites)
    task.start()
    # ... wait for task to complete in GEE task manager ...
    results = load_lc_change_export('/path/to/lc_change_2024.csv', '2024')

    Parameters
    ----------
    year : str
    test_sites : ee.FeatureCollection
    description : str, optional
        GEE task name and output filename. Defaults to 'lc_change_{year}'.
    folder : str
        Google Drive folder to write the CSV into.
    anom_lower, conf_lower, min_post_months, cross_year_post_months : see get_lc_change.

    Returns
    -------
    ee.batch.Task  (call .start() to submit)
    """
    yr = int(year)
    all_geom = test_sites.geometry()

    dist_mask = build_dist_mask(year, anom_lower, conf_lower)
    dw_pre = (
        build_dw_mode_composite(f'{yr - 1}-01-01', f'{yr - 1}-12-31', all_geom)
        .updateMask(dist_mask)
    )
    dist_month_img = build_dist_month_image(year, dist_mask, cross_year=False)
    dw_post = _build_post_dw_serverside(
        year, all_geom, dist_month_img, cross_year_post_months
    ).updateMask(dist_month_img.lte(12 + cross_year_post_months - min_post_months))

    pre_post = dw_pre.rename('pre').addBands(dw_post.rename('post'))
    encoded = pre_post.expression('b("pre") * 10 + b("post")').rename('transition')
    combined = pre_post.addBands(encoded)

    stats = combined.reduceRegions(
        collection=test_sites.select(['NAME', 'SITE_ID']),
        reducer=ee.Reducer.frequencyHistogram(),
        scale=30,
        tileScale=4,
    )

    if description is None:
        description = f'lc_change_{year}'

    return ee.batch.Export.table.toDrive(
        collection=stats,
        description=description,
        folder=folder,
        fileFormat='CSV',
    )


def load_lc_change_export(csv_path, year, min_disturbed_pixels=10):
    """
    Parse a CSV exported by export_lc_change_all_sites into the standard result dict.

    GEE serialises dict-type properties (pre, post, transition histograms) as
    JSON strings in the CSV. This function parses them back and returns the same
    {site_name: result_dict} format as get_lc_change_all_sites.

    Parameters
    ----------
    csv_path : str
        Local path to the CSV downloaded from Google Drive.
    year : str
        The disturbance year used when the export was created.
    min_disturbed_pixels : int
        Sites below this threshold are returned as skipped sentinels.
    """
    def _parse_prop(val):
        if val is None or (isinstance(val, float) and np.isnan(val)) or val == '':
            return {}
        try:
            return json.loads(val)
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            import ast
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass
        # GEE CSV format: {key=value, ...} with unquoted integer keys
        val = val.strip()
        if val.startswith('{') and val.endswith('}'):
            result = {}
            for pair in val[1:-1].split(','):
                pair = pair.strip()
                if '=' in pair:
                    k, v = pair.split('=', 1)
                    result[k.strip()] = float(v.strip())
            return result
        return {}

    df = pd.read_csv(csv_path)
    results = {}

    for _, row in df.iterrows():
        site_name = row.get('NAME', 'Unknown')
        site_id = row.get('SITE_ID')

        pre_raw = _parse_prop(row.get('pre'))
        post_raw = _parse_prop(row.get('post'))
        trans_raw = _parse_prop(row.get('transition'))

        pre_total = sum(int(v) for v in pre_raw.values())

        if pre_total < min_disturbed_pixels:
            results[site_name] = {
                'site_name': site_name,
                'site_id': site_id,
                'year': year,
                'skipped': True,
                'reason': f'{pre_total} disturbed pixels (threshold: {min_disturbed_pixels})',
            }
            continue

        results[site_name] = {
            'site_name': site_name,
            'site_id': site_id,
            'year': year,
            'pre_counts': _parse_class_hist(pre_raw),
            'post_counts': _parse_class_hist(post_raw),
            'transition_df': _parse_transition_df(trans_raw),
        }

    return results


def results_to_summary_df(results):
    """
    Flatten a results dict into a wide DataFrame with one row per site.

    Includes both valid sites and skipped sentinels so the full 30-site picture
    is in one place. Numeric columns are NaN for skipped sites.

    Columns
    -------
    site_name, site_id, year, skipped, skip_reason,
    total_pre_px      — disturbed pixels with valid prior-year DW coverage
    total_post_px     — disturbed pixels with valid post-disturbance coverage
    total_transition_px — pixels where both pre and post are valid (overlap)
    same_class_px     — diagonal of the transition matrix (disturbance within class)
    changed_class_px  — off-diagonal (actual land-cover conversion)
    pct_changed       — changed_class_px / total_transition_px × 100
    top_pre_class     — most common pre-disturbance class
    top_post_class    — most common post-disturbance class
    top_transition    — highest single cell in the matrix, e.g. 'Trees → Built Area'
    """
    rows = []
    for site_name, r in results.items():
        base = {
            'site_name': site_name,
            'site_id': r.get('site_id'),
            'year': r.get('year'),
            'skipped': r.get('skipped', False),
            'skip_reason': r.get('reason'),
        }
        if r.get('skipped'):
            rows.append({**base,
                         'total_pre_px': np.nan, 'total_post_px': np.nan,
                         'total_transition_px': np.nan, 'same_class_px': np.nan,
                         'changed_class_px': np.nan, 'pct_changed': np.nan,
                         'top_pre_class': None, 'top_post_class': None,
                         'top_transition': None})
            continue

        pre = r['pre_counts']
        post = r['post_counts']
        td = r['transition_df']
        total_trans = int(td.values.sum())
        same = int(np.diag(td.values).sum())
        changed = total_trans - same

        top_pre = max(pre, key=pre.get) if pre else None
        top_post = max(post, key=post.get) if post else None

        top_trans = None
        if td.values.max() > 0:
            i, j = np.unravel_index(td.values.argmax(), td.values.shape)
            top_trans = f'{td.index[i]} → {td.columns[j]}'

        rows.append({**base,
                     'total_pre_px': sum(pre.values()),
                     'total_post_px': sum(post.values()),
                     'total_transition_px': total_trans,
                     'same_class_px': same,
                     'changed_class_px': changed,
                     'pct_changed': round(changed / total_trans * 100, 1) if total_trans else np.nan,
                     'top_pre_class': top_pre,
                     'top_post_class': top_post,
                     'top_transition': top_trans})

    return pd.DataFrame(rows)


def results_to_transitions_df(results):
    """
    Flatten a results dict into a long DataFrame with one row per non-zero
    (site, pre_class, post_class) combination.

    Skipped sites are excluded. Zero-count cells are excluded.

    Columns
    -------
    site_name, site_id, year,
    pre_class    — DW class name before disturbance
    post_class   — DW class name after disturbance
    count        — pixel count for this transition at this site
    pct_of_site  — count / site's total_transition_px × 100
    same_class   — True when pre_class == post_class (within-class disturbance)

    Typical queries
    ---------------
    # Top 10 transitions across all sites
    df.sort_values('count', ascending=False).head(10)

    # All deforestation-to-development transitions
    df[(df.pre_class == 'Trees') & (df.post_class == 'Built Area')]

    # Only actual land-cover conversions (no same-class rows)
    df[~df.same_class]

    # Per-site % of pixels that actually changed class
    df.groupby('site_name').apply(
        lambda g: g.loc[~g.same_class, 'count'].sum() / g['count'].sum() * 100
    )
    """
    rows = []
    for site_name, r in results.items():
        if r.get('skipped'):
            continue
        td = r['transition_df']
        total_trans = int(td.values.sum())
        site_id = r.get('site_id')
        year = r['year']
        for pre_cls in td.index:
            for post_cls in td.columns:
                count = int(td.loc[pre_cls, post_cls])
                if count == 0:
                    continue
                rows.append({
                    'site_name': site_name,
                    'site_id': site_id,
                    'year': year,
                    'pre_class': pre_cls,
                    'post_class': post_cls,
                    'count': count,
                    'pct_of_site': round(count / total_trans * 100, 2) if total_trans else np.nan,
                    'same_class': pre_cls == post_cls,
                })

    return pd.DataFrame(rows)


def get_transition_mask(result, pre_class, post_class):
    """
    Return a GEE image masking pixels that transitioned from pre_class to post_class.

    Parameters
    ----------
    result : dict
        Output of get_lc_change.
    pre_class : int or str
        Pre-disturbance class — DW integer (0-8) or name (e.g. 'Built Area').
    post_class : int or str
        Post-disturbance class — DW integer (0-8) or name (e.g. 'Trees').

    Returns
    -------
    ee.Image  (1 = matching transition, masked elsewhere)
    """
    _name_to_id = {v: k for k, v in DW_CLASSES.items()}

    if isinstance(pre_class, str):
        pre_class = _name_to_id[pre_class]
    if isinstance(post_class, str):
        post_class = _name_to_id[post_class]

    return (
        result['dw_pre'].eq(pre_class)
        .And(result['dw_post'].eq(post_class))
        .selfMask()
        .rename('transition_mask')
    )
