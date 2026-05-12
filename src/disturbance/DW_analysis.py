"""
High-level orchestration for Dynamic World land-cover change analysis.

Typical usage
-------------
Single site (interactive / notebook):
    result = get_lc_change('2024', site_id, test_sites)
    summary = results_to_summary_df({'site': result})

All sites (async export to Google Drive):
    task = export_lc_change_all_sites('2024', test_sites)
    task.start()
    # … wait in GEE task manager, then download the CSV …
    results = load_lc_change_export('lc_change_2024.csv', '2024')

Post-processing:
    from disturbance.dw_results import results_to_summary_df, results_to_transitions_df
    summary_df    = results_to_summary_df(results)
    transitions_df = results_to_transitions_df(results)
"""
import json

import ee
import numpy as np
import pandas as pd

from disturbance.dw_builders import (
    _build_post_dw_serverside,
    build_dist_mask,
    build_dist_month_image,
    build_dw_mode_composite,
    build_monthly_dw_composites,
    build_per_pixel_pre_post,
)
from disturbance.dw_results import _parse_class_hist, _parse_transition_df


def get_lc_change(year, test_site_id, test_sites,
                  anom_lower=30, conf_lower=400,
                  min_post_months=2, min_disturbed_pixels=1, post_window=4):
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
    post_window : int
        Number of months from yr+1 to include as post-disturbance composites.
        Allows Nov/Dec pixels to accumulate enough post data.

    Returns
    -------
    dict with keys: site_name, year, pre_counts, post_counts, transition_df,
                    dw_pre, dw_post, dist_mask, dist_month_img, geometry.
    On skip: dict with keys: site_name, year, skipped=True, reason.
    """
    yr = int(year)
    site = test_sites.filter(ee.Filter.eq("SITE_ID", test_site_id))
    site_name = site.first().get("NAME").getInfo()
    geom = site.geometry()

    dist_mask = build_dist_mask(year, anom_lower, conf_lower)

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

    dw_pre = (
        build_dw_mode_composite(f'{yr - 1}-01-01', f'{yr - 1}-12-31', geom)
        .updateMask(dist_mask)
    )

    dist_month_img = build_dist_month_image(year, dist_mask, cross_year=False)
    monthly_comps = build_monthly_dw_composites(
        year, geom, cross_year=False, post_window=post_window
    )
    _, dw_post = build_per_pixel_pre_post(monthly_comps, dist_month_img, post_window)
    dw_post = dw_post.updateMask(
        dist_month_img.lte(12 + post_window - min_post_months)
    )

    def freq_hist(img):
        raw = img.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=geom, scale=30, maxPixels=1e9,
        ).getInfo().get('label_mode', {})
        return _parse_class_hist(raw)

    pre_hist = freq_hist(dw_pre)
    post_hist = freq_hist(dw_post)

    stacked = dw_pre.rename('pre').addBands(dw_post.rename('post'))
    encoded = stacked.expression('b("pre") * 10 + b("post")').rename('transition')
    trans_hist = encoded.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=geom, scale=30, maxPixels=1e9,
    ).getInfo().get('transition', {})

    return {
        'site_name': site_name,
        'year': year,
        'pre_counts': pre_hist,
        'post_counts': post_hist,
        'transition_df': _parse_transition_df(trans_hist),
        'dw_pre': dw_pre,
        'dw_post': dw_post,
        'dist_mask': dist_mask,
        'dist_month_img': dist_month_img,
        'geometry': geom,
    }


def export_lc_change_all_sites(year, test_sites, description=None, folder='EarthEngine',
                                anom_lower=30, conf_lower=400,
                                min_post_months=2, post_window=4):
    """
    Submit an async GEE Export task for multi-site LC change analysis.

    Builds the same combined image as get_lc_change but submits the
    reduceRegions result as an Export.table.toDrive task instead of calling
    .getInfo(). Export tasks get a 5-day compute budget vs. the ~5-minute
    .getInfo() timeout, making this the reliable path for the full site run.

    Parameters
    ----------
    year : str
    test_sites : ee.FeatureCollection
    description : str, optional
        GEE task name and output filename. Defaults to 'lc_change_{year}'.
    folder : str
        Google Drive folder to write the CSV into.
    anom_lower, conf_lower, min_post_months, post_window : see get_lc_change.

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
        year, all_geom, dist_month_img, post_window
    ).updateMask(dist_month_img.lte(12 + post_window - min_post_months))

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
    {site_name: result_dict} format expected by dw_results functions.

    Parameters
    ----------
    csv_path : str
        Local path to the CSV downloaded from Google Drive.
    year : str
        The disturbance year used when the export was created.
    min_disturbed_pixels : int
        Sites below this threshold are returned as skipped sentinels.

    Returns
    -------
    dict  {site_name: result_dict}
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
