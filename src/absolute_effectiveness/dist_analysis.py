import calendar
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


def build_monthly_dw_composites(year, geometry, cross_year=True):
    """
    Build monthly DW mode composites; returns dict {position: ee.Image}.

    A fully-masked fallback is merged into each collection before reducing
    so every composite has 1 band named 'label_mode' even when no DW images
    exist for that month and location.

    cross_year=True  → 24 composites: positions 1-12 = prior year,
                        positions 13-24 = disturbance year.
    cross_year=False → 12 composites: positions 1-12 = disturbance year only.
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
                  anom_lower=30, conf_lower=400, min_post_months=2):
    """
    Compare pre vs. post disturbance DynamicWorld land cover at disturbed pixels.

    PRE strategy  — annual composite for the full prior year (yr-1).
    POST strategy — per-pixel monthly composites for the disturbance year.
      Each pixel's post window starts the month after its individual VEGDISTDATE,
      filtered to pixels with >= min_post_months of post-disturbance data.

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
    """
    yr = int(year)
    site = test_sites.filter(ee.Filter.eq("SITE_ID", test_site_id))
    site_name = site.first().get("NAME").getInfo()
    geom = site.geometry()

    dist_mask = build_dist_mask(year, anom_lower, conf_lower)

    # PRE: full prior-year annual composite at disturbed pixels
    dw_pre = (
        build_dw_mode_composite(f'{yr - 1}-01-01', f'{yr - 1}-12-31', geom)
        .updateMask(dist_mask)
    )

    # POST: per-pixel monthly composites (disturbance year only)
    dist_month_img = build_dist_month_image(year, dist_mask, cross_year=False)
    monthly_comps = build_monthly_dw_composites(year, geom, cross_year=False)
    _, dw_post = build_per_pixel_pre_post(monthly_comps, dist_month_img)
    dw_post = dw_post.updateMask(dist_month_img.lte(12 - min_post_months))

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


def plot_lc_change(result):
    """Bar charts of pre/post class distributions + annotated transition heatmap."""
    classes = [DW_CLASSES[i] for i in range(9)]
    site_name = result['site_name']
    year = result['year']
    pre_yr = int(year) - 1

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle(
        f"{site_name}  —  {year} disturbance  |  DynamicWorld land cover change\n"
        f"Pre: {pre_yr} annual composite   Post: per-pixel monthly ({year})",
        fontsize=12,
    )

    pre_vals = [result['pre_counts'].get(c, 0) for c in classes]
    post_vals = [result['post_counts'].get(c, 0) for c in classes]

    for ax, vals, title in zip(
        axes[:2],
        [pre_vals, post_vals],
        [f'Pre-disturbance ({pre_yr} annual)\n(disturbed pixels only)',
         f'Post-disturbance ({year} per-pixel)\n(disturbed pixels only)'],
    ):
        ax.barh(classes, vals, color=DW_PALETTE)
        ax.set_title(title)
        ax.set_xlabel('Pixel count')
        ax.invert_yaxis()

    td = result['transition_df'].copy()

    if td.empty or td.values.sum() == 0:
        axes[2].text(0.5, 0.5, 'No overlapping pre/post pixels',
                     ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Transition matrix')
    else:
        im = axes[2].imshow(td.values, cmap='YlOrRd', aspect='auto')
        axes[2].set_xticks(range(len(td.columns)))
        axes[2].set_yticks(range(len(td.index)))
        axes[2].set_xticklabels(td.columns, rotation=45, ha='right', fontsize=9)
        axes[2].set_yticklabels(td.index, fontsize=9)
        axes[2].set_title('Transition matrix\n(row = pre class, col = post class)')
        axes[2].set_xlabel('Post-disturbance class')
        axes[2].set_ylabel('Pre-disturbance class')
        plt.colorbar(im, ax=axes[2], label='Pixel count', shrink=0.8)

        vmax = td.values.max() if td.values.max() > 0 else 1
        for i in range(len(td.index)):
            for j in range(len(td.columns)):
                val = td.values[i, j]
                if val > 0:
                    color = 'white' if val > vmax * 0.65 else 'black'
                    axes[2].text(j, i, str(val), ha='center', va='center',
                                 fontsize=8, color=color)

    plt.tight_layout()
    plt.show()
    return fig


def visualize_dw_on_map(result, map_obj=None):
    """Add pre/post composites and disturbance mask to a geemap Map."""
    import geemap
    site_name = result['site_name']
    year = result['year']
    pre_yr = int(year) - 1

    if map_obj is None:
        map_obj = geemap.Map()
        map_obj.add_basemap("Esri.WorldImagery")

    map_obj.centerObject(result['geometry'])
    map_obj.addLayer(result['geometry'], {'color': 'red'}, f"{site_name} boundary")
    map_obj.addLayer(result['dist_mask'].selfMask(), {'palette': ['FF0000']},
                     f"Dist mask {year}")
    map_obj.addLayer(result['dw_pre'], DW_VIS,
                     f"DW pre  ({pre_yr} annual, disturbed px)")
    map_obj.addLayer(result['dw_post'], DW_VIS,
                     f"DW post ({year} per-pixel, disturbed px)")
    return map_obj


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
