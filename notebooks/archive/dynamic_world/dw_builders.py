"""
Low-level GEE image-building functions for Dynamic World land-cover analysis.

All functions are stateless and return ee.Image objects. None require an
active reduceRegion call — they build the computation graph only.
"""
import calendar
from datetime import date, datetime, timedelta

import ee

from utils.dist_variables import FOLDERSET, VEGDISTDATE_EPOCH


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


def build_monthly_dw_composites(year, geometry, cross_year=True, post_window=4):
    """
    Build monthly DW mode composites; returns dict {position: ee.Image}.

    A fully-masked fallback is merged into each collection before reducing
    so every composite has 1 band named 'label_mode' even when no DW images
    exist for that month and location.

    cross_year=True  → 24 composites: positions 1-12 = prior year,
                        positions 13-24 = disturbance year.
    cross_year=False → 12 composites: positions 1-12 = disturbance year only.

    post_window > 0  → appends that many months of yr+1 (Jan, Feb, …) as
                        positions immediately after the disturbance-year block,
                        extending the post window for pixels disturbed late in
                        the year.
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

    if post_window > 0:
        base = 12 + month_offset
        for m in range(1, post_window + 1):
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
        _, days = calendar.monthrange(yr, m)
        for day in range(1, days + 1):
            ordinal = (date(yr, m, day) - VEGDISTDATE_EPOCH).days
            from_doys.append(ordinal)
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


def build_per_pixel_pre_post(monthly_composites, dist_month_img, post_window=4):
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
        monthly_composites[k].updateMask(
            dist_month_img.lt(k).And(dist_month_img.gte(k - post_window))
        )
        for k in positions
    ])

    dw_pre = pre_collection.reduce(ee.Reducer.mode()).rename('label_mode')
    dw_post = post_collection.reduce(ee.Reducer.mode()).rename('label_mode')
    return dw_pre, dw_post


def _build_post_dw_serverside(year, geometry, dist_month_img, post_window=4):
    """
    Build a post-disturbance DW mode composite via server-side per-image masking.

    Loads the full disturbance-year DW collection once and maps a masking
    function over it so each pixel only retains observations taken after its
    individual disturbance month. Optionally merges in the first N months of
    yr+1 (all observations in those months are post-disturbance by definition).

    This produces a shallower GEE computation graph than the monthly-composite
    approach and is required to avoid compute timeouts in reduceRegions calls
    over many sites simultaneously.
    """
    yr = int(year)

    def _mask_to_post(img):
        month = ee.Number(img.date().get('month'))
        return img.updateMask(
            dist_month_img.lt(month)
            .And(dist_month_img.gte(month.subtract(post_window)))
        )

    def _mask_next_year(img):
        effective_pos = ee.Number(img.date().get('month')).add(12)
        return img.updateMask(
            dist_month_img.lt(effective_pos)
            .And(dist_month_img.gte(effective_pos.subtract(post_window)))
        )

    post_col = (
        ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
        .filterDate(f'{yr}-01-01', f'{yr + 1}-01-01')
        .filterBounds(geometry)
        .select('label')
        .map(_mask_to_post)
    )

    if post_window > 0:
        end_date = datetime(yr + 1, post_window + 1, 1).strftime('%Y-%m-%d')
        next_year_col = (
            ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
            .filterDate(f'{yr + 1}-01-01', end_date)
            .filterBounds(geometry)
            .select('label')
            .map(_mask_next_year)
        )
        post_col = post_col.merge(next_year_col)

    return post_col.reduce(ee.Reducer.mode()).rename('label_mode')


def build_dw_habitat_raster(year, geometry, anom_lower=30, conf_lower=400, post_window=4):
    """
    Binary habitat raster for a given year, disturbance-aware.

    Undisturbed pixels use the annual DW mode; disturbed pixels use the
    post-disturbance DW mode (falling back to annual mode where no valid
    disturbance date exists). The result is remapped to 1 (habitat:
    trees / grass / flooded veg) and selfMasked, so non-habitat pixels
    are masked out.
    """
    dist_mask = build_dist_mask(year, anom_lower, conf_lower)

    yr = int(year)
    annual_dw = build_dw_mode_composite(
        f'{yr}-01-01', f'{yr + 1}-01-01', geometry
    )

    dist_month_img = build_dist_month_image(year, dist_mask)
    post_dw = _build_post_dw_serverside(year, geometry, dist_month_img, post_window)

    # post_dw only has values at disturbed pixels with a valid disturbance date;
    # annual_dw fills in everything else (including disturbed pixels without a date)
    combined = ee.ImageCollection([post_dw, annual_dw]).mosaic()

    return (
        combined
        .remap([0, 1, 2, 3, 4, 5, 6, 7, 8],
               [0, 1, 1, 1, 0, 0, 0, 0, 0])
        .selfMask()
        .rename('habitat')
    )
