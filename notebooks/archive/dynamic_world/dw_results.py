"""
Result-processing functions for Dynamic World land-cover change analysis.

These functions operate on the Python dicts and DataFrames returned by
dw_analysis.get_lc_change / load_lc_change_export. No GEE session is
required except for get_transition_mask, which returns a GEE image.
"""
import numpy as np
import pandas as pd

import ee

from utils.dist_variables import DW_CLASSES


# ------------------------------------------------------------------
# Internal parsers (used by dw_analysis to convert GEE histogram dicts)
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Summary / flattening functions
# ------------------------------------------------------------------

def results_to_summary_df(results):
    """
    Flatten a results dict into a wide DataFrame with one row per site.

    Includes both valid sites and skipped sentinels so the full site picture
    is in one place. Numeric columns are NaN for skipped sites.

    Parameters
    ----------
    results : dict
        Output of get_lc_change_all_sites or load_lc_change_export.

    Columns
    -------
    site_name, site_id, year, skipped, skip_reason,
    total_pre_px        — disturbed pixels with valid prior-year DW coverage
    total_post_px       — disturbed pixels with valid post-disturbance coverage
    total_transition_px — pixels where both pre and post are valid (overlap)
    same_class_px       — diagonal of the transition matrix (no class change)
    changed_class_px    — off-diagonal (actual land-cover conversion)
    pct_changed         — changed_class_px / total_transition_px × 100
    top_pre_class       — most common pre-disturbance class
    top_post_class      — most common post-disturbance class
    top_transition      — highest single cell in the matrix, e.g. 'Trees → Built Area'
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

    Skipped sites and zero-count cells are excluded.

    Parameters
    ----------
    results : dict
        Output of get_lc_change_all_sites or load_lc_change_export.

    Columns
    -------
    site_name, site_id, year,
    pre_class   — DW class name before disturbance
    post_class  — DW class name after disturbance
    count       — pixel count for this transition at this site
    pct_of_site — count / site's total_transition_px × 100
    same_class  — True when pre_class == post_class (within-class disturbance)

    Typical queries
    ---------------
    # Top 10 transitions across all sites
    df.sort_values('count', ascending=False).head(10)

    # All deforestation-to-development transitions
    df[(df.pre_class == 'Trees') & (df.post_class == 'Built Area')]

    # Only actual land-cover conversions (no same-class rows)
    df[~df.same_class]
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

    Requires an active Earth Engine session and a result dict from get_lc_change
    (which includes 'dw_pre' and 'dw_post' GEE images).

    Parameters
    ----------
    result : dict
        Output of get_lc_change (must contain 'dw_pre' and 'dw_post').
    pre_class : int or str
        Pre-disturbance class — DW integer (0-8) or name (e.g. 'Trees').
    post_class : int or str
        Post-disturbance class — DW integer (0-8) or name (e.g. 'Built Area').

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
