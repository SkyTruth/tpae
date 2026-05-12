import calendar
import json
from datetime import datetime

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.dist_variables import DW_CLASSES, DW_PALETTE, DW_VIS, FOLDERSET

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
