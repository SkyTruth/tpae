"""
Build the global 1 km covariate stack and save it as an Earth Engine asset
(COVARIATE_STACK_ASSET_ID), in two batch tasks: an export to GeoTIFFs in GCS
(COVARIATE_STACK_GCS_PREFIX), then an ingestion of those files into the asset.

Each covariate is averaged from its native pixels to the 1 km grid (EE_CRS_METERS at
CELL_SIZE_M) that cells are extracted on. Run once, and again after changing a
covariate; extract_cell_covariates.py then reads the asset.

    export_covariate_stack.py            export, then ingest (waits for both)
    export_covariate_stack.py --no-wait  start the export and exit
    export_covariate_stack.py --ingest   ingest a finished export
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

import ee
import gcsfs

from utils.variables import (
    PROJECT,
    EE_CRS_METERS,
    CELL_SIZE_M,
    MAX_PIXELS,
    COVARIATE_STACK_ASSET_ID,
    COVARIATE_STACK_GCS_PREFIX,
    COVARIATE_STACK_BANDS,
    GLO30_ASSET_ID,
    ATC_ASSET_ID,
    POP_ASSET_ID,
    HGFC_ASSET_ID,
    GAEZ_WHEAT_PATH,
    GAEZ_RICE_PATH,
    GAEZ_MAIZE_PATH,
    GAEZ_SOYBEAN_PATH,
    HUMAN_FOOTPRINT_PATH,
)

# Extent of EASE-Grid 2.0 Global (m), rounded out to whole cells
GLOBAL_EXTENT_M = [-17_368_000, -7_315_000, 17_368_000, 7_315_000]
# Pixels per side of each Earth Engine computation tile
SHARD_SIZE = 32
NODATA = -9999


def build_covariate_stack(ee_crs_1km):
    """Load covariate images and average each from native resolution to 1 km."""
    elevation_ic = ee.ImageCollection(GLO30_ASSET_ID).select("DEM")
    elevation = (
        elevation_ic.mosaic()
        .setDefaultProjection(elevation_ic.first().projection())
        .rename("elevation")
    )
    slope = ee.Terrain.slope(elevation).rename("slope")
    treecover2000 = ee.Image(HGFC_ASSET_ID).select("treecover2000")
    travel_time = ee.Image(ATC_ASSET_ID).select("accessibility").rename("travel_time")
    log_pop_density = (
        ee.Image(POP_ASSET_ID)
        .select("population_count")
        .add(1)  # handles zeros for log transform
        .log()
        .rename("log_pop_density")
    )
    human_footprint = ee.Image.loadGeoTIFF(HUMAN_FOOTPRINT_PATH).rename(
        "human_footprint"
    )

    wheat = ee.Image.loadGeoTIFF(GAEZ_WHEAT_PATH).rename("wheat")
    rice = ee.Image.loadGeoTIFF(GAEZ_RICE_PATH).rename("rice")
    maize = ee.Image.loadGeoTIFF(GAEZ_MAIZE_PATH).rename("maize")
    soybean = ee.Image.loadGeoTIFF(GAEZ_SOYBEAN_PATH).rename("soybean")

    ag_suitability = (
        wheat.addBands([rice, maize, soybean])
        .reduce(ee.Reducer.max())
        .rename("ag_suitability")
    )

    # Average native pixels within each 1 km cell
    def resample(img):
        return img.reduceResolution(
            reducer=ee.Reducer.mean(), maxPixels=65536
        ).reproject(ee_crs_1km)

    # Share of each cell that is land (1 = land, 2 = water, 0 = no data); not a covariate, just a flag
    land_frac = ee.Image(HGFC_ASSET_ID).select("datamask").eq(1).rename("land_frac")

    images = [
        elevation,
        slope,
        treecover2000,
        travel_time,
        log_pop_density,
        human_footprint,
        ag_suitability,
        land_frac,
    ]
    stack = ee.Image.cat([resample(img) for img in images])
    return stack.select(COVARIATE_STACK_BANDS).toFloat()


def load_covariate_stack(asset_id=COVARIATE_STACK_ASSET_ID):
    """The exported stack, with COVARIATE_STACK_BANDS, on its 1 km grid."""
    return ee.Image(asset_id).select(COVARIATE_STACK_BANDS)


def covariate_stack_updated(asset_id=COVARIATE_STACK_ASSET_ID):
    """When the exported stack was last written."""
    try:
        update_time = ee.data.getAsset(asset_id)["updateTime"]
    except ee.EEException as exc:
        raise FileNotFoundError(
            f"No covariate stack at {asset_id}; run export_covariate_stack.py first"
        ) from exc
    return datetime.fromisoformat(update_time)


def stack_gcs_files(fs, prefix=COVARIATE_STACK_GCS_PREFIX):
    """GCS paths of the exported GeoTIFF(s); large exports are split into tiles."""
    fs.invalidate_cache()
    return sorted(fs.glob(prefix.removeprefix("gs://") + "*.tif"))


def wait_for_task(task_id, what):
    """Wait for a batch task to finish; raise if it did not complete."""
    while (status := ee.data.getTaskStatus(task_id)[0])["state"] in (
        "UNSUBMITTED",
        "READY",
        "RUNNING",
    ):
        time.sleep(60)
        print(f"  {time.strftime('%H:%M')} {what}: {status['state']}")
    if status["state"] != "COMPLETED":
        raise RuntimeError(f"{what} {status['state']}: {status.get('error_message')}")
    eecu_hours = status.get("batch_eecu_usage_seconds", 0) / 3600
    print(f"{what} completed ({eecu_hours:,.1f} EECU-hours)")


def export_to_gcs(prefix=COVARIATE_STACK_GCS_PREFIX, extent_m=GLOBAL_EXTENT_M):
    """Start exporting the stack to GeoTIFFs at prefix; return the task ID.

    extent_m: [xmin, ymin, xmax, ymax] in EE_CRS_METERS.
    """
    ee_crs_1km = ee.Projection(EE_CRS_METERS).atScale(CELL_SIZE_M)

    # Remove an earlier export, so no stale tiles are left
    fs = gcsfs.GCSFileSystem()
    old_files = stack_gcs_files(fs, prefix)
    if old_files:
        print(f"Removing {len(old_files)} file(s) from an earlier export")
        fs.rm(old_files)

    bucket, file_prefix = prefix.removeprefix("gs://").split("/", 1)
    task = ee.batch.Export.image.toCloudStorage(
        image=build_covariate_stack(ee_crs_1km),
        description=f"covariate_stack_{Path(file_prefix).name}",
        bucket=bucket,
        fileNamePrefix=file_prefix,
        region=ee.Geometry.Rectangle(
            extent_m, proj=ee.Projection(EE_CRS_METERS), geodesic=False
        ),
        crs=EE_CRS_METERS,
        crsTransform=[CELL_SIZE_M, 0, 0, 0, -CELL_SIZE_M, 0],
        maxPixels=MAX_PIXELS,
        shardSize=SHARD_SIZE,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True, "noData": NODATA},
    )
    task.start()
    print(f"Started export task {task.id} to {prefix}*.tif")
    return task.id


def ingest_to_asset(
    prefix=COVARIATE_STACK_GCS_PREFIX, asset_id=COVARIATE_STACK_ASSET_ID
):
    """Start ingesting the exported GeoTIFFs at prefix into asset_id; return the task ID."""
    files = stack_gcs_files(gcsfs.GCSFileSystem(), prefix)
    if not files:
        raise FileNotFoundError(f"No exported stack at {prefix}; export it first")
    # The GeoTIFFs' CRS (EPSG:6933) can't be read by Earth Engine, so give it as WKT
    manifest = {
        "name": asset_id,
        "tilesets": [
            {
                "id": "stack",
                "crs": EE_CRS_METERS,
                "sources": [{"uris": [f"gs://{path}"]} for path in files],
            }
        ],
        "bands": [
            {"id": band, "tilesetId": "stack", "tilesetBandIndex": i}
            for i, band in enumerate(COVARIATE_STACK_BANDS)
        ],
        "missingData": {"values": [NODATA]},
        "pyramidingPolicy": "MEAN",
    }
    request_id = ee.data.newTaskId()[0]
    task_id = ee.data.startIngestion(request_id, manifest, allow_overwrite=True)["id"]
    print(f"Started ingesting {len(files)} file(s) into {asset_id} (task {task_id})")
    return task_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="start the export and exit; run --ingest once it has finished",
    )
    parser.add_argument(
        "--ingest", action="store_true", help="ingest a finished export into the asset"
    )
    args = parser.parse_args()
    ee.Initialize(project=PROJECT)

    if not args.ingest:
        task_id = export_to_gcs()
        if args.no_wait:
            print(
                "It runs on Earth Engine; check progress in the Code Editor Tasks tab "
                f"or with: poetry run earthengine --project {PROJECT} task info "
                f"{task_id}\nWhen it has finished, run this script with --ingest"
            )
            sys.exit()
        wait_for_task(task_id, "Export")
    wait_for_task(ingest_to_asset(), "Ingestion")
