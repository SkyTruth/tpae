"""
Build a global 1 km mask of the cells far enough from protected areas to be control
cells, and save it as an Earth Engine asset (CONTROL_MASK_ASSET_ID), in two batch
tasks: an export to GeoTIFFs in GCS (CONTROL_MASK_GCS_PREFIX), then an ingestion of
those files into the asset.

On the covariate stack's 1 km grid (EE_CRS_METERS at CELL_SIZE_M), a cell is 1 if all of
it is at least CONTROL_INNER_BUFFER_M from protected land, else 0. Protected land is
every PA and OECM in all realms (except UNESCO-MAB Biosphere Reserve PAs), painted on a
Web Mercator grid with their outlines, so PAs too small or thin to contain a pixel center
are still painted. Distances are measured on that grid with a distance transform, where
ground distance is pixel distance times cos(latitude), less the extents of the cell and
of the pixels. Cells more than about 78° from the equator are 0 unless protected land is
close by. Land is not masked.

For control sampling to read in place of painting PAs and OECMs in every request, which
is too slow where they are dense. Run once per WDPA release: the asset records the
WDPA and WDOECM update times it was built from.

    export_control_mask.py            export, then ingest (waits for both)
    export_control_mask.py --no-wait  start the export and exit
    export_control_mask.py --ingest   ingest a finished export
"""

import argparse
import json
import math
from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

import ee
import gcsfs

from covariates.export_covariate_stack import (
    GLOBAL_EXTENT_M,
    stack_gcs_files,
    wait_for_task,
)
from utils.variables import (
    PROJECT,
    GCS_BUCKET,
    EE_CRS_METERS,
    CELL_SIZE_M,
    MAX_PIXELS,
    CONTROL_INNER_BUFFER_M,
    PAS_ASSET_ID,
    OECMS_ASSET_ID,
    MAB_DESIGNATION,
)

CONTROL_MASK_ASSET_ID = f"projects/{PROJECT}/assets/control_mask_1km"
CONTROL_MASK_GCS_PREFIX = f"gs://{GCS_BUCKET}/control_mask/control_mask_1km"
CONTROL_MASK_BAND = "far_from_pas"
# Grid that protected land is painted on and distances are measured on
MERCATOR_CRS = "EPSG:3857"
MERCATOR_SCALE_M = 1000
# Width (pixels) of the painted PA and OECM outlines
OUTLINE_WIDTH_PX = 2
# Reach (pixels) of the distance transform; enough up to about 78° latitude
NEIGHBORHOOD_PX = 64
# Extra distance (m) required, for distance errors
DISTANCE_TOLERANCE_M = 100
# Pixels per side of each export computation tile
MASK_SHARD_SIZE = 256


def mask_features():
    """Every PA and OECM in all realms, except UNESCO-MAB Biosphere Reserve PAs, as in
    get_control_cells.py (server side)."""
    mab_pas = ee.Filter.And(
        ee.Filter.eq("DESIG_ENG", MAB_DESIGNATION), ee.Filter.eq("SITE_TYPE", "PA")
    )
    return (
        ee.FeatureCollection(
            [ee.FeatureCollection(PAS_ASSET_ID), ee.FeatureCollection(OECMS_ASSET_ID)]
        )
        .flatten()
        .filter(mab_pas.Not())
    )


def cell_half_diagonal():
    """Half the ground diagonal (m) of each 1 km EASE-Grid 2.0 cell, from the latitude of
    its center (server side)."""
    # Ground width per projected meter (1 at ±30°); height is its inverse
    k = (
        ee.Image.pixelLonLat()
        .select("latitude")
        .multiply(math.pi / 180)
        .cos()
        .divide(math.cos(math.radians(30)))
    )
    return k.pow(2).add(k.pow(-2)).sqrt().multiply(CELL_SIZE_M / 2)


def build_control_mask(features=None):
    """1 where all of a cell is at least CONTROL_INNER_BUFFER_M from protected land, else
    0 (server side).

    features: the polygons of protected land (default mask_features()).
    """
    features = mask_features() if features is None else features
    mercator = ee.Projection(MERCATOR_CRS).atScale(MERCATOR_SCALE_M)
    cos_lat = ee.Image.pixelLonLat().select("latitude").multiply(math.pi / 180).cos()
    # 1 in pixels with protected land: filled polygons, and outlines for small or thin ones
    protected = (
        ee.Image(0)
        .byte()
        .paint(features, 1)
        .paint(features, 1, OUTLINE_WIDTH_PX)
        .reproject(mercator)
    )
    # Ground distance (m) between pixel centers, to the nearest protected pixel
    distance = (
        protected.fastDistanceTransform(NEIGHBORHOOD_PX)
        .unmask(NEIGHBORHOOD_PX**2)
        .sqrt()
        .multiply(cos_lat.multiply(MERCATOR_SCALE_M))
        .reproject(mercator)
    )
    # Less the cell's half-diagonal and two pixels' half-diagonals
    pixel_half_diagonal = cos_lat.multiply(MERCATOR_SCALE_M * math.sqrt(2) / 2)
    min_distance = distance.subtract(cell_half_diagonal()).subtract(
        pixel_half_diagonal.multiply(2)
    )
    far = min_distance.gte(CONTROL_INNER_BUFFER_M + DISTANCE_TOLERANCE_M)
    return far.unmask(0).toByte().rename(CONTROL_MASK_BAND)


def load_control_mask(asset_id=CONTROL_MASK_ASSET_ID):
    """The exported mask, read on its own 1 km grid so each sample point reads the cell
    it is in."""
    mask = ee.Image(asset_id).select(CONTROL_MASK_BAND)
    return mask.reproject(mask.projection())


def release_path(prefix):
    """GCS path of the WDPA and WDOECM update times of the export at prefix."""
    return f"{prefix}_release.json"


def wdpa_update_times():
    """Update times of the current WDPA and WDOECM releases."""
    return {
        "wdpa_update_time": ee.data.getAsset(PAS_ASSET_ID)["updateTime"],
        "wdoecm_update_time": ee.data.getAsset(OECMS_ASSET_ID)["updateTime"],
    }


def export_to_gcs(prefix=CONTROL_MASK_GCS_PREFIX, extent_m=GLOBAL_EXTENT_M):
    """Start exporting the mask to GeoTIFFs at prefix, with the WDPA and WDOECM update
    times at release_path(prefix); return the task ID.

    extent_m: [xmin, ymin, xmax, ymax] in EE_CRS_METERS.
    """
    # Remove an earlier export, so no stale tiles are left
    fs = gcsfs.GCSFileSystem()
    old_files = stack_gcs_files(fs, prefix)
    if old_files:
        print(f"Removing {len(old_files)} file(s) from an earlier export")
        fs.rm(old_files)
    release = wdpa_update_times()
    with fs.open(release_path(prefix), "w") as f:
        json.dump(release, f, indent=2)

    bucket, file_prefix = prefix.removeprefix("gs://").split("/", 1)
    task = ee.batch.Export.image.toCloudStorage(
        image=build_control_mask(),
        description=Path(file_prefix).name,
        bucket=bucket,
        fileNamePrefix=file_prefix,
        region=ee.Geometry.Rectangle(
            extent_m, proj=ee.Projection(EE_CRS_METERS), geodesic=False
        ),
        crs=EE_CRS_METERS,
        crsTransform=[CELL_SIZE_M, 0, 0, 0, -CELL_SIZE_M, 0],
        maxPixels=MAX_PIXELS,
        shardSize=MASK_SHARD_SIZE,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    print(
        f"Started export task {task.id} to {prefix}*.tif "
        f"(WDPA updated {release['wdpa_update_time']})"
    )
    return task.id


def ingest_to_asset(prefix=CONTROL_MASK_GCS_PREFIX, asset_id=CONTROL_MASK_ASSET_ID):
    """Start ingesting the exported GeoTIFFs at prefix into asset_id, with their WDPA and
    WDOECM update times as properties; return the task ID."""
    fs = gcsfs.GCSFileSystem()
    files = stack_gcs_files(fs, prefix)
    if not files:
        raise FileNotFoundError(f"No exported mask at {prefix}; export it first")
    with fs.open(release_path(prefix)) as f:
        release = json.load(f)
    # The GeoTIFFs' CRS (EPSG:6933) can't be read by Earth Engine, so give it as WKT
    manifest = {
        "name": asset_id,
        "tilesets": [
            {
                "id": "mask",
                "crs": EE_CRS_METERS,
                "sources": [{"uris": [f"gs://{path}"]} for path in files],
            }
        ],
        "bands": [
            {"id": CONTROL_MASK_BAND, "tilesetId": "mask", "tilesetBandIndex": 0}
        ],
        # A coarser pixel is 1 only if every cell in it is
        "pyramidingPolicy": "MIN",
        "properties": release,
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
