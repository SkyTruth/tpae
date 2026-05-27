"""
Global covariate sampling, one tile at a time.

Each tile is exported as one CSV to GCS. Task state is persisted to a JSON
file so the pipeline can resume after crashes or rate-limit pauses.

State file schema:
{
  "tile_id": {
    "task_id": str,         # EE task ID
    "status": str,          # UNSUBMITTED | RUNNING | COMPLETED | FAILED
    "attempts": int,        # number of submission attempts
    "gcs_path": str,        # final GCS URI
    "error": str | None,
    "n_requested": int,     # total samples requested across all strata
  }
}
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import ee


@dataclass
class TileTaskState:
    tile_id: str
    task_id: Optional[str] = None
    status: str = "UNSUBMITTED"
    attempts: int = 0
    gcs_path: Optional[str] = None
    error: Optional[str] = None
    n_requested: int = 0


def build_sample_export_task(
    tile_id: str,
    tile_geom: ee.Geometry,
    tile_allocation: dict[int, int],
    covariates: ee.Image,
    strata_image: ee.Image,
    sample_bands: list[str],
    bucket: str,
    file_prefix: str,
    scale: int,
    projection: ee.Projection,
    seed: int = 42,
) -> tuple[ee.batch.Task, int]:
    """
    Build (but do not start) an EE export task for one tile.

    Uses a two-stage pattern:
      1. stratifiedSample on the strata band to get pixel locations
      2. sampleRegions on the covariate stack to extract values

    Parameters
    ----------
    tile_id : str
        Tile identifier (used in output filename).
    tile_geom : ee.Geometry
        Tile bounding box.
    tile_allocation : dict[int, int]
        {stratum_id: n_points} for this tile. Empty -> no task built.
    covariates : ee.Image
        Multi-band image with covariates + 'protected' band.
    strata_image : ee.Image
        Single-band image with stratum_id values, named 'strata'.
    sample_bands : list[str]
        Bands to extract at sample points (e.g., ['elevation', 'slope', ...]).
    bucket : str
        GCS bucket name (no gs:// prefix).
    file_prefix : str
        GCS path prefix, e.g. 'tpae/psm_samples/'.
    scale, projection : EE sampling parameters.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (task, n_requested) : tuple
        EE Task object (not yet started) and total samples requested.
    """
    class_values = list(tile_allocation.keys())
    class_points = list(tile_allocation.values())
    n_requested = sum(class_points)

    sample_pixels = strata_image.stratifiedSample(
        region=tile_geom,
        scale=scale,
        projection=projection,
        seed=seed,
        numPoints=1,  # ignored when classPoints provided
        classBand="strata",
        classValues=class_values,
        classPoints=class_points,
        dropNulls=True,
        geometries=True,
    )

    samples = covariates.select(sample_bands).sampleRegions(
        collection=sample_pixels,
        scale=scale,
        projection=projection,
        geometries=True,
    )

    # Tag every sample with its tile_id for downstream provenance
    samples = samples.map(lambda f: f.set("tile_id", tile_id))

    task = ee.batch.Export.table.toCloudStorage(
        collection=samples,
        description=f"psm_tile_{tile_id}",
        bucket=bucket,
        fileNamePrefix=f"{file_prefix}{tile_id}",
        fileFormat="CSV",
    )
    return task, n_requested


class TileTaskManager:
    """
    Manages EE export tasks for global tiled sampling.

    Persists state to a JSON file for resume-after-crash semantics.
    Polls EE for status updates. Supports retry on FAILED tasks up to a
    configurable attempt cap.
    """

    TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED"}

    def __init__(self, state_path: str | Path):
        self.state_path = Path(state_path)
        self.state: dict[str, TileTaskState] = self._load()

    def _load(self) -> dict[str, TileTaskState]:
        if not self.state_path.exists():
            return {}
        raw = json.loads(self.state_path.read_text())
        return {k: TileTaskState(**v) for k, v in raw.items()}

    def _save(self) -> None:
        serializable = {k: asdict(v) for k, v in self.state.items()}
        self.state_path.write_text(json.dumps(serializable, indent=2))

    def register(self, tile_id: str, n_requested: int, gcs_path: str) -> TileTaskState:
        """Register a new tile if not already tracked."""
        if tile_id not in self.state:
            self.state[tile_id] = TileTaskState(
                tile_id=tile_id,
                n_requested=n_requested,
                gcs_path=gcs_path,
            )
        return self.state[tile_id]

    def submit(self, tile_id: str, task: ee.batch.Task) -> None:
        """Start an EE task and record the task_id."""
        task.start()
        tile_state = self.state[tile_id]
        tile_state.task_id = task.id
        tile_state.status = "RUNNING"
        tile_state.attempts += 1
        self._save()

    def refresh_status(self, tile_id: str) -> str:
        """Query EE for current task status and update state."""
        tile_state = self.state[tile_id]
        if tile_state.task_id is None:
            return tile_state.status
        if tile_state.status in self.TERMINAL_STATUSES:
            return tile_state.status

        status_dict = ee.data.getOperation(f"projects/earthengine-legacy/operations/{tile_state.task_id}")
        # getOperation returns a structured response; we just want the state
        # Fallback: use ee.batch.Task.status() if getOperation isn't available
        try:
            done = status_dict.get("done", False)
            metadata = status_dict.get("metadata", {})
            ee_state = metadata.get("state", "RUNNING")
            error = status_dict.get("error")
            if error:
                tile_state.status = "FAILED"
                tile_state.error = error.get("message", str(error))
            elif done:
                tile_state.status = "COMPLETED"
            else:
                tile_state.status = ee_state
        except Exception as e:
            # Fall back to legacy status method
            task = ee.batch.Task(tile_state.task_id, None, None, None)
            legacy = task.status()
            tile_state.status = legacy.get("state", "UNKNOWN")
            if tile_state.status == "FAILED":
                tile_state.error = legacy.get("error_message")

        self._save()
        return tile_state.status

    def refresh_all(self) -> dict[str, int]:
        """Refresh status for all non-terminal tasks. Returns status counts."""
        counts: dict[str, int] = {}
        for tile_id, tile_state in self.state.items():
            if tile_state.status not in self.TERMINAL_STATUSES:
                self.refresh_status(tile_id)
            counts[tile_state.status] = counts.get(tile_state.status, 0) + 1
        return counts

    def pending_tiles(self) -> list[str]:
        """Tiles not yet completed (UNSUBMITTED, RUNNING, FAILED, etc.)."""
        return [
            tid
            for tid, st in self.state.items()
            if st.status not in {"COMPLETED"}
        ]

    def failed_tiles(self, max_attempts: int = 3) -> list[str]:
        """Failed tiles eligible for retry (under attempt cap)."""
        return [
            tid
            for tid, st in self.state.items()
            if st.status == "FAILED" and st.attempts < max_attempts
        ]

    def wait_until_done(
        self,
        poll_interval: int = 60,
        max_wait_hours: float = 12,
    ) -> dict[str, int]:
        """
        Block until all tasks reach terminal status or timeout elapses.

        Prints status summary every poll. Returns final status counts.
        """
        deadline = time.time() + max_wait_hours * 3600
        while time.time() < deadline:
            counts = self.refresh_all()
            n_done = counts.get("COMPLETED", 0) + counts.get("FAILED", 0) + counts.get("CANCELLED", 0)
            n_total = len(self.state)
            print(f"[{time.strftime('%H:%M:%S')}] {n_done}/{n_total} done — {counts}")
            if n_done >= n_total:
                return counts
            time.sleep(poll_interval)
        return self.refresh_all()
