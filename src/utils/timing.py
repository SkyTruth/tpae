"""
Time a pipeline script and record its Earth Engine compute (EECUs) in TIMING_LOG,
one row per run, so speed-up changes can be compared against earlier runs.

The script runs unchanged. Every Earth Engine request it makes is labeled with a
workload tag unique to the run; Cloud Monitoring reports EECU usage per tag, a few
minutes after the requests finish, so EECUs are filled in afterwards with --fill-eecu.

Usage (from the repo root):
    poetry run python src/utils/timing.py src/sites/clean_sites.py
    poetry run python src/utils/timing.py src/cells/get_treatment_cells.py
    poetry run python src/utils/timing.py src/cells/get_control_cells.py
    poetry run python src/utils/timing.py src/covariates/extract_cell_covariates.py --overwrite
    poetry run python src/utils/timing.py src/matching/MDM_match_cells.py
    poetry run python src/utils/timing.py src/absolute_effectiveness/absolute_effectiveness.py
    poetry run python src/utils/timing.py src/relative_effectiveness/relative_effectiveness.py
    poetry run python src/utils/timing.py --fill-eecu   # ~10 minutes after a run
    poetry run python src/utils/timing.py --show

Earth Engine caches results, so re-running an unchanged script can be faster and use
fewer EECUs than the first run. extract_cell_covariates.py skips up-to-date sites, so
time it with --overwrite.
"""

from pathlib import Path
import argparse
import runpy
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

import ee
import pandas as pd

from sites.site_selector import load_site_ids
from utils.variables import PROJECT, GCS_BUCKET, SITE_GROUP

TIMING_LOG = f"gs://{GCS_BUCKET}/timing/timings.csv"
EECU_METRIC = "earthengine.googleapis.com/project/cpu/usage_time"
# Cloud Monitoring reports EECU usage a few minutes after the requests finish
EECU_DELAY_MIN = 10


def git_info():
    """Current commit, branch, and whether there are uncommitted changes."""

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=_SRC.parent, capture_output=True, text=True
        ).stdout.strip()

    return {
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
    }


def read_log(path=TIMING_LOG):
    """Read the timing log, or an empty table if it does not exist yet."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()


def append_run(row, path=TIMING_LOG):
    """Add one run to the timing log."""
    log = pd.concat([read_log(path), pd.DataFrame([row])], ignore_index=True)
    log.to_csv(path, index=False)


def count_sites():
    """Number of sites in the active group, or None before clean_sites.py has run."""
    try:
        return len(load_site_ids())
    except FileNotFoundError:
        return None


def run_script(script, script_args):
    """Run a script as __main__ under a unique workload tag and log its run time."""
    script = Path(script).resolve()
    started = datetime.now(timezone.utc)
    run_tag = f"{script.stem.lower()}-{started:%Y%m%d-%H%M%S}"
    # Default tag, so it survives ee.Initialize and any resetWorkloadTag in the script
    ee.data.setDefaultWorkloadTag(run_tag)

    sys.argv = [str(script), *script_args]
    status = "success"
    elapsed_start, cpu_start = time.perf_counter(), time.process_time()
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            status = f"failed: exit {exc.code}"
            raise
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except BaseException as exc:
        status = f"failed: {type(exc).__name__}"
        raise
    finally:
        row = {
            "run_tag": run_tag,
            "script": script.name,
            "script_args": " ".join(script_args),
            "site_group": SITE_GROUP,
            "n_sites": count_sites(),
            **git_info(),
            "host": socket.gethostname(),
            "started_utc": started.isoformat(timespec="seconds"),
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "elapsed_seconds": round(time.perf_counter() - elapsed_start, 1),
            "cpu_seconds": round(time.process_time() - cpu_start, 1),
            "status": status,
            "eecu_seconds": None,
        }
        print(f"\n{'=' * 70}\nTiming {run_tag}: {status}, {row['elapsed_seconds']} s")
        try:
            append_run(row)
            print(f"Logged to {TIMING_LOG}")
            print(f"Run --fill-eecu in ~{EECU_DELAY_MIN} minutes to add EECUs")
        except Exception as exc:
            print(f"Could not write {TIMING_LOG} ({exc}); run was:\n{row}")


def query_eecu_seconds(monitoring, run_tag, started, finished):
    """Total EECU-seconds Cloud Monitoring reports for a workload tag."""
    now = datetime.now(timezone.utc)
    request = dict(
        name=f"projects/{PROJECT}",
        filter=(
            f'metric.type="{EECU_METRIC}" AND metric.labels.workload_tag="{run_tag}"'
        ),
        interval_startTime=(started - timedelta(minutes=5)).isoformat(),
        interval_endTime=min(now, finished + timedelta(minutes=30)).isoformat(),
    )
    total = 0.0
    while True:
        response = monitoring.projects().timeSeries().list(**request).execute()
        for series in response.get("timeSeries", []):
            for point in series["points"]:
                value = point["value"]
                total += float(value.get("doubleValue", value.get("int64Value", 0)))
        if not response.get("nextPageToken"):
            return total
        request["pageToken"] = response["nextPageToken"]


def fill_eecu(path=TIMING_LOG):
    """Add EECU-seconds to logged runs that finished at least EECU_DELAY_MIN ago."""
    import google.auth
    from googleapiclient.discovery import build

    log = read_log(path)
    if log.empty:
        print(f"No runs logged in {path}")
        return
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/monitoring.read"]
    )
    monitoring = build(
        "monitoring", "v3", credentials=credentials, cache_discovery=False
    )

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=EECU_DELAY_MIN)
    n_filled = 0
    for i, row in log[log["eecu_seconds"].isna()].iterrows():
        started = datetime.fromisoformat(row["started_utc"])
        finished = datetime.fromisoformat(row["finished_utc"])
        if finished > cutoff:
            print(f"{row['run_tag']}: finished < {EECU_DELAY_MIN} min ago, try later")
            continue
        eecu = query_eecu_seconds(monitoring, row["run_tag"], started, finished)
        log.loc[i, "eecu_seconds"] = round(eecu, 1)
        print(f"{row['run_tag']}: {eecu:,.1f} EECU-seconds")
        n_filled += 1

    if n_filled:
        log.to_csv(path, index=False)
        print(f"Updated {n_filled} run(s) in {path}")


def show_log(path=TIMING_LOG):
    """Print the timing log."""
    columns = [
        "run_tag",
        "site_group",
        "git_commit",
        "git_dirty",
        "elapsed_seconds",
        "cpu_seconds",
        "eecu_seconds",
        "status",
    ]
    log = read_log(path)
    print(log.reindex(columns=columns).to_string(index=False) if not log.empty else "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--fill-eecu", action="store_true", help="add EECUs to earlier runs"
    )
    parser.add_argument("--show", action="store_true", help="print the timing log")
    parser.add_argument("script", nargs="?", help="pipeline script to run and time")
    parser.add_argument(
        "script_args", nargs=argparse.REMAINDER, help="arguments for the script"
    )
    args = parser.parse_args()

    if args.fill_eecu:
        fill_eecu()
    elif args.show:
        show_log()
    elif args.script:
        run_script(args.script, args.script_args)
    else:
        parser.print_help()
