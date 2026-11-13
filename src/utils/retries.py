"""
Retry Earth Engine work that fails with transient errors, and report sites that still
failed. Other errors (e.g. missing inputs) fail on the first attempt.
"""

import time

import pandas as pd

from utils.variables import EE_MAX_ATTEMPTS, EE_RETRY_WAIT_S

# Substrings (lower case) of errors that can succeed on a later attempt
TRANSIENT_ERRORS = (
    "too many concurrent aggregations",
    "too many requests",
    "deadline exceeded",
    "timed out",
    "service unavailable",
    "internal error",
    "connection",
)
# Substrings (lower case) of errors that fail the same way on every attempt
PERMANENT_ERRORS = ("computation timed out",)


def is_transient(exc):
    """True if exc is an error that can succeed on a later attempt."""
    message = str(exc).lower()
    if any(error in message for error in PERMANENT_ERRORS):
        return False
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    return any(error in message for error in TRANSIENT_ERRORS)


def with_retries(
    fn, *args, label, attempts=EE_MAX_ATTEMPTS, wait_s=EE_RETRY_WAIT_S, **kwargs
):
    """Call fn, retrying transient errors up to attempts times in all."""
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == attempts or not is_transient(exc):
                raise
            wait = wait_s * 2 ** (attempt - 1)
            print(
                f"{label}: attempt {attempt}/{attempts} failed ({exc}); "
                f"retrying in {wait} s"
            )
            time.sleep(wait)


def has_error(row):
    """True if a results row has an error."""
    return pd.notna(row.get("error"))


def load_succeeded_rows(results_path, site_ids):
    """{site_id: row} of the sites in site_ids that succeeded in an earlier run."""
    try:
        previous = pd.read_csv(results_path)
    except FileNotFoundError:
        print(f"No earlier results at {results_path}; running every site")
        return {}
    previous = previous[previous["site_id"].isin(site_ids)]
    rows = {
        int(row["site_id"]): row
        for row in previous.to_dict("records")
        if not has_error(row)
    }
    print(f"Keeping {len(rows)} site(s) that succeeded in {results_path}")
    return rows


def report_failures(errors, n_total, label="sites"):
    """Print how many of n_total sites (or chunks) still failed after retrying, and why.

    errors: {site_id: error message} of the ones that failed.
    """
    print(
        f"\n{len(errors)} of {n_total} {label} still failed after retrying "
        f"(transient errors are tried up to {EE_MAX_ATTEMPTS} times)"
    )
    for key, error in errors.items():
        print(f"  {key}: {error}")
