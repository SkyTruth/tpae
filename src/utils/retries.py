"""
Retry Earth Engine work that fails for a reason that passes with time: a failed or
dropped connection, too many requests at once (HTTP 429), or Earth Engine being briefly
unavailable (HTTP 500, 502, 503). Everything else fails on the first attempt, including
requests too heavy to finish: Earth Engine's own errors (e.g. "Computation timed out",
"User memory limit exceeded") and requests with no response within EE_DEADLINE_MS.

Call set_request_limits after ee.Initialize, so the Earth Engine client does not retry
requests itself.
"""

import time

import ee
import googleapiclient.errors
import pandas as pd

from utils.variables import (
    EE_DEADLINE_MS,
    EE_MAX_ATTEMPTS,
    EE_RETRY_WAIT_S,
    EE_MAX_RETRY_WAIT_S,
    EE_RATE_LIMIT_MAX_WAIT_S,
)

# HTTP status of Earth Engine's "too many requests" errors
RATE_LIMITED_STATUS = 429
# HTTP statuses of Earth Engine errors that pass with time: too many requests, internal
# error, bad gateway, service unavailable
TRANSIENT_STATUSES = {RATE_LIMITED_STATUS, 500, 502, 503}


def set_request_limits():
    """Fail Earth Engine requests with no response within EE_DEADLINE_MS, and turn off
    the client's own retries (it retries timeouts too), leaving them to with_retries."""
    ee.data.setDeadline(EE_DEADLINE_MS)
    ee.data.setMaxRetries(0)


def http_status(exc):
    """HTTP status of an Earth Engine error, or None."""
    context = exc.__cause__ or exc.__context__
    if isinstance(context, googleapiclient.errors.HttpError):
        return context.resp.status
    return None


def is_transient(exc):
    """True if exc is a failed or dropped connection, or an Earth Engine error with a
    status in TRANSIENT_STATUSES."""
    return isinstance(exc, ConnectionError) or http_status(exc) in TRANSIENT_STATUSES


def with_retries(
    fn,
    *args,
    label,
    attempts=EE_MAX_ATTEMPTS,
    wait_s=EE_RETRY_WAIT_S,
    max_wait_s=EE_MAX_RETRY_WAIT_S,
    rate_limit_wait_s=EE_RATE_LIMIT_MAX_WAIT_S,
    **kwargs,
):
    """Call fn, retrying transient errors: rate limits until rate_limit_wait_s of
    waiting, and others up to attempts times in all."""
    attempt, waited, n_other = 1, 0, 0
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            rate_limited = http_status(exc) == RATE_LIMITED_STATUS
            n_other += not rate_limited
            if (
                not is_transient(exc)
                or n_other >= attempts
                or (rate_limited and waited >= rate_limit_wait_s)
            ):
                raise
            wait = min(wait_s * 2 ** (attempt - 1), max_wait_s)
            print(f"{label}: attempt {attempt} failed ({exc}); retrying in {wait} s")
            time.sleep(wait)
            attempt, waited = attempt + 1, waited + wait


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
        f"(rate limits are retried for up to {EE_RATE_LIMIT_MAX_WAIT_S // 60} min, "
        f"other transient errors up to {EE_MAX_ATTEMPTS} times)"
    )
    for key, error in errors.items():
        print(f"  {key}: {error}")
