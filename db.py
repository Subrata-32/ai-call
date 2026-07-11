import os
import time
import logging
from urllib.parse import urlparse
from supabase import create_client, Client

logger = logging.getLogger("db")

# ─── Columns added by supabase_migration_v2.sql ───────────────────────────────
_ANALYTICS_COLUMNS = {
    "sentiment", "was_booked", "interrupt_count",
    "estimated_cost_usd", "call_date", "call_hour", "call_day_of_week",
    "quality_score", "improvement_note",
}
_BASE_COLUMNS = {"phone_number", "duration_seconds", "transcript", "summary",
                 "recording_url", "caller_name"}

# ─── Singleton Supabase client (one per process, not per call) ─────────────────
# Avoids the overhead of creating a new client on every call.
_supabase_client: Client | None = None
_supabase_url:   str = ""
_supabase_key:   str = ""


def normalize_supabase_url(raw_url: str) -> str:
    """Convert dashboard/project URLs into the proper Supabase project API base."""
    raw_url = (raw_url or "").strip()
    if not raw_url:
        return ""

    if "/dashboard/project/" in raw_url:
        project_ref = raw_url.split("/dashboard/project/", 1)[1].split("/", 1)[0].split("?", 1)[0]
        if project_ref:
            return f"https://{project_ref}.supabase.co"

    parsed = urlparse(raw_url)
    host = parsed.netloc.lower()
    if host.endswith(".storage.supabase.co"):
        project_ref = host.split(".storage.supabase.co", 1)[0]
        return f"https://{project_ref}.supabase.co"

    if host.endswith(".supabase.co") or host == "supabase.co":
        return f"{parsed.scheme or 'https'}://{host}".rstrip("/")

    return raw_url.rstrip("/")


def build_supabase_storage_url(base_url: str, object_path: str) -> str:
    """Build a public storage URL from a Supabase project base URL."""
    base = normalize_supabase_url(base_url)
    if not base:
        return ""
    path = (object_path or "").lstrip("/")
    return f"{base}/storage/v1/object/public/{path}" if path else f"{base}/storage/v1/object/public"


def get_supabase() -> Client | None:
    """Return a singleton Supabase client; re-create only if credentials changed."""
    global _supabase_client, _supabase_url, _supabase_key
    raw_url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    normalized_url = normalize_supabase_url(raw_url)
    if not normalized_url or not key:
        return None
    # Re-create client only when credentials change (e.g. UI config override)
    if _supabase_client and normalized_url == _supabase_url and key == _supabase_key:
        return _supabase_client
    try:
        os.environ["SUPABASE_URL"] = normalized_url
        _supabase_client = create_client(normalized_url, key)
        _supabase_url    = normalized_url
        _supabase_key    = key
        return _supabase_client
    except Exception as e:
        logger.error(f"Failed to init Supabase client: {e}")
        _supabase_client = None
        return None


# ─── Retry helper ─────────────────────────────────────────────────────────────
_MAX_RETRIES  = 3
_RETRY_DELAYS = [1.0, 2.0, 4.0]


def _is_retryable(err_str: str) -> bool:
    transient = ("525", "ssl", "timeout", "connection", "network", "502", "503", "504")
    el = err_str.lower()
    return any(k in el for k in transient)


def _is_schema_error(err_str: str) -> bool:
    return "PGRST204" in err_str or "schema cache" in err_str.lower()


# ─── save_call_log ────────────────────────────────────────────────────────────

def save_call_log(
    phone: str,
    duration: int,
    transcript: str,
    summary: str = "",
    recording_url: str = "",
    caller_name: str = "",
    sentiment: str = "unknown",
    estimated_cost_usd: float | None = None,
    call_date: str | None = None,
    call_hour: int | None = None,
    call_day_of_week: str | None = None,
    was_booked: bool = False,
    interrupt_count: int = 0,
    quality_score: int | None = None,
    improvement_note: str = "",
) -> dict:
    """
    Insert a call log into Supabase.

    Strategy:
    1. Try with all columns (including analytics columns from migration_v2).
    2. If PGRST204 (column not in schema cache — migration not yet run),
       retry with only the base columns so the call is never silently lost.
    3. Retry up to 3× on transient SSL/network errors with exponential backoff.
    """
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        logger.info(f"Supabase not configured. Local log → {phone} {duration}s")
        return {"success": False, "message": "Supabase not configured"}

    supabase = get_supabase()
    if not supabase:
        return {"success": False, "message": "Supabase client failed"}

    # Build full payload
    full_data: dict = {
        "phone_number":     phone,
        "duration_seconds": duration,
        "transcript":       transcript,
        "summary":          summary,
        "sentiment":        sentiment,
        "was_booked":       was_booked,
        "interrupt_count":  interrupt_count,
    }
    if recording_url:                   full_data["recording_url"]      = recording_url
    if caller_name:                     full_data["caller_name"]         = caller_name
    if estimated_cost_usd is not None:  full_data["estimated_cost_usd"] = estimated_cost_usd
    if call_date:                       full_data["call_date"]           = call_date
    if call_hour is not None:           full_data["call_hour"]           = call_hour
    if call_day_of_week:                full_data["call_day_of_week"]    = call_day_of_week
    if quality_score is not None:       full_data["quality_score"]       = quality_score
    if improvement_note:                full_data["improvement_note"]    = improvement_note

    # Base-only payload (fallback if migration not run)
    base_data: dict = {k: v for k, v in full_data.items() if k not in _ANALYTICS_COLUMNS}

    def _try_insert(data: dict, label: str) -> dict:
        for attempt in range(_MAX_RETRIES):
            try:
                res = supabase.table("call_logs").insert(data).execute()
                logger.info(f"Saved call log for {phone} ({label})")
                return {"success": True, "data": res.data}
            except Exception as e:
                err = str(e)
                if _is_schema_error(err):
                    raise RuntimeError("SCHEMA_ERROR:" + err)
                if _is_retryable(err) and attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_DELAYS[attempt]
                    logger.warning(f"Transient error (attempt {attempt+1}), retrying in {delay}s: {err[:80]}")
                    time.sleep(delay)
                    continue
                logger.error(f"Failed to save call log ({label}): {e}")
                return {"success": False, "message": err}
        return {"success": False, "message": "Max retries exceeded"}

    # Attempt 1: full payload
    try:
        return _try_insert(full_data, "full")
    except RuntimeError as e:
        err = str(e)
        if "SCHEMA_ERROR" in err:
            logger.warning(
                "Analytics columns missing (run supabase_migration_v2.sql). "
                "Falling back to base columns for this call log."
            )
            return _try_insert(base_data, "base-fallback")
        raise


# ─── load_improvement_notes ───────────────────────────────────────────────────

async def load_improvement_notes(limit: int = 5) -> str:
    """
    Load improvement notes from recent low-quality calls.
    Returns a formatted string to inject into the system prompt as lessons learned.
    Used for self-training: poor calls generate notes, future calls learn from them.
    """
    supabase = get_supabase()
    if not supabase:
        return ""
    try:
        def _fetch():
            return (
                supabase.table("call_logs")
                .select("improvement_note, quality_score, created_at")
                .not_.is_("improvement_note", "null")
                .neq("improvement_note", "")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
        import asyncio
        res = await asyncio.to_thread(_fetch)
        notes = [r["improvement_note"] for r in (res.data or []) if r.get("improvement_note")]
        if not notes:
            return ""
        notes_block = "\n".join(f"- {n}" for n in notes)
        return (
            f"\n\n[LESSONS LEARNED FROM RECENT CALLS \u2014 APPLY THESE]\n"
            f"{notes_block}\n"
        )
    except Exception as e:
        logger.debug(f"[SELF-TRAIN] Could not load improvement notes: {e}")
        return ""



# ─── fetch_call_logs ──────────────────────────────────────────────────────────

def fetch_call_logs(limit: int = 50) -> list:
    supabase = get_supabase()
    if not supabase:
        return []
    for attempt in range(_MAX_RETRIES):
        try:
            res = (
                supabase.table("call_logs")
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return res.data
        except Exception as e:
            if _is_retryable(str(e)) and attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAYS[attempt])
                continue
            logger.error(f"Failed to fetch call logs: {e}")
            return []
    return []


# ─── fetch_bookings ───────────────────────────────────────────────────────────

def fetch_bookings() -> list:
    supabase = get_supabase()
    if not supabase:
        return []
    try:
        res = (
            supabase.table("call_logs")
            .select("id, phone_number, summary, created_at")
            .ilike("summary", "%Confirmed%")
            .order("created_at", desc=True)
            .limit(200)
            .execute()
        )
        return res.data
    except Exception as e:
        logger.error(f"Failed to fetch bookings: {e}")
        return []


# ─── fetch_stats ──────────────────────────────────────────────────────────────

def fetch_stats() -> dict:
    _empty = {"total_calls": 0, "total_bookings": 0, "avg_duration": 0, "booking_rate": 0}
    supabase = get_supabase()
    if not supabase:
        return _empty
    try:
        rows = (supabase.table("call_logs").select("duration_seconds, summary").execute()).data or []
        total = len(rows)
        bookings  = sum(1 for r in rows if "Confirmed" in r.get("summary", ""))
        durations = [r["duration_seconds"] for r in rows if r.get("duration_seconds")]
        avg_dur   = round(sum(durations) / len(durations)) if durations else 0
        rate      = round((bookings / total) * 100) if total else 0
        return {"total_calls": total, "total_bookings": bookings, "avg_duration": avg_dur, "booking_rate": rate}
    except Exception as e:
        logger.error(f"Failed to fetch stats: {e}")
        return _empty
