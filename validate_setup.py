"""
validate_setup.py — Run this BEFORE starting the agent.

Tests all API keys and dependencies one by one.
Usage:
    python validate_setup.py

All checks must pass (✅) before running:
    python agent.py start
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

load_dotenv()

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

errors   = []
warnings = []


def check(name: str, passed: bool, detail: str = "", is_warning: bool = False):
    icon = PASS if passed else (WARN if is_warning else FAIL)
    msg = f"{icon} {name}"
    if detail:
        msg += f": {detail}"
    print(msg)
    if not passed:
        if is_warning:
            warnings.append(name)
        else:
            errors.append(name)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Python version
# ─────────────────────────────────────────────────────────────────────────────
print("\n━━━ 1. Python Version ━━━")
pv = sys.version_info
check("Python >= 3.10", pv >= (3, 10), f"Python {pv.major}.{pv.minor}.{pv.micro}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Core imports
# ─────────────────────────────────────────────────────────────────────────────
print("\n━━━ 2. Core Package Imports ━━━")

packages = [
    ("livekit",              "livekit"),
    ("livekit-agents",       "livekit.agents"),
    ("livekit-plugins-openai","livekit.plugins.openai"),
    ("livekit-plugins-sarvam","livekit.plugins.sarvam"),
    ("openai",               "openai"),
    ("supabase",             "supabase"),
    ("httpx",                "httpx"),
    ("dotenv",               "dotenv"),
    ("pytz",                 "pytz"),
    ("tiktoken",             "tiktoken"),
    ("requests",             "requests"),
    ("sentry_sdk",           "sentry_sdk"),
]

for pkg_name, import_name in packages:
    try:
        __import__(import_name)
        check(pkg_name, True)
    except ImportError as e:
        check(pkg_name, False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Environment variables
# ─────────────────────────────────────────────────────────────────────────────
print("\n━━━ 3. Required Environment Variables ━━━")

REQUIRED = [
    ("LIVEKIT_URL",        "LiveKit server URL (wss://...)"),
    ("LIVEKIT_API_KEY",    "LiveKit API key"),
    ("LIVEKIT_API_SECRET", "LiveKit API secret"),
    ("SARVAM_API_KEY",     "Sarvam AI key (STT + TTS)"),
]

OPTIONAL = [
    ("GROQ_API_KEY",          "Groq LLM (free, recommended)"),
    ("GEMINI_API_KEY",        "Gemini LLM (free, best Hindi)"),
    ("CEREBRAS_API_KEY",      "Cerebras LLM (free, fastest)"),
    ("OPENAI_API_KEY",        "OpenAI LLM (paid fallback)"),
    ("CAL_API_KEY",           "Cal.com (booking calendar)"),
    ("CAL_EVENT_TYPE_ID",     "Cal.com event type ID"),
    ("SUPABASE_URL",          "Supabase URL (call logs)"),
    ("SUPABASE_KEY",          "Supabase anon key"),
    ("TWILIO_ACCOUNT_SID",    "Twilio account SID (phone)"),
    ("TWILIO_AUTH_TOKEN",     "Twilio auth token"),
    ("TWILIO_PHONE_NUMBER",   "Twilio phone number"),
    ("DEFAULT_TRANSFER_NUMBER","Human agent transfer number"),
    ("TELEGRAM_BOT_TOKEN",    "Telegram notifications"),
    ("TELEGRAM_CHAT_ID",      "Telegram chat ID"),
]

for key, desc in REQUIRED:
    val = os.environ.get(key, "")
    check(key, bool(val), desc if not val else f"{val[:8]}...")

print()
llm_keys = [os.environ.get(k, "") for k in ("GROQ_API_KEY", "CEREBRAS_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY")]
has_llm = any(llm_keys)
check("At least one LLM key set", has_llm,
      "Need GROQ_API_KEY, GEMINI_API_KEY, CEREBRAS_API_KEY, or OPENAI_API_KEY",
      is_warning=False)

print()
for key, desc in OPTIONAL:
    val = os.environ.get(key, "")
    check(key, bool(val), desc if not val else f"{val[:8]}...", is_warning=not bool(val))


# ─────────────────────────────────────────────────────────────────────────────
# 4. LiveKit connection
# ─────────────────────────────────────────────────────────────────────────────
print("\n━━━ 4. LiveKit Connection ━━━")

async def test_livekit():
    url    = os.environ.get("LIVEKIT_URL", "")
    key    = os.environ.get("LIVEKIT_API_KEY", "")
    secret = os.environ.get("LIVEKIT_API_SECRET", "")
    if not (url and key and secret):
        check("LiveKit API ping", False, "Credentials missing — skip")
        return
    try:
        from livekit import api as lk_api
        lk = lk_api.LiveKitAPI(url=url, api_key=key, api_secret=secret)
        rooms = await lk.room.list_rooms(lk_api.ListRoomsRequest())
        await lk.aclose()
        check("LiveKit API ping", True, f"Connected. Active rooms: {len(rooms.rooms)}")
    except Exception as e:
        check("LiveKit API ping", False, str(e)[:120])

asyncio.run(test_livekit())


# ─────────────────────────────────────────────────────────────────────────────
# 5. LLM test (whichever key is available)
# ─────────────────────────────────────────────────────────────────────────────
print("\n━━━ 5. LLM Test ━━━")

async def test_llm():
    import openai as _oa

    # Groq
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        try:
            c = _oa.AsyncOpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
            r = await c.chat.completions.create(
                model="llama-3.1-8b-instant", max_tokens=5, temperature=0.1,
                messages=[{"role": "user", "content": "Say: ok"}]
            )
            check("Groq LLM", True, f"Response: '{r.choices[0].message.content.strip()}'")
        except Exception as e:
            check("Groq LLM", False, str(e)[:120])

    # Gemini
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if gemini_key:
        try:
            c = _oa.AsyncOpenAI(
                api_key=gemini_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            r = await c.chat.completions.create(
                model="gemini-2.0-flash", max_tokens=5, temperature=0.1,
                messages=[{"role": "user", "content": "Say: ok"}]
            )
            check("Gemini LLM", True, f"Response: '{r.choices[0].message.content.strip()}'")
        except Exception as e:
            check("Gemini LLM", False, str(e)[:120])

    # Cerebras
    cerebras_key = os.environ.get("CEREBRAS_API_KEY", "")
    if cerebras_key:
        try:
            c = _oa.AsyncOpenAI(api_key=cerebras_key, base_url="https://api.cerebras.ai/v1")
            r = await c.chat.completions.create(
                model="llama-3.3-70b", max_tokens=5, temperature=0.1,
                messages=[{"role": "user", "content": "Say: ok"}]
            )
            check("Cerebras LLM", True, f"Response: '{r.choices[0].message.content.strip()}'")
        except Exception as e:
            check("Cerebras LLM", False, str(e)[:120])

    # OpenAI
    oai_key = os.environ.get("OPENAI_API_KEY", "")
    if oai_key:
        try:
            c = _oa.AsyncOpenAI(api_key=oai_key)
            r = await c.chat.completions.create(
                model="gpt-4o-mini", max_tokens=5, temperature=0.1,
                messages=[{"role": "user", "content": "Say: ok"}]
            )
            check("OpenAI LLM", True, f"Response: '{r.choices[0].message.content.strip()}'")
        except Exception as e:
            check("OpenAI LLM", False, str(e)[:120])

    if not any([groq_key, gemini_key, cerebras_key, oai_key]):
        check("LLM provider", False, "No LLM API keys found in .env")

asyncio.run(test_llm())


# ─────────────────────────────────────────────────────────────────────────────
# 6. Sarvam STT/TTS test
# ─────────────────────────────────────────────────────────────────────────────
print("\n━━━ 6. Sarvam AI (STT + TTS) ━━━")

async def test_sarvam():
    key = os.environ.get("SARVAM_API_KEY", "")
    if not key:
        check("Sarvam API key", False, "SARVAM_API_KEY not set")
        return
    try:
        import httpx
        resp = await httpx.AsyncClient(timeout=8.0).post(
            "https://api.sarvam.ai/text-to-speech",
            headers={"API-Subscription-Key": key, "Content-Type": "application/json"},
            json={
                "inputs": ["Hello"],
                "target_language_code": "en-IN",
                "speaker": "dev",
                "model": "bulbul:v3",
            }
        )
        if resp.status_code in (200, 201):
            check("Sarvam TTS", True, f"HTTP {resp.status_code} — audio generated")
        elif resp.status_code == 401:
            check("Sarvam TTS", False, "Invalid API key (401)")
        else:
            check("Sarvam TTS", False, f"HTTP {resp.status_code}: {resp.text[:80]}")
    except Exception as e:
        check("Sarvam TTS", False, str(e)[:120])

asyncio.run(test_sarvam())


# ─────────────────────────────────────────────────────────────────────────────
# 7. Supabase connection
# ─────────────────────────────────────────────────────────────────────────────
print("\n━━━ 7. Supabase Database ━━━")

def test_supabase():
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not (url and key):
        check("Supabase connection", False, "SUPABASE_URL or SUPABASE_KEY not set", is_warning=True)
        return
    try:
        from supabase import create_client
        sb = create_client(url, key)
        res = sb.table("call_logs").select("id").limit(1).execute()
        check("Supabase call_logs table", True, f"Connected. Rows sampled: {len(res.data)}")
    except Exception as e:
        err = str(e)
        if "relation" in err.lower() and "does not exist" in err.lower():
            check("Supabase call_logs table", False,
                  "Table missing — run supabase_setup.sql in Supabase SQL editor")
        else:
            check("Supabase connection", False, err[:120], is_warning=True)

test_supabase()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Cal.com booking
# ─────────────────────────────────────────────────────────────────────────────
print("\n━━━ 8. Cal.com Booking ━━━")

def test_calcom():
    key      = os.environ.get("CAL_API_KEY", "")
    event_id = os.environ.get("CAL_EVENT_TYPE_ID", "")
    if not key or not event_id:
        check("Cal.com credentials", False, "CAL_API_KEY or CAL_EVENT_TYPE_ID not set", is_warning=True)
        return
    try:
        import requests as req
        from datetime import date, timedelta
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        resp = req.get(
            "https://api.cal.com/v1/slots",
            params={
                "apiKey": key,
                "eventTypeId": event_id,
                "startTime": f"{tomorrow}T00:00:00.000Z",
                "endTime":   f"{tomorrow}T23:59:59.000Z",
            },
            timeout=5,
        )
        if resp.status_code == 200:
            slots = resp.json().get("data", {}).get("slots", {})
            total = sum(len(v) for v in slots.values())
            check("Cal.com slots API", True, f"Slots for tomorrow: {total}")
        elif resp.status_code == 401:
            check("Cal.com slots API", False, "Invalid API key (401)")
        else:
            check("Cal.com slots API", False, f"HTTP {resp.status_code}: {resp.text[:80]}", is_warning=True)
    except Exception as e:
        check("Cal.com slots API", False, str(e)[:120], is_warning=True)

test_calcom()


# ─────────────────────────────────────────────────────────────────────────────
# 9. Telegram notifications
# ─────────────────────────────────────────────────────────────────────────────
print("\n━━━ 9. Telegram Notifications ━━━")

def test_telegram():
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        check("Telegram", False, "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set", is_warning=True)
        return
    try:
        import requests as req
        resp = req.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": "✅ RapidX AI Voice Agent — validation test passed!"},
            timeout=5,
        )
        if resp.status_code == 200:
            check("Telegram notification", True, "Test message sent!")
        elif resp.status_code == 401:
            check("Telegram notification", False, "Invalid bot token (401)")
        else:
            check("Telegram notification", False, f"HTTP {resp.status_code}: {resp.text[:80]}", is_warning=True)
    except Exception as e:
        check("Telegram notification", False, str(e)[:120], is_warning=True)

test_telegram()


# ─────────────────────────────────────────────────────────────────────────────
# 10. Config file
# ─────────────────────────────────────────────────────────────────────────────
print("\n━━━ 10. Config File ━━━")

def test_config():
    import json
    for path in ("config.json", "configs/default.json"):
        if os.path.exists(path):
            try:
                with open(path) as f:
                    cfg = json.load(f)
                llm_p = cfg.get("llm_provider", "openai")
                llm_m = cfg.get("llm_model", "gpt-4o-mini")
                delay = cfg.get("stt_min_endpointing_delay", 0.2)
                check(f"config.json loaded ({path})", True,
                      f"LLM: {llm_p}/{llm_m} | endpointing: {delay}s")
                if float(delay) > 0.1:
                    check("  Endpointing delay", False,
                          f"Set to {delay}s — too high for low latency! Should be 0.05",
                          is_warning=True)
                return
            except Exception as e:
                check(f"config.json ({path})", False, str(e))
                return
    check("config.json", False, "File not found — create from template")

test_config()


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 50)
if errors:
    print(f"\n{FAIL} FAILED — {len(errors)} error(s) must be fixed before going live:")
    for e in errors:
        print(f"   • {e}")
    print()
    print("Fix the errors above, then run this script again.")
    sys.exit(1)
elif warnings:
    print(f"\n{WARN} READY WITH WARNINGS — {len(warnings)} optional item(s) not configured:")
    for w in warnings:
        print(f"   • {w}")
    print()
    print("Agent can start, but some features will be disabled.")
    print("\n🚀 Run: python agent.py start")
else:
    print(f"\n{PASS} ALL CHECKS PASSED — Agent is production ready!")
    print("\n🚀 Run: python agent.py start")
print("═" * 50 + "\n")
