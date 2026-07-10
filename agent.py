import os
import json
import logging
import certifi
import pytz
import re
import asyncio
import time
from collections import defaultdict
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Annotated

# Fix for macOS SSL certificate verification
os.environ["SSL_CERT_FILE"] = certifi.where()

# ── Sentry error tracking (#21) ───────────────────────────────────────────────
import sentry_sdk
_sentry_dsn = os.environ.get("SENTRY_DSN", "")
if _sentry_dsn:
    from sentry_sdk.integrations.asyncio import AsyncioIntegration
    sentry_sdk.init(
        dsn=_sentry_dsn,
        traces_sample_rate=0.1,
        integrations=[AsyncioIntegration()],
        environment=os.environ.get("ENVIRONMENT", "production"),
    )

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.getLogger("hpack").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

load_dotenv()
logger = logging.getLogger("outbound-agent")
logging.basicConfig(level=logging.INFO)

from livekit import api
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
    llm,
)
from livekit.plugins import openai, sarvam, groq
import db  # singleton Supabase client — imported here so it's available everywhere

CONFIG_FILE = "config.json"

# ── Rate limiting (#37) ───────────────────────────────────────────────────────
_call_timestamps: dict = defaultdict(list)
RATE_LIMIT_CALLS  = 5
RATE_LIMIT_WINDOW = 3600  # 1 hour

def is_rate_limited(phone: str) -> bool:
    if phone in ("unknown", "demo"):
        return False
    now = time.time()
    _call_timestamps[phone] = [t for t in _call_timestamps[phone] if now - t < RATE_LIMIT_WINDOW]
    if len(_call_timestamps[phone]) >= RATE_LIMIT_CALLS:
        return True
    _call_timestamps[phone].append(now)
    return False


# ── Config loader (#17 partial — per-client path awareness) ───────────────────
def get_live_config(phone_number: str | None = None):
    """Load config — tries per-client file first, then default config.json."""
    config = {}
    paths = []
    if phone_number and phone_number != "unknown":
        clean = phone_number.replace("+", "").replace(" ", "")
        paths.append(f"configs/{clean}.json")
    paths += ["configs/default.json", CONFIG_FILE]

    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    config = json.load(f)
                    logger.info(f"[CONFIG] Loaded: {path}")
                    break
            except Exception as e:
                logger.error(f"[CONFIG] Failed to read {path}: {e}")

    return {
        "agent_instructions":       config.get("agent_instructions", ""),
        "stt_min_endpointing_delay":config.get("stt_min_endpointing_delay", 0.05),
        "llm_model":                config.get("llm_model", "gpt-4o-mini"),
        "llm_provider":             config.get("llm_provider", "openai"),
        "llm_temperature":          float(config.get("llm_temperature", 0.3)),
        "max_completion_tokens":    int(config.get("max_completion_tokens", 80)),
        "tts_voice":                config.get("tts_voice", "kavya"),
        "tts_language":             config.get("tts_language", "hi-IN"),
        "tts_provider":             config.get("tts_provider", "sarvam"),
        "stt_provider":             config.get("stt_provider", "sarvam"),
        "stt_language":             config.get("stt_language", "unknown"),
        "lang_preset":              config.get("lang_preset", "multilingual"),
        "max_turns":                config.get("max_turns", 25),
        **config,
    }


# ── Token counter (#11) ───────────────────────────────────────────────────────
def count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


# ── IST time context ──────────────────────────────────────────────────────────
def get_ist_time_context() -> str:
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    today_str = now.strftime("%A, %B %d, %Y")
    time_str  = now.strftime("%I:%M %p")
    days_lines = []
    for i in range(7):
        day   = now + timedelta(days=i)
        label = "Today" if i == 0 else ("Tomorrow" if i == 1 else day.strftime("%A"))
        days_lines.append(f"  {label}: {day.strftime('%A %d %B %Y')} → ISO {day.strftime('%Y-%m-%d')}")
    days_block = "\n".join(days_lines)
    return (
        f"\n\n[SYSTEM CONTEXT]\n"
        f"Current date & time: {today_str} at {time_str} IST\n"
        f"Resolve ALL relative day references using this table:\n{days_block}\n"
        f"Always use ISO dates when calling save_booking_intent. Appointments in IST (+05:30).]"
    )


# ── Language presets ──────────────────────────────────────────────────────────
LANGUAGE_PRESETS = {
    "hinglish":    {"label": "Hinglish (Hindi+English)", "tts_language": "hi-IN", "tts_voice": "kavya",  "instruction": "Speak in natural Hinglish — mix Hindi and English like educated Indians do. Default to Hindi but use English words when more natural."},
    "hindi":       {"label": "Hindi",                   "tts_language": "hi-IN", "tts_voice": "ritu",   "instruction": "Speak only in pure Hindi. Avoid English words wherever a Hindi equivalent exists."},
    "english":     {"label": "English (India)",         "tts_language": "en-IN", "tts_voice": "dev",    "instruction": "Speak only in Indian English with a warm, professional tone."},
    "tamil":       {"label": "Tamil",                   "tts_language": "ta-IN", "tts_voice": "priya",  "instruction": "Speak only in Tamil. Use standard spoken Tamil for a professional context."},
    "telugu":      {"label": "Telugu",                  "tts_language": "te-IN", "tts_voice": "kavya",  "instruction": "Speak only in Telugu. Use clear, polite spoken Telugu."},
    "gujarati":    {"label": "Gujarati",                "tts_language": "gu-IN", "tts_voice": "rohan",  "instruction": "Speak only in Gujarati. Use polite, professional Gujarati."},
    "bengali":     {"label": "Bengali",                 "tts_language": "bn-IN", "tts_voice": "neha",   "instruction": "Speak only in Bengali (Bangla). Use standard, polite spoken Bengali."},
    "marathi":     {"label": "Marathi",                 "tts_language": "mr-IN", "tts_voice": "shubh",  "instruction": "Speak only in Marathi. Use polite, standard spoken Marathi."},
    "kannada":     {"label": "Kannada",                 "tts_language": "kn-IN", "tts_voice": "rahul",  "instruction": "Speak only in Kannada. Use clear, professional spoken Kannada."},
    "malayalam":   {"label": "Malayalam",               "tts_language": "ml-IN", "tts_voice": "ritu",   "instruction": "Speak only in Malayalam. Use polite, professional spoken Malayalam."},
    "multilingual":{"label": "Multilingual (Auto)",     "tts_language": "hi-IN", "tts_voice": "kavya",  "instruction": "Detect the caller's language from their first message and reply in that SAME language for the entire call. Supported: Hindi, Hinglish, English, Tamil, Telugu, Gujarati, Bengali, Marathi, Kannada, Malayalam. Switch if caller switches."},
}

def get_language_instruction(lang_preset: str) -> str:
    preset = LANGUAGE_PRESETS.get(lang_preset, LANGUAGE_PRESETS["multilingual"])
    return f"\n\n[LANGUAGE DIRECTIVE]\n{preset['instruction']}"


# ── Anti-hallucination rules (injected before every system prompt) ─────────────
# These are NON-NEGOTIABLE and cannot be overridden by config.json or UI.
ANTI_HALLUCINATION_RULES = (
    "[CRITICAL RULES — READ FIRST — ABSOLUTE PRIORITY — NO EXCEPTIONS]\n"
    "1. NEVER invent prices, package names, team members, client names, statistics, "
    "or dates. If you don't know it — DON'T say it.\n"
    "2. NEVER guess appointment availability. ALWAYS call the check_availability tool "
    "before naming any specific time slot.\n"
    "3. If asked something you do NOT know: say exactly — "
    "'Great question — our team will confirm that for you. Want me to schedule a quick call?'\n"
    "4. Every single response: MAX 2 short sentences. One sentence is better.\n"
    "5. Do NOT repeat back what the caller just said to you.\n"
    "6. Do NOT make any promises not explicitly stated in this system prompt.\n"
    "7. Use ONLY information explicitly given in this prompt or told to you by the caller.\n"
    "8. When uncertain about ANY fact — stay silent on it and redirect to booking.\n"
    "9. NEVER hallucinate a booking confirmation. Only confirm after save_booking_intent succeeds.\n"
    "10. If the caller is angry or abusive — calmly offer to transfer to a human.\n"
    "\n"
)


# ── External imports ──────────────────────────────────────────────────────────
from calendar_tools import get_available_slots, create_booking, cancel_booking
from notify import (
    notify_booking_confirmed,
    notify_booking_cancelled,
    notify_call_no_booking,
    notify_agent_error,
)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL CONTEXT — All AI-callable functions
# ══════════════════════════════════════════════════════════════════════════════

class AgentTools(llm.ToolContext):

    def __init__(self, caller_phone: str, caller_name: str = ""):
        super().__init__(tools=[])
        self.caller_phone        = caller_phone
        self.caller_name         = caller_name
        self.booking_intent: dict | None = None
        # Twilio SIP domain takes priority; VoBiz kept for backward compat
        self.sip_domain          = (
            os.getenv("TWILIO_SIP_DOMAIN") or
            os.getenv("VOBIZ_SIP_DOMAIN") or
            ""
        )
        self.ctx_api             = None
        self.room_name           = None
        self._sip_identity       = None

    # ── Tool: Transfer to Human ───────────────────────────────────────────
    @llm.function_tool(description="Transfer this call to a human agent. Use if: caller asks for human, is angry, or query is outside scope.")
    async def transfer_call(
    self,
    reason: Annotated[str, "Reason for transfer"] = "human_requested",
    ) -> str:
        logger.info("[TOOL] transfer_call triggered")
        destination = os.getenv("DEFAULT_TRANSFER_NUMBER", "").strip()
        if not destination:
            logger.warning("[TRANSFER] DEFAULT_TRANSFER_NUMBER not set")
            return "Transfer unavailable — no destination number configured."

        # Build SIP URI: Twilio → sip:number@trunk.pstn.twilio.com
        #               Generic → tel:+number (works with any trunk)
        if not destination.startswith("sip:") and not destination.startswith("tel:"):
            if self.sip_domain:
                # Strip non-digit chars except leading +
                clean = destination.lstrip("tel:").lstrip("sip:").replace(" ", "")
                number_only = "".join(c for c in clean if c.isdigit())
                destination = f"sip:{number_only}@{self.sip_domain}"
            else:
                # tel: URI — works with Twilio and most SIP trunks
                if not destination.startswith("+"):
                    destination = "+" + destination.lstrip("+")
                destination = f"tel:{destination}"

        logger.info(f"[TRANSFER] Routing to: {destination}")
        try:
            if self.ctx_api and self.room_name and self._sip_identity:
                await self.ctx_api.sip.transfer_sip_participant(
                    api.TransferSIPParticipantRequest(
                        room_name=self.room_name,
                        participant_identity=self._sip_identity,
                        transfer_to=destination,
                        play_dialtone=False,
                    )
                )
                logger.info("[TRANSFER] SIP transfer initiated successfully")
                return "Transfer initiated successfully."
            logger.warning("[TRANSFER] Missing ctx_api, room_name, or sip_identity")
            return "Unable to transfer right now — internal error."
        except Exception as e:
            logger.error(f"[TRANSFER] Failed: {e}")
            return "Transfer failed — please try again or call back."

    # ── Tool: End Call ────────────────────────────────────────────────────
    @llm.function_tool(description="End the call. Use ONLY when caller says bye/goodbye or after booking is fully confirmed.")
    async def end_call(
    self,
    reason: Annotated[str, "Reason for ending the call"] = "completed",
    ) -> str:
        logger.info("[TOOL] end_call triggered — hanging up.")
        try:
            if self.ctx_api and self.room_name and self._sip_identity:
                await self.ctx_api.sip.transfer_sip_participant(
                    api.TransferSIPParticipantRequest(
                        room_name=self.room_name,
                        participant_identity=self._sip_identity,
                        transfer_to="tel:+00000000",
                        play_dialtone=False,
                    )
                )
        except Exception as e:
            logger.warning(f"[END-CALL] SIP hangup failed: {e}")
        return "Call ended."

    # ── Tool: Save Booking Intent ─────────────────────────────────────────
    @llm.function_tool(description="Save booking intent after caller confirms appointment. Call this ONCE after you have name, phone, email, date, time.")
    async def save_booking_intent(
        self,
        start_time:  Annotated[str,  "ISO 8601 datetime e.g. '2026-03-01T10:00:00+05:30'"],
        caller_name: Annotated[str,  "Full name of the caller"],
        caller_phone:Annotated[str,  "Phone number of the caller"],
        notes:       Annotated[str,  "Any notes, email, or special requests"] = "",
    ) -> str:
        logger.info(f"[TOOL] save_booking_intent: {caller_name} at {start_time}")
        try:
            self.booking_intent = {
                "start_time":   start_time,
                "caller_name":  caller_name,
                "caller_phone": caller_phone,
                "notes":        notes,
            }
            self.caller_name = caller_name
            return f"Booking intent saved for {caller_name} at {start_time}. I'll confirm after the call."
        except Exception as e:
            logger.error(f"[TOOL] save_booking_intent failed: {e}")
            return "I had trouble saving the booking. Please try again."

    # ── Tool: Check Availability (#13) ────────────────────────────────────
    @llm.function_tool(description="Check available appointment slots for a given date. Call this when user asks about availability.")
    async def check_availability(
        self,
        date: Annotated[str, "Date to check in YYYY-MM-DD format e.g. '2026-03-01'"],
    ) -> str:
        logger.info(f"[TOOL] check_availability: date={date}")
        try:
            # get_available_slots is sync — run in thread to avoid blocking event loop
            import asyncio as _aio
            slots = await _aio.to_thread(get_available_slots, date)
            if not slots:
                return f"No available slots on {date}. Would you like to check another date?"
            # Use 'label' key (set by calendar_tools), fall back to 'time'
            slot_strings = [s.get("label") or s.get("time", str(s))[-8:][:5] for s in slots[:6]]
            return f"Available slots on {date}: {', '.join(slot_strings)} IST."
        except Exception as e:
            logger.error(f"[TOOL] check_availability failed: {e}")
            return "I'm having trouble checking the calendar right now."

    # ── Tool: Business Hours (#31) ────────────────────────────────────────
    @llm.function_tool(description="Check if the business is currently open and what the operating hours are.")
    async def get_business_hours(self) -> str:
        ist  = pytz.timezone("Asia/Kolkata")
        now  = datetime.now(ist)
        hours = {
            0: ("Monday",    "10:00", "19:00"),
            1: ("Tuesday",   "10:00", "19:00"),
            2: ("Wednesday", "10:00", "19:00"),
            3: ("Thursday",  "10:00", "19:00"),
            4: ("Friday",    "10:00", "19:00"),
            5: ("Saturday",  "10:00", "17:00"),
            6: ("Sunday",    None,    None),
        }
        day_name, open_t, close_t = hours[now.weekday()]
        current_time = now.strftime("%H:%M")
        if open_t is None:
            return "We are closed on Sundays. Next opening: Monday 10:00 AM IST."
        if open_t <= current_time <= close_t:
            return f"We are OPEN. Today ({day_name}): {open_t}–{close_t} IST."
        return f"We are CLOSED. Today ({day_name}): {open_t}–{close_t} IST."


# ══════════════════════════════════════════════════════════════════════════════
# AGENT CLASS
# ══════════════════════════════════════════════════════════════════════════════

class OutboundAssistant(Agent):

    def __init__(self, agent_tools: AgentTools, first_line: str = "", live_config: dict | None = None):
        tools = llm.find_function_tools(agent_tools)
        self._first_line  = first_line
        self._live_config = live_config or {}
        live_config_loaded = self._live_config

        base_instructions  = live_config_loaded.get("agent_instructions", "")
        ist_context        = get_ist_time_context()
        lang_preset        = live_config_loaded.get("lang_preset", "multilingual")
        lang_instruction   = get_language_instruction(lang_preset)

        # Anti-hallucination rules ALWAYS prepended — highest priority, cannot be overridden
        final_instructions = (
            ANTI_HALLUCINATION_RULES
            + base_instructions
            + ist_context
            + lang_instruction
        )

        # Token counter (#11)
        token_count = count_tokens(final_instructions)
        logger.info(f"[PROMPT] System prompt: {token_count} tokens")
        if token_count > 700:
            logger.warning(f"[PROMPT] Prompt exceeds 700 tokens — consider trimming for latency")

        super().__init__(instructions=final_instructions, tools=tools)

    async def on_enter(self):
        greeting = self._live_config.get(
            "first_line",
            self._first_line or (
                "Namaste! This is Aryan from RapidX AI — we help businesses automate with AI. "
                "Hmm, may I ask what kind of business you run?"
            )
        )
        await self.session.generate_reply(
            instructions=f"Say exactly this phrase: '{greeting}'"
        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

agent_is_speaking = False

async def entrypoint(ctx: JobContext):
    global agent_is_speaking

    # ── Connect ───────────────────────────────────────────────────────────
    await ctx.connect()
    logger.info(f"[ROOM] Connected: {ctx.room.name}")

    # ── Extract caller info ───────────────────────────────────────────────
    phone_number = None
    caller_name  = ""
    caller_phone = "unknown"

    # Try metadata first (outbound dispatch)
    metadata = ctx.job.metadata or ""
    if metadata:
        try:
            meta = json.loads(metadata)
            phone_number = meta.get("phone_number")
        except Exception:
            pass

    # Extract from SIP participants
    for identity, participant in ctx.room.remote_participants.items():
        # Name from caller ID (#32)
        if participant.name and participant.name not in ("", "Caller", "Unknown"):
            caller_name = participant.name
            logger.info(f"[CALLER-ID] Name from SIP: {caller_name}")
        if not phone_number:
            attr = participant.attributes or {}
            phone_number = attr.get("sip.phoneNumber") or attr.get("phoneNumber")
        if not phone_number and "+" in identity:
            import re as _re
            m = _re.search(r"\+\d{7,15}", identity)
            if m:
                phone_number = m.group()

    caller_phone = phone_number or "unknown"

    # ── Rate limiting (#37) ───────────────────────────────────────────────
    if is_rate_limited(caller_phone):
        logger.warning(f"[RATE-LIMIT] Blocked {caller_phone} — too many calls in 1h")
        return

    # ── Load config ───────────────────────────────────────────────────────
    live_config   = get_live_config(caller_phone)
    delay_setting = live_config.get("stt_min_endpointing_delay", 0.05)
    llm_model     = live_config.get("llm_model", "gpt-4o-mini")
    llm_provider  = live_config.get("llm_provider", "openai")
    tts_voice     = live_config.get("tts_voice", "kavya")
    tts_language  = live_config.get("tts_language", "hi-IN")
    tts_provider  = live_config.get("tts_provider", "sarvam")
    stt_provider  = live_config.get("stt_provider", "sarvam")
    stt_language  = live_config.get("stt_language", "unknown")  # auto-detect (#20)
    max_turns     = live_config.get("max_turns", 25)

    # Override OS env vars from UI config (includes Twilio keys)
    for key in [
        "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET",
        "OPENAI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY",
        "SARVAM_API_KEY",
        "CAL_API_KEY", "CAL_EVENT_TYPE_ID",
        "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
        "SUPABASE_URL", "SUPABASE_KEY",
        "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN",
        "TWILIO_PHONE_NUMBER", "TWILIO_SIP_DOMAIN",
        "DEFAULT_TRANSFER_NUMBER",
        "VOBIZ_SIP_DOMAIN",  # legacy fallback
    ]:
        val = live_config.get(key.lower(), "")
        if val:
            os.environ[key] = val

    # ── Caller memory (#15) ───────────────────────────────────────────────
    async def get_caller_history(phone: str) -> str:
        if phone == "unknown":
            return ""
        try:
            sb = db.get_supabase()
            if not sb:
                return ""
            result = (sb.table("call_logs")
                        .select("summary, created_at")
                        .eq("phone_number", phone)
                        .order("created_at", desc=True)
                        .limit(1)
                        .execute())
            if result.data:
                last = result.data[0]
                return f"\n\n[CALLER HISTORY: Last call {last['created_at'][:10]}. Summary: {last['summary']}]"
        except Exception as e:
            logger.warning(f"[MEMORY] Could not load history: {e}")
        return ""

    caller_history = await get_caller_history(caller_phone)
    if caller_history:
        logger.info(f"[MEMORY] Loaded caller history for {caller_phone}")
        # Append to live_config instructions
        live_config["agent_instructions"] = (live_config.get("agent_instructions","") + caller_history)

    # ── Self-training: inject improvement notes from low-quality past calls ───
    try:
        from db import load_improvement_notes
        improvement_notes = await load_improvement_notes(limit=5)
        if improvement_notes:
            live_config["agent_instructions"] = (
                live_config.get("agent_instructions", "") + improvement_notes
            )
            logger.info("[SELF-TRAIN] Improvement notes injected into system prompt")
    except Exception as _st_err:
        logger.debug(f"[SELF-TRAIN] Could not load improvement notes: {_st_err}")

    # ── Instantiate tools ─────────────────────────────────────────────────
    agent_tools = AgentTools(caller_phone=caller_phone, caller_name=caller_name)
    agent_tools._sip_identity = (
        f"sip_{caller_phone.replace('+','')}" if phone_number else "inbound_caller"
    )
    agent_tools.ctx_api   = ctx.api
    agent_tools.room_name = ctx.room.name

    # ══════════════════════════════════════════════════════════════════════
    # BUILD LLM — Smart provider selection
    # Priority (auto mode): Groq → Cerebras → Gemini → OpenAI
    # All providers use temperature=0.3 for zero hallucination
    # ══════════════════════════════════════════════════════════════════════
    llm_temperature     = float(live_config.get("llm_temperature", 0.3))
    max_completion_toks = int(live_config.get("max_completion_tokens", 80))

    # ── Auto-select best free provider based on available keys ────────────
    if llm_provider == "auto":
        if os.environ.get("GROQ_API_KEY"):
            llm_provider = "groq"
            llm_model    = llm_model or "llama-3.3-70b-versatile"
            logger.info("[LLM] Auto-selected: Groq (fastest free LLM ~40ms)")
        elif os.environ.get("CEREBRAS_API_KEY"):
            llm_provider = "cerebras"
            llm_model    = llm_model or "llama-3.3-70b"
            logger.info("[LLM] Auto-selected: Cerebras (world's fastest ~20ms)")
        elif os.environ.get("GEMINI_API_KEY"):
            llm_provider = "gemini"
            llm_model    = llm_model or "gemini-2.0-flash"
            logger.info("[LLM] Auto-selected: Gemini 2.0 Flash (best Hindi support)")
        else:
            llm_provider = "openai"
            llm_model    = llm_model or "gpt-4o-mini"
            logger.info("[LLM] Auto-selected: OpenAI GPT-4o-mini (fallback)")

    # ── Build the chosen provider ─────────────────────────────────────────
    if llm_provider == "groq":
        # 🥇 GROQ — Free, ~40ms, Llama 3.3 70B quality
        # Free tier: 14,400 req/day | console.groq.com
        _groq_key = os.environ.get("GROQ_API_KEY", "")
        if not _groq_key:
            logger.warning("[LLM] GROQ_API_KEY missing — falling back to OpenAI")
            agent_llm = openai.LLM(
                model="gpt-4o-mini",
                max_completion_tokens=max_completion_toks,
                temperature=llm_temperature,
            )
        else:
            agent_llm = groq.LLM(
                model=llm_model or "llama-3.3-70b-versatile",
                api_key=_groq_key,
                max_completion_tokens=max_completion_toks,
                temperature=llm_temperature,
            )
        logger.info(f"[LLM] Groq | model={llm_model} | temp={llm_temperature} | max_tokens={max_completion_toks}")

    elif llm_provider == "cerebras":
        # ⚡ CEREBRAS — Free, ~20ms (world's fastest!), custom AI chips
        # Free tier available | cloud.cerebras.ai
        _cerebras_key = os.environ.get("CEREBRAS_API_KEY", "")
        if not _cerebras_key:
            logger.warning("[LLM] CEREBRAS_API_KEY missing — falling back to Groq/OpenAI")
            _fallback_key = os.environ.get("GROQ_API_KEY", "")
            if _fallback_key:
                agent_llm = groq.LLM(
                    model="llama-3.3-70b-versatile",
                    api_key=_fallback_key,
                    max_completion_tokens=max_completion_toks,
                    temperature=llm_temperature,
                )
            else:
                agent_llm = openai.LLM(
                    model="gpt-4o-mini",
                    max_completion_tokens=max_completion_toks,
                    temperature=llm_temperature,
                )
        else:
            # Cerebras uses OpenAI-compatible API
            agent_llm = openai.LLM(
                model=llm_model or "llama-3.3-70b",
                base_url="https://api.cerebras.ai/v1",
                api_key=_cerebras_key,
                max_completion_tokens=max_completion_toks,
                temperature=llm_temperature,
            )
        logger.info(f"[LLM] Cerebras | model={llm_model} | temp={llm_temperature} | max_tokens={max_completion_toks}")

    elif llm_provider == "gemini":
        # 🌍 GEMINI — Free 1M tokens/day, best Hindi/multilingual support
        # Free tier | aistudio.google.com
        _gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if not _gemini_key:
            logger.warning("[LLM] GEMINI_API_KEY missing — falling back to Groq/OpenAI")
            _fallback_key = os.environ.get("GROQ_API_KEY", "")
            if _fallback_key:
                agent_llm = groq.LLM(
                    model="llama-3.3-70b-versatile",
                    api_key=_fallback_key,
                    max_completion_tokens=max_completion_toks,
                    temperature=llm_temperature,
                )
            else:
                agent_llm = openai.LLM(
                    model="gpt-4o-mini",
                    max_completion_tokens=max_completion_toks,
                    temperature=llm_temperature,
                )
        else:
            # Gemini OpenAI-compatible endpoint
            agent_llm = openai.LLM(
                model=llm_model or "gemini-2.0-flash",
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=_gemini_key,
                max_completion_tokens=max_completion_toks,
                temperature=llm_temperature,
            )
        logger.info(f"[LLM] Gemini | model={llm_model} | temp={llm_temperature} | max_tokens={max_completion_toks}")

    elif llm_provider == "claude":
        # Anthropic Claude (paid)
        _anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        agent_llm = openai.LLM(
            model=llm_model or "claude-haiku-3-5-latest",
            base_url="https://api.anthropic.com/v1/",
            api_key=_anthropic_key,
            max_completion_tokens=max_completion_toks,
            temperature=llm_temperature,
        )
        logger.info(f"[LLM] Claude | model={llm_model} | temp={llm_temperature} | max_tokens={max_completion_toks}")

    else:
        # OpenAI GPT-4o-mini (default paid fallback)
        agent_llm = openai.LLM(
            model=llm_model or "gpt-4o-mini",
            max_completion_tokens=max_completion_toks,
            temperature=llm_temperature,
        )
        logger.info(f"[LLM] OpenAI | model={llm_model} | temp={llm_temperature} | max_tokens={max_completion_toks}")

    # ── Build STT (#1 16kHz, #20 auto-detect, #9 Deepgram) ──────────────
    if stt_provider == "deepgram":
        try:
            from livekit.plugins import deepgram
            agent_stt = deepgram.STT(
                model="nova-2-general",
                language="multi",        # multilingual mode
                interim_results=False,
            )
            logger.info("[STT] Using Deepgram Nova-2")
        except ImportError:
            logger.warning("[STT] deepgram plugin not installed — falling back to Sarvam")
            agent_stt = sarvam.STT(
                language=stt_language,
                model="saaras:v3",
                mode="translate",
                flush_signal=True,
                sample_rate=16000,
            )
    else:
        agent_stt = sarvam.STT(
            language=stt_language,      # "unknown" = auto-detect (#20)
            model="saaras:v3",
            mode="translate",
            flush_signal=True,
            sample_rate=16000,          # force 16kHz (#1)
        )
        logger.info("[STT] Using Sarvam Saaras v3")

    # ── Build TTS (#2 24kHz, #10 ElevenLabs) ────────────────────────────
    if tts_provider == "elevenlabs":
        try:
            from livekit.plugins import elevenlabs
            _el_voice_id = live_config.get("elevenlabs_voice_id", "21m00Tcm4TlvDq8ikWAM")
            agent_tts = elevenlabs.TTS(
                model="eleven_turbo_v2_5",
                voice_id=_el_voice_id,
            )
            logger.info(f"[TTS] Using ElevenLabs Turbo v2.5 — voice: {_el_voice_id}")
        except ImportError:
            logger.warning("[TTS] elevenlabs plugin not installed — falling back to Sarvam")
            agent_tts = sarvam.TTS(
                target_language_code=tts_language,
                model="bulbul:v3",
                speaker=tts_voice,
                speech_sample_rate=24000,
            )
    else:
        agent_tts = sarvam.TTS(
            target_language_code=tts_language,
            model="bulbul:v3",
            speaker=tts_voice,
            speech_sample_rate=24000,          # force 24kHz (#2)
        )
        logger.info(f"[TTS] Using Sarvam Bulbul v3 — voice: {tts_voice} lang: {tts_language}")

    # ── Sentence chunker — first sentence only for minimum perceived latency ──
    def before_tts_cb(agent_response: str) -> str:
        """Extract only the first sentence to start TTS as fast as possible."""
        text = agent_response.strip()
        if not text:
            return text
        # Handle English . ! ? and Hindi ।  — split on sentence boundaries
        parts = re.split(r'(?<=[।\.!?])\s+', text)
        first = parts[0].strip() if parts else text
        # Safety cap: 250 chars max to avoid TTS timeouts
        if len(first) > 250:
            # Truncate at word boundary
            first = first[:250].rsplit(" ", 1)[0] + "..."
        return first

    # ── Turn counter + auto-close (#29) ──────────────────────────────────
    turn_count    = 0
    interrupt_count = 0  # (#30)

    # ── Build agent ───────────────────────────────────────────────────────
    agent = OutboundAssistant(
        agent_tools=agent_tools,
        first_line=live_config.get("first_line", ""),
        live_config=live_config,
    )

    # ── Build session (#3 noise cancellation attempted) ───────────────────
    try:
        from livekit.agents import noise_cancellation as nc
        _noise_cancel = nc.BVC()
        logger.info("[AUDIO] BVC noise cancellation enabled")
    except Exception:
        _noise_cancel = None
        logger.info("[AUDIO] BVC not available — running without noise cancellation")

    room_input = RoomInputOptions(close_on_disconnect=False)
    if _noise_cancel:
        try:
            room_input = RoomInputOptions(close_on_disconnect=False, noise_cancellation=_noise_cancel)
        except Exception:
            room_input = RoomInputOptions(close_on_disconnect=False)

    session = AgentSession(
        stt=agent_stt,
        llm=agent_llm,
        tts=agent_tts,
        turn_detection="stt",
        min_endpointing_delay=float(delay_setting),  # 0.05s — optimized for low latency
        allow_interruptions=True,
    )

    await session.start(room=ctx.room, agent=agent, room_input_options=room_input)

    # ── TTS pre-warm (#12) ────────────────────────────────────────────────
    try:
        await session.tts.prewarm()
        logger.info("[TTS] Pre-warmed successfully")
    except Exception as e:
        logger.debug(f"[TTS] Pre-warm skipped: {e}")

    logger.info("[AGENT] Session live — waiting for caller audio.")
    # Use timezone-aware datetime so .astimezone() works correctly in shutdown hook
    call_start_time = datetime.now(pytz.timezone("Asia/Kolkata"))

    # ── Recording → Supabase Storage ─────────────────────────────────────
    egress_id = None
    try:
        rec_api = api.LiveKitAPI(
            url=os.environ["LIVEKIT_URL"],
            api_key=os.environ["LIVEKIT_API_KEY"],
            api_secret=os.environ["LIVEKIT_API_SECRET"],
        )
        egress_resp = await rec_api.egress.start_room_composite_egress(
            api.RoomCompositeEgressRequest(
                room_name=ctx.room.name,
                audio_only=True,
                file_outputs=[api.EncodedFileOutput(
                    file_type=api.EncodedFileType.OGG,
                    filepath=f"recordings/{ctx.room.name}.ogg",
                    s3=api.S3Upload(
                        access_key=os.environ["SUPABASE_S3_ACCESS_KEY"],
                        secret=os.environ["SUPABASE_S3_SECRET_KEY"],
                        bucket="call-recordings",
                        region=os.environ.get("SUPABASE_S3_REGION", "ap-south-1"),
                        endpoint=os.environ["SUPABASE_S3_ENDPOINT"],
                        force_path_style=True,
                    )
                )]
            )
        )
        egress_id = egress_resp.egress_id
        await rec_api.aclose()
        logger.info(f"[RECORDING] Started egress: {egress_id}")
    except Exception as e:
        logger.warning(f"[RECORDING] Failed to start recording: {e}")

    # ── Upsert active_calls (#38) ─────────────────────────────────────────
    async def upsert_active_call(status: str):
        try:
            sb = db.get_supabase()
            if sb:
                sb.table("active_calls").upsert({
                    "room_id":     ctx.room.name,
                    "phone":       caller_phone,
                    "caller_name": caller_name,
                    "status":      status,
                    "last_updated": datetime.utcnow().isoformat(),
                }).execute()
        except Exception as e:
            logger.debug(f"[ACTIVE-CALL] {e}")

    await upsert_active_call("active")

    # ── Real-time transcript streaming (#33) ─────────────────────────────
    async def _log_transcript(role: str, content: str):
        try:
            sb = db.get_supabase()
            if sb:
                sb.table("call_transcripts").insert({
                    "call_room_id": ctx.room.name,
                    "phone":        caller_phone,
                    "role":         role,
                    "content":      content,
                }).execute()
        except Exception as e:
            logger.debug(f"[TRANSCRIPT-STREAM] {e}")

    # ── Session event handlers ────────────────────────────────────────────
    @session.on("agent_speech_started")
    def _agent_speech_started(ev):
        global agent_is_speaking
        agent_is_speaking = True

    @session.on("agent_speech_finished")
    def _agent_speech_finished(ev):
        global agent_is_speaking
        agent_is_speaking = False

    # Interrupt logging (#30)
    @session.on("agent_speech_interrupted")
    def _on_interrupted(ev):
        nonlocal interrupt_count
        interrupt_count += 1
        logger.info(f"[INTERRUPT] Agent interrupted. Total: {interrupt_count}")


    FILLER_WORDS = {
        "okay.", "okay", "ok", "uh", "hmm", "hm", "yeah", "yes",
        "no", "um", "ah", "oh", "right", "sure", "fine", "good",
        "haan", "han", "theek", "theek hai", "accha", "ji", "ha",
    }

    @session.on("user_speech_committed")
    def on_user_speech_committed(ev):
        nonlocal turn_count
        global agent_is_speaking

        transcript = ev.user_transcript.strip()
        transcript_lower = transcript.lower().rstrip(".")

        if agent_is_speaking:
            logger.debug(f"[FILTER-ECHO] Dropped: '{transcript}'")
            return
        if not transcript or len(transcript) < 3:
            return
        if transcript_lower in FILLER_WORDS:
            logger.debug(f"[FILTER-FILLER] Dropped: '{transcript}'")
            return

        # Real-time transcript stream
        asyncio.create_task(_log_transcript("user", transcript))

        # Turn counter + auto-close (#29)
        turn_count += 1
        logger.info(f"[TRANSCRIPT] Turn {turn_count}/{max_turns}: '{transcript}'")
        if turn_count >= max_turns:
            logger.info(f"[LIMIT] Reached {max_turns} turns — wrapping up")
            asyncio.create_task(
                session.generate_reply(
                    instructions="Politely wrap up: thank the caller, say they can call back anytime, and say a warm goodbye."
                )
            )

    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant):
        global agent_is_speaking
        logger.info(f"[HANGUP] Participant disconnected: {participant.identity}")
        agent_is_speaking = False
        asyncio.create_task(unified_shutdown_hook(ctx))

    # ══════════════════════════════════════════════════════════════════════
    # POST-CALL SHUTDOWN HOOK
    # ══════════════════════════════════════════════════════════════════════

    async def unified_shutdown_hook(shutdown_ctx: JobContext):
        logger.info("[SHUTDOWN] Sequence started.")

        duration = int((datetime.now() - call_start_time).total_seconds())

        # ── Quality score for self-training ──────────────────────────────────
        def _compute_quality_score(booked: bool, sent: str, dur: int, interrupts: int) -> int:
            """Score 0-10. Used to auto-extract lessons for future calls."""
            score = 5  # baseline
            if booked:            score += 3
            if sent == "positive": score += 2
            elif sent == "frustrated": score -= 2
            elif sent == "negative":   score -= 1
            if dur > 90:          score += 1
            if dur < 15:          score -= 2  # call too short — likely error
            score -= min(2, interrupts // 3)  # heavy interruption penalty
            return max(0, min(10, score))

        # Booking
        booking_status_msg = "No booking"
        if agent_tools.booking_intent:
            from calendar_tools import async_create_booking
            intent = agent_tools.booking_intent
            result = await async_create_booking(
                start_time=intent["start_time"],
                caller_name=intent["caller_name"] or "Unknown Caller",
                caller_phone=intent["caller_phone"],
                notes=intent["notes"],
            )
            if result.get("success"):
                notify_booking_confirmed(
                    caller_name=intent["caller_name"],
                    caller_phone=intent["caller_phone"],
                    booking_time_iso=intent["start_time"],
                    booking_id=result.get("booking_id"),
                    notes=intent["notes"],
                    tts_voice=tts_voice,
                    ai_summary="",
                )
                booking_status_msg = f"Booking Confirmed: {result.get('booking_id')}"
            else:
                booking_status_msg = f"Booking Failed: {result.get('message')}"
        else:
            notify_call_no_booking(
                caller_name=agent_tools.caller_name,
                caller_phone=agent_tools.caller_phone,
                call_summary="Caller did not schedule during this call.",
                tts_voice=tts_voice,
                duration_seconds=duration,
            )

        # Build transcript
        transcript_text = ""
        try:
            messages = agent.chat_ctx.messages
            if callable(messages):
                messages = messages()
            lines = []
            for msg in messages:
                if getattr(msg, "role", None) in ("user", "assistant"):
                    content = getattr(msg, "content", "")
                    if isinstance(content, list):
                        content = " ".join(str(c) for c in content if isinstance(c, str))
                    lines.append(f"[{msg.role.upper()}] {content}")
            transcript_text = "\n".join(lines)
        except Exception as e:
            logger.error(f"[SHUTDOWN] Transcript read failed: {e}")
            transcript_text = "unavailable"

        # Sentiment analysis (#14) — use whichever LLM key is available
        sentiment = "unknown"
        if transcript_text and transcript_text != "unavailable":
            _oai_key = (
                os.environ.get("OPENAI_API_KEY", "") or
                os.environ.get("GROQ_API_KEY", "")   # fallback: use Groq for sentiment too
            )
            if _oai_key:
                try:
                    import openai as _oai
                    # Use Groq if no OpenAI key (cheaper + faster)
                    if not os.environ.get("OPENAI_API_KEY") and os.environ.get("GROQ_API_KEY"):
                        _client = _oai.AsyncOpenAI(
                            api_key=os.environ["GROQ_API_KEY"],
                            base_url="https://api.groq.com/openai/v1",
                        )
                        _sentiment_model = "llama-3.1-8b-instant"
                    else:
                        _client = _oai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
                        _sentiment_model = "gpt-4o-mini"
                    resp = await _client.chat.completions.create(
                        model=_sentiment_model, max_tokens=5, temperature=0.1,
                        messages=[{"role": "user", "content":
                            f"Classify this call transcript as ONE word only: positive, neutral, negative, or frustrated.\n\n{transcript_text[:800]}"}]
                    )
                    raw = resp.choices[0].message.content.strip().lower()
                    # Validate — only accept known values
                    sentiment = raw if raw in ("positive", "neutral", "negative", "frustrated") else "neutral"
                    logger.info(f"[SENTIMENT] {sentiment}")
                except Exception as e:
                    logger.warning(f"[SENTIMENT] Failed: {e}")

        # Cost estimation (#34)
        def estimate_cost(dur: int, chars: int) -> float:
            return round(
                (dur / 60) * 0.002 +
                (dur / 60) * 0.006 +
                (chars / 1000) * 0.003 +
                (chars / 4000) * 0.0001,
                5
            )
        estimated_cost = estimate_cost(duration, len(transcript_text))
        logger.info(f"[COST] Estimated: ${estimated_cost}")

        # Analytics timestamps (#19)
        ist = pytz.timezone("Asia/Kolkata")
        # call_start_time is already timezone-aware (set to IST at session start)
        call_dt = call_start_time if call_start_time.tzinfo else call_start_time.replace(tzinfo=ist)

        # Stop recording
        recording_url = ""
        if egress_id:
            try:
                stop_api = api.LiveKitAPI(
                    url=os.environ["LIVEKIT_URL"],
                    api_key=os.environ["LIVEKIT_API_KEY"],
                    api_secret=os.environ["LIVEKIT_API_SECRET"],
                )
                await stop_api.egress.stop_egress(api.StopEgressRequest(egress_id=egress_id))
                await stop_api.aclose()
                recording_url = (
                    f"{os.environ.get('SUPABASE_URL','')}/storage/v1/object/public/"
                    f"call-recordings/recordings/{ctx.room.name}.ogg"
                )
                logger.info(f"[RECORDING] Stopped. URL: {recording_url}")
            except Exception as e:
                logger.warning(f"[RECORDING] Stop failed: {e}")

        # Update active_calls to completed (#38)
        await upsert_active_call("completed")

        # n8n webhook (#39)
        _n8n_url = os.getenv("N8N_WEBHOOK_URL")
        if _n8n_url:
            try:
                import httpx
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: httpx.post(_n8n_url, json={
                        "event":        "call_completed",
                        "phone":        caller_phone,
                        "caller_name":  agent_tools.caller_name,
                        "duration":     duration,
                        "booked":       bool(agent_tools.booking_intent),
                        "sentiment":    sentiment,
                        "summary":      booking_status_msg,
                        "recording_url":recording_url,
                        "interrupt_count": interrupt_count,
                    }, timeout=5.0)
                )
                logger.info("[N8N] Webhook triggered")
            except Exception as e:
                logger.warning(f"[N8N] Webhook failed: {e}")

        # ── Compute quality score now that we have sentiment & booking result ──
        quality_score = _compute_quality_score(
            booked=bool(agent_tools.booking_intent),
            sent=sentiment,
            dur=duration,
            interrupts=interrupt_count,
        )
        logger.info(f"[QUALITY] Score {quality_score}/10 — booked={bool(agent_tools.booking_intent)} sentiment={sentiment}")

        # ── Auto-generate improvement note for low-quality calls (self-training) ──
        improvement_note = ""
        if quality_score < 5 and transcript_text and transcript_text != "unavailable":
            try:
                import openai as _oai
                _client = _oai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
                _resp = await _client.chat.completions.create(
                    model="gpt-4o-mini", max_tokens=80,
                    temperature=0.3,
                    messages=[{"role": "user", "content": (
                        f"This AI voice call scored {quality_score}/10 (low). "
                        f"Sentiment: {sentiment}. Booked: {bool(agent_tools.booking_intent)}.\n"
                        f"Transcript (first 600 chars):\n{transcript_text[:600]}\n\n"
                        "In one sentence, what should the agent do differently next time?"
                    )}]
                )
                improvement_note = _resp.choices[0].message.content.strip()
                logger.info(f"[SELF-TRAIN] Improvement note: {improvement_note}")
            except Exception as _e:
                logger.debug(f"[SELF-TRAIN] Note generation skipped: {_e}")

        # ── Save to Supabase ─────────────────────────────────────────────────
        from db import save_call_log
        save_call_log(
            phone=caller_phone,
            duration=duration,
            transcript=transcript_text,
            summary=booking_status_msg,
            recording_url=recording_url,
            caller_name=agent_tools.caller_name or "",
            sentiment=sentiment,
            estimated_cost_usd=estimated_cost,
            call_date=call_dt.date().isoformat(),
            call_hour=call_dt.hour,
            call_day_of_week=call_dt.strftime("%A"),
            was_booked=bool(agent_tools.booking_intent),
            interrupt_count=interrupt_count,
            quality_score=quality_score,
            improvement_note=improvement_note,
        )

    ctx.add_shutdown_callback(unified_shutdown_hook)


# ══════════════════════════════════════════════════════════════════════════════
# WORKER ENTRY
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="outbound-caller",
    ))
