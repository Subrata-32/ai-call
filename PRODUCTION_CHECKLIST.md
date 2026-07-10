# ══════════════════════════════════════════════════════════════════════════════
# RapidX AI Voice Agent — Production Checklist
# Complete this BEFORE going live. Run validate_setup.py after each step.
# ══════════════════════════════════════════════════════════════════════════════

## Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 2 — Fill in .env
Copy `.env` and fill in ALL the values:

### 🔴 REQUIRED (agent won't start without these)
| Variable | Where to get |
|---|---|
| `LIVEKIT_URL` | [cloud.livekit.io](https://cloud.livekit.io) → Settings |
| `LIVEKIT_API_KEY` | Same page |
| `LIVEKIT_API_SECRET` | Same page |
| `SARVAM_API_KEY` | [app.sarvam.ai](https://app.sarvam.ai) |

### 🟡 LLM — Pick at least ONE (Groq is free and fastest)
| Variable | Where to get | Speed |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) — Free | ~40ms |
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) — Free 1M/day | ~80ms |
| `CEREBRAS_API_KEY` | [cloud.cerebras.ai](https://cloud.cerebras.ai) — Free | ~20ms |
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) — Paid | ~150ms |

### 📞 Phone / SIP — Twilio
| Variable | Where to get |
|---|---|
| `TWILIO_ACCOUNT_SID` | [console.twilio.com](https://console.twilio.com) |
| `TWILIO_AUTH_TOKEN` | Same page |
| `TWILIO_PHONE_NUMBER` | Buy number at Twilio → Phone Numbers |
| `TWILIO_SIP_DOMAIN` | Elastic SIP Trunking → your trunk → Termination URI |
| `DEFAULT_TRANSFER_NUMBER` | Your human agent number (+91...) |

### 📅 Booking
| Variable | Where to get |
|---|---|
| `CAL_API_KEY` | [app.cal.com/settings/developer/api-keys](https://app.cal.com/settings/developer/api-keys) |
| `CAL_EVENT_TYPE_ID` | Your event URL number |

### 🗄️ Database
| Variable | Where to get |
|---|---|
| `SUPABASE_URL` | [app.supabase.com](https://app.supabase.com) → Settings → API |
| `SUPABASE_KEY` | Same page (anon key) |

### 📲 Notifications (optional)
| Variable | Where to get |
|---|---|
| `TELEGRAM_BOT_TOKEN` | @BotFather on Telegram |
| `TELEGRAM_CHAT_ID` | Send message to bot → `api.telegram.org/bot<TOKEN>/getUpdates` |

## Step 3 — Run Supabase Migrations
In Supabase SQL Editor, run these IN ORDER:
1. `supabase_setup.sql`
2. `supabase_migration_v2.sql`
3. `supabase_migration_v3.sql` ← new (quality score + self-training)

## Step 4 — Configure Twilio SIP Trunk
1. Go to [Twilio Console](https://console.twilio.com) → Elastic SIP Trunking
2. Create a SIP Trunk
3. Set **Origination URI** to your LiveKit SIP endpoint:
   - Format: `sip:your-project.sip.livekit.cloud`
   - Find in: [LiveKit Dashboard](https://cloud.livekit.io) → SIP → Inbound Trunk
4. Set **Termination URI** (for call transfers):
   - Note the domain (e.g., `your-trunk.pstn.twilio.com`)
   - Put it in `.env` as `TWILIO_SIP_DOMAIN`
5. Attach your Twilio phone number to the trunk

## Step 5 — Validate Everything
```bash
python validate_setup.py
```
All checks must show ✅ (errors) or ⚠️ (optional warnings) before proceeding.

## Step 6 — Test Locally
```bash
python agent.py start
```
Then call your Twilio number. You should hear the greeting within 2 seconds.

## Step 7 — Deploy to Production
See `COOLIFY_DEPLOYMENT.md` or `VERCEL_DEPLOYMENT.md` for deployment options.

---

## Twilio → LiveKit SIP Flow
```
Caller dials your Twilio number
    → Twilio routes to LiveKit SIP endpoint (via SIP trunk)
        → LiveKit creates a room
            → agent.py entrypoint() fires
                → AI agent answers the call
```

## config.json Quick Reference
```json
{
  "llm_provider": "groq",              // groq | cerebras | gemini | openai | auto
  "llm_model": "llama-3.3-70b-versatile",
  "llm_temperature": 0.3,              // 0.3 = low hallucination
  "max_completion_tokens": 80,         // keep responses short for voice
  "stt_min_endpointing_delay": 0.05,  // 0.05s = lowest latency
  "tts_voice": "kavya",               // kavya | ritu | dev | priya ...
  "tts_language": "hi-IN",            // hi-IN | en-IN | ta-IN ...
  "lang_preset": "multilingual"        // multilingual | hinglish | hindi | english ...
}
```
