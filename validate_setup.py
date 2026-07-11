import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

ok = 0
warn = 0
fail = 0


def check(name: str, passed: bool, detail: str = "", is_warning: bool = False) -> None:
    global ok, warn, fail
    if passed:
        ok += 1
        print(f"[OK]   {name}" + (f" - {detail}" if detail else ""))
    elif is_warning:
        warn += 1
        print(f"[WARN] {name}" + (f" - {detail}" if detail else ""))
    else:
        fail += 1
        print(f"[FAIL] {name}" + (f" - {detail}" if detail else ""))


print("RapidX AI Voice Agent - OpenAI setup validation\n")

check("Python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0])

for pkg_name, import_name in [
    ("livekit", "livekit"),
    ("livekit-agents", "livekit.agents"),
    ("livekit-plugins-openai", "livekit.plugins.openai"),
    ("openai", "openai"),
    ("supabase", "supabase"),
    ("httpx", "httpx"),
    ("dotenv", "dotenv"),
]:
    try:
        __import__(import_name)
        check(pkg_name, True)
    except ImportError as exc:
        check(pkg_name, False, str(exc))

required_env = [
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_TRANSCRIPTION_MODEL",
    "OPENAI_TTS_MODEL",
    "OPENAI_TTS_VOICE",
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
]

for key in required_env:
    value = os.environ.get(key, "")
    check(key, bool(value), "set" if value else "missing")

for key in [
    "OPENAI_BASE_URL",
    "CAL_API_KEY",
    "CAL_EVENT_TYPE_ID",
    "SUPABASE_URL",
    "SUPABASE_KEY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
]:
    check(key, bool(os.environ.get(key, "")), "optional", is_warning=True)


async def test_openai() -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("OPENAI_MODEL", "")
    if not api_key or not model:
        check("OpenAI chat ping", False, "OPENAI_API_KEY or OPENAI_MODEL missing")
        return

    try:
        import openai

        kwargs = {"api_key": api_key}
        base_url = os.environ.get("OPENAI_BASE_URL", "")
        if base_url:
            kwargs["base_url"] = base_url
        client = openai.AsyncOpenAI(**kwargs)
        response = await client.chat.completions.create(
            model=model,
            max_tokens=5,
            temperature=0,
            messages=[{"role": "user", "content": "Say: ok"}],
        )
        check("OpenAI chat ping", True, response.choices[0].message.content.strip())
    except Exception as exc:
        check("OpenAI chat ping", False, str(exc)[:160])


asyncio.run(test_openai())

print(f"\nSummary: {ok} ok, {warn} warnings, {fail} failures")
sys.exit(1 if fail else 0)
