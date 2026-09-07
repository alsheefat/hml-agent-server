import os
import re
import time
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# ============================================================
# FREE-ONLY AI ENGINES
# Public-facing server: only genuinely free-tier engines are used
# here, since usage scales with however many people use the app.
# Paid engines (GPT, Grok, Perplexity, aimlapi) stay in the private
# Telegram bot only, never here.
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY", "")

SYSTEM_PROMPT = (
    "You are HML Agent, a free AI assistant made by Shahrij Al Sheefat Heemel. "
    "Understand Bangla, Banglish, and English, and reply in whichever the user "
    "used. Be direct, useful, and match answer length to the question's need — "
    "short questions get short answers."
)

REQUEST_TIMEOUT = 12


def call_gemini(message):
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={
                "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": message}]}]
            },
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        print(f"[GEMINI] HTTP {response.status_code} {response.text[:200]}")
        return None
    except Exception as e:
        print("[GEMINI ERROR]", e)
        return None


def _call_openai_compatible(url, api_key, model, message):
    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message}
                ]
            },
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        print(f"[{url}] HTTP {response.status_code} {response.text[:200]}")
        return None
    except Exception as e:
        print(f"[{url} ERROR]", e)
        return None


def call_groq(message):
    return _call_openai_compatible(
        "https://api.groq.com/openai/v1/chat/completions",
        GROQ_API_KEY, "llama-3.3-70b-versatile", message
    )


def call_mistral(message):
    return _call_openai_compatible(
        "https://api.mistral.ai/v1/chat/completions",
        MISTRAL_API_KEY, "mistral-small-latest", message
    )


def call_cerebras(message):
    return _call_openai_compatible(
        "https://api.cerebras.ai/v1/chat/completions",
        CEREBRAS_API_KEY, "llama-3.3-70b", message
    )


def call_kimi(message):
    return _call_openai_compatible(
        "https://api.moonshot.ai/v1/chat/completions",
        MOONSHOT_API_KEY, "moonshot-v1-8k", message
    )


# Priority order: Gemini -> Groq -> Mistral -> Cerebras -> Kimi.
# Strict sequential fallback — only try the next engine if the
# previous one genuinely failed, same pattern as the Telegram bot.
ENGINE_CHAIN = [
    ("Gemini", call_gemini),
    ("Groq", call_groq),
    ("Mistral", call_mistral),
    ("Cerebras", call_cerebras),
    ("Kimi", call_kimi),
]


def clean_answer(text):
    if not text:
        return None
    text = text.strip()
    if not text:
        return None
    # Strip any leaked provider name so the agent always presents as
    # HML Agent, regardless of which underlying engine answered.
    provider_names = ["Gemini", "Google", "Groq", "Mistral", "Cerebras", "Kimi", "Moonshot"]
    for name in sorted(provider_names, key=len, reverse=True):
        text = re.sub(re.escape(name), "HML Agent", text, flags=re.IGNORECASE)
    return text


@app.route("/", methods=["GET"])
def health_check():
    # Used by UptimeRobot to keep the free Render service awake.
    return jsonify({"status": "ok", "service": "HML Agent backend"})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "message is required"}), 400

    if len(message) > 4000:
        message = message[:4000]

    for name, engine in ENGINE_CHAIN:
        started = time.perf_counter()
        raw_answer = engine(message)
        elapsed = time.perf_counter() - started

        answer = clean_answer(raw_answer)
        if answer:
            print(f"[SUCCESS] {name} {elapsed:.2f}s")
            return jsonify({
                "answer": answer,
                "engine": "HML Agent",
                "time": round(elapsed, 2)
            })

    return jsonify({"error": "All engines failed. Please try again shortly."}), 503


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
