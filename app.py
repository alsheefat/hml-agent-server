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
    if not GEMINI_API_KEY:
        return None, "no API key configured"
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={
                "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": message}]}]
            },
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"], None
        return None, f"HTTP {response.status_code}: {response.text[:300]}"
    except Exception as e:
        return None, f"exception: {e}"


def _call_openai_compatible(url, api_key, model, message):
    if not api_key:
        return None, "no API key configured"
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
            return response.json()["choices"][0]["message"]["content"], None
        return None, f"HTTP {response.status_code}: {response.text[:300]}"
    except Exception as e:
        return None, f"exception: {e}"


def call_groq(message):
    return _call_openai_compatible(
        "https://api.groq.com/openai/v1/chat/completions",
        GROQ_API_KEY, "openai/gpt-oss-120b", message
    )


def call_mistral(message):
    return _call_openai_compatible(
        "https://api.mistral.ai/v1/chat/completions",
        MISTRAL_API_KEY, "mistral-small-latest", message
    )


def call_cerebras(message):
    return _call_openai_compatible(
        "https://api.cerebras.ai/v1/chat/completions",
        CEREBRAS_API_KEY, "gpt-oss-120b", message
    )


def call_kimi(message):
    return _call_openai_compatible(
        "https://api.moonshot.ai/v1/chat/completions",
        MOONSHOT_API_KEY, "kimi-k2.6", message
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
    provider_names = ["Gemini", "Google", "Groq", "Mistral", "Cerebras", "Kimi", "Moonshot", "GPT-OSS", "GPT OSS", "OpenAI", "GPT"]
    for name in sorted(provider_names, key=len, reverse=True):
        text = re.sub(re.escape(name), "HML Agent", text, flags=re.IGNORECASE)
    return text


def ask_engines(prompt):
    """Runs a prompt through the priority engine chain (Gemini -> Groq ->
    Mistral -> Cerebras -> Kimi) and returns (answer, failures_dict).
    Shared by both the plain /chat route and the agent task pipeline."""
    failures = {}
    for name, engine in ENGINE_CHAIN:
        raw_answer, error_detail = engine(prompt)
        answer = clean_answer(raw_answer)
        if answer:
            return answer, failures
        failures[name] = error_detail or "no answer returned"
    return None, failures


# ============================================================
# WEB SEARCH (DuckDuckGo HTML — free, no API key)
#
# DuckDuckGo has no official search API; this uses their public HTML
# results page, which is free and needs no key but is unofficial and
# can rate-limit or change format without notice. If it fails, the
# agent falls back to answering from its own knowledge instead of
# blocking the whole response on search working.
# ============================================================

def web_search(query, max_results=5):
    """Returns a list of {title, snippet, url} dicts, or an empty list
    if search fails for any reason."""
    try:
        response = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (compatible; HMLAgent/1.0)"},
            timeout=10
        )
        if response.status_code != 200:
            return []

        results = []
        # Lightweight parse without a heavy HTML library — DuckDuckGo's
        # HTML results wrap each result in a "result__a" link and a
        # "result__snippet" span; this pulls both with regex.
        titles_urls = re.findall(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            response.text, re.DOTALL
        )
        snippets = re.findall(
            r'class="result__snippet"[^>]*>(.*?)</a>',
            response.text, re.DOTALL
        )

        def strip_tags(html_text):
            return re.sub(r"<[^>]+>", "", html_text).strip()

        for i in range(min(len(titles_urls), max_results)):
            url, title_html = titles_urls[i]
            snippet_html = snippets[i] if i < len(snippets) else ""
            results.append({
                "title": strip_tags(title_html),
                "snippet": strip_tags(snippet_html),
                "url": url
            })

        return results

    except Exception as e:
        print("[WEB SEARCH ERROR]", e)
        return []


def needs_web_search(message):
    """Fast heuristic check for whether a question likely needs current/
    live information, so we only pay the search-latency cost when it's
    actually likely to help. Not perfect, but avoids searching for every
    single message."""
    triggers = [
        "today", "now", "current", "latest", "recent", "this week",
        "this year", "price of", "weather", "news", "who is the",
        "score", "stock", "exchange rate", "election", "released",
        "update", "২০২৬", "আজ", "এখন", "সর্বশেষ"
    ]
    lowered = message.lower()
    return any(t in lowered for t in triggers)


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

    prompt = message

    # If the question looks like it needs current/live info, search
    # first and hand the results to the engine as extra context.
    search_results = []
    if needs_web_search(message):
        search_results = web_search(message)
        if search_results:
            context_lines = []
            for r in search_results:
                context_lines.append(f"- {r['title']}: {r['snippet']} ({r['url']})")
            prompt = (
                f"{message}\n\n"
                "Here are current web search results that may help answer "
                "this — use them if relevant, ignore them if not:\n"
                + "\n".join(context_lines)
            )

    answer, failures = ask_engines(prompt)

    if answer:
        response_data = {
            "answer": answer,
            "engine": "HML Agent",
        }
        if search_results:
            response_data["sources"] = [r["url"] for r in search_results]
        return jsonify(response_data)

    return jsonify({
        "error": "All engines failed. Please try again shortly.",
        "details": failures
    }), 503


@app.route("/agent", methods=["POST"])
def agent_task():
    """Multi-step task execution: breaks a goal into a short plan, then
    works through each step (searching the web when a step needs current
    info), and returns both the plan and a final combined result."""
    data = request.get_json(silent=True) or {}
    goal = (data.get("goal") or data.get("message") or "").strip()

    if not goal:
        return jsonify({"error": "goal is required"}), 400

    if len(goal) > 2000:
        goal = goal[:2000]

    # Step 1: ask the model to break the goal into a short numbered plan.
    planning_prompt = (
        "Break this task into a short numbered plan of at most 4 concrete "
        "steps. Reply with ONLY the numbered list, one step per line, no "
        "extra commentary.\n\nTask: " + goal
    )
    plan_text, plan_failures = ask_engines(planning_prompt)

    if not plan_text:
        return jsonify({
            "error": "Could not create a plan for this task.",
            "details": plan_failures
        }), 503

    steps = []
    for line in plan_text.strip().split("\n"):
        cleaned = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
        if cleaned:
            steps.append(cleaned)

    if not steps:
        steps = [goal]

    # Step 2: work through each step, searching the web for it if it
    # looks like it needs current info, then asking the engine chain.
    step_results = []
    for step in steps:
        step_prompt = step
        sources = []

        if needs_web_search(step):
            results = web_search(step)
            if results:
                sources = [r["url"] for r in results]
                context_lines = [f"- {r['title']}: {r['snippet']}" for r in results]
                step_prompt = (
                    f"{step}\n\nCurrent web search results:\n"
                    + "\n".join(context_lines)
                )

        step_answer, _ = ask_engines(step_prompt)
        step_results.append({
            "step": step,
            "result": step_answer or "Could not complete this step.",
            "sources": sources
        })

    # Step 3: synthesize all step results into one final answer.
    synthesis_prompt = (
        f"The original task was: {goal}\n\n"
        "Here is the work done on each step:\n"
        + "\n\n".join(
            f"Step: {r['step']}\nResult: {r['result']}" for r in step_results
        )
        + "\n\nWrite ONE clear final answer for the user that completes "
        "the original task, using the step results above. Do not mention "
        "'steps' or that this was broken into a plan — just give the "
        "final, complete answer naturally."
    )
    final_answer, final_failures = ask_engines(synthesis_prompt)

    if not final_answer:
        # Fall back to just showing the step results if synthesis fails.
        final_answer = "\n\n".join(r["result"] for r in step_results)

    return jsonify({
        "answer": final_answer,
        "engine": "HML Agent",
        "plan": steps,
        "steps": step_results
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
