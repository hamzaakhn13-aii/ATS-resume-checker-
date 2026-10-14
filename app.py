"""
AI Resume ATS Checker
---------------------
Upload a resume (PDF or DOCX), optionally paste a job description, and get:
  * an ATS score (0-100) with a section-wise breakdown
  * strengths, problems and concrete improvements
  * missing keywords and rewrite suggestions

AI model: Google Gemini Flash (via the `google-genai` SDK).
"""

import io
import json
import os
import re
import time

import streamlit as st
from docx import Document
from pypdf import PdfReader

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DEFAULT_MODEL = "gemini-3.8-flash"      # change here, or set GEMINI_MODEL in secrets/env
FALLBACK_MODELS = ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
                   "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]  # tried in order if the main model fails
RETRY_DELAYS = [2, 5]                     # seconds to wait between retries on temporary overload (503/500/504)
MAX_FILE_MB = 5
MIN_TEXT_CHARS = 150                    # below this we assume a scanned/empty PDF
MAX_RESUME_CHARS = 30000                # safety cap on what we send to the model
MAX_JD_CHARS = 8000

SCORE_CATEGORIES = [
    "Formatting & Structure",
    "Keywords & Skills",
    "Work Experience",
    "Education",
    "Impact & Achievements",
    "Language & Grammar",
]

PROMPT_TEMPLATE = """You are an expert ATS (Applicant Tracking System) analyst and senior recruiter.
Evaluate the resume below and return ONLY a valid JSON object (no markdown, no commentary).

Scoring rules:
- Be realistic and strict. Most resumes score between 45 and 80. Do not inflate.
- Judge parseability: clear section headings, consistent dates, simple layout, no tables/columns/graphics clues.
- Judge content: quantified achievements, strong action verbs, relevant keywords, clarity, grammar.
{jd_rule}

Return this exact JSON shape:
{{
  "overall_score": <integer 0-100>,
  "summary": "<2-3 sentence overall verdict>",
  "category_scores": {{
{category_lines}
  }},
  "strengths": ["<short point>", "..."],
  "weaknesses": ["<short point>", "..."],
  "improvements": [
    {{"priority": "High|Medium|Low", "section": "<resume section>", "issue": "<what is wrong>", "fix": "<specific action to take>"}}
  ],
  "missing_keywords": ["<keyword>", "..."],
  "rewrite_examples": [
    {{"original": "<weak bullet copied from the resume>", "improved": "<stronger rewritten bullet>"}}
  ]
}}

Give 3-6 strengths, 3-6 weaknesses, 5-10 improvements (High priority first),
up to 12 missing keywords, and 2-4 rewrite examples taken from the actual resume.
Never invent experience, employers or numbers that are not implied by the resume;
in rewrite examples use placeholders like [X%] where a metric is unknown.

{jd_block}
=== RESUME START ===
{resume}
=== RESUME END ===
"""


# --------------------------------------------------------------------------
# Text extraction
# --------------------------------------------------------------------------
def extract_text_from_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise ValueError("This PDF is password protected. Please upload an unlocked copy.")
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def extract_text_from_docx(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    # Resumes often keep content in tables; read those too.
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def extract_resume_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        text = extract_text_from_pdf(data)
    elif name.endswith(".docx"):
        text = extract_text_from_docx(data)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or DOCX file.")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


# --------------------------------------------------------------------------
# Prompt + response handling
# --------------------------------------------------------------------------
def build_prompt(resume_text: str, job_description: str = "") -> str:
    jd = (job_description or "").strip()[:MAX_JD_CHARS]
    if jd:
        jd_rule = ("- A job description is provided. Weight keyword match and relevance to it heavily, "
                   "and list the important JD keywords missing from the resume.")
        jd_block = f"=== JOB DESCRIPTION START ===\n{jd}\n=== JOB DESCRIPTION END ===\n"
    else:
        jd_rule = "- No job description is provided: judge general ATS-readiness and common industry keywords."
        jd_block = ""
    category_lines = ",\n".join(f'    "{c}": <integer 0-100>' for c in SCORE_CATEGORIES)
    return PROMPT_TEMPLATE.format(
        jd_rule=jd_rule,
        jd_block=jd_block,
        category_lines=category_lines,
        resume=resume_text[:MAX_RESUME_CHARS],
    )


def _to_int(value, default=0) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


def _str_list(value) -> list:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def parse_model_json(raw: str) -> dict:
    """Parse the model output into a dict, tolerating ```json fences and stray text."""
    if not raw or not raw.strip():
        raise ValueError("The AI returned an empty response.")
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("Could not find JSON in the AI response.")
        data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("Unexpected AI response format.")
    return data


def normalize_result(data: dict) -> dict:
    """Validate and clean the model output so the UI never crashes on odd data."""
    cat_in = data.get("category_scores") if isinstance(data.get("category_scores"), dict) else {}
    categories = {c: _to_int(cat_in.get(c)) for c in SCORE_CATEGORIES}

    overall = data.get("overall_score")
    if overall is None and categories:
        overall = sum(categories.values()) / len(categories)

    improvements = []
    for item in data.get("improvements") or []:
        if not isinstance(item, dict):
            continue
        priority = str(item.get("priority", "Medium")).strip().capitalize()
        if priority not in ("High", "Medium", "Low"):
            priority = "Medium"
        improvements.append({
            "priority": priority,
            "section": str(item.get("section", "General")).strip() or "General",
            "issue": str(item.get("issue", "")).strip(),
            "fix": str(item.get("fix", "")).strip(),
        })
    order = {"High": 0, "Medium": 1, "Low": 2}
    improvements.sort(key=lambda i: order[i["priority"]])

    rewrites = []
    for item in data.get("rewrite_examples") or []:
        if isinstance(item, dict) and item.get("original") and item.get("improved"):
            rewrites.append({"original": str(item["original"]).strip(),
                             "improved": str(item["improved"]).strip()})

    return {
        "overall_score": _to_int(overall),
        "summary": str(data.get("summary", "")).strip(),
        "category_scores": categories,
        "strengths": _str_list(data.get("strengths")),
        "weaknesses": _str_list(data.get("weaknesses")),
        "improvements": improvements,
        "missing_keywords": _str_list(data.get("missing_keywords")),
        "rewrite_examples": rewrites,
    }


# --------------------------------------------------------------------------
# Gemini call
# --------------------------------------------------------------------------
def call_gemini(prompt: str, api_key: str, model: str) -> str:
    """Call Gemini, trying the fallback models if the primary one fails."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    # NOTE: no temperature/top_p/top_k - newer Gemini models reject them.
    config = types.GenerateContentConfig(response_mime_type="application/json")

    models_to_try = [model] + [m for m in FALLBACK_MODELS if m != model]
    errors = []
    for m in models_to_try:
        fatal = False
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                response = client.models.generate_content(model=m, contents=prompt, config=config)
                text = getattr(response, "text", None)
                if text:
                    return text
                errors.append(f"[{m}] empty response (it may have been blocked)")
                break
            except Exception as e:  # noqa: BLE001 - we want to surface any API error nicely
                msg = str(e).lower()
                # Bad key / quota problems will not be fixed by another model.
                if any(k in msg for k in ("api key", "api_key", "permission", "unauthenticated", "quota", "429")):
                    errors.append(f"[{m}] {e}")
                    fatal = True
                    break
                # Temporary overload: wait and retry the same model.
                temporary = any(k in msg for k in ("503", "500", "504", "unavailable", "overloaded",
                                                   "high demand", "deadline", "timed out"))
                if temporary and attempt < len(RETRY_DELAYS):
                    time.sleep(RETRY_DELAYS[attempt])
                    continue
                errors.append(f"[{m}] {e}")
                break
        if fatal:
            break
    raise RuntimeError(" || ".join(errors) if errors else "Unknown error calling Gemini.")


def analyze_resume(resume_text: str, job_description: str, api_key: str, model: str) -> dict:
    prompt = build_prompt(resume_text, job_description)
    raw = call_gemini(prompt, api_key, model)
    return normalize_result(parse_model_json(raw))


def friendly_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if "api key" in low or "api_key" in low or "unauthenticated" in low or "permission" in low:
        return "Your Gemini API key looks invalid or lacks permission. Please check it."
    if "quota" in low or "429" in low or "resource_exhausted" in low:
        return "Gemini rate limit / quota reached. Wait a minute and try again."
    if "503" in low or "unavailable" in low or "high demand" in low:
        return ("Google's Gemini servers are busy right now (temporary). "
                "The app already retried automatically; please wait a minute and click Analyze again.")
    if "blocked" in low or "safety" in low:
        return "The AI blocked this request. Try a different file."
    return f"Something went wrong: {msg}"


# --------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------
def score_label(score: int):
    if score >= 80:
        return "Excellent", "green"
    if score >= 65:
        return "Good", "orange"
    if score >= 50:
        return "Needs work", "orange"
    return "Poor", "red"


def get_secret(name: str, default: str = "") -> str:
    """Read from Streamlit secrets, then environment variables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets file present
        pass
    return os.environ.get(name, default)


def render_results(result: dict) -> None:
    score = result["overall_score"]
    label, color = score_label(score)

    st.divider()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("ATS Score", f"{score} / 100")
        st.markdown(f"**Verdict:** :{color}[{label}]")
    with c2:
        st.progress(score / 100)
        if result["summary"]:
            st.write(result["summary"])

    st.subheader("Score breakdown")
    cols = st.columns(2)
    for i, (cat, val) in enumerate(result["category_scores"].items()):
        with cols[i % 2]:
            st.write(f"**{cat}** — {val}/100")
            st.progress(val / 100)

    left, right = st.columns(2)
    with left:
        st.subheader("Strengths")
        for s in result["strengths"] or ["No strengths reported."]:
            st.markdown(f"- {s}")
    with right:
        st.subheader("Weaknesses")
        for w in result["weaknesses"] or ["No weaknesses reported."]:
            st.markdown(f"- {w}")

    st.subheader("Recommended improvements")
    icons = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}
    if not result["improvements"]:
        st.info("No specific improvements returned.")
    for imp in result["improvements"]:
        with st.expander(f"{icons[imp['priority']]} {imp['priority']} · {imp['section']}"):
            st.markdown(f"**Issue:** {imp['issue']}")
            st.markdown(f"**Fix:** {imp['fix']}")

    if result["missing_keywords"]:
        st.subheader("Missing keywords")
        st.write(" ".join(f"`{k}`" for k in result["missing_keywords"]))

    if result["rewrite_examples"]:
        st.subheader("Rewrite examples")
        for ex in result["rewrite_examples"]:
            st.markdown(f"**Before:** {ex['original']}")
            st.markdown(f"**After:** {ex['improved']}")
            st.write("")

    report = json.dumps(result, indent=2)
    st.download_button("Download report (JSON)", report, "ats_report.json", "application/json")


# --------------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="AI Resume ATS Checker", page_icon="📄", layout="wide")
    st.title("📄 AI Resume ATS Checker")
    st.caption("Upload your resume and get an ATS score with actionable improvements.")

    secret_key = get_secret("GEMINI_API_KEY")
    model = get_secret("GEMINI_MODEL", DEFAULT_MODEL)

    with st.sidebar:
        st.header("Settings")
        if secret_key:
            st.success("API key loaded from secrets.")
            api_key = secret_key
        else:
            api_key = st.text_input("Gemini API key", type="password",
                                    help="Get a free key at https://aistudio.google.com/apikey")
        st.caption(f"Model: `{model}`")
        st.caption("Your resume is sent to Google's Gemini API for analysis and is not stored by this app.")

    uploaded = st.file_uploader("Upload resume (PDF or DOCX)", type=["pdf", "docx"])
    job_description = st.text_area(
        "Job description (optional, but gives a much more accurate score)",
        height=160, placeholder="Paste the job description here...")

    if st.button("Analyze resume", type="primary", disabled=uploaded is None):
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar.")
            return
        data = uploaded.getvalue()
        if len(data) > MAX_FILE_MB * 1024 * 1024:
            st.error(f"File is larger than {MAX_FILE_MB} MB.")
            return
        try:
            with st.spinner("Reading your resume..."):
                text = extract_resume_text(uploaded.name, data)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the file: {e}")
            return

        if len(text) < MIN_TEXT_CHARS:
            st.error("Very little text was found. If this is a scanned/image PDF, "
                     "ATS systems cannot read it either — export a text-based PDF or DOCX.")
            return

        try:
            with st.spinner("Analyzing with Gemini..."):
                result = analyze_resume(text, job_description, api_key, model)
            st.session_state["result"] = result
        except Exception as e:  # noqa: BLE001
            st.error(friendly_error(e))
            return

    if "result" in st.session_state:
        render_results(st.session_state["result"])


if __name__ == "__main__":
    main()
