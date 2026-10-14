# ATS-resume-checker-# 📄 AI Resume ATS Checker

Upload a resume (PDF or DOCX), optionally paste a job description, and get an **ATS score (0-100)**,
a score breakdown, strengths/weaknesses, prioritized improvements, missing keywords and rewrite examples.
Powered by **Google Gemini Flash** and built with **Streamlit**.

## Features
- PDF and DOCX support (DOCX tables are read too)
- Optional job description for a role-specific score and keyword gap
- Section-wise scores: formatting, keywords, experience, education, impact, grammar
- Prioritized fixes (High / Medium / Low) and before/after bullet rewrites
- Detects scanned/image PDFs that ATS systems cannot read
- Download the report as JSON

## Project files
| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## Run locally
1. Get a free Gemini API key: https://aistudio.google.com/apikey
2. Install and run:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Provide your key in **one** of these ways:
   - paste it into the sidebar field when the app opens, or
   - create `.streamlit/secrets.toml` (never commit this file):
     ```toml
     GEMINI_API_KEY = "your-key-here"
     ```
   - or set an environment variable: `export GEMINI_API_KEY=your-key-here`

## Choosing the model
The default is `gemini-3.8-flash`, with an automatic fallback to `gemini-3.7-flash`.
To use a different model, add this to secrets: `GEMINI_MODEL = "model-name"`.
Model names change over time; check https://ai.google.dev/gemini-api/docs/models

## Deploy on Streamlit Community Cloud
1. Push this project to a **public** GitHub repo (see below).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → choose your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**. You get a public `*.streamlit.app` URL.

## Notes and limitations
- The ATS score is an AI estimate, not the output of a real ATS. Use it as guidance.
- Resumes are sent to Google's Gemini API for analysis. The app itself does not store them.
- Scanned (image-only) PDFs are rejected because no text can be extracted.
