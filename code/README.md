# Message Notification Router

Routes every message in `../dataset/messages.csv` to `notify` / `digest` / `mute` and writes
`../dataset/output.csv`. See `../README.md` and `../ARCHITECTURE.md` for the full approach.

## Approach

A deterministic Python feature layer (`features/`) joins `users.csv`, `groups.csv`,
`group_members.csv`, `business_accounts.csv`, `user_business_history.csv`,
`message_history.csv`, and `message_events.csv` to build per-message context: recipient
engagement/dismissal history, group trust/mute state or business verification/opt-in state,
quiet hours, and a scored shortlist of real candidate evidence messages from this user's own
history. Image and voice-note media are described/transcribed via Gemini before classification
(`media/`). A single Gemini call per message (`llm/`) then makes the final judgment - structured
JSON output, `evidence_message_ids` restricted at the schema level to the real shortlisted
candidates (so hallucinated IDs are impossible), and a system prompt with an explicit guardrail
against messages that try to instruct the router directly. Responses are cached by message/media
ID so reruns are free and deterministic, and a rule-based fallback (`llm/fallback.py`) covers any
row where the API call fails after retries.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate       # .venv\Scripts\activate on Windows
pip install -r requirements.txt
export GEMINI_API_KEY=...       # required for real classification + media understanding
```

Optional overrides: `ROUTER_LLM_MODEL`, `ROUTER_MEDIA_MODEL` (default `gemini-3.6-flash`).

Without `GEMINI_API_KEY` set, the pipeline still runs end-to-end using the deterministic
rule-based fallback classifier in `llm/fallback.py` (lower quality, but always produces a
schema-valid `output.csv` - useful for smoke-testing the plumbing without API cost).

## Run

```bash
python main.py                    # writes ../dataset/output.csv
python -m evaluation.main         # scores the pipeline against ../dataset/sample_messages.csv
```

(From the repo root, the evaluation module can also be run as `python -m code.evaluation.main`.)

`.cache/` (gitignored) holds the LLM and media response caches. Delete it to force fresh API
calls; otherwise reruns are fast and byte-identical.

## Layout

```
code/
├── main.py                 # entry point
├── config.py                # paths, model names, thresholds (env-overridable)
├── pipeline.py               # shared route_messages(), used by main.py and evaluation/
├── output_writer.py          # output.csv schema + writer
├── data/                     # CSV loaders + typed row/DataBundle schema
├── features/                 # user profile, conversation context, quiet hours,
│                              # evidence shortlisting, signal aggregation, output validation
├── media/                    # format sniffing, Gemini image/audio description, caching
├── llm/                      # prompt, few-shot examples, structured-output schema,
│                              # Gemini client, rule-based fallback, response cache
└── evaluation/                # scoring harness against sample_messages.csv
```
