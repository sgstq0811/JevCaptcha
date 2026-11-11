# JevCaptcha: Enumerate, Ask, Aggregate

Code and per-challenge results for *Enumerate, Ask, Aggregate: Solving CAPTCHAs with Typed Decisions*.

The agent solves visual CAPTCHAs with eight interaction skills (cycle, select, swap, point, order, drag, hold, type).
Each skill enumerates the actions a widget allows, asks a typed decision model
([Clef-flash](https://huggingface.co/Cloudflare/clef-flash)) one closed question per candidate, and combines the
returned probabilities with a fixed aggregation rule (verify-argmax, attribute matching, joint inference over a rotation
group, pairwise equality for swaps, assignment, generate-then-verify for text, largest log-odds gap for tile sets).
The agent is not told the CAPTCHA family; it routes each challenge from the instruction, the picture and the kinds of
controls on the page. No model is fine-tuned.

## Layout

| Path | Content |
|---|---|
| `code/ocw_skills.py` | skill library (`SkillAgent`) and the Open CaptchaWorld adapter (browser) |
| `code/cog_skills.py` | COGNITION adapter (static tasks) |
| `code/hal_skills.py` | Halligan adapter |
| `code/mca_skills.py` | MCA-Bench adapter (held out; page description and answer format only) |
| `code/ocw_policy.py`, `code/hal_policy.py` | question templates, candidate generators and type-specific pipelines |
| `code/clef.py` | Clef / Clef-flash wrapper; `code/gen_backend.py` the same interface on a generative VLM (Qwen3.5-9B) |
| `code/tools.py` | Florence-2 OCR candidates for the `type` skill (disk-cached) |
| `code/ocw_generic.py` | step-wise agent baseline |
| `code/cog_direct.py`, `code/mca_direct.py` | direct-prompting baselines |
| `code/api_direct.py` | direct prompting of API models (OpenAI-compatible endpoint in `API_BASE`, key file in `API_KEY_FILE`) |
| `code/cog_tasks.py`, `code/hal_score_official.py`, `code/mca_tasks.py` | official task builders and scorers |
| `code/theory_data.py`, `code/theory_analysis.py` | records and analysis for the aggregation-rule checks |
| `results/` | per-challenge results of the frozen library (see below) |

## Data

All paths are relative to `$JEVCAPTCHA_ROOT` (default `~/captcha_jev`):

- **Open CaptchaWorld**: the official app at the commit of the paper version (225 challenges) in `ocw_v1_app/`, and the current release in `OpenCaptchaWorld/`; serve with `python app.py` and point `OCW_URL` / `OCW_GT` at it.
- **COGNITION**: the USENIX Security '26 artifact (Zenodo 20406852) in `cognition/cognition-usenixsec26-artifact/`; build tasks with `python code/cog_tasks.py dump results/cog_tasks.json` in the artifact's own environment.
- **Halligan**: the USENIX Security '25 benchmark (Zenodo 15709075) in `data/halligan/bench/`.
- **MCA-Bench**: the released dataset in `mca/` and the official `tojson` outputs in `mca_json/` (`code/mca_build.py`).
- A word list at `data/lexicon/american-english` (e.g. `/usr/share/dict/american-english`).

## Running

```bash
pip install -r requirements.txt && playwright install chromium
export JEVCAPTCHA_ROOT=~/captcha_jev
cd code
# Open CaptchaWorld (app running on port 7861)
OCW_URL=http://127.0.0.1:7861 OCW_GT=$JEVCAPTCHA_ROOT/ocw_v1_app/captcha_data python ocw_skills.py --out ../results/ocw225.jsonl
# COGNITION main split, then score with the artifact's scorer
python cog_skills.py --split main --out ../results/cog_main.json
python cog_tasks.py score $PWD/../results/cog_tasks.json $PWD/../results/cog_main.json
# Halligan (all 26 families), then score with the official verifier
python hal_skills.py --out ../results/hal.jsonl
python hal_score_official.py ../results/hal.jsonl
# MCA-Bench official test split (scored inline with the official rules)
python mca_skills.py --out ../results/mca.jsonl
```

`--model Qwen/Qwen3.5-9B` runs the same library with the generative backbone (`GEN_SCORE=first` for first-token scoring).
`ROUTE_FROM=<results file>` replaces the router with the routes of another run (crossed router x scorer study).

## Results in `results/`

| File | Set | Correct |
|---|---|---|
| `ocw225_frozen_run2.jsonl` | OCW paper version, full run of the frozen library | 136 / 225 |
| `cog_v17_full.json` | COGNITION full set (main split 189 / 303; outside it 57 / 75) | 246 / 378 |
| `hal_v17_all.jsonl` | Halligan | 1753 / 2600 |
| `ocw_new245_v17.jsonl` | OCW current release outside the paper version's pools | 118 / 245 |
| `ocw_new245_typed.jsonl` | same, type-specific pipelines | 138 / 245 |
| `cog_ext_v17.json` | three COGNITION families unseen during design | 17 / 80 |
| `mca_v17.jsonl` | MCA-Bench official test split | 414 / 2000 |
| `hal_qf_all.jsonl` | Halligan, Qwen3.5-9B with first-token scoring | 1507 / 2600 |
| `hal_x_*.jsonl` | crossed router x scorer on the 150 differently routed Halligan challenges | |
| `ocw_new245_qwen.jsonl`, `cog_rest_qwen.json`, `cog_ext_qwen.json`, `mca_qwenlib.jsonl` | held-out sets, same library with Qwen3.5-9B (MCA with first-token scoring) | 96 / 245, 29 / 75, 2 / 80, 309 / 2000 |
| `cog_rest_direct.json`, `mca_direct_qwen9b.jsonl` | held-out sets, Qwen3.5-9B direct prompting (MCA coordinates are 0-1000; rescale to pixels before scoring) | 18 / 75, 472 / 2000 |
| `api_*_gpt-6-sol.*`, `api_*_gemini-3.1-pro-preview*` | held-out sets, gpt-6-sol / Gemini-3.1-Pro direct prompting (MCA Gemini coordinates rescaled to pixels in `_px`) | COGNITION outside main 41 / 30 of 75, unseen families 48 / 62 of 80, MCA 1129 / 862 of 2000 |
| `ocw_new245_stepwise.jsonl` | OCW new challenges, step-wise agent | 49 / 245 |
| `ocwnew_n_*.jsonl`, `cogrest_n_*.json`, `mca_n_*.jsonl` | held-out sets, library with one rule replaced by a naive rule (`ABL=indep/top1/gap/assign/nover`) | |

## Responsible use

A general CAPTCHA solver weakens a security control. The code is released so that CAPTCHA designers can test new widgets
against enumerate-and-verify solvers; use it only on benchmarks and systems you are authorised to test.
