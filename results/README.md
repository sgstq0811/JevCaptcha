# Results

Per-challenge outputs of the frozen library and of the baselines. `.jsonl` files hold one challenge per line: OCW and MCA-Bench lines carry the official verdict, Halligan lines the final state (scored with `code/hal_score_official.py`); `.json` files map each COGNITION key to the answer (scored with `code/cog_tasks.py score`). `cog_v17_full_log.jsonl` holds the route and the skill log of every COGNITION challenge outside the main split.

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
