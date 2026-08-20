# Cost/time report -- experiment_smoke (profile: full)

## Measured

| stage | runs | input tok | output tok | USD | wall s | cache hits | flags |
|---|---|---|---|---|---|---|---|
| mining | 3 | 33,203 | 3,492 | $0.003198 | 110.5 | 0 | - |
| attribution | 0 | 5,310 | 203 | $0.000539 | 13.3 | 0 | - |
| proposal | 0 | 1,581 | 159 | $0.000107 | 5.1 | 0 | - |
| validation | 12 | 322,003 | 18,755 | $0.03 | 544.7 | 0 | - |
| eval | 24 | 1,470,728 | 84,763 | $0.15 | 3,416.5 | 0 | lower-bound resumed x8 |

### Per-run means (the extrapolation basis)

| environment | length | runs | mean in | mean out | mean USD | mean s | flags |
|---|---|---|---|---|---|---|---|
| graphwalks | long | 4 | 22,349.2 | 1,254.5 | $0.002253 | 33.8 | - |
| graphwalks | short | 23 | 25,681.2 | 1,415.0 | $0.002364 | 40.8 | - |
| oolong_pairs | long | 4 | 137,188.0 | 8,099.8 | $0.01 | 237.8 | - |
| oolong_pairs | short | 8 | 74,639.8 | 4,630.9 | $0.007815 | 254.3 | lower-bound |

Disk: 379,779,821 B over 39 run(s) = 9,737,944 B/run; projected 236.340 GB for the full experiment.

## Extrapolation

Runs/round = 1,456 (m*n_in + v(K+1)(n_in+n_ho) + p_merge*v(n_in+n_ho)); 15 round(s) = 21,840 optimization runs. Eval grid = 3 condition(s) x (short + long) x repetitions = 2,430 runs.

### point projection

| leg | context | runs | cost drift | input tok | output tok | basis |
|---|---|---|---|---|---|---|
| optimization | short | 21,840 | - | 560,876,838 | 30,904,550 | graphwalks/short (23 run(s)) |
| eval_short | short | 720 | - | 27,587,265 | 1,616,354 | graphwalks+oolong_pairs/short (31 run(s)) |
| eval_long | long | 1,710 | - | 136,404,349 | 7,997,884 | graphwalks+oolong_pairs/long (8 run(s)) |

Total: 724,868,452 in / 40,518,787 out

### pessimistic projection

| leg | context | runs | cost drift | input tok | output tok | basis |
|---|---|---|---|---|---|---|
| optimization | short | 21,840 | x7 | 4,101,387,116 | 225,988,154 | graphwalks/short (23 run(s)) |
| eval_short | short | 720 | - | 27,587,265 | 1,616,354 | graphwalks+oolong_pairs/short (31 run(s)) |
| eval_long | long | 1,710 | - | 136,404,349 | 7,997,884 | graphwalks+oolong_pairs/long (8 run(s)) |

Total: 4,265,378,729 in / 235,602,391 out

## Scenarios

| scenario | kind | USD (point) | USD (pessimistic) | eligible | notes |
|---|---|---|---|---|---|
| api_promo | api | $69.74 | $409.55 | yes | configured pricing tier: $0.08/1M in, $0.29/1M out |
| api_list | api | $84.64 | $497.22 | yes | configured pricing tier: $0.1/1M in, $0.3/1M out |
| 1x_a100_int4 | gpu | $238.61 | $982.27 | no | provenance is marked unvalidated, so its throughput input cannot support a recommendation; this profile is scenario-only; changes model numerics; Unvalidated estimate; AWQ/GPTQ-INT4 weights ~17GB fit 262k KV on one A100 80GB |
| h100_sxm_rented | gpu | $321.65 | $1,398.22 | yes | Median H100 SXM on-demand marketplace rate, 2026-08 (Lambda/RunPod listings) |
| 2x_a100_tp2_rented | gpu | $429.50 | $1,768.08 | no | provenance is marked unvalidated, so its throughput input cannot support a recommendation; this profile is scenario-only; Unvalidated estimate from A100 80GB marketplace rates 2026-08; BF16 needs 2xA100 at 262k (weights ~61GB + KV ~24.6GB/seq) |

## Recommendation

**api_promo** at $69.74 -- the cheapest scenario passing every validity gate.

## Warnings

- lower-bound usage in oolong_pairs/short: terminated run(s) contributed, so its per-run mean understates the true cost
- measured evaluation covers 2 condition(s) while the full-experiment grid assumes 3 (report.eval_conditions); the projection uses the configured grid
