# Analysis and evidence rules

These rules apply to everything under `analysis/` and add to the root `AGENTS.md`.

## Evidence hierarchy

1. Immutable source identity and exact command.
2. Machine-readable receipt with thresholds, warnings, and failures.
3. Untouched raw/baseline/candidate artifacts.
4. Independent measurements and visual review.
5. Narrative conclusion scoped to the measured evidence.

Never reverse that order by writing a conclusion first and selecting supporting images afterward.

## Output policy

- Start a new narrative receipt from `EXPERIMENT_TEMPLATE.md` and keep its machine-readable evidence beside it.
- Every run gets a new hash/timestamp directory. Never overwrite a frozen release receipt or proof image.
- Keep intermediate tuning directories clearly distinguishable from canonical results.
- Store commands, baseline/candidate/core/shared-module/validator hashes, source SHA, scene catalog ID, PTS/ROI, decoded frame count, runtime, thresholds, metrics, warnings, and failures.
- For Motion/ImageScout release runs, pass every relevant candidate, baseline, and shared core through repeatable `m5_v3_validation.py --code-file` arguments so the receipt embeds their hashes.
- Preserve raw and derived artifacts separately. Labels belong outside measured image pixels.
- Surface write failures and partial/incomplete output explicitly.

## Canonical July 14 results

- Index: `flight_review_20260714/README.md`.
- Historical pre-fix audit: `flight_review_20260714/flight_system_audit.md`.
- Motion ISR Rev3 and ImageScout Rev3 receipt: `flight_review_20260714/v3_implementation_results.md`; it retains known failures and is not an all-green release.
- Historical SuperRes e13 automatic receipt: `flight_review_20260714/superres_v3_release_e13/superres_v3_validation.json`, status `PASS_METRICS_REVIEW_REQUIRED`.
- Current bounded 2x SuperRes evidence is dual and lives under `flight_review_20260714/superres_v3_regional_mps_0c4fbed0_20260717/`: the direct actual-Rev1 receipt is `PASS_METRICS_REVIEW_REQUIRED` with zero failures and two fixture-length warnings, and the independent source-honesty receipt is `PASS_METRICS_REVIEW_REQUIRED` with zero failures and zero warnings. Cite both and retain the operator-review requirement. `superres_v3_regional_mps_b6959382_20260717/` is `FAIL / SUPERSEDED` receipt-binding history; `superres_v3_mps_quality_4d600463_20260716/` and `superres_v3_quality_soak_d251ec99_20260716/` remain controlled earlier history.
- Directories named `repair`, `probe`, `tune`, `final_full`, `final_current`, or `final_frozen` are development history unless the index explicitly promotes them.

## Visual review

- Inspect original-resolution images, not only resized contact sheets.
- Check blur, ringing, halos, double contours, ghosting, checkerboard texture, color shifts, clipping, unsupported edges, and whether the change is actually useful.
- SuperRes delta energy must stay localized to source-supported structure; do not infer readable text or identifying detail that the source did not contain.
- Automatic image metrics cannot clear the human-review gate by themselves.

## Claims

- Report exact failures and warnings even when other metrics improve.
- Separate source smoke, bounded proxy validation, controlled A/B, and release acceptance.
- Do not present historical intermediate artifacts as current code truth.
- Do not say “every frame” when the audit means every recoverable decoded frame.

## Storage

- Most generated candidate directories are local evidence, not automatic Git content.
- Prefer committing small catalogs, Markdown conclusions, JSON receipts, CSV curves, and selected proof sheets; do not bulk-stage gigabytes of transient output.
