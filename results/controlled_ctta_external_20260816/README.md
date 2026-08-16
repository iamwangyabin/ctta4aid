# External Controlled CTTA Results

This package contains final, audited continual-stream results for the three target-only external benchmarks. Each benchmark uses the GenImage SD v1.4 ResNet-50 source detector recorded in `source_model.json`; external images are never used for source training.

Every suite has three fixed seeds, seven Controlled CTTA methods, 1,000 selected real and 1,000 selected fake images per generator, 750 adaptation samples per class, and a disjoint 250-sample final holdout per class. Target labels are available only to the evaluator.

The per-seed summaries retain the complete online metrics. For each seed, `per_seed/seed{n}_online_manifest.csv` and `per_seed/seed{n}_final_holdout_manifest.csv` record the exact sample identity and order used for online adaptation and final holdout evaluation, respectively. The canonical Source manifests were verified byte-for-byte against the corresponding manifests from all seven methods before import; the committed copies preserve the same entries and order with normalized LF line endings. Their `sample_id` fields resolve to the Arrow logical image paths. The AUC and accuracy JSON/CSV files provide cross-seed aggregates. `external_overview.csv` is the compact final table across all suites.

OpenSDID uses only its global `entire/` scope. AIGCDetectionBenchmark contains one selected DALL-E2 JPEG that was fully decodable only under PIL truncated-image recovery; its decoded pixels were re-encoded as PNG and the conversion remains disclosed in `aigc_detection_benchmark/run_metadata.json`.

## Data Provenance

`data_provenance.json` locks the raw artifacts and Arrow bundles actually used for these runs with portable SHA-256 tree fingerprints. It also records the official source identities against which the local raw artifacts were verified. A rerun is the same data snapshot only when the recorded raw and Arrow fingerprints match.
