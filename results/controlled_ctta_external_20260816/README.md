# External Controlled CTTA Results

This package contains final, audited continual-stream results for the three target-only external benchmarks. Each benchmark uses the GenImage SD v1.4 ResNet-50 source detector recorded in `source_model.json`; external images are never used for source training.

Every suite has three fixed seeds, seven Controlled CTTA methods, 1,000 selected real and 1,000 selected fake images per generator, 750 adaptation samples per class, and a disjoint 250-sample final holdout per class. Target labels are available only to the evaluator.

The per-seed summaries retain the complete online metrics. The AUC and accuracy JSON/CSV files provide cross-seed aggregates. `external_overview.csv` is the compact final table across all suites.

OpenSDID uses only its global `entire/` scope. AIGCDetectionBenchmark contains one selected DALL-E2 JPEG that was fully decodable only under PIL truncated-image recovery; its decoded pixels were re-encoded as PNG and the conversion remains disclosed in `aigc_detection_benchmark/run_metadata.json`.

## Data Provenance

`data_provenance.json` locks the raw artifacts and Arrow bundles actually used for these runs with portable SHA-256 tree fingerprints. A rerun is the same data snapshot only when the recorded raw and Arrow fingerprints match. It deliberately distinguishes those locked local snapshots from any upstream revision that was not captured at download time.
