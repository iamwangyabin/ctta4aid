# GenImage Controlled CTTA Results

This package contains the completed three-seed continual-stream results used
for Tables 2 and 3. The seven Controlled CTTA methods use the ResNet-50 source
detector recorded in source_model.json for this campaign; this is a property
of the recorded run, rather than a framework-wide requirement on future
experiments.

The stream order is BigGAN, ADM, GLIDE, Stable Diffusion v1.5, VQDM, Wukong,
and Midjourney. Each domain contributes 750 real and 750 fake online samples,
followed by a disjoint final holdout of 250 real and 250 fake samples. Target
labels are available only to the evaluator.

The genimage_continual/continual_auc_* and
genimage_continual/continual_accuracy_* files contain cross-seed tables and
full aggregates. The per_seed directory preserves each complete online summary
and the canonical Source online and final-holdout manifests. Before import,
those manifests were verified byte-for-byte against the corresponding
manifests of all seven methods. The committed copies use normalized LF line
endings while preserving every manifest entry and order. Their hashes, row
counts, and zero online/holdout overlap are recorded in run_metadata.json; each
sample_id identifies the exact Arrow logical image path and order used by a run.

The campaign completed on 2026-08-12. Its archived execution snapshot did not
retain Git metadata, so this package records the observed checkpoint digest,
protocol, summaries, and exact sample manifests without asserting a source-code
commit identity.
