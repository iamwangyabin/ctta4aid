# P0 official raw UFD source

The original UFD Google Drive aggregate is no longer available. The UFD README
also points to the dataset scripts maintained by the authors of the earlier
CNNDetection work. Their current official `download_testset.sh` downloads
`CNN_synth_testset.zip` from the `sywang/CNNDetection` Hugging Face repository.

The 20,052,866,587-byte archive is downloading with resume support on 4090-1.
The host cannot reach `huggingface.co` directly because of its network setup, so
the transfer uses the Hugging Face mirror resolver. The mirror redirects to the
same upstream Xet object recorded in `source_manifest.json`; after the transfer,
the archive size and SHA-256 digest will be frozen before extraction.

Once extracted, P0 will compare file counts, paths, image bytes, and per-domain
prediction behavior against the existing Arrow copy. This provides stronger
evidence than treating the now-dead UFD Drive link as an unresolved data-source
ambiguity.

The byte audit is implemented in `scripts/compare_ufd_raw_arrow.py`. For the P0
diagnostic it compares `crn`, `imle`, `san`, and `seeingdark` from the
ForenSynths Arrow root. `guided` belongs to the separately released Ojha
diffusion bundle and is therefore not expected inside this CNNDetection archive.
The same tool can then audit all 11 CNNDetection domains before P1.

Before the official archive finished downloading, the comparator was exercised
against the byte-exact ImageFolder export already used by the P0 control. Ten
samples from each of the four ForenSynths P0 domains matched their Arrow payloads
exactly, including the deterministic aggregate hashes. The project test suite
then passed all 54 tests on A6000.

`scripts/postprocess_ufd_official_archive.sh` is waiting on the active transfer.
It verifies the exact archive size, records SHA-256, tests the ZIP structure, and
extracts with idle I/O priority so the running IAPL inference keeps precedence.

The separate official `diffusion_datasets.zip` archive is complete and verified:
917,979,875 bytes, SHA-256
`916cc972810e51f6ca1be30d7a85c84230310a4e4b141a90c47f20ee310ac0fc`, and
no ZIP errors. UFD's pinned `dataset_paths.py` pairs `imagenet/0_real` with
`guided/1_fake`. With that explicit alias, all 2,000 Guided files in the Ojha
Arrow copy match the official archive byte for byte, with no missing or extra
paths. This rules out Arrow serialization and the Guided dataset copy as causes
of the remaining metric differences.
