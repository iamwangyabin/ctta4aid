# P0 official raw UFD source

The original UFD Google Drive aggregate is no longer available. The UFD README
also points to the dataset scripts maintained by the authors of the earlier
CNNDetection work. Their current official `download_testset.sh` downloads
`CNN_synth_testset.zip` from the `sywang/CNNDetection` Hugging Face repository.

The 20,052,866,587-byte archive was downloaded with resume support on 4090-1.
The host cannot reach `huggingface.co` directly because of its network setup, so
the transfer used the Hugging Face mirror resolver. The mirror redirected to the
same upstream Xet object recorded in `source_manifest.json`.

P0 then compared file counts, paths, and image bytes against the existing Arrow
copy. This provides stronger evidence than treating the now-dead UFD Drive link
as an unresolved data-source ambiguity.

The byte audit is implemented in `scripts/compare_ufd_raw_arrow.py`. For the P0
diagnostic it compares `crn`, `imle`, `san`, and `seeingdark` from the
ForenSynths Arrow root. `guided` belongs to the separately released Ojha
diffusion bundle and is therefore not expected inside this CNNDetection archive.
The same tool also completed the full 11-domain CNNDetection audit before P1.

Before the official archive finished downloading, the comparator was exercised
against the byte-exact ImageFolder export already used by the P0 control. Ten
samples from each of the four ForenSynths P0 domains matched their Arrow payloads
exactly, including the deterministic aggregate hashes. The project test suite
then passed all 54 tests on A6000.

`scripts/postprocess_ufd_official_archive.sh` handled the completed transfer. It
verified the exact archive size, recorded SHA-256, tested the ZIP structure, and
extracted with idle I/O priority so the running IAPL inference kept precedence.

The separate official `diffusion_datasets.zip` archive is complete and verified:
917,979,875 bytes, SHA-256
`916cc972810e51f6ca1be30d7a85c84230310a4e4b141a90c47f20ee310ac0fc`, and
no ZIP errors. UFD's pinned `dataset_paths.py` pairs `imagenet/0_real` with
`guided/1_fake`. With that explicit alias, all 2,000 Guided files in the Ojha
Arrow copy match the official archive byte for byte, with no missing or extra
paths. This rules out Arrow serialization and the Guided dataset copy as causes
of the remaining metric differences.

The full diffusion audit uses the path pairings frozen in UFD's pinned
`dataset_paths.py`: Guided uses ImageNet real images, while the three LDM,
three GLIDE, and DALL-E domains use LAION real images. All 16,000 released files
across the eight domains match the Ojha Arrow payloads byte for byte. UFD notes
that its paper evaluated 10,000 randomly sampled images per diffusion domain but
released 1,000 real and 1,000 fake images per domain; this reproduction can only
claim exactness against that public 2,000-image-per-domain release.

The CNNDetection archive is now also complete. Its exact size matches the Hub
metadata, SHA-256 is
`d87eeff4eb6d1061f57620aa1bd54e699a18cc9860fcdc6a55bf4cf643008d85`
(the same value as the linked ETag), and the ZIP test and extraction succeeded.
All 26,326 official files across CRN, IMLE, SAN, and SeeingDark match the
ForenSynths Arrow payloads byte for byte, with no missing or extra paths.

The complete pre-P1 audit extends this result to all 11 CNNDetection domains:
72,353 of 72,353 files match exactly, with zero missing paths, extra paths,
byte mismatches, label/path mismatches, or metadata MD5 mismatches. The full
per-domain counts and deterministic manifest hashes are recorded in
`official_forensynths_11domain_arrow_comparison.json`.

Across the complete public 19-domain UFD release, 88,353 of 88,353 files now
have an exact official-archive match in the Arrow datasets. This closes the
public-data-copy ambiguity before the full P1 protocol run.

Together with the 2,000-file Guided audit, all five P0 abnormal domains are now
proven to use the official released image bytes. Their residual metric gaps
cannot be attributed to Arrow serialization, ImageFolder decoding, or a
different dataset copy; the remaining live hypothesis is execution protocol,
especially rank sampling, rank RNG, and accumulating BatchNorm buffers.
