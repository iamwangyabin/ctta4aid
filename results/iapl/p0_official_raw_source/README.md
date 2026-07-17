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
