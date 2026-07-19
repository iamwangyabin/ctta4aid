# P2 original GenImage eight-domain rerun

This run uses the original public `genimage_test.zip`, the released IAPL SD1.4
checkpoint, and the domain order and inference arguments from the authors'
`tta_genimage.sh`. It does not use the CAIDBench proxy exports.

The source ZIP passed a complete CRC test before transfer. All three compute
hosts hold the same 25,408,572,820-byte ZIP with SHA256
`1ee98d0958a5905b4e1b2f7f44bb384069704cad8f70e741512e1911e29dba97`.
The structured extractor read all 100,000 members through Python's CRC-checking
ZIP stream and produced identical extraction manifests on all hosts. Every
domain has 6,000 real and 6,000 fake images except SD1.5, which has 8,000 of
each.

The official protocol is 8 ranks, 32 views, 2 TTA steps, selection fraction
0.2, learning rate 0.005, OIS, smoothing, 8 data workers, seed 100, and the
released SD1.4 checkpoint. As in P0/P1, the eight ranks are mapped to three
physical GPUs using the validated NCCL 2.30 shared-GPU mode. This preserves the
official rank sampler and seed semantics but is not an eight-GPU topology.

The run started at 2026-07-19 07:42 +08 and completed at 17:12 +08. All eight
ranks completed the initial distributed barrier and all eight domains finished
without fatal errors.

| Domain | Accuracy | AP | Real Acc. | Fake Acc. |
|---|---:|---:|---:|---:|
| ADM | 85.53 | 98.29 | 99.77 | 71.30 |
| BigGAN | 98.69 | 99.65 | 99.80 | 97.58 |
| GLIDE | 95.95 | 99.45 | 99.68 | 92.22 |
| Midjourney | 95.74 | 99.12 | 99.78 | 91.70 |
| SD1.4 | 99.91 | 99.94 | 99.82 | 100.00 |
| SD1.5 | 99.74 | 99.87 | 99.59 | 99.89 |
| VQDM | 98.79 | 99.69 | 99.67 | 97.92 |
| Wukong | 99.82 | 99.90 | 99.67 | 99.97 |
| **Mean** | **96.77** | **99.49** | **99.72** | **93.82** |

Full-precision mAcc is 96.7714%, 0.0714 percentage points above the paper's
rounded 96.7%. Full-precision mAP is 99.4895%, 0.0105 points below the paper's
99.5%. Both metrics meet the one-percentage-point reproduction target. The
100,000 prediction records cover all 100,000 unique test images with no
distributed padding duplicates because every domain size is divisible by eight.

`comparison.json` contains full-precision metrics. `predictions/` contains the
sampler indices, labels, and probabilities for every image. The eight rank logs,
three launcher logs, source/config hashes, and completion checks are recorded in
`run_manifest.json`.
