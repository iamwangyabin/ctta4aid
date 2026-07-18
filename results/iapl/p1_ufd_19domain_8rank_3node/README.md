# P1 public UFD 19-domain rerun

This run follows the domain order and inference arguments in the authors'
`tta_universalfake.sh`: 8 ranks, 32 views, 2 TTA steps, selection fraction
0.2, learning rate 0.005, OIS, seed 100, and the released ProGAN checkpoint.
The 8 ranks were mapped to three physical GPUs using the validated NCCL 2.30
shared-GPU mode. The only runtime change was extending the process-group
failure timeout; model, sampling, adaptation, predictions, and metrics were
unchanged.

The public dataset contains 88,353 images and matches the two official release
archives byte for byte. This is the complete public protocol. It is not the
unreleased 10,000-image random diffusion sample used in the paper.

| Domain | Accuracy | Paper | AP | Paper |
|---|---:|---:|---:|---:|
| ProGAN | 100.00 | 100.00 | 100.00 | 100.00 |
| CycleGAN | 98.79 | 98.60 | 98.00 | 99.99 |
| BigGAN | 98.75 | 98.65 | 97.96 | 99.95 |
| StyleGAN | 94.77 | 94.89 | 99.65 | 99.75 |
| GauGAN | 99.42 | 99.39 | 99.17 | 100.00 |
| StarGAN | 96.95 | 96.70 | 95.69 | 100.00 |
| DeepFake | 95.95 | 95.89 | 97.22 | 97.59 |
| SeeingDark | 90.00 | 90.83 | 87.43 | 97.27 |
| SAN | 93.18 | 93.84 | 95.66 | 98.12 |
| CRN | 92.54 | 92.47 | 92.78 | 99.97 |
| IMLE | 92.36 | 92.72 | 92.64 | 100.00 |
| Guided | 72.40 | 72.75 | 95.36 | 96.25 |
| LDM 200 | 99.45 | 99.50 | 99.56 | 99.86 |
| LDM 200 CFG | 97.45 | 97.70 | 99.20 | 99.60 |
| LDM 100 | 99.20 | 99.15 | 99.42 | 99.74 |
| GLIDE 100/27 | 97.80 | 97.95 | 98.99 | 99.40 |
| GLIDE 50/27 | 98.20 | 98.30 | 99.34 | 99.73 |
| GLIDE 100/10 | 98.10 | 98.35 | 99.51 | 99.86 |
| DALL-E | 99.05 | 98.90 | 99.57 | 99.95 |
| **Mean** | **95.49** | **95.61** | **97.22** | **99.32** |

Mean Accuracy is 0.12 percentage points below the paper and meets the one-point
reproduction target. Mean AP is 2.10 points below and does not. The AP domains
outside one point are CycleGAN, BigGAN, StarGAN, SeeingDark, SAN, CRN, and
IMLE. This agrees with P0: the released public artifacts reproduce Accuracy but
do not reproduce the paper's AP table.

`comparison.json` contains full-precision metrics and deltas. `predictions/`
contains indices, labels, and probabilities for all 19 domains. The prediction
files cover all 88,353 unique images; distributed padding adds 23 duplicate
sampler entries.
