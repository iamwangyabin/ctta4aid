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

Status: data preparation complete; launcher preflight and experiment startup in
progress. The paper reference is 96.7% mAcc and 99.5% mAP.
