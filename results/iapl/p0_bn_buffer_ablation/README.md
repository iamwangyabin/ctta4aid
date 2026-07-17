# P0 IAPL buffer-state ablation

This directory isolates two stateful behaviors in the released IAPL inference
loop: BatchNorm running buffers that persist between samples, and the default
DDP broadcast of model buffers. The diagnostic patch is opt-in and leaves the
authors' released behavior unchanged unless an environment switch is set.

The single-rank control runs SAN followed by SeeingDark with
`IAPL_RESET_BN_PER_SAMPLE=1`. Relative to the byte-identical Arrow baseline,
resetting buffers lowers SAN Accuracy/AP by 1.83/2.59 percentage points and
raises SeeingDark by 0.83/1.33 points. The two-domain mean AP is 0.63 points
lower, so buffer accumulation is real but not a general remedy.

The four-rank SAN controls use the same shared A6000 topology as the earlier
SAN smoke run. Disabling DDP buffer broadcast changes Accuracy/AP by
+0.23/+0.006 points. Resetting BatchNorm buffers before every sample changes
them by 0.00/-0.045 points. Both AP effects are negligible compared with the
5.61-point five-domain gap to the paper.

Together with the exact-byte data audit, Arrow/ImageFolder controls, and the
completed eight-rank run, these results rule out dataset copying, the Arrow
backend, rank-faithful sampling, DDP buffer broadcasts, and simple BatchNorm
buffer carry-over as the main AP discrepancy. The remaining discrepancy is
between the paper's reported AP and what the released checkpoint, public data,
and released inference path produce; it should be reported as an artifact-level
reproduction gap rather than hidden by protocol changes.
