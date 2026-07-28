# P4 IAPL inference ablations

P4 uses the released ProGAN checkpoint and the complete public 19-domain UFD
release. This matches the dataset used for the paper's inference ablations and
avoids mixing P3's trained-checkpoint seed instability into a module study.
Every run keeps the official eight-rank 4+2+2 layout, seed 100, domain order,
learning rate, per-image reset behavior, and public 88,353-image sample set.

The baseline is 32 views, two tuning steps, six selected views, averaged
entropy, and optimal-input selection (OIS). Each variant changes one factor.
The order starts with cheaper view and step settings, then follows the expected
number of view forwards through OIS, selection-count, entropy, a newly profiled
baseline, and finally the most expensive three-step run. P1's baseline
predictions remain the performance reference,
but the baseline is rerun because P1 predates timing and memory capture.

The runtime patch adds an exact selected-view count, averaged/pointwise entropy
selection, and per-domain timing plus CUDA peak-memory records. The launcher
also samples total physical-GPU memory once per second. Internal memory is
reported per rank; host memory is reported separately because multiple ranks
share each physical GPU.

The paper reports averaged versus pointwise entropy in Table 8 and tuning steps
1/2/3 in Table 9. Its implementation details fix 32 views, six selected views,
two steps, and learning rate 0.005. P4 reproduces those controllable inference
choices and extends them with view-count, selected-count, OIS, latency, and
VRAM curves. No weak or failed run will be removed.

## Execution status

The first ordered variant, `views8`, started at 2026-07-28 15:09 CST on the
planned eight-rank A6000/3090/4090-2 layout. All ranks passed the distributed
barrier, rank 0 entered the 12,764-row `crn` domain, all three GPUs reached
100% utilization, and the launch audit found no traceback, NCCL warning, CUDA
error, or out-of-memory event. Outputs are being written under
`outputs/iapl_official/p4_ufd_ablation/views8` on each host.

The preflight did expose three reproducible setup failures before launch: the
server shell has no bare `python`, the runtime source requires the project
`PYTHONPATH`, and upstream IAPL disables argparse's built-in `--help`. These
are preserved in `preflight_20260728.json`; the final checks use the pinned
conda interpreter, the project source path, an early 4090 compatibility-library
export, and static parser-option checks before constructing a real Arrow
dataset view.
