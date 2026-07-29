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

The 15:45 snapshot preserves the first three completed domains while `biggan`
continues. Their macro Acc/AP are 97.2408%/98.0795%; `crn`, `cyclegan`, and
`dalle` Acc are 94.1808%, 98.7915%, and 98.7500%. The measured cluster
throughput is stable at about 8.96-8.98 images/s and the rank-local PyTorch
peak is 3,534 MiB allocated. Physical-GPU peaks observed so far are 17,399 MiB
on A6000, 8,768 MiB on 3090, and 9,270 MiB on 4090-2. The A6000 carries four
ranks, so its approximately 0.89 s rank-local latency is the cluster bottleneck;
these latency figures must be compared only under the same 4+2+2 layout.

The 16:45 snapshot extends the audit to ten completed domains while `imle`
continues. The ten-domain macro Acc/AP are 95.0350%/98.2757%. The weak
`guided` result is preserved without restart or selection: 70.9500% Acc and
95.1482% AP, caused by 43.0000% fake accuracy despite 98.9000% real accuracy.
Throughput, rank-local memory, physical-GPU peaks, and the error-free eight-rank
state remain consistent with the first snapshot.

`views8` completed all 19 domains at 17:55:55 CST with 88,353 unique samples.
Final macro Acc/AP are 95.6007%/98.1885%, real/fake accuracy are
97.2529%/93.9446%, and no rank log contains a runtime error. Relative to the
unprofiled 32-view P1 reference, Acc improves by 0.1084 percentage points and
AP by 0.9711 points, while fake accuracy drops by 0.7636 points. This is a
valid ablation outcome rather than a claim that eight views universally
outperform the baseline, because the random augmentation stream also changes.

The 19 profiled domains contain 88,376 samples after distributed padding and
9,850.67 seconds of summed critical-path domain time, giving 8.9692 unique
images/s overall. Weighted bottleneck-rank latency is 891.36 ms/image. Peak
PyTorch allocation/reservation is 3,533.84/3,836 MiB per rank; final host peaks
remain 17,399/8,768/9,270 MiB for A6000/3090/4090-2.

After the views8 final audit and push, `views16` passed the real Arrow creator
preflight on all three hosts and started at 18:00:55 CST. All eight ranks
crossed the barrier and rank0 entered `crn`; the initial A6000 reading is
24,095 MiB at 100% utilization with a 5,168 MiB per-rank PyTorch peak. No P4
variant was skipped or overlapped.

The 19:15 snapshot preserves the first four completed `views16` domains while
`deepfake` continues. Macro Acc/AP are 97.4545%/97.6009%, respectively 0.1824
and 0.5667 percentage points below `views8` on the same four domains. Throughput
falls to 5.23-5.26 images/s, about 58.5% of the corresponding `views8` rate.
Per-rank PyTorch allocation rises to 5,168.46 MiB and host peaks rise to
24,095/12,116/12,618 MiB. This is the expected compute/memory penalty and is
preserved before the remaining domains determine the final accuracy tradeoff.

The 20:45 snapshot reaches ten domains while `imle` continues. On the same ten
domains, `views16` is 0.0901 percentage points higher than `views8` in Acc but
0.2437 points lower in AP; real accuracy is 0.2356 points lower and fake
accuracy 0.4159 points higher. `guided` remains weak and is preserved at
72.10% Acc / 95.32% AP. The compute and memory profile remains stable, so no
restart or protocol change is justified before the final nine domains.

`views16` completed all 19 domains at 22:42:40 CST. Final macro Acc is
95.5999%, statistically unchanged from `views8` in this single fixed-seed
comparison (-0.0008 percentage points), while AP falls by 0.4999 points.
Real accuracy falls 0.4280 points and fake accuracy rises 0.4278 points. The
larger view set therefore changes the threshold tradeoff but does not improve
macro Acc.

Its overall throughput is 5.2513 images/s, only 58.5% of `views8`, and weighted
bottleneck-rank latency rises from 891.36 to 1,522.30 ms/image. Per-rank
allocated memory rises 46.3% to 5,168.46 MiB; host peaks are
24,095/12,116/12,618 MiB. On this protocol, 16 views are dominated by 8 views
on AP, latency, throughput, and memory, with effectively tied Acc.

After the views16 final audit and push, `steps1` passed its three-host 32-view
Arrow preflight and started at 22:50:27 CST. All eight ranks crossed the
distributed barrier and entered model execution. This is the first direct
tuning-step ablation and keeps every other baseline factor fixed.

The 23:45 snapshot preserves the first three `steps1` domains while `biggan`
continues. Macro Acc/AP are 96.6848%/97.7624%. Against the original two-step P1
run on these same domains, Acc is 0.1078 percentage points lower but AP is
0.9828 points higher, driven primarily by `crn`; this provisional reversal is
kept without restart. Throughput is 5.54-5.57 images/s, weighted latency about
1.44 s/image, and per-rank allocation is 8,437.55 MiB. Host peaks are
37,171/18,652/19,154 MiB, making 32-view memory cost visible even with one
tuning step.

The 01:16 snapshot reaches ten completed domains while `imle` continues. On
the same ten domains, `steps1` is 0.0522 percentage points higher than the P1
two-step reference in Acc and 0.5257 points higher in AP; real and fake
accuracy are respectively 0.0193 and 0.0850 points higher. This remains a
provisional fixed-seed comparison rather than a final step-count conclusion.
The weak `guided` result is retained at 72.8500% Acc / 95.9568% AP, including
46.7000% fake accuracy.

Compute remains stable across the ten completed domains at 5.51-5.58 images/s
and 1.43-1.45 s bottleneck-rank latency per image. Per-rank allocation and
reservation remain 8,437.55/8,776 MiB, and host peaks remain
37,171/18,652/19,154 MiB. All eight rank logs are error-free, so the run
continues unchanged through the remaining nine domains before `ois_off`.

Attempt 1 did not survive the next domain. At 01:24:26 CST the 3090 host's
`/data` NVMe began timing out on reads; its controller reset failed, the kernel
disabled the device at 01:26:52, and EXT4 aborted the journal. The two ranks
reading Arrow from that filesystem both exited with SIGBUS. Rank 7 then
reported the expected downstream `ncclRemoteError`; the remaining ranks were
stopped after the complete failure state was copied. This is a physical
storage failure, not a CUDA OOM, host-memory shortage, or model error.

Only the first ten domains remain valid. Although rank 0 finished its local
`imle` loop, the failed peers never completed the distributed gather, so no
`imle` result is accepted. The run is not resumed from that boundary because
a new process would reset the augmentation RNG stream. All 19 domains are
being restarted from seed 100 and the failed attempt remains archived.

Attempt 2 started at 02:29:04 CST after all three replacement-node Arrow
preflights passed. The eight-rank logical protocol is unchanged, but healthy
4090-1 now carries ranks 4-5 in place of the failed 3090; A6000 retains ranks
0-3 and 4090-2 retains ranks 6-7. The replacement node needed the same isolated
580.159.03 NVIDIA compatibility libraries already validated on 4090-2. All
eight ranks crossed the barrier and entered `crn`; no subsequent P4 variant
was started.

The 03:46 attempt-2 snapshot preserves four completed domains while
`deepfake` continues. Macro Acc/AP are 97.2057%/97.9478%. Against failed
attempt 1 on exactly those domains, the replacement run differs by only
-0.0141/-0.0080 percentage points; the small hardware-dependent drift is kept
without selecting either attempt. Relative to P1's two-step run on the same
domains, one step is 0.0761 points lower in Acc and 0.8734 points higher in AP.

Throughput is stable at 5.50-5.51 images/s, bottleneck-rank latency is
1.45-1.46 s/image, and per-rank allocation/reservation remain
8,437.55/8,776 MiB. Physical peaks are 37,179/23,731/19,162 MiB on
A6000/4090-1/4090-2. The 4090-1 figure includes 5,277 MiB of stale driver
accounting present before launch, so rank-local PyTorch memory is the valid
cross-run comparison. Eight-rank logs remain free of runtime errors.

The 04:46 snapshot reaches ten completed domains and enters `imle`. Macro
Acc/AP are 95.1660%/98.3126%. Against failed attempt 1 on exactly those ten
domains, Acc changes by +0.0141 percentage points and AP by -0.0007 points;
the replacement run is effectively identical rather than a selected retry.
Against P1's two-step result on those domains, one step is +0.0662 points in
Acc and +0.5250 points in AP. This remains provisional until all domains
complete.

The weak `guided` result is preserved at 72.9000% Acc / 95.9346% AP, including
46.9000% fake accuracy. Throughput remains 5.44-5.51 images/s, rank-local peak
allocation remains 8,437.55 MiB, all three physical-memory peaks are unchanged,
and all eight rank logs remain free of execution errors.

The 05:48 snapshot reaches fourteen completed domains and continues into
`progan`. Macro Acc/AP are 95.6894%/98.3107%. Against P1's two-step run on the
same domains, one step changes Acc by only +0.0141 percentage points while AP
is +0.5465 points; real/fake accuracy change by -0.0521/+0.0803 points. The
newly completed `imle` result is retained at 91.9408% Acc / 94.9137% AP, with
83.8866% real accuracy and 100% fake accuracy.

The extended run remains stable at 5.44-5.51 images/s and 1.45-1.47 s
bottleneck-rank latency per image. Rank-local and physical-GPU memory peaks are
unchanged, all fourteen prediction files cover their complete unique sample
sets, and no execution error appears in any of the eight rank logs.

Attempt 2 then completed `progan` at 100% Acc/AP, bringing its accepted total
to fifteen domains and its partial macro Acc/AP to 95.9767%/98.4234%. During
the following `san` domain, 4090-1 became unreachable over both its LAN and
Tailscale addresses. A6000 and 4090-2 independently reported an incomplete ARP
neighbor and no route to 4090-1. Existing NCCL sockets remained stale.

Rank 0 completed its local `san` loop, but the eight-rank gather did not finish
and no `san` prediction JSON exists, so `san` is explicitly rejected. The six
reachable ranks remained alive without a logged runtime error at 06:48; their
100% GPU readings are not treated as progress because NCCL wait kernels can
remain active. The processes are left intact through the configured 7,200 s
collective timeout in case the link recovers. If it does not, the final timeout
state will be archived and all nineteen domains will be restarted rather than
splicing predictions from a new RNG stream.

A non-destructive attempt-3 fallback is now prepared. The 3090 GPU and healthy
system disk remain online, while its failed `/data` NVMe still reports a 0-byte
device and an ext4 `shutdown` mount and is not read. Instead, a user-local
SSHFS 3.7.3 binary mounts the complete verified UFD Arrow copy from 3070x2
read-only over the LAN. The real Arrow loader preflight passes all 19 domains,
88,353 samples, label counts, row mappings, and first-image decoding without
writing to either dataset.

If attempt 2 reaches its collective timeout, attempt 3 can therefore restore
the original A6000 4 + 3090 2 + 4090-2 2 GPU layout without waiting for the
offline 4090-1 or touching the failed NVMe. The remote storage backend will be
recorded explicitly; the A6000 remains the four-rank timing bottleneck, but the
final profile must still verify that SSHFS does not move the bottleneck to the
3090 ranks.

Attempt 2 reached the expected NCCL watchdog failure at 08:08-08:09 CST.
Ranks 1-3 and 6-7 independently report `ALLREDUCE` sequence 193 exceeding the
configured 7,200 s timeout; rank 0 reports the resulting remote-peer error.
All reachable processes exited and both A6000 and 4090-2 returned to 0% GPU.
The final logs and monitors are archived, while `san` remains rejected and the
fifteen earlier prediction files remain the only accepted partial results.

Attempt 3 started from seed 100 at 08:23 CST after all three hosts passed the
real 32-view Arrow creator preflight. It restores the original A6000 4 + 3090
2 + 4090-2 2 GPU layout. The 3090 reads the verified 3070x2 Arrow copy through
the read-only SSHFS mount and never accesses its failed `/data` filesystem.
All runtime and launcher hashes match, all eight ranks crossed the barrier,
rank 0 entered `crn`, and the initial memory/utilization readings are
37,179/18,652/19,162 MiB at 100%. No later variant has been started.

The 09:19 attempt-3 snapshot preserves `crn`, `cyclegan`, and `dalle` while
`biggan` continues. Their macro Acc/AP are 96.6717%/97.7579%. Against attempts
1 and 2 on exactly these domains, attempt 3 differs by at most 0.0131
percentage points in Acc and 0.0100 points in AP, so the SSHFS recovery has not
introduced a meaningful prediction shift.

Throughput is 5.40-5.43 images/s and bottleneck-rank latency is 1.47-1.48 s per
image. The critical rank is on A6000 for every completed domain; neither 3090
SSHFS rank determines wall time. Rank-local allocation remains 8,437.55 MiB,
host peaks remain 37,179/18,652/19,162 MiB, the SSHFS mount remains live, and
all eight rank logs are free of execution errors.
