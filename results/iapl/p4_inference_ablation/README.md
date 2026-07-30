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

The 09:48 snapshot adds `biggan` and `deepfake`; `gaugan` is now running. The
five-domain macro Acc/AP are 96.9494%/97.8619%. Relative to attempts 1 and 2 on
the same domains, the maximum difference is 0.0166 percentage points in Acc
and 0.0232 points in AP. This remains consistent with ordinary runtime
variation rather than a storage-backend regression.

All five completed domains still place their critical path on A6000. Cluster
throughput is 5.40-5.46 images/s, bottleneck latency is 1.47-1.48 s per image,
the 3090 read-only SSHFS mount is live, and all eight rank logs remain free of
execution errors.

The 10:30 snapshot adds `gaugan`, `glide_50_27`, and `glide_100_10`; the next
domain is `glide_100_27`. Across the first eight domains, one-step TTA reaches
97.5734% Acc / 98.4958% AP. It is only 0.0264 percentage points below the P1
two-step protocol in Acc while gaining 0.5547 AP points on the same domains.

Attempt-3 reproducibility remains tight: against attempts 1 and 2, the maximum
eight-domain difference is 0.0279 Acc points and 0.0114 AP points. Every domain
still has an A6000 critical rank, throughput remains 5.40-5.47 images/s, the
SSHFS mount is live, and no execution error appears in any of the eight rank
logs.

Remote monitoring recovered at 11:16 without an experiment interruption. The
11:21 snapshot adds `glide_100_27`, `guided`, and `imle`; `ldm_100` is running.
The eleven-domain one-step macro is 94.8494% Acc / 98.0033% AP. Against P1's
two-step protocol on the same domains, Acc differs by only -0.0009 percentage
points while AP improves by 0.6836 points.

Attempt 3 remains reproducible after passing attempt 1's failure boundary. Its
shared first ten domains differ from attempt 1 by at most 0.0133 Acc points and
0.0082 AP points, while all eleven domains differ from attempt 2 by 0.0235 and
0.0004 points. All critical ranks remain on A6000, all GPUs are active, the
SSHFS mount remains live, and all eight rank logs remain error-free.

At 12:03, attempt 3 completed and gathered `san`, thereby passing attempt 2's
4090-1 outage boundary rather than merely repeating rank 0's local loop. The
12:05 snapshot also includes `ldm_100`, `ldm_200`, `ldm_200_cfg`, `progan`, and
`seeingdark`; only `stargan` and `stylegan` remain.

The seventeen-domain macro is 95.4605% Acc / 97.7813% AP. Against P1 two-step
on the same domains, one-step is 0.0113 Acc points and 0.6171 AP points higher.
Against attempt 2's shared first fifteen domains, the differences are 0.0205
Acc points and 0.0008 AP points. All critical ranks remain on A6000; the three
hosts, SSHFS mount, prediction counts, and eight-rank error audit remain clean.

At 12:17, `stargan` completed and the final `stylegan` domain started. The
eighteen-domain macro is 95.5460% Acc / 97.7375% AP. One-step is 0.0135 Acc
points and 0.6550 AP points above P1 two-step on the same domains. The final
domain started with all three GPUs active; the 3090 SSHFS mount and all eight
rank logs remain healthy.

`steps1` completed all 19 domains at 12:53:06 after 4h29m48s. Final Acc/AP are
95.5064%/97.8391%, with 96.2234% real and 94.7879% fake Accuracy. Relative to
the paper's Table 9 T=1 row, this is +1.0364 Acc points and -0.6909 AP points.
Relative to the local P1 two-step run, it is effectively tied in Acc
(+0.0141 points) and improves AP by 0.6217 points.

The sample-level P1 comparison uses the same 88,353 unique indices and labels.
It finds 431 threshold disagreements (0.4878%), 50,043 exactly equal
probabilities, and a weighted mean absolute probability difference of 0.00558.
The full run has no rank error, all processes exited, and all three GPUs
returned to idle. Both failed attempts remain preserved. `ois_off` is next.

After the final `steps1` audit and push, `ois_off` passed fresh preflight on all
three hosts and launched at 12:59:58 on port 29662. This changes only OIS from
true to false; views, TTA steps, selected views, entropy, seed, rank layout,
domain order, checkpoint, and data remain fixed. All eight ranks crossed the
barrier, rank 0 entered `crn`, and initial A6000/3090/4090-2 memory is
37,175/18,652/19,154 MiB at 100% utilization. The 3090 read-only SSHFS mount
remains live and no later variant has started.

The 14:17 snapshot preserves the first completed `ois_off` domain while
`cyclegan` continues. On `crn`, disabling OIS gives 92.1758% Acc and 98.8495%
AP. Relative to the local two-step P1 result on the same 12,764 unique samples,
Acc falls 0.3603 percentage points while AP rises 6.0744 points. This large
single-domain AP change is provisional and is preserved without restart or
selection until all nineteen domains determine the macro result.

The profiled `crn` throughput is 3.0001 images/s and bottleneck-rank latency is
2.666 s/image. Per-rank allocation/reservation is 8,437.54/8,776 MiB and host
peaks are 37,175/18,652/19,154 MiB. All critical ranks are on A6000, the 3090
SSHFS mount remains live, all prediction counts match distributed padding and
the unique dataset size, and no execution error appears in any rank log.

The 14:47 snapshot extends `ois_off` to three completed domains while `biggan`
continues. The provisional macro Acc/AP are 96.4887%/99.2424%. Against P1 on
the same `crn`, `cyclegan`, and `dalle` samples, disabling OIS is 0.3038 Acc
points lower and 2.4628 AP points higher. The AP gain is no longer confined to
one stored prediction file, but the remaining sixteen domains are still
required before attributing it to OIS.

All three domains run at 2.9985-3.0001 images/s with 2.661-2.668 s
bottleneck-rank latency. Memory peaks are unchanged and every critical rank is
on A6000, so the 3090 SSHFS data path is not limiting the run. The three-domain
snapshot covers 17,406 unique samples with matching indices and labels, and all
eight rank logs remain free of execution errors.

The 15:47 snapshot preserves five completed domains while `gaugan` continues.
The provisional macro Acc/AP are 96.7359%/98.7912%. Against P1 on those five
domains, disabling OIS is 0.2797 Acc points lower and 1.6887 AP points higher;
against `steps1`, it is 0.2135 Acc points lower and 0.9292 AP points higher.
The AP advantage is narrowing as more domains arrive, so the run remains
unselected and unchanged rather than treating the early `crn` gain as final.

All five domains remain tightly grouped at 2.9985-3.0001 images/s and
2.661-2.668 s bottleneck-rank latency. Rank-local and physical-GPU memory peaks
are unchanged, every critical rank is on A6000, the SSHFS mount is healthy,
and 26,811 unique samples plus all eight logs pass the snapshot audit.

The 17:17 snapshot reaches ten completed domains while `imle` continues. The
ten-domain macro Acc/AP are 94.8979%/98.4421%. Against P1 on the same domains,
disabling OIS is 0.2019 Acc points lower and 0.6545 AP points higher. Against
`steps1`, it is 0.2408 Acc points lower and only 0.1369 AP points higher, so the
large early AP advantage has mostly disappeared.

The weak `guided` result is retained without restart: 71.9500% Acc and
91.7111% AP, including 45.1000% fake accuracy. Its AP is 3.6506 points below
P1, which is the clearest negative OIS-off result so far. The transient
`gaugan` slowdown expands the throughput range to 2.9078-3.0001 images/s, but
all critical ranks remain on A6000 and rank-local memory stays unchanged. The
A6000 physical monitor briefly reaches 38,778 MiB while the other host peaks
remain unchanged; all 44,811 unique samples, SSHFS state, and rank logs pass
the audit.

The 19:17 snapshot reaches fourteen completed domains while `progan`
continues. Macro Acc/AP recover to 95.5077%/98.7569%. Against P1 on the same
domains, disabling OIS is 0.1676 Acc points lower and 0.9926 AP points higher;
against `steps1`, the differences are -0.1597 Acc and +0.4453 AP points.

The recovery is driven in part by `imle`, where OIS-off gives 92.1288% Acc and
98.8687% AP. Relative to P1, its Acc is 0.2271 points lower but AP is 6.2283
points higher, the opposite AP direction from the preserved weak `guided`
result. Both outcomes remain in the run. Throughput spans 2.9078-3.0047
images/s after the earlier transient slowdown, all critical ranks remain on
A6000, and 63,575 unique samples, the SSHFS mount, memory records, and all rank
logs pass the audit.

The 19:47 snapshot reaches seventeen completed domains and continues into
`stargan`. Macro Acc/AP are now 94.5387%/98.2081%. Relative to P1 on the same
domains, OIS-off is 0.9104 Acc points lower despite AP being 1.0438 points
higher. Real accuracy rises 0.6591 points while fake accuracy falls 2.4900
points, exposing a substantial threshold tradeoff rather than a uniform gain.

The main cause is a preserved `san` collapse: 74.7727% Acc and 89.1415% AP,
with fake accuracy falling to 52.5114%. Relative to P1, this is -18.4091 Acc,
-6.5173 AP, and -37.4429 fake-accuracy points on the complete 438-image domain.
The distributed gather completed successfully on the healthy A6000/3090/
4090-2 layout, so this is an accepted model result rather than a repeat of the
earlier 4090-1 outage during attempt 2. No run is restarted or selected away.
All 72,373 unique samples, profiles, memory monitors, SSHFS state, and rank logs
pass the audit.

After `stargan` completed and `stylegan` started, another user's process began
sharing the A6000 at 19:59:32 and allocated 2,976 MiB at 20:00:44. Raw A6000
host-memory samples after that boundary are therefore retained but not
attributed wholly to P4; the four P4 ranks still sum to about 37,144 MiB and
their internal CUDA allocation remains valid. `stargan` averaged 2.6633 s per
iteration, within the earlier range, but its latter portion and `stylegan`
timing are conservatively flagged as potentially shared-GPU affected. The
other user's process is not modified.

A single Tailscale SSH probe to 4090-2 also timed out at 20:18, but retry
succeeded seconds later. Both peers reached its LAN address with 0% packet
loss, NCCL sockets remained established, the GPU stayed at 100% utilization,
and no rank error or experiment interruption occurred. This is recorded as a
control-plane observation rather than an experiment failure.

`ois_off` completed all 19 domains at 21:14:52 CST after 8h14m54s. Final
Acc/AP are 94.6641%/98.2870%, with 96.8499% real and 92.4680% fake Accuracy.
Relative to the matched local P1 OIS-on run, disabling OIS lowers Acc by
0.8282 points and fake accuracy by 2.2402 points, while raising AP by 1.0696
points and real accuracy by 0.5749 points. The 1,092 threshold disagreements
on identical indices and labels confirm that OIS primarily changes calibration
rather than uniformly improving ranking.

The severe `san` result remains the clearest evidence for OIS: disabling it
costs 18.4091 Acc points, 6.5173 AP points, and 37.4429 fake-accuracy points on
that domain. OIS-off improves other domains, notably `seeingdark`, `imle`, and
`crn`, so the accepted conclusion is a tradeoff rather than universal module
dominance. The run processes all 88,353 unique samples, all rank processes
exit cleanly, and no execution error appears in any rank log.

The final critical-path sum is 29,602.64 s, or 2.9846 unique images/s and
2,679.70 ms weighted bottleneck-rank latency. Per-rank peak allocation and
reservation remain 8,437.54/8,776 MiB. Raw physical peaks are
40,166/18,652/19,154 MiB, but the A6000 value includes the documented shared
2,976 MiB process; the clean P4 plateau is 37,175 MiB. Every critical rank is
on A6000. This completes ordered variant 4; `select2` is next and will not
start until this audit is committed and pushed.

After commit `4e5d40c` was pushed, `select2` passed the three-host code, NCCL,
checkpoint, empty-output, port, and real 12,764-row `crn` Arrow preflight. The
3090 read-only SSHFS backend and 4090-2 compatibility libraries remain healthy,
and those two GPUs are idle. Launch is intentionally held at the final clean-GPU
gate because another user's A6000 training process is now actively consuming
2,986 MiB and 46-49% sampled SM utilization. Starting the four A6000 ranks would
contaminate the latency and host-memory curves that P4 is meant to measure. The
other process is not modified; `select2` remains unlaunched and will be retried
when the required A6000 is clean.

At 03:17:47 CST the shared process had exited without intervention and A6000
returned to 17 MiB and 0% utilization. Fresh three-host output, port, GPU, and
read-only SSHFS checks passed, so `select2` started worker-first at 03:18:49
and completed the eight-rank launch at 03:19:19 on port 29663. All ranks crossed
the barrier, rank 0 entered `crn`, and initial A6000/3090/4090-2 memory reached
37,167/18,652/19,158 MiB at 100% utilization. The first rank-0 iteration took
7.3004 s during warm-up, rank-local peak allocation reached 8,437 MiB, and no
error appears in any of the eight logs. Later variants remain unlaunched.

The 04:47 snapshot preserves the first two completed `select2` domains while
the run enters `dalle`. Partial Acc/AP are 94.1064%/94.7412%, respectively
1.5573/0.6449 points below the matched P1 baseline domains. The difference is
concentrated in `crn`, where selecting two views gives 89.9123% Acc and
92.0658% AP, down 2.6237/0.7093 points; fake accuracy remains 100%, while real
accuracy falls by 5.2459 points. All 15,406 unique indices and labels match P1,
with 360 threshold disagreements. Both completed domains run at about 2.95
unique images/s, every critical rank is on A6000, and the eight-rank, SSHFS,
GPU-memory, and log audits remain healthy. The weak early result is retained
without restart or selection.

The 06:19 snapshot extends `select2` to five completed domains while `gaugan`
runs. Partial Acc/AP recover to 96.3388%/96.8035%, but remain 0.6768/0.2990
points below P1 on the same domains. Real accuracy is 1.4599 points lower while
fake accuracy is 0.1071 points higher. The gap remains dominated by `crn`;
`dalle` matches P1 accuracy and adds 0.0999 AP points, while `deepfake` adds
0.0555 Acc points. All 26,811 unique samples match P1 indices and labels, with
398 threshold disagreements. Throughput stays within 2.9520-2.9632 images/s,
all critical ranks remain on A6000, and no host, SSHFS, memory, or rank-log
failure is detected.

The 07:18 snapshot reaches eight completed `select2` domains and enters
`glide_100_27`. Partial Acc/AP are 97.2005%/97.7374%, now 0.3993/0.2037 points
below matched P1. Real accuracy remains 1.0149 points lower, while fake accuracy
is 0.2169 points higher. Of the new domains, `glide_100_10` improves Acc/AP by
0.3500/0.1251 points and `glide_50_27` improves Acc by 0.0500 points, partially
offsetting the persistent `crn` loss. All 40,811 sample indices and labels match
P1. Throughput remains 2.9345-2.9632 images/s, all critical ranks remain on
A6000, and the distributed, storage, GPU, and log audits remain clean.

The 07:48 snapshot reaches ten completed `select2` domains and enters `imle`.
Partial Acc/AP are 95.0004%/97.6275%, only 0.0994/0.1602 points below matched
P1. The apparent Acc recovery is driven by `guided`: its absolute result is
still weak at 74.40% Acc and 95.40% AP, but it improves P1 by 2.00 Acc points
and 4.30 fake-accuracy points. Real accuracy remains 0.8519 points below P1,
while fake accuracy is 0.6535 points higher. All 44,811 indices and labels
match, and the throughput, rank-local memory, host memory, SSHFS, and eight-log
audits remain stable.

The 09:48 snapshot reaches fourteen completed `select2` domains and runs
`progan`. Partial Acc/AP are 95.4638%/97.5420%, now 0.2115/0.2223 points below
matched P1. The renewed gap comes from `imle`, which records 89.8888% Acc and
91.3541% AP, down 2.4671/1.2863 points and 4.9327 real-accuracy points. The
three completed LDM domains remain near P1 and slightly improve Acc. All 63,575
indices and labels match. Throughput remains 2.9319-2.9632 images/s; all
critical ranks are on A6000, and the distributed, storage, memory, and log
audits stay clean.

The 10:18 snapshot reaches seventeen completed `select2` domains and runs
`stargan`. Partial Acc/AP are 95.0484%/96.7281%, 0.4007/0.4361 points below
matched P1. `progan` is effectively unchanged and `san` loses only 0.2273 Acc
points, but `seeingdark` falls to 86.3889% Acc and 84.0614% AP, down
3.6111/3.3731 points and 7.2222 real-accuracy points. This weak result is kept.
All 72,373 indices and labels match P1; the small-domain overhead expands the
throughput range to 2.7465-2.9632 images/s, while memory, SSHFS, GPU, and all
eight rank logs remain healthy.

`select2` completed all 19 domains at 11:39:06 CST after 8h19m47s. Final
Acc/AP are 95.1162%/96.8206%, with 95.0867% real and 95.1451% fake Accuracy.
Against the matched P1 six-view reference, selecting two views loses
0.3761 Acc points, 0.3968 AP points, and 1.1884 real-accuracy points while
gaining 0.4369 fake-accuracy points. The largest losses are `seeingdark`,
`crn`, and `imle`; `guided` instead gains 2.00 Acc points. All 88,353 unique
indices and labels match P1, with 1,015 threshold disagreements and a 0.01151
weighted mean absolute probability delta.

The final critical-path sum is 29,925.58 s, or 2.9524 unique images/s and
2,708.93 ms weighted bottleneck-rank latency. Per-rank peak allocation and
reservation are 8,437.55/8,776 MiB, and clean physical peaks are
37,167/18,652/19,158 MiB on A6000/3090/4090-2. All eight rank logs and three
launcher logs are error-free, the 3090 read-only SSHFS remained mounted, and
all processes exited. The accepted conclusion is that two selected views are
not performance-neutral: the six-view default remains the better balanced
operating point. This completes ordered variant 5; `select4` is next and will
not start until this final audit is committed and pushed.

After commit `f3f1506` was pushed, `select4` passed fresh empty-output, code,
NCCL, checkpoint, port, GPU, and real 12,764-row `crn` Arrow checks on all
three hosts. The 3090 read-only SSHFS remained healthy and the failed `/data`
device was not accessed; 4090-2 used its audited driver compatibility library.
Ranks 4/5 started at 11:58:42, ranks 6/7 at 11:59:00, and ranks 0-3 completed
the worker-first launch at 11:59:15 on port 29664. All eight ranks crossed the
barrier and entered `crn`; rank 0's first iteration took 4.9484 s. Initial
A6000/3090/4090-2 memory is 37,175/18,652/19,162 MiB at 100% utilization,
with no startup error. Later variants remain unlaunched.

The 13:18 snapshot preserves the first completed `select4` domain while the
run enters `cyclegan`. On `crn`, selecting four views gives 91.6432% Acc and
92.1722% AP. This is 0.8929/0.6030 points below P1's six-view reference, but
1.7309/0.1064 points above `select2`; the real-accuracy deltas are -1.7852 and
+3.4607 points, respectively, while fake accuracy remains 100%. All 12,764
unique indices and labels match both references. The P1 comparison has 154
threshold disagreements and a 0.01240 mean absolute probability delta.
`crn` runs at 2.9483 unique images/s with 2,712.53 ms bottleneck-rank latency.
All three GPUs remain fully utilized, 3090's read-only SSHFS is healthy, and
the eight rank logs contain no execution error. This provisional degradation
is retained without restart or selection.

The 13:48 snapshot extends `select4` to `crn`, `cyclegan`, and `dalle` while
`biggan` runs. Partial Acc/AP are 96.3738%/96.5628%, 0.4187/0.2168 points
below matched P1 but 0.6195/0.1798 points above `select2`. Real accuracy is
0.7371 points below P1 and 1.4381 points above `select2`; fake accuracy is
0.10/0.20 points lower. `cyclegan` recovers 0.3776 Acc points over `select2`,
whereas `dalle` loses 0.25 Acc points against both references. All 17,406
unique sample indices and labels match both controls. The P1 comparison has
166 threshold disagreements and a 0.00981 weighted probability MAD. Domain
throughput remains 2.9472-2.9565 images/s, all critical ranks remain on A6000,
and the three-host storage, GPU, memory, and log audits stay clean.

The 14:48 snapshot reaches five completed `select4` domains and runs `gaugan`.
Partial Acc/AP are 96.7531%/96.9622%, 0.2625/0.1403 points below matched P1
but 0.4143/0.1586 points above `select2`. Real accuracy is 0.4796 points below
P1 and 0.9802 points above `select2`; fake accuracy is lower by 0.0452/0.1522
points. `biggan` is nearly identical to P1 and gains 0.25 Acc points over
`select2`; `deepfake` differs by less than 0.04 Acc points from either control.
All 26,811 sample indices and labels match both references, with 180 P1
threshold disagreements and a 0.00703 weighted probability MAD. Throughput,
rank-local memory, physical memory, SSHFS, and all eight logs remain stable.

The 15:48 snapshot reaches seven completed `select4` domains and enters
`glide_100_10`. Partial Acc/AP rise to 97.3408%/97.5981%, 0.1875/0.1194
points below matched P1 and 0.3188/0.1312 points above `select2`. Real accuracy
is 0.3712 points below P1 but 0.7745 points above `select2`; fake accuracy is
essentially tied with P1 and 0.1373 points below `select2`. `gaugan` recovers
0.16 Acc points over `select2`, and `glide_50_27` matches its Acc while adding
0.1055 AP points. All 38,811 sample indices and labels match both references.
The P1 comparison has 194 threshold disagreements and a 0.00527 weighted MAD.
Throughput, memory, the read-only SSHFS, and every rank log remain healthy.

The 16:18 snapshot reaches ten completed `select4` domains and enters `imle`.
Partial Acc/AP are 95.0385%/97.6713%, 0.0613/0.1163 points below matched P1
and 0.0382/0.0439 points above `select2`. `guided` remains weak at 73.00% Acc
and 94.96% AP: it gains 0.60 Acc points over P1 but loses 1.40 points to
`select2`. All 44,811 sample indices and labels match both references, with
234 P1 threshold disagreements and a 0.00547 weighted probability MAD.

The raw A6000 monitor briefly rose from the 37,175 MiB P4 plateau to 39,240
MiB during `glide_100_27` from 15:55:43 to 15:55:54, then returned in one
sample. Rank-local allocation/reservation stayed at 8,437.55/8,776 MiB and no
log error occurred. A later process snapshot shows only the four P4 ranks at
37,175 MiB. Because no process snapshot exists during the event, the extra
2,065 MiB is preserved as an unattributed transient rather than assigned to
P4 or another user. `glide_100_27` timing is not an outlier but is
conservatively flagged; the run continues without restart.

During `imle`, the raw A6000 monitor recorded three later shared-memory
intervals. The first was already at 39,448 MiB when observed at 16:57:50,
peaked at 41,788 MiB, and cleared at 16:59:07; the second held 41,398 MiB
from 16:59:43 to 17:09:14. Neither interval has a contemporaneous process
snapshot and both remain unattributed. The third began at 17:09:19 and a
17:21:48 snapshot identified another user's `train_sharepara_moe_0406_loss.py`
process holding 4,530 MiB alongside the four unchanged P4 ranks. It was
observed only and not modified. Predictions remain valid, but `imle` timing
is flagged as shared-GPU contaminated. The raw 41,788 MiB peak, clean 37,175
MiB P4 plateau, and rank-local CUDA allocation are reported separately; the
run continues without restart.

The 17:49 snapshot reaches thirteen completed domains and runs `ldm_200_cfg`.
Partial Acc/AP are 95.4505%/97.5280%, 0.0882/0.1257 points below matched P1
but 0.1626/0.1025 points above `select2`. `imle` is 91.7215% Acc and 92.1722%
AP: it is 0.6344/0.4683 points below P1 but recovers 1.8327/0.8180 points over
`select2`. `ldm_100` matches P1 Accuracy, and `ldm_200` is 0.10 points above
P1. All 61,575 unique indices and labels match both references, with 365 P1
threshold disagreements and a 0.00624 weighted probability delta.

The external A6000 process remained present through the new domains. At
17:49:18 it began changing its allocation and by 17:52:38 held 1,554 MiB,
with raw host use at 38,734 MiB; the four P4 ranks remained at their unchanged
37,144 MiB sum. Correspondingly, `imle`, `ldm_100`, and `ldm_200` throughput
is 2.9303/2.9163/2.9151 images/s versus the earlier 2.9472-2.9601 range.
Their predictions are retained, but timing is explicitly shared-GPU
contaminated. The external process was not modified. 4090-2's default NVML
still has its known library mismatch, while the audited 580.159.03 library
reports 19,162 MiB at 100%; both ranks and all logs remain healthy.
At 18:00:01 the same external process had re-expanded to 4,514 MiB and raw
A6000 use returned to 41,694 MiB, so `ldm_200_cfg` timing is flagged as well.
