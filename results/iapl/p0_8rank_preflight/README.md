# P0 three-node eight-rank preflight

The A6000 host stores the synchronized UFD Arrow copy under the `20260717`
directory, while both 4090 hosts use the `20260716` directory. The directory
names differ, but all metadata files, all three Ojha shards, and the first,
middle, and last ForenSynths shards have matching SHA-256 hashes on all three
hosts.

The manual-rank launcher now chooses the first complete local UFD Arrow root
instead of assuming that every host uses the same dated directory name. The
single-GPU Arrow and ImageFolder controls continue to use each 4090 host's
existing `cl` environment. The eight-rank diagnostic uses the separate
`caid-gemini-compat` environment because multiple ranks share each physical GPU
and therefore require the explicitly validated NCCL 2.30 runtime.

The launcher's non-running preflight passed on all hosts for the intended rank
mapping: ranks 0-3 on A6000, ranks 4-5 on 4090-1, and ranks 6-7 on 4090-2.
Every host resolved its local dataset root, IAPL repository, model checkpoint,
OpenAI CLIP checkpoint, and NCCL 2.30.7 library successfully.
