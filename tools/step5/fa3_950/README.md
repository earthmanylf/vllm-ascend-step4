# Retained FA3 950 native ABI repair

The patch is against the exact native source retained in the immutable base
image recorded by `source-identity.json`. Its source hashes, rather than a
potentially different checkout with the same family label, are authoritative.

It restores the two-boolean FAInfer specialization, publishes the compiled
ABI version, and passes independent uint64 K/V block strides through the host
launch and paged QK/PV address calculation. The required scheduler-capacity
backport and its dependent source changes are included in this previously
compiled patch; it is not a patch against an arbitrary current Flash main.

```bash
python tools/step5/fa3_950/apply.py --source /path/to/exact/native/source
# Build the patched source on Ascend 950 with formal CANN 9.2.
python tools/step5/fa3_950/check_library.py /path/to/compiled/native.so
```

`apply.py` preflights every source hash and a patch dry run. Applying twice
or applying to source drift is an error. The checker requires the exact
BF16 paged specialization, including both booleans, all pointer arguments
and the final `ffjjmm` argument signature on aarch64. Searching only for
`FAInfer`, a partial prefix, or the old `ffjj` signature is insufficient.

The native patch is source-controlled here because the available native
source is an exported tree, not an independently verifiable Git worktree.
No synthetic Flash Git commit is claimed. No native binary is committed.

Library symbols are compilation evidence only. The final BLNHC device probe
must show matching kernel events and numerical results for the actual loaded
library; complete model/MTP service acceptance is a separate requirement.
