# UNVALIDATED Step5 compact-summary candidate

Status: design reviewed; implementation incomplete; not hardware validated. This patch is archived on the requested branch but is NOT applied to production source. No service/image/runtime behavior changes merely by syncing it.

Base production paths: vLLM-Ascend143f3649d8181edae635091c5fe809eb51ff992c (later commits modify CI only), vLLMca5e43ddcd5b04c91e3de451702eecc1632faea8.

The candidate separates the 4KiB summary backing from the homogeneous96KiB main/SWA/ring backing, shares one global block-ID namespace and charges BOTH allocations through the builder. It does not disable MTP, relax null/zero/descriptor guards or use an unsafe heterogeneous layer-major overlay.

Do not apply or build for production before host isolation/descriptor/rank-shrink/null/zero/COW tests and isolated four-node MTP numerical and memory acceptance. The new builder and allocator must be paired; worker profile budget scaling has not been fully integrated. Prefix caching, multimodal, offload and PP/PCP/DCP paths are intentionally outside the candidate scope. BS128 is not validated by this patch.

The patch is a development snapshot, not a ready-to-merge performance fix.
