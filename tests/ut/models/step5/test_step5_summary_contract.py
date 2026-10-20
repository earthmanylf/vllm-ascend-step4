"""Step5 summary-cache contracts (TailRing architecture).

PLACEHOLDER - to be written by the L6 TailRing rewrite plan.

The SWF-journal sources (test_summary_cache_contract.py,
test_summary_update_dispatch_contract.py, test_summary_scratch_contract.py)
test the request-journal undo log, which vLLM main removed in favour of
Step5CSATailRing. They must be rewritten against main's
Step5SparseSummaryCache(mean) / update_decode / update_verify /
update_prefill API, not copied.

See design spec section 6.
"""
