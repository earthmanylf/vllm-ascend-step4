# SPDX-License-Identifier: Apache-2.0
"""Compare real Step5 PD with local prefill on the same running D service."""

import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from transformers import AutoTokenizer


@pytest.fixture(scope="module")
def services(request):
    prefill = request.config.getoption("--step5-prefill-url")
    decode = request.config.getoption("--step5-decode-url")
    tokenizer = request.config.getoption("--step5-tokenizer")
    if not all((prefill, decode, tokenizer)):
        pytest.skip("Pass Step5 P/D URLs and tokenizer to run this external-service test.")
    tokens = AutoTokenizer.from_pretrained(tokenizer).encode(
        "Explain why paged attention must preserve both key and value tensors. ", add_special_tokens=False
    )
    assert tokens
    return prefill.rstrip("/"), decode.rstrip("/"), request.config.getoption("--step5-served-model"), tokens


def complete(url, payload):
    response = httpx.post(f"{url}/v1/completions", json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


def check_handoff(services, length):
    prefill, decode, model, seed_tokens = services
    prompt = (seed_tokens * ((length + len(seed_tokens) - 1) // len(seed_tokens)))[:length]
    payload = dict(model=model, prompt=prompt, temperature=0, max_tokens=16, ignore_eos=True, logprobs=1)
    # Distinct salts prevent the baseline from satisfying D's remote load
    # with its own prefix cache. P/D share a salt for the transferred request.
    pd_payload = dict(payload, cache_salt=uuid.uuid4().hex)
    produced = complete(prefill, dict(pd_payload, max_tokens=1, kv_transfer_params={"do_remote_decode": True}))
    metadata = produced["kv_transfer_params"]
    prefix_tokens = max(0, (length - 128) // 128 * 128)
    assert metadata["step5_pd_version"] == 1
    assert metadata["step5_prefix_tokens"] == prefix_tokens
    assert metadata["step5_original_prompt_tokens"] == length
    if prefix_tokens:
        assert any(metadata["remote_block_ids"])
    else:
        assert not any(metadata["remote_block_ids"])
    transferred = complete(decode, dict(pd_payload, kv_transfer_params=metadata))
    baseline = complete(decode, dict(payload, cache_salt=uuid.uuid4().hex))
    actual, expected = transferred["choices"][0], baseline["choices"][0]
    assert actual["text"] == expected["text"]
    assert actual["finish_reason"] == expected["finish_reason"]
    assert actual["logprobs"]["tokens"] == expected["logprobs"]["tokens"]
    assert actual["logprobs"]["token_logprobs"] == pytest.approx(expected["logprobs"]["token_logprobs"], abs=0.05)


@pytest.mark.parametrize("length", [7, 8, 9, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025, 8193])
def test_region_page_and_chunk_boundaries(services, length):
    check_handoff(services, length)


def test_concurrent_handoffs_and_block_reuse(services):
    with ThreadPoolExecutor(max_workers=4) as executor:
        for _ in executor.map(lambda length: check_handoff(services, length), [1025, 257, 2049, 513] * 4):
            pass
