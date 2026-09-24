# SPDX-License-Identifier: Apache-2.0


def pytest_addoption(parser):
    group = parser.getgroup("step5-pd")
    group.addoption("--step5-prefill-url", default=None)
    group.addoption("--step5-decode-url", default=None)
    group.addoption("--step5-tokenizer", default=None)
    group.addoption("--step5-served-model", default="step5")
