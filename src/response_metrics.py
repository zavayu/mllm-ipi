"""Explicit response metrics for controlled target-output experiments.

Exact matching strips leading/trailing whitespace only. Case, punctuation,
quotes, and additional commentary remain significant.
"""


def response_metrics(expected_response: str, response: str) -> dict[str, bool]:
    if not expected_response.strip():
        raise ValueError("expected_response cannot be blank")
    return {
        "target_containment": expected_response.casefold() in response.casefold(),
        "exact_target_response": expected_response.strip() == response.strip(),
    }
