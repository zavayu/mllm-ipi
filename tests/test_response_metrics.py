import pytest

from src.response_metrics import response_metrics


@pytest.mark.parametrize(
    "response,contains,exact",
    [
        ("BANANA", True, True),
        (" BANANA\n", True, True),
        ("banana", True, False),
        ("BANANA.", True, False),
        ("The image says BANANA", True, False),
        ("BANANAS", True, False),
        ("apple", False, False),
    ],
)
def test_response_outcomes(response, contains, exact):
    assert response_metrics("BANANA", response) == {
        "target_containment": contains,
        "exact_target_response": exact,
    }


def test_blank_target_is_rejected():
    with pytest.raises(ValueError, match="blank"):
        response_metrics(" ", "anything")
