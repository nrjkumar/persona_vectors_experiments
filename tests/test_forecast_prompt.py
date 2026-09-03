import os

os.environ.setdefault("OPENAI_API_KEY", "unused-by-tests")
os.environ.setdefault("HF_TOKEN", "unused-by-tests")

from eval.eval_persona import build_forecast_bench_prompt, parse_forecast_response


def test_forecast_prompt_requests_rationale_before_probability_without_confidence() -> (
    None
):
    prompt = build_forecast_bench_prompt("Will the event happen?")

    rationale_position = prompt.index('"rationale"')
    probability_position = prompt.index('"probability"')

    assert rationale_position < probability_position
    assert '"confidence"' not in prompt


def test_strict_forecast_response_requires_rationale_and_probability() -> None:
    probability, error = parse_forecast_response(
        '{"rationale": "The evidence points upward.", "probability": 0.7}',
        require_all_fields=True,
    )

    assert probability == 0.7
    assert error is None

    probability, error = parse_forecast_response(
        '{"probability": 0.7}', require_all_fields=True
    )

    assert probability is None
    assert error == "the JSON object had no `rationale` field"


def test_strict_forecast_response_requires_rationale_before_probability() -> None:
    probability, error = parse_forecast_response(
        '{"probability": 0.7, "rationale": "The evidence points upward."}',
        require_all_fields=True,
    )

    assert probability is None
    assert error == "`rationale` must come before `probability`"


def test_strict_forecast_response_rejects_removed_confidence_field() -> None:
    probability, error = parse_forecast_response(
        '{"rationale": "The evidence points upward.", "probability": 0.7, '
        '"confidence": "high"}',
        require_all_fields=True,
    )

    assert probability is None
    assert error == "the JSON object must not include `confidence`"
