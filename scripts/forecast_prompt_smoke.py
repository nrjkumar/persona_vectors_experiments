#!/usr/bin/env python3
"""Run three resolved ForecastBench questions through the structured prompt."""

import json
import os
import tempfile
import urllib.request
from pathlib import Path

# eval.eval_persona currently initializes credentials at import time, although
# this smoke test uses neither service. Remove the placeholders before loading
# the public Hugging Face model so they are not sent as credentials.
os.environ.setdefault("OPENAI_API_KEY", "not-used-by-forecast-smoke")
os.environ.setdefault("HF_TOKEN", "not-used-by-forecast-smoke")

import torch

from eval.eval_persona import (
    ForecastBenchExample,
    load_forecast_bench_dataset,
    parse_forecast_response,
)
from eval.model_utils import load_model

os.environ.pop("HF_TOKEN", None)

DATASET_BASE = (
    "https://raw.githubusercontent.com/forecastingresearch/"
    "forecastbench-datasets/main/datasets"
)


def download_json(relative_path):
    with urllib.request.urlopen(f"{DATASET_BASE}/{relative_path}") as response:
        return json.load(response)


def generate(model, tokenizer, conversation, max_new_tokens=220):
    prompt = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    answer = tokenizer.decode(
        output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return prompt, answer


def main():
    questions = download_json("question_sets/2024-07-21-human.json")
    resolutions = download_json("resolution_sets/2024-07-21_resolution_set.json")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        question_path = temp_path / "questions.json"
        resolution_path = temp_path / "resolutions.json"
        question_path.write_text(json.dumps(questions), encoding="utf-8")
        resolution_path.write_text(json.dumps(resolutions), encoding="utf-8")
        records = load_forecast_bench_dataset(
            question_path,
            resolution_path=resolution_path,
            require_resolution=True,
        )[:3]

    if len(records) != 3:
        raise RuntimeError(f"Expected exactly 3 resolved questions, got {len(records)}")

    model_name = os.environ.get(
        "FORECAST_SMOKE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"
    )
    print(f"model={model_name}")
    print(f"device={torch.cuda.get_device_name(0)}")
    model, tokenizer = load_model(model_name)

    parsed_count = 0
    for index, record in enumerate(records, 1):
        example = ForecastBenchExample(record)
        conversation = [{"role": "user", "content": example.prompt}]
        prompt, answer = generate(model, tokenizer, conversation)
        probability, error = parse_forecast_response(answer, require_all_fields=True)
        attempts = 1

        if error is not None:
            attempts += 1
            prompt, answer = generate(
                model,
                tokenizer,
                example._retry_conversation(answer, error),
            )
            probability, error = parse_forecast_response(
                answer, require_all_fields=True
            )

        parsed_count += error is None
        row = example.build_rows(
            [prompt], [answer], [probability], [error], [attempts]
        )[0]
        print(
            "FORECAST_RESULT "
            + json.dumps(
                {
                    "index": index,
                    "question": record["prompt"],
                    "forecast_due_date": record["forecast_due_date"],
                    "gold": record["gold"],
                    "answer": answer,
                    "predicted_probability": row["predicted_probability"],
                    "brier_score": row["brier_score"],
                    "parse_attempts": attempts,
                    "parse_error": error,
                },
                ensure_ascii=False,
            )
        )

    print(f"parsed={parsed_count}/{len(records)}")
    if parsed_count != len(records):
        raise RuntimeError("At least one structured forecast remained unparseable")


if __name__ == "__main__":
    main()
