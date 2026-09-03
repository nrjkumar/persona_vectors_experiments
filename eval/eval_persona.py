import asyncio
import json
import logging
import math
import os
import random
import re

import pandas as pd
import torch
from tqdm import tqdm, trange

from activation_steer import ActivationSteerer
from config import setup_credentials
from eval.model_utils import load_model, load_vllm_model
from eval.prompts import Prompts
from judge import OpenAiJudge, set_judge_requests_per_minute

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)

# Set up credentials and environment
config = setup_credentials()


def sample_steering(
    model,
    tokenizer,
    conversations,
    vector,
    layer,
    coef,
    bs=20,
    top_p=1,
    max_tokens=500,
    temperature=1,
    min_tokens=1,
    steering_type="response",
):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    prompts = []
    for messages in conversations:
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    outputs = []
    for i in trange(0, len(prompts), bs):
        batch = prompts[i : i + bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}
        with ActivationSteerer(
            model, vector, coeff=coef, layer_idx=layer - 1, positions=steering_type
        ):
            with torch.no_grad():
                output = model.generate(
                    **tokenized_batch,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    min_new_tokens=min_tokens,
                )
        prompt_len = tokenized_batch["input_ids"].shape[1]
        output = [
            tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in output
        ]
        outputs.extend(output)
    return prompts, outputs


def sample(
    model,
    tokenizer,
    conversations,
    top_p=1,
    max_tokens=500,
    temperature=1,
    min_tokens=1,
    lora_path=None,
):
    # vLLM is an optional GPU dependency. Import it only on the vLLM path so
    # prompt construction, parsing, scoring, and Transformers-based smoke tests
    # remain usable in the base environment.
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token],
        min_tokens=min_tokens,
    )

    texts = []
    for i, messages in enumerate(conversations):
        texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    generate_kwargs = {"sampling_params": sampling_params, "use_tqdm": True}
    if lora_path:
        from vllm.lora.request import LoRARequest

        completions = model.generate(
            texts,
            **generate_kwargs,
            lora_request=LoRARequest("default", 1, lora_path=lora_path),
        )
    else:
        completions = model.generate(texts, **generate_kwargs)
    answers = [completion.outputs[0].text for completion in completions]
    return texts, answers


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


# A question whose text still contains one of these after substitution would be
# shown to the model verbatim, which is what produced malformed prompts before.
FORECAST_PLACEHOLDER_RE = re.compile(r"\{[a-z_]+\}")

# `freeze_datetime_value` holds the market's own probability at freeze time. It
# is part of the official ForecastBench context, but it hands the model most of
# the answer on market-sourced questions, so it is opt-in.
FORECAST_FREEZE_VALUE_KEYS = (
    "freeze_datetime_value",
    "freeze_datetime_value_explanation",
)


def _id_key(value):
    """
    Make a ForecastBench id hashable.

    Combination questions carry a list of ids (5992 of 7561 rows in the
    2024-07-21 resolution set); single questions carry a string.
    """
    return tuple(value) if isinstance(value, list) else value


def _read_forecast_json(path, split="latest"):
    """Read a ForecastBench .json/.jsonl file, or find one inside a directory."""
    from pathlib import Path

    path = Path(path)
    if path.is_dir():
        candidates = [
            path / f"{split}.jsonl",
            path / f"{split}.json",
            path / "latest.jsonl",
            path / "latest.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break

    if not path.exists():
        raise FileNotFoundError(f"Forecast-bench file not found: {path}")

    if path.suffix == ".jsonl":
        return load_jsonl(str(path))
    if path.suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)
    raise ValueError(
        f"Unsupported Forecast-bench file type: {path.suffix}. Use .json or .jsonl."
    )


def load_forecast_bench_resolutions(resolution_path):
    """
    Index a ForecastBench resolution set by (source, id, resolution_date).

    Only rows with resolved=True carry an outcome. Unresolved rows also have a
    `resolved_to` value, but it is the question's current market price rather
    than ground truth -- in the 2024-07-21 set the 303 unresolved rows are
    exactly the 303 with a fractional `resolved_to`, while every resolved row is
    0.0 or 1.0. Scoring against those would be scoring against a market.
    """
    loaded = _read_forecast_json(resolution_path)
    rows = loaded.get("resolutions", []) if isinstance(loaded, dict) else loaded

    # (source, id) -> {resolution_date: resolved_to}. Nested rather than a flat
    # (source, id, date) key because market-sourced questions carry
    # resolution_dates == "N/A" while their resolution row still has a concrete
    # date, so those have to be matched on (source, id) alone.
    index = {}
    n_resolved = 0
    unresolved = 0
    for row in rows:
        if not row.get("resolved"):
            unresolved += 1
            continue
        key = (row.get("source"), _id_key(row.get("id")))
        index.setdefault(key, {})[row.get("resolution_date")] = row.get("resolved_to")
        n_resolved += 1

    print(
        f"ForecastBench resolutions: {n_resolved} resolved across {len(index)} "
        f"questions, {unresolved} unresolved (skipped)"
    )
    return index


def _lookup_forecast_resolution(resolutions, source, question_id, resolution_date):
    """
    Find the outcome for one (question, horizon).

    Returns (gold, resolution_date, ambiguous). When the question declares no
    horizon of its own -- market questions, whose resolution_dates is "N/A" --
    the single resolution row for that question is used and its date adopted.
    More than one candidate is reported as ambiguous rather than guessed at.
    """
    by_date = resolutions.get((source, _id_key(question_id)))
    if not by_date:
        return None, resolution_date, False
    if resolution_date is not None:
        return by_date.get(resolution_date), resolution_date, False
    if len(by_date) == 1:
        only_date, gold = next(iter(by_date.items()))
        return gold, only_date, False
    return None, resolution_date, True


def _expand_forecast_bench_questions(
    payload, resolutions, require_resolution, sources, include_freeze_value
):
    """
    Expand an official ForecastBench question set into one item per horizon.

    A question with a `resolution_dates` list is really several questions -- one
    per date -- each with its own recorded outcome, and its text carries a
    {resolution_date} slot to be filled per horizon. The 2024-07-21 human set
    expands from 200 questions to 968 items, 578 of which have an outcome.
    """
    forecast_due_date = payload.get("forecast_due_date")
    question_set = payload.get("question_set", "")
    raw_questions = payload["questions"]

    if isinstance(sources, str):
        sources = [s.strip() for s in sources.split(",") if s.strip()]
    sources = set(sources) if sources else None

    normalized = []
    n_scorable = 0
    n_unmatched = 0
    n_filtered = 0
    n_ambiguous = 0

    for i, record in enumerate(raw_questions):
        source = record.get("source")
        if sources is not None and source not in sources:
            n_filtered += 1
            continue

        question_id = record.get("id", f"forecast_{i}")
        horizons = record.get("resolution_dates")
        horizons = horizons if isinstance(horizons, list) else [None]

        for resolution_date in horizons:
            text = record.get("question")
            if text is None:
                raise ValueError(f"Missing question text in Forecast-bench item {i}")
            if resolution_date:
                text = text.replace("{resolution_date}", resolution_date)
            if forecast_due_date:
                text = text.replace("{forecast_due_date}", forecast_due_date)

            leftover = FORECAST_PLACEHOLDER_RE.search(text)
            if leftover:
                raise ValueError(
                    f"Unsubstituted placeholder {leftover.group(0)} in question "
                    f"{source}/{question_id}: {text!r}. The question set expects "
                    f"this filled from its date fields; showing it to the model "
                    f"verbatim would be a malformed prompt."
                )

            gold, resolution_date, ambiguous = _lookup_forecast_resolution(
                resolutions, source, question_id, resolution_date
            )
            if ambiguous:
                n_ambiguous += 1
            if gold is None:
                n_unmatched += 1
                if require_resolution:
                    continue
            else:
                n_scorable += 1

            metadata = dict(record)
            if not include_freeze_value:
                for key in FORECAST_FREEZE_VALUE_KEYS:
                    metadata.pop(key, None)
            metadata["question"] = text
            if resolution_date:
                metadata["resolution_date"] = resolution_date
            if forecast_due_date:
                metadata["forecast_due_date"] = forecast_due_date

            normalized.append(
                {
                    # horizons share an id, so the date is part of the item id
                    "example_id": "/".join(
                        str(part)
                        for part in (source, question_id, resolution_date)
                        if part is not None
                    ),
                    "question_id": question_id,
                    "source": source,
                    "resolution_date": resolution_date,
                    "forecast_due_date": forecast_due_date,
                    "prompt": text,
                    "gold": gold,
                    # resolved outcomes are strictly 0.0/1.0, so these are binary
                    "task_type": "binary" if gold is not None else "forecast",
                    "metadata": metadata,
                }
            )

    label = question_set or "ForecastBench question set"
    print(
        f"ForecastBench {label}: {len(raw_questions)} questions -> "
        f"{len(normalized)} items"
    )
    print(f"  scorable (resolved outcome found): {n_scorable}")
    print(f"  no resolution in set:              {n_unmatched}")
    if n_ambiguous:
        print(
            f"  ambiguous (no declared horizon but several resolutions): "
            f"{n_ambiguous}"
        )
    if n_filtered:
        print(f"  questions filtered out by --sources: {n_filtered}")
    if not resolutions:
        print(
            "  NOTE: no resolution set was supplied, so nothing can be scored. "
            "Pass --resolution_path to join outcomes."
        )
    return normalized


def _normalize_legacy_forecast_records(raw_records, as_of_date=None):
    """
    Normalize a non-official question file (a bare list, or {'data': [...]} /
    {'forecasts': [...]}) where each record is expected to carry its own gold.
    """
    normalized = []
    for i, record in enumerate(raw_records):
        prompt = (
            record.get("prompt")
            or record.get("question")
            or record.get("title")
            or record.get("question_title")
            or record.get("name")
            or record.get("input")
            or record.get("question_text")
            or record.get("forecasting_question")
        )
        if prompt is None:
            prompt = record.get("resolution_criteria")

        gold = record.get("gold")
        for key in (
            "answer",
            "target",
            "resolved",
            "resolution",
            "outcome",
            "result",
            "resolution_value",
            "resolved_to",
        ):
            if gold is None:
                gold = record.get(key)

        task_type = record.get("task_type")
        if task_type is None:
            if isinstance(gold, bool):
                task_type = "binary"
            elif isinstance(gold, (int, float)):
                task_type = "numeric"
            else:
                task_type = "forecast"

        if prompt is None:
            raise ValueError(f"Missing prompt/question/input in Forecast-bench item {i}")
        metadata = dict(record)
        if as_of_date and not metadata.get("forecast_due_date"):
            metadata["forecast_due_date"] = as_of_date
        normalized.append(
            {
                "example_id": (
                    record.get("id")
                    or record.get("_id")
                    or record.get("question_id")
                    or record.get("slug")
                    or f"forecast_{i}"
                ),
                "question_id": record.get("id") or record.get("question_id"),
                "source": record.get("source"),
                "resolution_date": record.get("resolution_date"),
                "forecast_due_date": record.get("forecast_due_date") or as_of_date,
                "prompt": prompt,
                "gold": gold,
                "task_type": task_type,
                "metadata": metadata,
            }
        )
    return normalized


def load_forecast_bench_dataset(
    dataset_path,
    resolution_path=None,
    split="latest",
    require_resolution=False,
    sources=None,
    include_freeze_value=False,
    as_of_date=None,
):
    """
    Load Forecast-bench examples and normalize them to a small common schema.

    An official question set ({'forecast_due_date', 'question_set',
    'questions'}) is expanded to one item per resolution horizon and joined
    against `resolution_path` for outcomes; other shapes are read as-is with
    gold expected on each record.

    `as_of_date` supplies a forecast due date for question files that carry none
    of their own; official question sets take theirs from the file.
    """
    loaded = _read_forecast_json(dataset_path, split=split)

    resolutions = (
        load_forecast_bench_resolutions(resolution_path) if resolution_path else {}
    )

    if isinstance(loaded, dict) and "questions" in loaded:
        return _expand_forecast_bench_questions(
            loaded,
            resolutions,
            require_resolution=require_resolution,
            sources=sources,
            include_freeze_value=include_freeze_value,
        )

    if isinstance(loaded, dict):
        for key in ("data", "forecasts"):
            if key in loaded:
                return _normalize_legacy_forecast_records(
                    loaded[key], as_of_date=as_of_date
                )
        raise ValueError(
            "Unsupported Forecast-bench JSON format: expected a dict with a "
            "'questions' array (official ForecastBench), 'data', or 'forecasts'."
        )

    if isinstance(loaded, list):
        return _normalize_legacy_forecast_records(loaded, as_of_date=as_of_date)

    raise ValueError(
        "Unsupported Forecast-bench JSON format: expected "
        "{'questions': [...]} or {'data': [...]} or {'forecasts': [...]} "
        "or a top-level list."
    )


def load_benchmark_examples(
    benchmark="persona",
    trait=None,
    version="eval",
    dataset_path=None,
    resolution_path=None,
    split="latest",
    require_resolution=False,
    sources=None,
    include_freeze_value=False,
    as_of_date=None,
    temperature=1,
    persona_instructions_type=None,
    assistant_name=None,
    judge_model="gpt-4.1-mini-2025-04-14",
    eval_type="0_100",
    use_few_shot_examples=False,
    few_shot_example_count=0,
):
    if benchmark == "persona":
        if trait is None:
            raise ValueError("trait is required when benchmark='persona'")
        return load_persona_questions(
            trait,
            temperature=temperature,
            persona_instructions_type=persona_instructions_type,
            assistant_name=assistant_name,
            judge_model=judge_model,
            eval_type=eval_type,
            version=version,
            use_few_shot_examples=use_few_shot_examples,
            few_shot_example_count=few_shot_example_count,
        )
    if benchmark == "forecast_bench":
        if dataset_path is None:
            raise ValueError("dataset_path is required when benchmark='forecast_bench'")
        return [
            ForecastBenchExample(record, temperature=temperature)
            for record in load_forecast_bench_dataset(
                dataset_path,
                resolution_path=resolution_path,
                split=split,
                require_resolution=require_resolution,
                sources=sources,
                include_freeze_value=include_freeze_value,
                as_of_date=as_of_date,
            )
        ]
    raise ValueError(f"Unknown benchmark: {benchmark}")


class Question:
    def __init__(
        self,
        id: str,
        paraphrases: list[str],
        judge_prompts: dict,
        temperature: float = 1,
        system: str = None,
        judge: str = "gpt-4o",
        judge_eval_type: str = "0_100",
        **ignored_extra_args,
    ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.judges = {
            metric: OpenAiJudge(
                judge,
                prompt,
                eval_type=judge_eval_type if metric != "coherence" else "0_100",
            )
            for metric, prompt in judge_prompts.items()
        }

    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        conversations = [[dict(role="user", content=i)] for i in paraphrases]
        if self.system:
            conversations = [
                [dict(role="system", content=self.system)] + c for c in conversations
            ]
        return paraphrases, conversations

    async def eval(
        self,
        llm,
        tokenizer,
        coef,
        vector=None,
        layer=None,
        max_tokens=300,
        n_per_question=70,
        steering_type="last",
        lora_path=None,
    ):
        paraphrases, conversations = self.get_input(n_per_question)
        if coef != 0:
            prompts, answers = sample_steering(
                llm,
                tokenizer,
                conversations,
                vector,
                layer,
                coef,
                temperature=self.temperature,
                max_tokens=max_tokens,
                steering_type=steering_type,
            )
        else:
            prompts, answers = sample(
                llm,
                tokenizer,
                conversations,
                temperature=self.temperature,
                max_tokens=max_tokens,
                lora_path=lora_path,
            )
        df = pd.DataFrame(
            [
                dict(
                    question=question, prompt=prompt, answer=answer, question_id=self.id
                )
                for question, answer, prompt in zip(paraphrases, answers, prompts)
            ]
        )
        for score, judge in self.judges.items():
            scores = await asyncio.gather(
                *[
                    judge(question=question, answer=answer)
                    for question, answer in zip(paraphrases, answers)
                ]
            )
            df[score] = scores
        return df


def _normalize_forecast_text(text):
    if text is None:
        return ""
    return " ".join(str(text).strip().lower().replace("_", " ").split())


def _format_forecast_context(metadata):
    """Return optional ForecastBench fields as readable prompt context."""
    field_labels = [
        ("source_intro", "About this question"),
        ("background", "Background"),
        ("description", "Description"),
        ("body", "Question details"),
        ("resolution_criteria", "Resolution criteria"),
        ("market_info_resolution_criteria", "Market resolution criteria"),
        ("fine_print", "Fine print"),
        ("resolution_date", "Resolution date"),
        ("question_type", "Question type"),
        ("source", "Source"),
        ("url", "URL"),
        # Present only with --include_freeze_value; see FORECAST_FREEZE_VALUE_KEYS.
        ("freeze_datetime_value", "Value at freeze time"),
        ("freeze_datetime_value_explanation", "Freeze value explanation"),
        ("market_info_open_datetime", "Market opened"),
        ("market_info_close_datetime", "Market closes"),
        ("resolve_time", "Resolve time"),
        ("close_time", "Close time"),
        ("created_time", "Created time"),
        ("categories", "Categories"),
    ]
    blocks = []
    for key, label in field_labels:
        value = metadata.get(key)
        if value in (None, "", [], {}):
            continue
        # ForecastBench writes the literal string "N/A" for absent fields
        if isinstance(value, str) and value.strip().upper() == "N/A":
            continue
        if isinstance(value, (list, tuple, set)):
            value = ", ".join(map(str, value))
        elif isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        blocks.append(f"{label}:\n{value}")
    return "\n\n".join(blocks)


def build_forecast_bench_prompt(question, metadata=None):
    """
    Build a structured ForecastBench prompt that asks the model for a calibrated
    probability and a machine-parseable JSON answer.

    The date anchor is the question set's own `forecast_due_date` -- the deadline
    the human forecasters answered by, and the date the question text itself
    refers to via {forecast_due_date}. It is deliberately not the wall-clock
    date: stamping the run date in told the model it was 2026 while asking it to
    forecast something decided in 2024, and made reruns non-reproducible. The
    line is omitted when no due date is known rather than invented.
    """
    metadata = metadata or {}
    context = _format_forecast_context(metadata)
    context_block = f"\n\nAdditional context:\n{context}" if context else ""
    forecast_due_date = metadata.get("forecast_due_date")
    due_date_block = (
        f"\n\nForecast due date: {forecast_due_date}" if forecast_due_date else ""
    )
    return f"""You are a careful, calibrated forecasting model.{due_date_block}

Task: Estimate the probability that the following forecasting question resolves YES/TRUE. Use all supplied context and resolution criteria. If the question is not binary, interpret the requested outcome as stated and put your best estimate in the same `probability` field when possible.

Forecasting question:
{question}{context_block}

Return ONLY valid JSON with this exact schema:
{{
  "rationale": "one concise sentence explaining the main drivers",
  "probability": <number from 0 to 1>
}}

Do not include markdown, extra text, ranges, percentages, or multiple probabilities."""


def _extract_first_number(text):
    if text is None:
        return None

    match = re.search(r"-?\d+(?:\.\d+)?", str(text))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_forecast_response(answer, require_all_fields=False):
    """
    Strictly parse a ForecastBench response.

    Returns (probability, error). On success `error` is None; on failure
    `probability` is None and `error` is a short explanation of what was wrong,
    phrased so it can be fed back to the model in a retry turn.

    Nothing is ever inferred from prose. A response that does not match the
    requested schema is refused, not guessed at: reading a probability out of
    an arbitrary number in free text silently fabricates confident forecasts
    (e.g. "the January 5 deadline" -> 0.05, "a 1 in 10 chance" -> 1.0).

    Set require_all_fields=True to also require `rationale`. By default it is
    validated only when present, since it does not affect the score and
    rejecting an otherwise-valid forecast over it costs a generation for
    nothing.
    """
    if answer is None:
        return None, "the response was empty"

    text = str(answer).strip()
    if not text:
        return None, "the response was empty"

    # Strip a markdown code fence if the model wrapped its JSON in one. This
    # unwraps the payload; it does not guess at its content.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"the response was not valid JSON ({exc.msg})"

    if not isinstance(parsed, dict):
        return None, "the response was not a JSON object"

    if require_all_fields and "confidence" in parsed:
        return None, "the JSON object must not include `confidence`"

    if (
        require_all_fields
        and "rationale" in parsed
        and "probability" in parsed
        and list(parsed).index("rationale") > list(parsed).index("probability")
    ):
        return None, "`rationale` must come before `probability`"

    if "probability" not in parsed:
        return None, "the JSON object had no `probability` field"

    raw = parsed["probability"]
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return None, f"`probability` was {type(raw).__name__}, not a number"
    try:
        probability = float(raw)
    except (TypeError, ValueError):
        return None, f"`probability` was {raw!r}, which is not a number"
    if probability != probability or probability in (float("inf"), float("-inf")):
        return None, f"`probability` was {raw!r}, which is not a finite number"
    if not 0.0 <= probability <= 1.0:
        return None, (
            f"`probability` was {probability}, outside the required range 0 to 1"
        )

    rationale = parsed.get("rationale")
    if rationale is None:
        if require_all_fields:
            return None, "the JSON object had no `rationale` field"
    elif not isinstance(rationale, str) or not rationale.strip():
        return None, "`rationale` was not a non-empty string"

    return probability, None


def _extract_forecast_probability(answer):
    """Probability in [0, 1] from a schema-conforming answer, else None."""
    return parse_forecast_response(answer)[0]


def _coerce_probability(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().rstrip("%")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value > 1 and value <= 100:
        value /= 100.0
    if 0 <= value <= 1:
        return value
    return None


def _normalize_binary_gold(gold):
    """Convert common ForecastBench resolution labels to 0/1 outcomes."""
    if gold is None:
        return None
    if isinstance(gold, dict):
        for key in (
            "outcome",
            "resolution",
            "result",
            "value",
            "status",
            "answer",
            "resolved_to",
        ):
            if key in gold:
                normalized = _normalize_binary_gold(gold[key])
                if normalized is not None:
                    return normalized
        return None
    if isinstance(gold, bool):
        return 1.0 if gold else 0.0
    if isinstance(gold, (int, float)) and not isinstance(gold, bool):
        if float(gold) in {0.0, 1.0}:
            return float(gold)
    text = _normalize_forecast_text(gold)
    yes_values = {
        "yes",
        "true",
        "1",
        "y",
        "resolved yes",
        "resolve yes",
        "yes resolved",
        "positive",
    }
    no_values = {
        "no",
        "false",
        "0",
        "n",
        "resolved no",
        "resolve no",
        "no resolved",
        "negative",
    }
    ambiguous_values = {
        "ambiguous",
        "annulled",
        "cancelled",
        "canceled",
        "unresolved",
        "unknown",
        "not resolved",
        "void",
    }
    if text in yes_values:
        return 1.0
    if text in no_values:
        return 0.0
    if text in ambiguous_values:
        return None
    probability = _coerce_probability(text)
    if probability in {0.0, 1.0}:
        return probability
    return None


def _brier_scores(predicted_probability, gold):
    """
    Score a binary forecast from an already-parsed probability. Returns the
    prediction plus Brier score (lower is better), 1-Brier forecasting score
    (higher is better), and log score.

    Separate from _forecast_probability_scores so the retry loop in
    ForecastBenchExample.eval can score the probability it already parsed
    instead of parsing the answer a second time.
    """
    outcome = _normalize_binary_gold(gold)
    if predicted_probability is None or outcome is None:
        return {
            "predicted_probability": predicted_probability,
            "brier_score": None,
            "forecasting_score": None,
            "log_score": None,
        }
    brier = (predicted_probability - outcome) ** 2
    clipped = min(1 - 1e-15, max(1e-15, predicted_probability))
    log_score = outcome * math.log(clipped) + (1 - outcome) * math.log(1 - clipped)
    return {
        "predicted_probability": predicted_probability,
        "brier_score": brier,
        "forecasting_score": 1.0 - brier,
        "log_score": log_score,
    }


def _forecast_probability_scores(answer, gold):
    """Parse an answer strictly, then Brier-score it."""
    return _brier_scores(parse_forecast_response(answer)[0], gold)


def print_forecast_parse_summary(df):
    """
    Report how many responses matched the requested schema.

    Worth reading before the scores: a low parse rate means the scores below are
    computed over a biased subset, and the refusal reasons say why.
    """
    if "predicted_probability" not in df.columns:
        return
    parsed = int(df["predicted_probability"].notna().sum())
    print(f"Parsed probabilities: {parsed}/{len(df)} ({parsed / max(len(df), 1):.1%})")

    if "parse_attempts" in df.columns:
        attempts = df["parse_attempts"].dropna()
        if not attempts.empty:
            retried = int((attempts > 1).sum())
            print(
                f"Needed a retry: {retried}/{len(df)} "
                f"(mean {attempts.mean():.2f} generations per row)"
            )

    if "parse_error" in df.columns:
        failures = df[df["parse_error"].notna()]
        if not failures.empty:
            print(f"Unparseable after retries: {len(failures)}/{len(df)}. Reasons:")
            for reason, count in failures["parse_error"].value_counts().head(5).items():
                print(f"  {count:>5}  {reason}")


def print_forecast_bench_summary(df):
    """Print aggregate ForecastBench metrics when scored rows are available."""
    print_forecast_parse_summary(df)

    scored = df
    if "forecasting_score" in scored.columns:
        scored = scored[scored["forecasting_score"].notna()]
    else:
        scored = scored.iloc[0:0]

    if scored.empty:
        print(
            "ForecastBench: no scored rows found. This usually means the dataset "
            "does not include resolved gold outcomes, or model probabilities could "
            "not be parsed."
        )
        return

    print(f"ForecastBench scored rows: {len(scored)}/{len(df)}")
    print(
        f"forecasting_score (1 - Brier, higher better): "
        f"{scored['forecasting_score'].mean():.4f} +- {scored['forecasting_score'].std():.4f}"
    )
    print(
        f"brier_score (lower better): "
        f"{scored['brier_score'].mean():.4f} +- {scored['brier_score'].std():.4f}"
    )
    print(
        f"log_score (higher/less negative better): "
        f"{scored['log_score'].mean():.4f} +- {scored['log_score'].std():.4f}"
    )


def _score_forecast_answer(answer, gold, task_type=None):
    task_type = _normalize_forecast_text(task_type)
    answer_text = _normalize_forecast_text(answer)

    if gold is None:
        return None

    if isinstance(gold, bool) or task_type in {"binary", "yes_no", "boolean"}:
        gold_text = "yes" if bool(gold) else "no"
        yes_aliases = {"yes", "true", "likely", "probable", "1", "y"}
        no_aliases = {"no", "false", "unlikely", "0", "n"}
        if answer_text in yes_aliases:
            return 1.0 if gold_text == "yes" else 0.0
        if answer_text in no_aliases:
            return 1.0 if gold_text == "no" else 0.0
        # The answer is not a recognizable yes/no label, so this scorer cannot
        # adjudicate it. Return None (missing) rather than 0.0, which would be
        # indistinguishable from a confidently wrong forecast.
        return None

    if isinstance(gold, (int, float)) or task_type in {"numeric", "number", "continuous"}:
        gold_value = float(gold)
        predicted_value = _extract_first_number(answer)
        if predicted_value is None:
            return None
        abs_error = abs(predicted_value - gold_value)
        return 1.0 / (1.0 + abs_error)

    if isinstance(gold, (list, tuple, set)):
        gold_candidates = {_normalize_forecast_text(item) for item in gold}
        return 1.0 if answer_text in gold_candidates else 0.0

    gold_text = _normalize_forecast_text(gold)
    if task_type in {"multiple_choice", "mcq", "classification", "categorical"}:
        return 1.0 if answer_text == gold_text else 0.0

    if answer_text == gold_text:
        return 1.0

    predicted_number = _extract_first_number(answer)
    gold_number = _extract_first_number(gold)
    if predicted_number is not None and gold_number is not None:
        return 1.0 / (1.0 + abs(predicted_number - gold_number))

    return 1.0 if gold_text in answer_text or answer_text in gold_text else 0.0


def _generate_with_parse_retries(
    generate,
    conversations,
    retry_conversation,
    max_parse_retries=2,
    require_all_fields=False,
):
    """
    Generate, strictly parse, and re-ask whatever failed, over a flat list.

    `generate(conversations) -> (prompts, answers)` and
    `retry_conversation(index, bad_answer, error) -> conversation`.

    Shared by the per-example path and the batched run-level path so the two
    produce identical rows; the only difference between them is where the batch
    boundary falls.
    """
    prompts, answers = generate(conversations)
    prompts, answers = list(prompts), list(answers)

    probabilities = [None] * len(answers)
    errors = [None] * len(answers)
    attempts = [1] * len(answers)
    pending = []
    for i, answer in enumerate(answers):
        probabilities[i], errors[i] = parse_forecast_response(
            answer, require_all_fields=require_all_fields
        )
        if errors[i] is not None:
            pending.append(i)

    for _ in range(max_parse_retries):
        if not pending:
            break
        retry_prompts, retry_answers = generate(
            [retry_conversation(i, answers[i], errors[i]) for i in pending]
        )
        still_failing = []
        for slot, i in enumerate(pending):
            attempts[i] += 1
            prompts[i] = retry_prompts[slot]
            answers[i] = retry_answers[slot]
            probabilities[i], errors[i] = parse_forecast_response(
                retry_answers[slot], require_all_fields=require_all_fields
            )
            if errors[i] is not None:
                still_failing.append(i)
        pending = still_failing

    return prompts, answers, probabilities, errors, attempts


class ForecastBenchExample:
    def __init__(self, record, temperature=0.0):
        self.id = record["example_id"]
        self.raw_question = record["prompt"]
        self.gold = record.get("gold")
        self.task_type = record.get("task_type", "forecast")
        self.question_id = record.get("question_id")
        self.source = record.get("source")
        self.resolution_date = record.get("resolution_date")
        self.forecast_due_date = record.get("forecast_due_date")
        self.metadata = record.get("metadata", {})
        self.prompt = build_forecast_bench_prompt(self.raw_question, self.metadata)
        self.temperature = temperature
        self.system = None

    def get_input(self, n_per_question):
        paraphrases = [self.prompt for _ in range(n_per_question)]
        conversations = [[dict(role="user", content=self.prompt)] for _ in paraphrases]
        return paraphrases, conversations

    def _retry_conversation(self, bad_answer, error):
        """
        Re-ask the question with the parse failure appended.

        A from-scratch retry would be wasted: generation on this path is greedy,
        so it would reproduce the same unparseable answer byte for byte.
        Appending the failure changes the prompt, so the model gets a real
        second attempt even at temperature 0.
        """
        return [
            dict(role="user", content=self.prompt),
            dict(role="assistant", content=bad_answer),
            dict(
                role="user",
                content=(
                    f"Your previous response could not be parsed: {error}. "
                    "Reply with ONLY the JSON object described above and nothing "
                    "else."
                ),
            ),
        ]

    def make_generate(
        self,
        llm,
        tokenizer,
        coef,
        vector=None,
        layer=None,
        max_tokens=300,
        steering_type="last",
        lora_path=None,
    ):
        """Return a `generate(conversations) -> (prompts, answers)` callable."""

        def generate(conversations):
            if coef != 0:
                return sample_steering(
                    llm,
                    tokenizer,
                    conversations,
                    vector,
                    layer,
                    coef,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    steering_type=steering_type,
                )
            return sample(
                llm,
                tokenizer,
                conversations,
                temperature=self.temperature,
                max_tokens=max_tokens,
                lora_path=lora_path,
            )

        return generate

    def build_rows(self, prompts, answers, probabilities, errors, attempts):
        """Assemble output rows from generated answers and their parse results."""
        rows = []
        for i, answer in enumerate(answers):
            probability_scores = _brier_scores(probabilities[i], self.gold)
            rows.append(
                dict(
                    question=self.raw_question,
                    prompt=prompts[i],
                    answer=answer,
                    example_id=self.id,
                    question_id=self.question_id,
                    source=self.source,
                    resolution_date=self.resolution_date,
                    gold=self.gold,
                    task_type=self.task_type,
                    predicted_probability=probability_scores["predicted_probability"],
                    brier_score=probability_scores["brier_score"],
                    forecasting_score=probability_scores["forecasting_score"],
                    log_score=probability_scores["log_score"],
                    # `score` is always 1 - Brier so it is safe to average.
                    # The string/numeric match heuristic is kept separately
                    # rather than folded in on a different scale.
                    score=probability_scores["forecasting_score"],
                    match_score=_score_forecast_answer(
                        answer, self.gold, self.task_type
                    ),
                    parse_attempts=attempts[i],
                    parse_error=errors[i],
                )
            )
        return rows

    async def eval(
        self,
        llm,
        tokenizer,
        coef,
        vector=None,
        layer=None,
        max_tokens=300,
        n_per_question=1,
        steering_type="last",
        lora_path=None,
        max_parse_retries=2,
        require_all_fields=False,
    ):
        generate = self.make_generate(
            llm,
            tokenizer,
            coef,
            vector=vector,
            layer=layer,
            max_tokens=max_tokens,
            steering_type=steering_type,
            lora_path=lora_path,
        )
        _paraphrases, conversations = self.get_input(n_per_question)
        prompts, answers, probabilities, errors, attempts = (
            _generate_with_parse_retries(
                generate,
                conversations,
                lambda i, bad_answer, error: self._retry_conversation(
                    bad_answer, error
                ),
                max_parse_retries=max_parse_retries,
                require_all_fields=require_all_fields,
            )
        )
        return pd.DataFrame(
            self.build_rows(prompts, answers, probabilities, errors, attempts)
        )


def eval_forecast_bench_batched(
    examples,
    llm,
    tokenizer,
    coef,
    vector=None,
    layer=None,
    max_tokens=300,
    n_per_question=1,
    steering_type="last",
    lora_path=None,
    max_parse_retries=2,
    require_all_fields=False,
):
    """
    Evaluate every ForecastBench example in one batch instead of one at a time.

    The sequential path issues one generation call per example (968 calls for
    the 2024-07-21 human set); this flattens every example x sample into a
    single call, and each retry round into one more. Rows are assembled by the
    same ForecastBenchExample.build_rows as the sequential path, over the same
    shared retry core, so the two produce identical output for identical
    generations -- only the batch boundary differs.
    """
    if not examples:
        return pd.DataFrame()

    temperatures = {example.temperature for example in examples}
    if len(temperatures) > 1:
        raise ValueError(
            f"Batched generation needs one temperature for the whole run, got "
            f"{sorted(temperatures)}. Use the sequential path for mixed "
            f"temperatures."
        )

    # flat index -> owning example, so a retry can rebuild the right prompt
    owners = []
    conversations = []
    for example in examples:
        _paraphrases, example_conversations = example.get_input(n_per_question)
        for conversation in example_conversations:
            owners.append(example)
            conversations.append(conversation)

    generate = examples[0].make_generate(
        llm,
        tokenizer,
        coef,
        vector=vector,
        layer=layer,
        max_tokens=max_tokens,
        steering_type=steering_type,
        lora_path=lora_path,
    )

    print(
        f"Batched generation: {len(conversations)} prompts from "
        f"{len(examples)} examples in one call"
    )
    prompts, answers, probabilities, errors, attempts = _generate_with_parse_retries(
        generate,
        conversations,
        lambda i, bad_answer, error: owners[i]._retry_conversation(bad_answer, error),
        max_parse_retries=max_parse_retries,
        require_all_fields=require_all_fields,
    )

    rows = []
    offset = 0
    for example in examples:
        stop = offset + n_per_question
        rows.extend(
            example.build_rows(
                prompts[offset:stop],
                answers[offset:stop],
                probabilities[offset:stop],
                errors[offset:stop],
                attempts[offset:stop],
            )
        )
        offset = stop
    return pd.DataFrame(rows)


def a_or_an(word):
    return "an" if word[0].lower() in "aeiou" else "a"


def load_persona_questions(
    trait,
    temperature=1,
    persona_instructions_type=None,
    assistant_name=None,
    judge_model="gpt-4.1-mini-2025-04-14",
    eval_type="0_100",
    version="eval",
    use_few_shot_examples=False,
    few_shot_example_count=0,
):
    if few_shot_example_count < 0 or few_shot_example_count > 30:
        raise ValueError("few_shot_example_count must be between 0 and 30")
    trait_data = json.load(
        open(f"data_generation/trait_data_{version}/{trait}.json", "r")
    )
    judge_prompts = {}
    prompt_template = trait_data["eval_prompt"]
    few_shot_examples = []
    if use_few_shot_examples and few_shot_example_count > 0:
        available_examples = trait_data.get("few_shot_examples", [])
        few_shot_examples = random.sample(
            available_examples,
            k=min(few_shot_example_count, len(available_examples)),
        )
    if few_shot_examples:
        example_blocks = []
        for idx, example in enumerate(few_shot_examples, start=1):
            example_blocks.append(
                f"Example {idx}:\n"
                f"[QUESTION START]\n{example['question']}\n[QUESTION END]\n\n"
                f"[ANSWER START]\n{example['answer']}\n[ANSWER END]\n\n"
                f"Score: {example['score']}"
            )
        prompt_template = (
            prompt_template
            + "\n\nFew-shot examples:\n\n"
            + "\n\n".join(example_blocks)
            + "\n\nNow evaluate the final question and answer below."
        )
    judge_prompts[trait] = prompt_template
    judge_prompts["coherence"] = Prompts[f"coherence_{eval_type}"]
    raw_questions = trait_data["questions"]
    questions = []
    for i, question in enumerate(raw_questions):
        if persona_instructions_type is not None:
            persona_instructions = [
                x[persona_instructions_type] for x in trait_data["instruction"]
            ]
            for k, instruction in enumerate(persona_instructions):
                if assistant_name is None:
                    if persona_instructions_type == "pos":
                        assistant_name = trait
                    else:
                        assistant_name = "helpful"
                system = f"You are {a_or_an(assistant_name)} {assistant_name} assistant. {instruction}"
                questions.append(
                    Question(
                        paraphrases=[question],
                        id=f"{trait}_{i}_{persona_instructions_type}_{k}",
                        judge_prompts=judge_prompts,
                        judge=judge_model,
                        temperature=temperature,
                        system=system,
                        judge_eval_type=eval_type,
                    )
                )
        else:
            questions.append(
                Question(
                    paraphrases=[question],
                    id=f"{trait}_{i}",
                    judge_prompts=judge_prompts,
                    judge=judge_model,
                    temperature=temperature,
                    judge_eval_type=eval_type,
                )
            )
    return questions


async def eval_batched(
    questions,
    llm,
    tokenizer,
    coef,
    vector=None,
    layer=None,
    n_per_question=100,
    max_concurrent_judges=70,
    max_tokens=300,
    steering_type="last",
    lora_path=None,
):
    """Batch process all questions together for faster inference"""
    # Collect all prompts from all questions
    all_paraphrases = []
    all_conversations = []
    question_indices = []
    for i, question in enumerate(questions):
        paraphrases, conversations = question.get_input(n_per_question)
        all_paraphrases.extend(paraphrases)
        all_conversations.extend(conversations)
        question_indices.extend([i] * len(paraphrases))

    # Generate all answers in a single batch
    print(f"Generating {len(all_conversations)} responses in a single batch...")
    if coef != 0:
        prompts, answers = sample_steering(
            llm,
            tokenizer,
            all_conversations,
            vector,
            layer,
            coef,
            temperature=questions[0].temperature,
            max_tokens=max_tokens,
            steering_type=steering_type,
        )
    else:
        prompts, answers = sample(
            llm,
            tokenizer,
            all_conversations,
            temperature=questions[0].temperature,
            max_tokens=max_tokens,
            lora_path=lora_path,
        )

    # Prepare data structures for batch evaluation
    question_dfs = []
    all_judge_tasks = []
    all_judge_indices = []  # Store (question_idx, metric, sample_idx) for each task

    print("Preparing judge evaluation tasks...")
    for i, question in enumerate(questions):
        # Get this question's data
        indices = [j for j, idx in enumerate(question_indices) if idx == i]
        q_paraphrases = [all_paraphrases[j] for j in indices]
        q_prompts = [prompts[j] for j in indices]
        q_answers = [answers[j] for j in indices]

        # Create dataframe for this question
        df = pd.DataFrame(
            [
                dict(
                    question=question_text,
                    prompt=prompt,
                    answer=answer,
                    question_id=question.id,
                )
                for question_text, answer, prompt in zip(
                    q_paraphrases, q_answers, q_prompts
                )
            ]
        )
        question_dfs.append(df)

        # Collect all judge tasks
        for metric, judge in question.judges.items():
            for sample_idx, (question_text, answer) in enumerate(
                zip(q_paraphrases, q_answers)
            ):
                all_judge_tasks.append((judge, question_text, answer))
                all_judge_indices.append((i, metric, sample_idx))

    # Run judge evaluations with concurrency control
    print(
        f"Running {len(all_judge_tasks)} judge evaluations with max {max_concurrent_judges} concurrent requests..."
    )
    all_results = [None] * len(all_judge_tasks)  # Pre-allocate results array

    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent_judges)

    async def run_with_semaphore(task_idx, judge, question_text, answer):
        async with semaphore:
            result = await judge(question=question_text, answer=answer)
            return task_idx, result

    # Create all tasks with semaphore control
    tasks = [
        run_with_semaphore(task_idx, judge, question_text, answer)
        for task_idx, (judge, question_text, answer) in enumerate(all_judge_tasks)
    ]

    # Process tasks in batches with progress bar
    with tqdm(total=len(tasks), desc="Judge evaluations") as pbar:
        for task in asyncio.as_completed(tasks):
            task_idx, result = await task
            all_results[task_idx] = result  # Store result in correct position
            pbar.update(1)

    # Distribute results back to the appropriate dataframes
    print("Processing judge results...")
    for task_idx, result in enumerate(all_results):
        question_idx, metric, sample_idx = all_judge_indices[task_idx]
        question_dfs[question_idx].loc[sample_idx, metric] = result

    return question_dfs


def main(
    model,
    trait,
    output_path,
    benchmark="persona",
    dataset_path=None,
    split="latest",
    coef=0,
    vector_path=None,
    layer=None,
    steering_type="response",
    max_tokens=300,
    n_per_question=100,
    batch_process=True,
    max_concurrent_judges=70,
    persona_instruction_type=None,
    assistant_name=None,
    judge_model="gpt-4.1-mini-2025-04-14",
    version="extract",
    use_few_shot_examples=False,
    few_shot_example_count=0,
    judge_requests_per_minute=4800,
    overwrite=False,
    max_parse_retries=2,
    require_all_fields=False,
    temperature=None,
    resolution_path=None,
    require_resolution=False,
    sources=None,
    include_freeze_value=False,
    as_of_date=None,
    batch_generation=False,
):
    if few_shot_example_count < 0 or few_shot_example_count > 30:
        raise ValueError("few_shot_example_count must be between 0 and 30")
    if judge_requests_per_minute <= 0:
        raise ValueError("judge_requests_per_minute must be positive")
    set_judge_requests_per_minute(judge_requests_per_minute)

    if os.path.exists(output_path) and not overwrite:
        print(f"Output path {output_path} already exists, skipping...")
        df = pd.read_csv(output_path)
        if benchmark == "forecast_bench":
            print_forecast_bench_summary(df)
        else:
            for trait_name in [trait, "coherence"]:
                if trait_name in df.columns:
                    print(f"{trait_name}:  {df[trait_name].mean():.2f} +- {df[trait_name].std():.2f}")
        return

    print(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # An explicit --temperature wins; otherwise derive it from n_per_question as
    # before (greedy for a single sample, 1.0 when sampling repeatedly).
    if temperature is None:
        temperature = 0.0 if n_per_question == 1 else 1.0

    if benchmark == "forecast_bench" and n_per_question > 1 and temperature == 0:
        raise ValueError(
            f"n_per_question={n_per_question} with temperature=0 draws "
            f"{n_per_question} identical forecasts per question, at "
            f"{n_per_question}x the cost of one. Use n_per_question=1, or pass "
            f"--temperature > 0 to sample genuinely different forecasts."
        )

    if coef != 0:
        llm, tokenizer = load_model(model)
        lora_path = None
        vector = torch.load(vector_path, weights_only=False)[layer]

    else:
        llm, tokenizer, lora_path = load_vllm_model(model)
        vector = None

    questions = load_benchmark_examples(
        benchmark=benchmark,
        trait=trait,
        version=version,
        dataset_path=dataset_path,
        resolution_path=resolution_path,
        split=split,
        require_resolution=require_resolution,
        sources=sources,
        include_freeze_value=include_freeze_value,
        as_of_date=as_of_date,
        temperature=temperature,
        persona_instructions_type=persona_instruction_type,
        assistant_name=assistant_name,
        judge_model=judge_model,
        eval_type="0_100",
        use_few_shot_examples=use_few_shot_examples,
        few_shot_example_count=few_shot_example_count,
    )
    if benchmark == "forecast_bench":
        # The persona async path exists to run many judge calls concurrently;
        # this benchmark has no judge. Generation batching is separate, and is
        # opt-in via --batch_generation until it has been measured on a GPU.
        batch_process = False

    if benchmark == "forecast_bench" and batch_generation:
        outputs = eval_forecast_bench_batched(
            questions,
            llm,
            tokenizer,
            coef,
            vector=vector,
            layer=layer,
            max_tokens=max_tokens,
            n_per_question=n_per_question,
            steering_type=steering_type,
            lora_path=lora_path,
            max_parse_retries=max_parse_retries,
            require_all_fields=require_all_fields,
        )
    elif batch_process:
        print(f"Batch processing {len(questions)} '{benchmark}' examples...")
        outputs_list = asyncio.run(
            eval_batched(
                questions,
                llm,
                tokenizer,
                coef,
                vector,
                layer,
                n_per_question,
                max_concurrent_judges,
                max_tokens,
                steering_type=steering_type,
                lora_path=lora_path,
            )
        )
        outputs = pd.concat(outputs_list)
    else:
        outputs = []
        # Strict-parse retries only exist on the forecast_bench path; the
        # persona Question.eval does not take these arguments.
        eval_kwargs = {}
        if benchmark == "forecast_bench":
            eval_kwargs = dict(
                max_parse_retries=max_parse_retries,
                require_all_fields=require_all_fields,
            )
        for question in tqdm(questions, desc=f"Processing {benchmark} examples"):
            outputs.append(
                asyncio.run(
                    question.eval(
                        llm,
                        tokenizer,
                        coef,
                        vector,
                        layer,
                        max_tokens,
                        n_per_question,
                        steering_type=steering_type,
                        lora_path=lora_path,
                        **eval_kwargs,
                    )
                )
            )
        outputs = pd.concat(outputs)
    outputs.to_csv(output_path, index=False)
    print(output_path)
    if benchmark == "persona":
        for trait_name in [trait, "coherence"]:
            print(
                f"{trait_name}:  {outputs[trait_name].mean():.2f} +- {outputs[trait_name].std():.2f}"
            )
    else:
        print(f"Saved {len(outputs)} Forecast-bench rows to {output_path}")
        print_forecast_bench_summary(outputs)

    # Explicitly destroy vLLM engine to free GPU memory
    # vLLM v0.10+ uses V1 engine with engine_core subprocess
    import gc
    import time

    try:
        # vLLM v0.10+ V1 engine: shutdown the engine_core subprocess
        if hasattr(llm, "llm_engine") and hasattr(llm.llm_engine, "engine_core"):
            llm.llm_engine.engine_core.shutdown()
        elif hasattr(llm, "llm_engine") and hasattr(llm.llm_engine, "shutdown"):
            llm.llm_engine.shutdown()
        elif hasattr(llm, "shutdown"):
            llm.shutdown()
    except Exception as e:
        print(f"Warning during vLLM engine shutdown: {e}")
    del llm

    # Destroy PyTorch distributed process groups (NCCL/Gloo) created by vLLM
    # This is critical to prevent memory leaks between runs
    # See: https://pytorch.org/docs/stable/distributed.html#shutdown
    try:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            print("Destroyed distributed process group")
    except Exception:
        # This is expected to fail sometimes when no process group exists
        pass

    # Force garbage collection and CUDA cleanup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)  # Brief wait for subprocess to fully terminate
    print("vLLM cleanup complete")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
