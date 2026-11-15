"""Exercise schema, chat-template, and datasets 5 JSON APIs on tiny fixtures."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from dataset.schema import TrainingExample, load_examples, output_items_to_text, parse_output_text
from reward import QMDRewardFunction, score_expansion_detailed

ROOT = Path(__file__).resolve().parents[1]


def example(query="fixture query"):
    return {"query": query, "output": [["hyde", "A synthetic fixture document describes query tokens and a portable test answer."], ["lex", "fixture tokens"], ["vec", "answer the fixture query using synthetic documents"]]}


def test_schema_and_reward_contracts(tmp_path):
    row = example()
    fixture = tmp_path / "examples.jsonl"
    fixture.write_text(json.dumps(row) + "\n")
    loaded = load_examples(fixture)
    assert len(loaded) == 1
    assert loaded[0].output_as_lists() == row["output"]
    text = output_items_to_text(loaded[0].output)
    assert parse_output_text(text) == row["output"]
    assert score_expansion_detailed(row["query"], text)["percentage"] > 50
    assert score_expansion_detailed(row["query"], "assistant: invalid output")["percentage"] == 0
    assert len(QMDRewardFunction()([text], ["Expand this search query: fixture query"])) == 1
    with pytest.raises(ValidationError):
        TrainingExample.model_validate({"query": "", "output": []})
    with pytest.raises(ValidationError):
        TrainingExample.model_validate({"query": "fixture", "output": [["invalid-type", "content"]]})


def test_prepare_data_cli_and_datasets_json(tiny_tokenizer, tmp_path):
    from datasets import load_dataset

    tokenizer_dir = tmp_path / "tokenizer"
    tiny_tokenizer.save_pretrained(tokenizer_dir)
    fixture = tmp_path / "input.jsonl"
    fixture.write_text("\n".join(json.dumps(example(q)) for q in ["fixture query", "FIXTURE QUERY", "synthetic answer", "test document"]) + "\n")
    output = tmp_path / "prepared"
    env = dict(os.environ, QMD_BASE_MODEL=str(tokenizer_dir))
    result = subprocess.run(
        [sys.executable, "dataset/prepare_data.py", "--input", str(fixture), "--output", str(output), "--split", "0.34"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Deduplicated: 4 -> 3" in result.stdout
    data = load_dataset("json", data_files={"train": str(output / "train.jsonl"), "validation": str(output / "val.jsonl")})
    assert len(data["train"]) == 1 and len(data["validation"]) == 2
    assert all(key in data["train"].column_names for key in ("text", "messages", "query", "output"))
    assert data["train"][0]["messages"][0]["content"].startswith("/no_think Expand this search query:")
    assert data["train"][0]["messages"][1]["content"].startswith("hyde:")
