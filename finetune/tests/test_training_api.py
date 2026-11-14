"""Offline contracts for the exact locked Transformers/TRL/PEFT stack."""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(relative):
    spec = importlib.util.spec_from_file_location("fixture_" + relative.replace("/", "_").replace(".", "_"), ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script", ["jobs/sft.py", "experiments/lfm2/sft_lfm2.py", "experiments/grpo/grpo.py"])
def test_job_config_constructors(script, tmp_path):
    module = load_module(script)  # Must be import-safe: no downloads or login.
    config = module.build_config(output_dir=str(tmp_path), push_to_hub=False, use_cpu=True, bf16=False)
    assert config.use_cpu and not config.bf16
    assert config.report_to == []
    if script != "experiments/grpo/grpo.py":
        assert config.max_length == 512
        assert config.get_warmup_steps(100) == 3


def test_all_training_call_keywords_match_installed_apis():
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer, GRPOConfig, GRPOTrainer

    apis = {api.__name__: inspect.signature(api).parameters for api in [LoraConfig, SFTConfig, SFTTrainer, GRPOConfig, GRPOTrainer]}
    checked = 0
    for name in ["train.py", "train_unsloth.py", "jobs/sft.py", "experiments/lfm2/sft_lfm2.py", "experiments/grpo/grpo.py"]:
        for node in ast.walk(ast.parse((ROOT / name).read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in apis:
                unknown = {kw.arg for kw in node.keywords if kw.arg} - apis[node.func.id].keys()
                assert not unknown, (name, node.lineno, node.func.id, unknown)
                checked += 1
    assert checked >= 12


@pytest.mark.parametrize("name", ["sft.yaml", "sft_local.yaml", "sft-lfm2.yaml"])
def test_main_sft_yaml_config_contract(name, tmp_path):
    import yaml

    module = load_module("train.py")
    cfg = yaml.safe_load((ROOT / "configs" / name).read_text())
    config = module.build_sft_config(cfg, output_dir=str(tmp_path), use_cpu=True, bf16=False, push_to_hub=False)
    assert config.learning_rate == 2e-4
    assert config.get_warmup_steps(100) == 3
    assert config.project == "qmd-query-expansion"


@pytest.mark.parametrize("module_name", ["reward.py", "jobs/eval_common.py"])
def test_grpo_reward_input_shapes(module_name):
    module = load_module(module_name)
    reward = module.QMDRewardFunction()
    prompt = "/no_think Expand this search query: fixture query"
    text = "hyde: A synthetic document describes fixture query tokens for this offline test.\nlex: fixture tokens\nvec: answer the fixture query with synthetic documents"
    plain = reward(completions=[text], prompts=[prompt], completion_ids=[[1]], trainer_state=None)
    chat = reward(completions=[[{"role": "assistant", "content": text}]], prompts=[[{"role": "user", "content": prompt}]], completion_ids=[[1]])
    assert plain == chat
    assert len(plain) == 1 and 0 <= plain[0] <= 1


def test_sft_adapter_save_reload_and_eval_generation(tiny_model, tiny_tokenizer, tmp_path):
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer

    base_dir = tmp_path / "base"
    tiny_model.save_pretrained(base_dir)
    tiny_tokenizer.save_pretrained(base_dir)
    model = AutoModelForCausalLM.from_pretrained(base_dir, local_files_only=True)
    dataset = Dataset.from_list([{"text": "user: fixture query\nassistant: lex: fixture tokens"}] * 2)
    args = SFTConfig(
        output_dir=str(tmp_path / "adapter"), use_cpu=True, bf16=False,
        max_steps=1, per_device_train_batch_size=1, gradient_accumulation_steps=1,
        learning_rate=1e-3, max_length=32, warmup_steps=0.03,
        gradient_checkpointing=False, optim="adamw_torch", report_to="none",
        save_strategy="no", eval_strategy="no", disable_tqdm=True, dataset_num_proc=None,
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=dataset, processing_class=tiny_tokenizer,
        peft_config=LoraConfig(
            task_type="CAUSAL_LM", r=2, lora_alpha=4,
            target_modules=["q_proj", "v_proj"], modules_to_save=["embed_tokens", "lm_head"],
            ensure_weight_tying=True,
        ),
    )
    result = trainer.train()
    assert result.global_step == 1 and math.isfinite(result.training_loss)
    trainer.save_model()
    tiny_tokenizer.save_pretrained(args.output_dir)
    assert (Path(args.output_dir) / "adapter_model.safetensors").is_file()
    evaluator = load_module("eval.py")
    loaded, tokenizer = evaluator.load_model(args.output_dir)
    assert loaded.device.type == "cpu"
    outputs = evaluator.generate_batch(loaded, tokenizer, ["fixture query"], max_new_tokens=2, max_time=0)
    assert len(outputs) == 1 and isinstance(outputs[0], str)


def test_grpo_cpu_one_step(tiny_model, tiny_tokenizer, tmp_path):
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from trl import GRPOConfig, GRPOTrainer
    from reward import QMDRewardFunction

    calls = []
    scorer = QMDRewardFunction()
    def reward(prompts, completions, **kwargs):
        values = scorer(prompts=prompts, completions=completions, **kwargs)
        calls.append((len(completions), values))
        return values
    args = GRPOConfig(
        output_dir=str(tmp_path / "grpo"), use_cpu=True, bf16=False,
        max_steps=1, per_device_train_batch_size=2, gradient_accumulation_steps=1,
        num_generations=2, max_completion_length=3, beta=0.0,
        gradient_checkpointing=False, optim="adamw_torch", report_to="none",
        save_strategy="no", disable_tqdm=True, generation_kwargs={"min_new_tokens": 2},
    )
    trainer = GRPOTrainer(
        model=get_peft_model(tiny_model, LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"])),
        args=args, processing_class=tiny_tokenizer,
        train_dataset=Dataset.from_list([{"prompt": "Expand this search query: fixture query"}] * 2),
        reward_funcs=[reward],
    )
    result = trainer.train()
    assert result.global_step == 1 and math.isfinite(result.training_loss)
    assert calls and all(count == 2 and len(values) == 2 for count, values in calls)


def test_qwen35_compatible_cli_native_backend(tiny_tokenizer, tmp_path, monkeypatch):
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    # Exercise the real Qwen3.5 hybrid architecture, not a Qwen3 stand-in.
    config = Qwen3_5TextConfig(
        vocab_size=len(tiny_tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=1, linear_num_value_heads=1,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=128, pad_token_id=1, eos_token_id=2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]},
    )
    torch.manual_seed(17)
    tiny_model = Qwen3_5ForCausalLM(config).to("cpu")
    tiny_model.set_attn_implementation("eager")
    module = load_module("train_unsloth.py")
    base_dir = tmp_path / "native-base"
    tiny_model.save_pretrained(base_dir)
    tiny_tokenizer.save_pretrained(base_dir)
    data_file = tmp_path / "synthetic.jsonl"
    data_file.write_text("\n".join(json.dumps({"text": "user: fixture query\nassistant: lex: fixture tokens"}) for _ in range(4)) + "\n")
    output = tmp_path / "native-adapter"
    monkeypatch.setitem(module.MODEL_MAP, "0.8B", str(base_dir))
    monkeypatch.setattr(sys, "argv", ["train_unsloth.py", "--model", "0.8B", "--epochs", "1", "--batch-size", "1", "--grad-accum", "1", "--data", str(data_file), "--output", str(output), "--device", "cpu", "--no-gguf", "--no-eval"])
    module.main()
    assert (output / "adapter_model.safetensors").is_file()
    assert (output / "tokenizer.json").is_file()


def test_lfm2_lora_forward_backward(tiny_tokenizer):
    import torch
    from transformers import Lfm2Config, Lfm2ForCausalLM
    from peft import LoraConfig, get_peft_model

    config = Lfm2Config(
        vocab_size=len(tiny_tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        full_attn_idxs=[1], block_multiple_of=8, block_auto_adjust_ff_dim=False,
        max_position_embeddings=64,
    )
    model = get_peft_model(Lfm2ForCausalLM(config).to("cpu"), LoraConfig(
        task_type="CAUSAL_LM", r=2, lora_alpha=4,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"],
    ))
    tokens = torch.tensor([[3, 6, 7, 8, 9]])
    loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss
    loss.backward()
    assert math.isfinite(loss.item())
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_trackio_local_logging_and_callback(tiny_model, tmp_path, monkeypatch):
    import trackio
    from transformers import TrainingArguments, TrainerState
    from transformers.integrations import TrackioCallback

    original_init = trackio.init
    calls = []
    def offline_init(**kwargs):
        calls.append(kwargs)
        assert kwargs["space_id"] is None and kwargs["bucket_id"] is None
        return original_init(**kwargs, embed=False, auto_log_gpu=False, auto_log_cpu=False, capture_console=False)
    monkeypatch.setattr(trackio, "init", offline_init)
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, bf16=False,
                             report_to="trackio", project="qmd-synthetic-contract", run_name="cpu-fixture")
    callback = TrackioCallback()
    callback.setup(args, TrainerState(is_world_process_zero=True), tiny_model)
    trackio.log({"fixture/contract_probe": 1})
    trackio.finish()
    assert calls[0]["project"] == "qmd-synthetic-contract"


@pytest.mark.parametrize("argv", [
    ["train.py", "sft", "--config", "configs/sft.yaml", "--dry-run"],
    ["train_unsloth.py", "--model", "0.8B", "--dry-run"],
    ["jobs/sft.py", "--dry-run"],
    ["jobs/eval.py", "--dry-run", "--sft-only"],
    ["experiments/grpo/grpo.py", "--dry-run"],
    ["experiments/lfm2/sft_lfm2.py", "--dry-run"],
    ["eval.py", "--help"],
    ["eval_retrieval.py", "--help"],
    ["convert_gguf.py", "--help"],
    ["convert_onnx.py", "--help"],
    ["dataset/prepare_data.py", "--help"],
])
def test_offline_cli_previews(argv):
    result = subprocess.run([sys.executable, *argv], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
