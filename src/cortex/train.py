"""Night school: the thinking model learns from the brain's own day. Runs in the trainer container
(trainer/Dockerfile), with the graphics card, while the brain sleeps.

It learns with KTO, from replies each marked good or bad by what came of them, which is what the
brain has: no pairs of a better and a worse answer to one question, only how each of its own
answers went. What is learned is a LoRA adapter, some millions of weights beside the model's
billions, which the model server loads on top of the model it was trained on. The model itself
is never changed, so a night that teaches it something wrong can simply be thrown away.

A model that sees and hears (Gemma 4) learns in its language layers only: what it is taught is
the brain's replies, in words. Its adapter is not folded in: llama.cpp's server loads an adapter
beside the model, so it is only turned into the server's own format (`--gguf`).

usage: python train.py --base /models/granite-4.1-8b --lessons lessons.jsonl --out /brain/mind/night-3
       python train.py --base /models/granite-4.1-8b --out /brain/mind/night-3 --fold
       python train.py --base /models/gemma-4-E4B-it --out /brain/mind/night-3 --gguf
Last line printed: @@cortex-train@@ and a JSON report.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

MARK = "@@cortex-train@@"


def main() -> None:
    ask = argparse.ArgumentParser()
    ask.add_argument("--base", required=True)
    ask.add_argument("--lessons", default="")
    ask.add_argument("--out", required=True)
    ask.add_argument("--max-len", type=int, default=2048)
    ask.add_argument("--rate", type=float, default=2e-5)
    ask.add_argument("--rank", type=int, default=16)
    ask.add_argument("--limit", type=int, default=400, help="at most this many lessons a night")
    ask.add_argument("--fold", action="store_true", help="fold the adapter in --out into the model, and stop")
    ask.add_argument("--gguf", action="store_true", help="turn the adapter in --out into llama.cpp's format, and stop")
    args = ask.parse_args()
    if args.fold:
        fold(args.out, args.out)
        return
    if args.gguf:
        to_gguf(args.base, args.out)
        return
    started = time.time()

    sees = json.loads((Path(args.base) / "config.json").read_text())["model_type"].startswith("gemma")

    lessons = [json.loads(line) for line in Path(args.lessons).read_text(encoding="utf-8").splitlines()][-args.limit:]
    plan = Path(args.lessons).with_name("plan.json")  # in a long sleep, later rounds learn more passes over them
    epochs = max(1, min(8, int(json.loads(plan.read_text())["epochs"]))) if plan.is_file() else 1
    good = sum(lesson["label"] for lesson in lessons)
    bad = len(lessons) - good
    if not good or not bad:
        print(MARK + json.dumps({"ok": False, "why": f"needs good and bad lessons both: {good} good, {bad} bad"}))
        return
    if sees:
        seeing(args, lessons, epochs, started)
        return
    from unsloth import FastLanguageModel  # before trl and torch's own: it patches what it trains with

    import torch
    from datasets import Dataset
    from trl import KTOConfig, KTOTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        args.base, max_seq_length=args.max_len, load_in_4bit=True, dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=7,
    )
    data = Dataset.from_list([{k: lesson[k] for k in ("prompt", "completion", "label")} for lesson in lessons])
    # The brain fails more often than it succeeds: the two kinds are weighed so that each counts alike.
    config = KTOConfig(
        output_dir=str(Path(args.out) / "work"), per_device_train_batch_size=2, gradient_accumulation_steps=4,  # KTO needs two at once
        learning_rate=args.rate, num_train_epochs=epochs, max_length=args.max_len, max_completion_length=768,
        desirable_weight=1.0, undesirable_weight=good / bad, logging_steps=5, save_strategy="no",
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(), report_to=[],
        gradient_checkpointing=True, remove_unused_columns=False,
    )
    trainer = KTOTrainer(model=model, args=config, train_dataset=data, processing_class=tokenizer)
    result = trainer.train()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(MARK + json.dumps({
        "ok": True, "lessons": len(lessons), "good": good, "bad": bad, "loss": round(float(result.training_loss), 4),
        "minutes": round((time.time() - started) / 60, 1), "memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }))


# What stays off the card for a model that sees and hears: the table of per-layer embeddings (2.8
# billion numbers that are only looked up, 5.6 GB as they are) and the eyes and ears, which the
# lessons do not teach. Training loaders keep all of it on the card, where an 8 GB card has no
# room for it beside the decoder.
AWAY = ("embed_tokens_per_layer", "vision_tower", "audio_tower", "embed_vision", "embed_audio")
SLICE = 128  # positions whose word scores are worked out at once: 262,144 words each
BETA = 0.1  # how far KTO lets the model move from what it was, per unit of reward


def looked_up_here(table) -> None:
    """Look a table of embeddings up in the machine's memory and send only what was looked up to
    the card. Parked in memory by the loader, it would be copied whole onto the card at every use."""
    import torch
    from accelerate.hooks import remove_hook_from_module
    from accelerate.utils import set_module_tensor_to_device

    hook = getattr(table, "_hf_hook", None)
    weights = getattr(hook, "weights_map", None)
    remove_hook_from_module(table, recurse=True)
    for name, tensor in [*table.named_parameters(), *table.named_buffers()]:
        try:
            set_module_tensor_to_device(table, name, "cpu", value=weights[name])
        except (KeyError, TypeError):  # not among the parked weights, such as a scale made when it was built
            if tensor.device.type != "meta":
                set_module_tensor_to_device(table, name, "cpu")
    table.requires_grad_(False)
    table.register_forward_pre_hook(lambda module, args: tuple(a.to("cpu") if torch.is_tensor(a) else a for a in args))
    table.register_forward_hook(lambda module, args, out: out.to("cuda", non_blocking=True))


def seeing(args, lessons: list[dict], epochs: int, started: float) -> None:
    """KTO for Gemma 4, by hand. TRL's trainer works out a score for every word of a 262,144-word
    vocabulary at every position at once, and keeps them: 4 GB more for 2 x 2048 tokens. Here only
    the reply's tokens are scored, a slice at a time, and the scores are worked out again on the
    way back instead of being kept. What it is compared with, the model as it was, is the same
    model with the adapter switched off. The loss is KTO's: a good reply is pushed above, a bad one
    below, the reference point, which is how far the model has moved on average (never below 0)."""
    import random

    import torch
    from accelerate import init_empty_weights
    from peft import LoraConfig, get_peft_model
    from torch.utils.checkpoint import checkpoint
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

    config = AutoConfig.from_pretrained(args.base)
    with init_empty_weights():
        empty = AutoModelForImageTextToText.from_config(config)
    placement = {
        name: "cpu" if any(f".{away}" in f".{name}" for away in AWAY) else 0
        for name, _ in empty.named_modules() if name.count(".") == 2 or name == "lm_head"
    }
    del empty
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True, llm_int8_skip_modules=[*AWAY, "lm_head", "embed_tokens"],
    )
    model = AutoModelForImageTextToText.from_pretrained(args.base, quantization_config=quant, device_map=placement, dtype=torch.bfloat16)
    looked_up_here(model.model.language_model.embed_tokens_per_layer)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.rank, lora_dropout=0.0, bias="none",
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    ))
    model.train()
    inner = model.base_model.model  # the multimodal model, with the adapter woven into it
    words = inner.lm_head.weight  # tied to the input embeddings, on the card
    cap = getattr(config.get_text_config(), "final_logit_softcapping", None)

    def encode(lesson: dict) -> tuple[torch.Tensor, int] | None:
        prompt = tokenizer.apply_chat_template(lesson["prompt"], tokenize=False, add_generation_prompt=True)
        whole = tokenizer.apply_chat_template(lesson["prompt"] + lesson["completion"], tokenize=False)
        asked = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        said = tokenizer(whole, add_special_tokens=False)["input_ids"][len(asked):][:768]
        if not said:
            return None
        asked = asked[-(args.max_len - len(said)):]  # what it was shown is cut from the front, the reply never
        return torch.tensor([asked + said], device="cuda"), len(said)

    def scored(h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = (h @ words.T).float()
        if cap:
            logits = torch.tanh(logits / cap) * cap
        return torch.log_softmax(logits, -1).gather(-1, target[:, None]).squeeze(-1).sum()

    def reply_logp(ids: torch.Tensor, said: int) -> torch.Tensor:
        hidden = inner.model(input_ids=ids).last_hidden_state[0, -said - 1:-1]
        target = ids[0, -said:]
        total = hidden.new_zeros((), dtype=torch.float32)
        for at in range(0, said, SLICE):
            total = total + checkpoint(scored, hidden[at:at + SLICE], target[at:at + SLICE], use_reentrant=False)
        return total

    encoded = [(encode(lesson), lesson["label"]) for lesson in lessons]
    encoded = [(item, label) for item, label in encoded if item is not None]
    good = sum(label for _, label in encoded)
    bad = len(encoded) - good
    reference = []
    with torch.no_grad(), model.disable_adapter():  # the model as it was, once for every reply
        for (ids, said), _ in encoded:
            reference.append(float(reply_logp(ids, said)))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.rate, weight_decay=0.0)
    weight = {True: 1.0, False: good / max(bad, 1)}  # the brain fails more than it succeeds: each kind counts alike
    moved, losses, step, accumulate = 0.0, [], 0, 8
    order = list(range(len(encoded)))
    for _ in range(epochs):
        random.Random(7 + step).shuffle(order)
        for index in order:
            (ids, said), label = encoded[index]
            ratio = reply_logp(ids, said) - reference[index]
            point = max(moved, 0.0)  # KTO's reference point: how far the model has moved, on average
            value = torch.sigmoid(BETA * (ratio - point)) if label else torch.sigmoid(BETA * (point - ratio))
            loss = weight[label] * (1 - value)
            (loss / accumulate).backward()
            moved = 0.9 * moved + 0.1 * float(ratio.detach())
            losses.append(float(loss.detach()))
            step += 1
            if step % accumulate == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                print(f"step {step}: loss {sum(losses[-accumulate:]) / accumulate:.4f}", flush=True)
    if step % accumulate:
        optimizer.step()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(MARK + json.dumps({
        "ok": True, "lessons": len(encoded), "good": good, "bad": bad,
        "loss": round(sum(losses) / max(len(losses), 1), 4), "loss_last": round(sum(losses[-16:]) / max(len(losses[-16:]), 1), 4),
        "minutes": round((time.time() - started) / 60, 1), "memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }))


def fold(adapter: str, out: str) -> None:
    """Fold a learned adapter into a copy of the model and make that copy as small as the model the
    server runs: the server no longer loads an adapter beside a model. A run of its own, after the
    training has let go of its memory: done in the same run, the copying ran out of memory."""
    started, work, model_file, why = time.time(), Path("/fold"), Path(out) / "model.gguf", ""
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(adapter, max_seq_length=2048, load_in_4bit=True)
        model.save_pretrained_merged(str(work / "merged"), tokenizer, save_method="merged_16bit")
        del model
        for step in (
            [sys.executable, "/opt/llama.cpp/convert_hf_to_gguf.py", str(work / "merged"), "--outtype", "f16",
             "--outfile", str(work / "f16.gguf")],
            ["/opt/llama.cpp/build/bin/llama-quantize", str(work / "f16.gguf"), str(model_file), "Q4_K_M"],
        ):
            done = subprocess.run(step, capture_output=True, text=True)
            if done.returncode:
                raise RuntimeError(done.stderr[-600:] or done.stdout[-600:])
    except Exception as exc:
        why = f"folding it into the model failed: {str(exc)[-500:]}"
    finally:
        subprocess.run(["rm", "-rf", str(work / "merged"), str(work / "f16.gguf")])
    print(MARK + json.dumps({
        "ok": model_file.is_file() and not why, "file": model_file.name if model_file.is_file() else "",
        "fold_minutes": round((time.time() - started) / 60, 1), "why": why,
    }))

def to_gguf(base: str, out: str) -> None:
    """Turn a learned adapter into llama.cpp's format, for its server to load beside the model."""
    started, adapter, why = time.time(), Path(out) / "adapter.gguf", ""
    done = subprocess.run(
        [sys.executable, "/opt/llama.cpp/convert_lora_to_gguf.py", out, "--base", base, "--outtype", "f16",
         "--outfile", str(adapter)],
        capture_output=True, text=True,
    )
    if done.returncode or not adapter.is_file():
        why = f"turning the adapter into llama.cpp's format failed: {(done.stderr or done.stdout)[-600:]}"
    print(MARK + json.dumps({
        "ok": not why, "file": adapter.name if adapter.is_file() else "", "gguf_minutes": round((time.time() - started) / 60, 1),
        "why": why,
    }))


if __name__ == "__main__":
    main()
