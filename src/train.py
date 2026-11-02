"""A-D whole-image ablations; local phrase/region training is not implemented."""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
import transformers
import PIL
import safetensors
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

from src.data import BananaDataset, collate, verify_disjoint
from src.model import BlipExperiment

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {"A": "none", "B": "coarse", "C": "coarse_caption", "D": "caption"}


def config_path(value):
    p = Path(value)
    return p if p.is_absolute() else (ROOT / p).resolve()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metrics(truth, predicted):
    matrix = [[0] * 3 for _ in range(3)]
    for t, p in zip(truth, predicted):
        matrix[t][p] += 1
    per_class = []
    for c in range(3):
        tp = matrix[c][c]
        fp = sum(matrix[r][c] for r in range(3)) - tp
        fn = sum(matrix[c]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"precision": precision, "recall": recall, "f1": f1,
                          "support": sum(matrix[c])})
    return {"accuracy": sum(matrix[c][c] for c in range(3)) / max(1, len(truth)),
            "macro_f1": sum(x["f1"] for x in per_class) / 3,
            "per_class": per_class, "confusion_matrix": matrix, "n": len(truth)}


def make_loader(data, batch, shuffle=False, seed=0, workers=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(data, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      collate_fn=collate, generator=generator, pin_memory=True)


def negative_texts(dataset, labels, rng):
    # Category labels are used ONLY for sampling; this is noisy image-text
    # supervision, especially for D where visual descriptions can overlap.
    negatives = []
    for label in labels:
        other = rng.choice([x for x in range(3) if x != label])
        index = rng.choice(dataset.by_label[other])
        negatives.append(dataset.text(index))
    return negatives


def image_tensor(processor, images, device):
    return processor.image_processor(images=images, return_tensors="pt")["pixel_values"].to(
        device, non_blocking=True)


def evaluate(model, processor, dataset, device, batch_size, amp, max_items=None):
    if max_items is not None:
        dataset = Subset(dataset, range(min(max_items, len(dataset))))
    loader = make_loader(dataset, batch_size)
    truth, predicted = [], []
    model.eval()
    with torch.no_grad():
        for rows in loader:
            pixels = image_tensor(processor, rows["image"], device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(pixels)["class_logits"]
            truth.extend(rows["label"])
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
    return metrics(truth, predicted)


def token_stats(tokenizer, dataset, max_length):
    if dataset.text_mode == "none":
        return None
    lengths = []
    for start in range(0, len(dataset), 256):
        strings = [dataset.text(i) for i in range(start, min(start + 256, len(dataset)))]
        encoded = tokenizer(strings, add_special_tokens=True, truncation=False, padding=False,
                            return_length=True, verbose=False)
        lengths.extend(encoded["length"])
    return {"count": len(lengths), "max": max(lengths), "limit": max_length,
            "truncated_count": sum(x > max_length for x in lengths),
            "truncated_fraction": sum(x > max_length for x in lengths) / len(lengths)}


def atomic_save(obj, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temp)
    os.replace(temp, path)


def log_event(path, data):
    line = json.dumps(data, ensure_ascii=False, allow_nan=False)
    print(line, flush=True)
    if path is not None:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def run(cfg, group, smoke=False, resume=False):
    seed = int(cfg["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; refusing unintended CPU training")
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    data_root = config_path(cfg["data_root"])
    checkpoint = config_path(cfg["checkpoint_dir"])
    output = config_path(cfg["output_root"]) / group
    log = None
    if not smoke:
        if output.exists() and not resume:
            raise FileExistsError(f"Output exists; use --resume or a new output_root: {output}")
        output.mkdir(parents=True, exist_ok=True)
        log = output / "events.jsonl"
    train_data = BananaDataset(data_root, "train", GROUPS[group])
    val_data = BananaDataset(data_root, "validation", "none")
    test_data = BananaDataset(data_root, "test", "none")
    verify_disjoint([train_data, val_data, test_data])
    index_hash = sha256(data_root / "datasets" / "banana" / "index.jsonl")
    weight_hash = sha256(checkpoint / "model.safetensors")
    if weight_hash != cfg["checkpoint_safetensors_sha256"]:
        raise ValueError("Local converted checkpoint hash differs from the frozen config")
    processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True, use_fast=False)
    model = BlipExperiment(str(checkpoint), use_text=(group != "A"), rank=cfg["rank"],
                           alpha=cfg["alpha"], dropout=cfg["lora_dropout"],
                           last_layers=cfg["last_layers"]).to(device)
    # Salesforce BLIP retrieval uses [ENC] (not BERT's [CLS]) as the first
    # token in the image-text matching branch. The distributed tokenizer file
    # omits these two added tokens although the checkpoint has their embeddings.
    tokenizer = processor.tokenizer
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    enc_token_id = tokenizer.convert_tokens_to_ids("[ENC]")
    if (tokenizer.bos_token_id != model.blip.config.text_config.bos_token_id
            or enc_token_id == tokenizer.unk_token_id
            or len(tokenizer) != model.blip.config.text_config.vocab_size):
        raise ValueError("BLIP special token IDs do not match checkpoint embeddings")
    max_allowed = model.blip.config.text_config.max_position_embeddings
    if cfg["max_text_tokens"] > max_allowed:
        raise ValueError(f"Token limit exceeds model position limit {max_allowed}")
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    classifier = [p for n, p in trainable if n.startswith("classifier.")]
    adapters = [p for n, p in trainable if not n.startswith("classifier.")]
    optimizer = torch.optim.AdamW([
        {"params": classifier, "lr": cfg["learning_rate_classifier"]},
        {"params": adapters, "lr": cfg["learning_rate_lora_itm"]}],
        weight_decay=cfg["weight_decay"])
    batches = math.ceil(len(train_data) / cfg["batch_size"])
    updates_per_epoch = math.ceil(batches / cfg["gradient_accumulation"])
    total_updates = updates_per_epoch * cfg["epochs"]
    warmup = max(1, int(total_updates * cfg["warmup_fraction"]))
    def lr_scale(step):
        if step < warmup:
            return max(1e-8, step / warmup)
        progress = min(1.0, (step - warmup) / max(1, total_updates - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    amp = bool(cfg["mixed_precision"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch, best_f1 = 0, -1.0
    if resume:
        state = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if (state["config"] != cfg or state["group"] != group
                or state["index_sha256"] != index_hash or state["weight_sha256"] != weight_hash):
            raise ValueError("Resume config, dataset or model differs from checkpoint")
        model.load_trainable_state(state["trainable"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_epoch, best_f1 = state["epoch"], state["best_f1"]
    info = {"event": "start", "group": group, "smoke": smoke,
            "data_counts": [len(train_data), len(val_data), len(test_data)],
            "index_sha256": index_hash, "weight_sha256": weight_hash,
            "checkpoint_repo": cfg["checkpoint_repo"], "checkpoint_revision": cfg["checkpoint_revision"],
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "versions": {"python": platform.python_version(), "transformers": transformers.__version__,
                         "pillow": PIL.__version__, "safetensors": safetensors.__version__,
                         "pyyaml": yaml.__version__},
            "targets": model.targets, "trainable_names": [n for n, _ in trainable],
            "trainable_params": sum(p.numel() for _, p in trainable),
            "total_params": sum(p.numel() for p in model.parameters()),
            "text_stats_train": token_stats(tokenizer, train_data, cfg["max_text_tokens"]),
            "itm_enc_token_id": enc_token_id,
            "config": cfg}
    log_event(log, info)
    if not smoke:
        with (output / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=True)
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic()
        loader = make_loader(train_data, cfg["batch_size"], True, seed + epoch, cfg["num_workers"])
        rng = random.Random(seed + 10000 + epoch)
        optimizer.zero_grad(set_to_none=True)
        running_cls, running_global, seen = 0.0, 0.0, 0
        for step, rows in enumerate(loader):
            pixels = image_tensor(processor, rows["image"], device)
            labels = torch.tensor(rows["label"], device=device, dtype=torch.long)
            kwargs = {}
            if group != "A":
                neg = negative_texts(train_data, rows["label"], rng)
                tokens = tokenizer(rows["text"] + neg, return_tensors="pt",
                                   padding="max_length" if smoke else True,
                                   truncation=True, max_length=cfg["max_text_tokens"])
                tokens["input_ids"][:, 0] = enc_token_id
                kwargs = {"input_ids": tokens["input_ids"].to(device),
                          "attention_mask": tokens["attention_mask"].to(device),
                          "text_image_indices": torch.arange(len(labels), device=device).repeat(2)}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                outputs = model(pixels, **kwargs)
                loss_cls = F.cross_entropy(outputs["class_logits"], labels)
                loss_global = torch.zeros((), device=device)
                if group != "A":
                    itm_target = torch.cat([torch.ones(len(labels), device=device, dtype=torch.long),
                                            torch.zeros(len(labels), device=device, dtype=torch.long)])
                    loss_global = F.cross_entropy(outputs["itm_logits"], itm_target)
                total = loss_cls + cfg["global_loss_weight"] * loss_global
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite loss: group={group} epoch={epoch+1} batch={step}")
            window_start = (step // cfg["gradient_accumulation"]) * cfg["gradient_accumulation"]
            window_size = min(cfg["gradient_accumulation"], len(loader) - window_start)
            if smoke:
                window_size = min(window_size, 2)
            scaler.scale(total / window_size).backward()
            if step == 0:
                grad_names = [n for n, p in trainable if p.grad is not None and torch.isfinite(p.grad).all()]
                if not any("qkv.b.weight" in n for n in grad_names):
                    raise RuntimeError("No finite visual LoRA gradient")
                if group != "A" and not any("crossattention.self.value.b.weight" in n for n in grad_names):
                    raise RuntimeError("No finite fusion LoRA gradient")
                log_event(log, {"event": "gradient_check", "group": group, "epoch": epoch + 1,
                                "finite_gradient_count": len(grad_names)})
            if ((step + 1) % cfg["gradient_accumulation"] == 0 or step + 1 == len(loader)
                    or (smoke and step == 1)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            n = len(labels)
            seen += n
            running_cls += loss_cls.item() * n
            running_global += loss_global.item() * n
            if (step + 1) % 100 == 0:
                log_event(log, {"event": "progress", "group": group, "epoch": epoch + 1,
                                "batch": step + 1, "batches": len(loader), "seen": seen,
                                "cls_loss": running_cls / seen, "global_loss": running_global / seen,
                                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3)})
            if smoke and step >= 1:
                break
        result = evaluate(model, processor, val_data, device, cfg["batch_size"], amp,
                          max_items=6 if smoke else None)
        log_event(log, {"event": "epoch_end", "group": group, "epoch": epoch + 1,
                        "seen": seen, "cls_loss": running_cls / seen,
                        "global_loss": running_global / seen, "val": result,
                        "seconds": round(time.monotonic() - begin, 2),
                        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3)})
        if smoke:
            # Verify adapter serialization/load without writing a checkpoint.
            if not optimizer.state:
                raise RuntimeError("Smoke optimizer step did not create optimizer state")
            current = model.trainable_state()
            with tempfile.TemporaryDirectory(prefix="blip-smoke-") as directory:
                probe = Path(directory) / "checkpoint.pt"
                atomic_save({"trainable": current, "optimizer": optimizer.state_dict(),
                             "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                             "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, probe)
                loaded = torch.load(probe, map_location="cpu", weights_only=True)
            name, param = next((n, p) for n, p in model.named_parameters() if p.requires_grad)
            with torch.no_grad():
                param.add_(1)
            model.load_trainable_state(loaded["trainable"])
            optimizer.load_state_dict(loaded["optimizer"])
            scheduler.load_state_dict(loaded["scheduler"])
            scaler.load_state_dict(loaded["scaler"])
            if not torch.equal(param.detach().cpu(), current[name]):
                raise RuntimeError("Adapter state restoration failed")
            print("SMOKE_OK", group, flush=True)
            return
        is_best = result["macro_f1"] > best_f1
        if is_best:
            best_f1 = result["macro_f1"]
        state = {"epoch": epoch + 1, "group": group, "config": cfg,
                 "best_f1": best_f1, "trainable": model.trainable_state(),
                 "index_sha256": index_hash, "weight_sha256": weight_hash,
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all()}
        atomic_save(state, output / "last.pt")
        if is_best:
            atomic_save(state, output / "best.pt")
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_trainable_state(best["trainable"])
    test = evaluate(model, processor, test_data, device, cfg["batch_size"], amp)
    log_event(log, {"event": "test_end", "group": group, "best_epoch": best["epoch"],
                    "best_validation_macro_f1": best["best_f1"], "test": test})
    with (output / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--group", choices=GROUPS, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-batch-size", type=int)
    parser.add_argument("--smoke-workers", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.resume:
        parser.error("--smoke and --resume cannot be combined")
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.smoke_batch_size is not None:
        if not args.smoke or args.smoke_batch_size < 1:
            parser.error("--smoke-batch-size requires --smoke and a positive value")
        cfg["batch_size"] = args.smoke_batch_size
    if args.smoke_workers is not None:
        if not args.smoke or args.smoke_workers < 0:
            parser.error("--smoke-workers requires --smoke and a nonnegative value")
        cfg["num_workers"] = args.smoke_workers
    run(cfg, args.group, smoke=args.smoke, resume=args.resume)


if __name__ == "__main__":
    main()
