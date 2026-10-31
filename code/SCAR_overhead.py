"""SCAR Stage-3 inference computational overhead benchmark.

Derived from the user's existing SCAR/SBDADB evaluation script. Run this file
in the SAME project directory as resnet.py, mobilenet.py, densenet.py,
generator.py and utils.py. No training is performed.

The source evaluator is intentionally untouched. This benchmark reuses its
checkpoint/data partition and PGD + trigger-generation recipe, but measures
ONLY inference on pre-created clean/poisoned images, excluding attack creation,
dataset loading, metrics calculation, and model/checkpoint loading.

Benchmarked methods:
  Baseline          : F(x)
  SF_CBA            : F(x), JPEG50 + F(JPEG50(x)), cosine detection (no AR)
  SCAR_AlwaysAR     : original Stage-3 implementation, AR on every image
  SCAR_Conditional  : Stage-3 Algorithm-1 implementation, AR only if flagged

Both SCAR variants must produce exactly the same predictions and CBA flags.
Results include separate Clean/Poisoned trigger rates, latency, overhead,
throughput and GPU memory. The conditional routing decision itself is timed.
"""

import argparse
import io
import os
import random
import re
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from resnet import get_resnet
from mobilenet import MobileNetV2
from densenet import DenseNet
from generator import Autoencoder
from utils import set_random_seed

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark SCAR Stage-3 inference overhead")
    p.add_argument("--exp_root", default="experiment_results_collection_01")
    p.add_argument("--experiment", default=None,
                   help="Only evaluate this exact experiment subfolder name (default: all)")
    p.add_argument("--csv_out", default="scar_overhead.csv")
    p.add_argument("--dataset", default="cifar10",
                   choices=["cifar10", "cifar100", "tinyimagenet"])
    p.add_argument("--model", default="resnet10")
    p.add_argument("--sim_threshold", type=float, default=0.8)
    p.add_argument("--q_recover_list", type=int, nargs="+", default=[70, 40])
    p.add_argument("--detect_quality", type=int, default=50)
    p.add_argument("--stage", type=int, default=3,
                   help="Only Stage 3 is implemented for the overhead comparison")
    p.add_argument("--client_num", type=int, default=100)
    p.add_argument("--client_batch", type=int, default=1,
                   help="Inference batch size; use 1 for per-query latency")
    p.add_argument("--max_clients", type=int, default=1,
                   help="Limit models per experiment, 0 for all")
    p.add_argument("--client_ids", type=int, nargs="*", default=None,
                   help="Optional explicit client IDs, e.g. --client_ids 0 5 20")
    p.add_argument("--max_samples_per_client", type=int, default=64,
                   help="Max clean images/client (and corresponding poisoned), 0 for all")
    p.add_argument("--repeats", type=int, default=3,
                   help="Timing repeats per input batch")
    p.add_argument("--warmup", type=int, default=5,
                   help="Untimed calls per method before the first measurement/client")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--shuffle", type=int, default=0, choices=[0, 1],
                   help="Match client mapping used by the original evaluator")
    p.add_argument("--attack_target_class", type=int, default=0)
    
    args = p.parse_args()
    
    if args.stage != 3:
        p.error("Only --stage 3 supports conditional AR overhead measurement")
        
    if args.client_batch < 1 or args.repeats < 1 or args.warmup < 0:
        p.error("--client_batch and --repeats must be positive; --warmup >= 0")
        
    if args.max_clients < 0 or args.max_samples_per_client < 0:
        p.error("--max_clients and --max_samples_per_client must be >= 0")
        
    if any(not (1 <= q <= 100) for q in [args.detect_quality, *args.q_recover_list]):
        p.error("JPEG quality must be between 1 and 100")
        
    return args


def create_model(model_name, num_classes):
    model_name = model_name.lower()
    if "resnet" in model_name:
        size = int(model_name.replace("resnet", "")) if model_name != "resnet" else 10
        return get_resnet(size=size, num_classes=num_classes)
    
    if model_name == "mobilenet":
        return MobileNetV2(n_classes=num_classes)
    
    if model_name == "densenet":
        return DenseNet(nClasses=num_classes)
    
    raise ValueError(f"Unsupported model: {model_name}")


def apply_jpeg_defense(images, quality=50):
    """Exactly the PIL/BytesIO approach from the uploaded evaluator."""
    restored_list = []
    for img in images:
        pil_img = torchvision.transforms.ToPILImage()(img.cpu())
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        
        with Image.open(buffer) as jpeg_pil:
            restored_list.append(torchvision.transforms.ToTensor()(jpeg_pil))
            
    return torch.stack(restored_list).to(images.device)


def pgd_attack(model, images, labels):
    """Same 1-step PGD recipe as the original evaluation script."""
    epsilon = 4.0 / 255.0
    alpha = 4.0 / 255.0
    adversarial = images.detach().clone() + torch.zeros_like(images).uniform_(-epsilon, epsilon)
    adversarial = adversarial.clamp(0, 1)
    
    with torch.enable_grad():
        adversarial.requires_grad_(True)
        logits = model(adversarial)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        adversarial = adversarial + alpha * adversarial.grad.sign()
        eta = (adversarial - images).clamp(-epsilon, epsilon)
        adversarial = (images + eta).clamp(0, 1).detach()
        
    return adversarial


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def detect(model, images, quality, threshold):
    """Returns baseline prediction and CBA suspicious-mask for each image."""
    original_logits = model(images)
    original_preds = original_logits.argmax(dim=1)
    
    jpeg_images = apply_jpeg_defense(images, quality=quality)
    jpeg_logits = model(jpeg_images)
    
    similarity = F.cosine_similarity(original_logits, jpeg_logits, dim=1)
    flagged = similarity < threshold
    
    return original_preds, flagged


def infer_baseline(model, images, args):
    preds = model(images).argmax(dim=1)
    return preds, None


def infer_sf_cba(model, images, args):
    preds, flagged = detect(model, images, args.detect_quality, args.sim_threshold)
    return preds, flagged


def recover_predictions(model, images, orig_preds, qualities):
    """Q fusion and original-class masking, identical to Stage 3."""
    probs_sum = None
    for q in qualities:
        filtered = apply_jpeg_defense(images, quality=q)
        probs = torch.softmax(model(filtered), dim=1)
        probs_sum = probs if probs_sum is None else probs_sum + probs
        
    mean_probs = probs_sum / len(qualities)
    # Use batch-local indices; unlike the uploaded original, they reside on
    # the correct device even when running on a GPU.
    rows = torch.arange(images.size(0), device=orig_preds.device)
    mean_probs[rows, orig_preds] = -float("inf")
    
    return mean_probs.argmax(dim=1)


def infer_scar_always_ar(model, images, args):
    """Original Stage-3 behavior: run AR for EVERY image then torch.where."""
    base_preds, flagged = detect(model, images, args.detect_quality, args.sim_threshold)
    recovery_preds = recover_predictions(model, images, base_preds, args.q_recover_list)
    final_preds = torch.where(flagged, recovery_preds, base_preds)
    
    return final_preds, flagged


def infer_scar_conditional(model, images, args):
    """Algorithm-1 behavior: invoke Q70/Q40 AR ONLY for CBA-flagged images."""
    base_preds, flagged = detect(model, images, args.detect_quality, args.sim_threshold)
    final_preds = base_preds.clone()
    
    # `.any()` changes control flow, so with CUDA the needed sync is included
    # in the latency measurement. For a batch, only suspicious rows go to AR.
    if bool(flagged.any().item()):
        suspect_images = images[flagged]
        suspect_base_preds = base_preds[flagged]
        recovery_preds = recover_predictions(model, suspect_images, suspect_base_preds, args.q_recover_list)
        final_preds[flagged] = recovery_preds
        
    return final_preds, flagged


METHODS = {
    "Baseline": infer_baseline,
    "SF_CBA": infer_sf_cba,
    "SCAR_AlwaysAR": infer_scar_always_ar,
    "SCAR_Conditional": infer_scar_conditional,
}


def time_inference(fn, model, images, args, device):
    """Time only the prediction path, including CPU JPEG and GPU synchronizations."""
    sync_device(device)
    if device.type == "cuda":
        before_alloc = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        
    else:
        before_alloc = None
        
    start = time.perf_counter()
    
    with torch.inference_mode():
        preds, flagged = fn(model, images, args)
        
    sync_device(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    
    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated(device)
        peak_gpu_mb = peak_alloc / (1024.0 ** 2)
        extra_gpu_mb = max(0.0, peak_alloc - before_alloc) / (1024.0 ** 2)
        
    else:
        peak_gpu_mb = float("nan")
        extra_gpu_mb = float("nan")
        
    return preds, flagged, elapsed_ms, peak_gpu_mb, extra_gpu_mb


def warmup_model(model, images, args, device):
    if args.warmup == 0:
        return
    
    with torch.inference_mode():
        for _ in range(args.warmup):
            for fn in METHODS.values():
                fn(model, images, args)
                
    sync_device(device)


def sort_experiment_keys(folder_name):
    bad_match = re.search(r"Bad(\d+)", folder_name)
    rate_match = re.search(r"Rate([\d.]+)", folder_name)
    bad_num = int(bad_match.group(1)) if bad_match else -1
    rate_num = float(rate_match.group(1)) if rate_match else -1.0
    
    return (bad_num, rate_num)


def load_dataset(dataset_name):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
    ])
    if dataset_name == "cifar10":
        return torchvision.datasets.CIFAR10("./data", train=False, download=False, transform=transform), 10
    
    if dataset_name == "cifar100":
        return torchvision.datasets.CIFAR100("./data", train=False, download=False,transform=transform), 100
    
    return torchvision.datasets.ImageFolder("./data/tiny-imagenet-200/val", transform=transform), 200


def safe_percent(numerator, denominator):
    return 100.0 * numerator / denominator if denominator else float("nan")


def summary_rows(stats, args, device):
    """Generate per-client and pooled rows. Pooled values weight actual queries."""
    rows = []
    for (experiment, cid, input_kind), block in stats.items():
        counts = block["counts"]
        n = counts["images"]
        
        if n == 0:
            continue
        
        base_time = sum(block["methods"]["Baseline"]["ms"])
        base_calls = block["methods"]["Baseline"]["images_timed"]
        base_ms_img = base_time / base_calls
        
        for method, data in block["methods"].items():
            ms_values = np.array(data["ms_per_image"], dtype=float)
            mean_ms_img = sum(data["ms"]) / data["images_timed"]
            
            rows.append({
                "Experiment": experiment,
                "Client": cid,
                "Input": input_kind,
                "Method": method,
                "Num_Images": n,
                "Batch_Size": args.client_batch,
                "Repeats": args.repeats,
                "Device": str(device),
                "Model": args.model,
                "Detect_Quality": args.detect_quality,
                "Threshold": args.sim_threshold,
                "Recover_Qualities": str(args.q_recover_list),
                "Mean_Latency_ms_per_image": mean_ms_img,
                "P95_Batch_Normalized_ms_per_image": float(np.percentile(ms_values, 95)),
                "Throughput_images_per_s": 1000.0 / mean_ms_img,
                "Overhead_vs_Baseline_pct": (mean_ms_img / base_ms_img - 1.0) * 100.0,
                "Peak_GPU_Allocated_MB": max(data["peak_mb"], default=float("nan")),
                "Peak_Extra_GPU_Allocated_MB": max(data["extra_mb"], default=float("nan")),
                "AR_Trigger_Rate_pct": safe_percent(counts["ar_triggered"], n),
                "Clean_ACC_Baseline_pct": safe_percent(counts["base_clean_correct"], n) if input_kind == "Clean" else float("nan"),
                "Clean_ACC_SCAR_pct": safe_percent(counts["scar_clean_correct"], n) if input_kind == "Clean" else float("nan"),
                "Poison_ASR_Baseline_pct": safe_percent(counts["base_attack_success"], counts["poison_non_target"]) if input_kind == "Poisoned" else float("nan"),
                "Poison_ASR_SCAR_pct": safe_percent(counts["scar_attack_success"], counts["poison_non_target"]) if input_kind == "Poisoned" else float("nan"),
                "Prediction_Mismatches_Original_vs_Conditional": counts["prediction_mismatch"],
            })
            
    return rows


def make_block():
    return {
        "counts": defaultdict(int),
        "methods": {method: {"ms": [], "ms_per_image": [], "images_timed": 0,
                             "peak_mb": [], "extra_mb": []} for method in METHODS},
    }


def add_timing(block, method, elapsed_ms, n, peak_gpu_mb, extra_gpu_mb, repeats):
    m = block["methods"][method]
    m["ms"].append(elapsed_ms)
    m["ms_per_image"].append(elapsed_ms / n)
    m["images_timed"] += n
    
    if not np.isnan(peak_gpu_mb):
        m["peak_mb"].append(peak_gpu_mb)
        
    if not np.isnan(extra_gpu_mb):
        m["extra_mb"].append(extra_gpu_mb)


def run_benchmark():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.exp_root)
    
    if not root.is_dir():
        raise FileNotFoundError(f"Experiment root does not exist: {root}")
    
    set_random_seed(args.seed)
    dataset, num_classes = load_dataset(args.dataset)
    
    if args.experiment is None:
        experiments = sorted([p for p in root.iterdir() if p.is_dir()],
                             key=lambda p: sort_experiment_keys(p.name), reverse=True)
    else:
        experiments = [root / args.experiment]
        if not experiments[0].is_dir():
            raise FileNotFoundError(f"Experiment not found: {experiments[0]}")

    stats = {}
    print(f"Device={device}, model={args.model}, data={args.dataset}, "
          f"batch={args.client_batch}, repeat={args.repeats}")
    
    if args.client_batch != 1:
        print("NOTE: Latency is normalized per image from BATCH timing; "
              "it is not single-query latency.")
        
    for exp_dir in experiments:
        generator_path = exp_dir / "trained_trigger_generator.pth"
        indices_path = exp_dir / "client_test_data_indices.pt"
        
        if not generator_path.is_file() or not indices_path.is_file():
            print(f"SKIP {exp_dir.name}: generator or index file missing")
            continue

        available_models = {}
        bad_paths = []
        for model_path in exp_dir.iterdir():
            if model_path.suffix != ".pth":
                continue
            
            if "bad_attacker" in model_path.name:
                bad_paths.append(model_path)
                
            if "clean_client" in model_path.name or "bad_attacker" in model_path.name:
                match = re.search(r"_(\d+)\.pth$", model_path.name)
                
                if match:
                    available_models[int(match.group(1))] = model_path
                    
        if not available_models or not bad_paths:
            print(f"SKIP {exp_dir.name}: client/surrogate model missing")
            continue

        generator = Autoencoder().to(device)
        generator.load_state_dict(torch.load(generator_path, map_location=device, weights_only=True))
        generator.eval()
        surrogate = create_model(args.model, num_classes).to(device)
        surrogate.load_state_dict(torch.load(sorted(bad_paths)[0], map_location=device, weights_only=True))
        surrogate.eval()
        model = create_model(args.model, num_classes).to(device)
        client_indices = torch.load(indices_path, map_location="cpu", weights_only=False)

        # Preserve original evaluator's optional shuffle mapping.
        mapping = list(range(args.client_num))
        if args.shuffle:
            random.shuffle(mapping)
            
        selected_ids = sorted(available_models)
        if args.client_ids is not None:
            selected_ids = [cid for cid in selected_ids if cid in args.client_ids]
            
        if args.max_clients:
            selected_ids = selected_ids[:args.max_clients]
            
        print(f"Experiment={exp_dir.name}: selected clients={selected_ids}")

        for cid in tqdm(selected_ids, desc=f"Benchmark {exp_dir.name}"):
            model.load_state_dict(torch.load(available_models[cid], map_location=device, weights_only=True))
            model.eval()
            original_idx = mapping[cid]
            loader = DataLoader(dataset, batch_size=args.client_batch,
                                sampler=SubsetRandomSampler(client_indices[original_idx]),
                                drop_last=False)
            warmed_up = False
            samples_done = 0
            
            for clean_cpu, labels_cpu in loader:
                if args.max_samples_per_client and samples_done >= args.max_samples_per_client:
                    break
                
                if args.max_samples_per_client:
                    remaining = args.max_samples_per_client - samples_done
                    clean_cpu = clean_cpu[:remaining]
                    labels_cpu = labels_cpu[:remaining]
                    
                if clean_cpu.numel() == 0:
                    break
                
                n = clean_cpu.size(0)
                samples_done += n
                # Poison production is OUTSIDE the timer and unchanged in method.
                clean_on_device = clean_cpu.to(device)
                labels_on_device = labels_cpu.to(device)
                adv = pgd_attack(surrogate, clean_on_device, labels_on_device)
                
                with torch.no_grad():
                    trigger = generator(clean_on_device) / 255.0 * 4.0
                    poisoned_cpu = (adv + trigger).clamp(0, 1).detach().cpu()
                    
                del adv, trigger, clean_on_device, labels_on_device
                sync_device(device)

                for kind, cpu_images in (("Clean", clean_cpu), ("Poisoned", poisoned_cpu)):
                    images = cpu_images.to(device)
                    labels = labels_cpu.to(device)
                    key = (exp_dir.name, cid, kind)
                    
                    if key not in stats:
                        stats[key] = make_block()
                        
                    block = stats[key]
                    block["counts"]["images"] += n
                    
                    if not warmed_up:
                        warmup_model(model, images, args, device)
                        # Warmup the model once per client, before any measured calls.
                        # Subsequent input types share warmed-up model/kernel/cache.

                    predictions = {}
                    masks = {}
                    # Rotate method order every batch to reduce systematic bias.
                    order = list(METHODS.items())
                    
                    if (samples_done // args.client_batch) % 2 == 0:
                        order.reverse()
                        
                    for method, fn in order:
                        for repeat_idx in range(args.repeats):
                            preds, flags, ms, peak_mb, extra_mb = time_inference(fn, model, images, args, device)
                            add_timing(block, method, ms, n, peak_mb, extra_mb, args.repeats)
                            
                            if repeat_idx == 0:
                                predictions[method] = preds.detach().clone()
                                masks[method] = flags.detach().clone() if flags is not None else None

                    a = predictions["SCAR_AlwaysAR"]
                    b = predictions["SCAR_Conditional"]
                    
                    flagged_a = masks["SCAR_AlwaysAR"]
                    flagged_b = masks["SCAR_Conditional"]
                    
                    mismatches = (a != b).sum().item()
                    flag_mismatch = (flagged_a != flagged_b).sum().item()
                    block["counts"]["prediction_mismatch"] += mismatches
                    
                    if mismatches or flag_mismatch:
                        raise AssertionError(
                            f"Old/new SCAR disagree: {exp_dir.name}, client={cid}, "
                            f"input={kind}, different predictions={mismatches}, "
                            f"different CBA flags={flag_mismatch}")
                        
                    block["counts"]["ar_triggered"] += flagged_b.sum().item()
                    
                    if kind == "Clean":
                        block["counts"]["base_clean_correct"] += (predictions["Baseline"] == labels).sum().item()
                        block["counts"]["scar_clean_correct"] += (b == labels).sum().item()
                        
                    else:
                        non_target = labels != args.attack_target_class
                        block["counts"]["poison_non_target"] += non_target.sum().item()
                        block["counts"]["base_attack_success"] += (predictions["Baseline"][non_target] == args.attack_target_class).sum().item()
                        block["counts"]["scar_attack_success"] += (b[non_target] == args.attack_target_class).sum().item() 
                        
                    del images, labels
                    sync_device(device)
                    
                warmed_up = True
                del poisoned_cpu

    if not stats:
        raise RuntimeError("No benchmark samples found. Check paths and model filenames.")
    
    rows = summary_rows(stats, args, device)
    df = pd.DataFrame(rows)
    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Pooled across clients, by experiment and input; pooling is QUERY-weighted.
    pooled = {}
    for (exp, cid, kind), block in stats.items():
        poolkey = (exp, "ALL", kind)
        dest = pooled.setdefault(poolkey, make_block())
        
        for k, v in block["counts"].items():
            dest["counts"][k] += v
            
        for method in METHODS:
            source = block["methods"][method]
            target = dest["methods"][method]
            
            for list_key in ("ms", "ms_per_image", "peak_mb", "extra_mb"):
                target[list_key].extend(source[list_key])
                
            target["images_timed"] += source["images_timed"]
            
    pooled_df = pd.DataFrame(summary_rows(pooled, args, device))
    pool_path = csv_path.with_name(csv_path.stem + "_pooled.csv")
    pooled_df.to_csv(pool_path, index=False, encoding="utf-8-sig")

    preview_columns = ["Experiment", "Input", "Method", "Num_Images",
                       "Mean_Latency_ms_per_image", "Overhead_vs_Baseline_pct",
                       "AR_Trigger_Rate_pct", "Peak_GPU_Allocated_MB"]
    
    print("\n=== Pooled Results (query-weighted) ===")
    print(pooled_df[preview_columns].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nPer-client results: {csv_path}")
    print(f"Pooled results:     {pool_path}")
    print("Note: attack generation, model loading and CSV writing are excluded from timing.")
    print("Note: peak allocated GPU memory counts resident model/input allocations too;")
    print("      extra allocated memory is shown in Peak_Extra_GPU_Allocated_MB.")


if __name__ == "__main__":
    run_benchmark()
