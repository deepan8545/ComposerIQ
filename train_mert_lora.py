"""
MERT LoRA Fine-Tuning (P6)
------------------------------
Fine-tunes MERT-v1-95M on benchmark engagement labels using LoRA adapters
attached to the last 4 transformer layers' q_proj and v_proj.

The fine-tuned model learns to embed audio such that the vector distance
directly correlates with engagement quality, not just musical similarity.

Requirements (install separately):
    pip install peft

Usage:
    python train_mert_lora.py

Output:
    models/mert_lora/  — directory with LoRA adapter weights

After training, mert_embedder.py automatically loads the adapter if the
directory exists, so no other code changes are needed.
"""
import json
import sys
import time
import numpy as np
from pathlib import Path

MODELS_DIR    = Path(__file__).parent / "models"
LORA_DIR      = MODELS_DIR / "mert_lora"
MANIFEST_PATH = Path(__file__).parent / "benchmark_manifest.json"
CLUSTER_STATS_PATH = Path(__file__).parent / "cluster_stats.json"


def check_dependencies():
    """Verify required packages are available."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        missing.append("peft (pip install peft)")

    try:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
    except ImportError:
        missing.append("transformers")

    if missing:
        print("[LoRA] Missing dependencies:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)


def load_training_data() -> list[dict]:
    """Load processed training data from models/training_data.json (built by train_calibrator.py)."""
    training_path = MODELS_DIR / "training_data.json"
    if not training_path.exists():
        print("[LoRA] No training_data.json found. Run train_calibrator.py first.")
        sys.exit(1)

    with open(training_path) as f:
        data = json.load(f)

    print(f"[LoRA] Loaded {len(data)} training records")
    return data


def load_manifest_audio_map() -> dict[str, str]:
    """Build {track_id: audio_path} map from benchmark_manifest.json."""
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    audio_map: dict[str, str] = {}
    for cluster in manifest.values():
        for t in cluster.get("tracks", []):
            if t.get("audio_path") and t.get("track_id"):
                audio_map[t["track_id"]] = t["audio_path"]
    return audio_map


def build_dataset(training_data: list[dict], audio_map: dict[str, str]):
    """
    Build (audio_path, quality_label) pairs from training records.
    Only includes records with real audio files.
    """
    pairs = []
    skipped = 0

    for rec in training_data:
        tid   = rec.get("track_id", "")
        label = float(rec.get("quality_label", 0.0))
        audio = audio_map.get(tid, "")

        if not audio:
            skipped += 1
            continue

        full_path = Path(__file__).parent / audio
        if not full_path.exists():
            skipped += 1
            continue

        pairs.append((str(full_path), label))

    print(f"[LoRA] Dataset: {len(pairs)} tracks with audio ({skipped} skipped)")
    return pairs


def train(pairs: list, n_epochs: int = 5, lr: float = 1e-4, batch_size: int = 4):
    """
    Fine-tune MERT with LoRA on engagement labels.

    Architecture:
      MERT-v1-95M → LoRA on last 4 layers' q_proj + v_proj
      → mean-pool last hidden state → Linear(768, 1) → MSE loss

    Training is intentionally conservative (few epochs, small LR) to
    avoid catastrophic forgetting — the base MERT embeddings should remain
    musically meaningful.
    """
    import torch
    import torch.nn as nn
    import librosa
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    from peft import LoraConfig, get_peft_model, TaskType

    MODELS_DIR.mkdir(exist_ok=True)
    LORA_DIR.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[LoRA] Device: {device}")

    # ---- Load base MERT model ----
    print("[LoRA] Loading MERT-v1-95M base model...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        "m-a-p/MERT-v1-95M", trust_remote_code=True
    )
    base_model = AutoModel.from_pretrained(
        "m-a-p/MERT-v1-95M", trust_remote_code=True
    )

    # ---- Attach LoRA to last 4 transformer layers ----
    n_layers = len(base_model.encoder.layers)
    target_layers = [f"encoder.layers.{i}" for i in range(n_layers - 4, n_layers)]
    target_modules = []
    for layer_prefix in target_layers:
        target_modules.extend([
            f"{layer_prefix}.attention.q_proj",
            f"{layer_prefix}.attention.v_proj",
        ])

    lora_config = LoraConfig(
        r=8,                           # LoRA rank
        lora_alpha=16,                 # scaling factor
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    print(f"[LoRA] Attaching LoRA adapters to: {target_modules}")
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()
    peft_model = peft_model.to(device)

    # ---- Regression head ----
    head = nn.Linear(768, 1).to(device)

    # ---- Optimiser ----
    params = list(peft_model.parameters()) + list(head.parameters())
    optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=n_epochs * (len(pairs) // batch_size + 1)
    )
    loss_fn = nn.MSELoss()

    # ---- Normalise labels to 0-1 range ----
    labels  = np.array([p[1] for p in pairs], dtype=np.float32)
    y_min   = labels.min()
    y_max   = labels.max()
    y_range = y_max - y_min + 1e-8

    # ---- Training loop ----
    print(f"\n[LoRA] Training {n_epochs} epochs on {len(pairs)} tracks...")
    best_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        peft_model.train()
        head.train()
        epoch_losses = []

        # Shuffle each epoch
        indices = np.random.permutation(len(pairs))
        batches = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]

        for batch_idx, batch in enumerate(batches):
            optimiser.zero_grad()
            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
            valid_count = 0

            for i in batch:
                audio_path, label_raw = pairs[i]
                label_norm = (label_raw - y_min) / y_range

                try:
                    y, _ = librosa.load(audio_path, sr=24000, mono=True)
                    # Truncate to 30s to keep training fast
                    y = y[: 24000 * 30]
                    inputs = processor(y, sampling_rate=24000, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    # Call the inner MERT model directly to avoid PEFT's forward
                    # wrapper injecting `input_ids` (an LLM concept MERT rejects).
                    out = peft_model.base_model.model(
                        **inputs, output_hidden_states=True, return_dict=True
                    )
                    # Mean-pool last 4 hidden layers
                    hidden = torch.stack(out.hidden_states[-4:]).mean(dim=0)
                    emb    = hidden.mean(dim=1)  # (1, 768)

                    pred  = head(emb).squeeze()  # scalar
                    y_tgt = torch.tensor(label_norm, dtype=torch.float32, device=device)
                    loss  = loss_fn(pred, y_tgt)
                    batch_loss = batch_loss + loss
                    valid_count += 1

                except Exception as e:
                    print(f"  [LoRA] Skipping {Path(audio_path).name}: {e}")
                    continue

            if valid_count > 0:
                avg_loss = batch_loss / valid_count
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimiser.step()
                scheduler.step()
                epoch_losses.append(float(avg_loss.detach()))

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        print(f"  Epoch {epoch}/{n_epochs}: loss={mean_loss:.4f}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            # Save best checkpoint
            peft_model.save_pretrained(str(LORA_DIR))
            torch.save(head.state_dict(), str(LORA_DIR / "head.pt"))

    # Save label normalisation params for inference
    import json as _json
    norm_params = {"y_min": float(y_min), "y_max": float(y_max)}
    with open(LORA_DIR / "norm_params.json", "w") as fp:
        _json.dump(norm_params, fp)

    print(f"\n[LoRA] Training complete. Best loss: {best_loss:.4f}")
    print(f"[LoRA] Adapter saved to: {LORA_DIR}")


def main():
    print("=" * 64)
    print("  ComposerIQ — MERT LoRA Fine-Tuning (P6)")
    print("  Trains LoRA adapters on benchmark engagement labels")
    print("=" * 64)

    check_dependencies()

    print("\n[Step 1] Loading training data...")
    training_data = load_training_data()
    audio_map     = load_manifest_audio_map()

    print("\n[Step 2] Building dataset...")
    pairs = build_dataset(training_data, audio_map)

    if len(pairs) < 5:
        print(f"[FAILED] Only {len(pairs)} tracks with audio. Need at least 5.")
        sys.exit(1)

    print(f"\n[Step 3] Fine-tuning MERT with LoRA...")
    start = time.time()
    train(pairs)
    elapsed = time.time() - start

    print(f"\n{'=' * 64}")
    print(f"  DONE in {elapsed:.0f}s")
    print(f"  Adapter:  {LORA_DIR}")
    print(f"  Restart the app — mert_embedder.py will auto-load the adapter.")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
