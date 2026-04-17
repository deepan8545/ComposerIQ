"""
MERT Embedder — singleton with optional LoRA adapter (P6)

On startup, checks for models/mert_lora/. If found (and PEFT is installed),
loads the LoRA adapter on top of base MERT-v1-95M for engagement-tuned
embeddings. Otherwise falls back to vanilla MERT (current behaviour).
"""
import numpy as np
import librosa
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_LORA_DIR = Path(__file__).parent / "models" / "mert_lora"


class MERTEmbedder:
    def __init__(self):
        self._model     = None
        self._processor = None
        self._lora_loaded = False

    def _load(self):
        if self._model is not None:
            return

        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-95M", trust_remote_code=True
        )
        base_model = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-95M", trust_remote_code=True
        )

        # ---- Attempt to load LoRA adapter (P6) ----
        if _LORA_DIR.exists() and (_LORA_DIR / "adapter_config.json").exists():
            try:
                from peft import PeftModel
                self._model = PeftModel.from_pretrained(base_model, str(_LORA_DIR))
                self._lora_loaded = True
                print("[MERT] LoRA adapter loaded from", _LORA_DIR)
            except Exception as e:
                print(f"[MERT] LoRA load failed ({e}), using base MERT")
                self._model = base_model
        else:
            self._model = base_model

        self._model.eval()
        if not self._lora_loaded:
            print("[MERT] Loaded base MERT-v1-95M (no LoRA adapter found)")

    def embed(self, audio_path: str) -> np.ndarray:
        import torch
        self._load()
        y, _ = librosa.load(audio_path, sr=24000, mono=True)
        inputs = self._processor(
            y, sampling_rate=24000, return_tensors="pt"
        )
        with torch.no_grad():
            # Filter to only keys MERT accepts — processor may add extras
            safe = {k: v for k, v in inputs.items() if k in ("input_values", "attention_mask")}
            # Use base_model.model to bypass PEFT's forward wrapper (avoids input_ids injection)
            inner = self._model.base_model.model if self._lora_loaded else self._model
            outputs = inner(**safe, output_hidden_states=True, return_dict=True)
        # Average last 4 hidden layers for richer representation
        hidden = torch.stack(outputs.hidden_states[-4:]).mean(dim=0)
        embedding = hidden.mean(dim=1).squeeze().numpy()
        return embedding.astype(np.float32)  # 768-dim


# Singleton — import this everywhere; model is pre-loaded at import time
embedder = MERTEmbedder()
embedder._load()
