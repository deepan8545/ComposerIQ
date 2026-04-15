# ComposerIQ

ComposerIQ is an advanced, AI-driven audio analysis and scoring pipeline designed to evaluate music tracks based on algorithmic engagement drivers. It evaluates tracks, identifies hooks, measures harmonic features, detects structural segments, and generates detailed counterfactual "what-if" fixes to help musicians improve their tracks for streaming success.

## Core Architecture

ComposerIQ implements a 4-layer AI architecture:
1. **Raw Feature Extraction (Layer 1):** Uses `librosa` to extract tempo, key, energy curves, and spectral factors.
2. **MIR Analysis (Layer 2):** Calculates genre-conditional percentiles (e.g., skip risk percentile) using contextual cluster data.
3. **Scoring Engine (Layer 3):** Calibrates a final 0-100 score utilizing XGBoost dynamically blending 6 unique weighted engagement signals.
4. **Visibility Agent (Layer 4):** A LangChain orchestrator using Claude (Sonnet/Opus) to generate a deeply personalized and fully quantified markdown report.

## Recent Engine Improvements

ComposerIQ has recently been upgraded with 6 core AI intelligence capabilities:

- **P1: Learned Calibration (XGBoost):** Retrained scoring system that dynamically predicts engagement using a 5-fold cross-validated XGBoost regressor, instead of naive static formulas.
- **P2: MERT Hook Detection:** Utilizes MERT-v1-95M (a transformer audio model) and frame-level hidden state tracking to detect hooks and chorus drops far more accurately than amplitude-based techniques.
- **P3: Harmonic Complexity Features:** Employs chroma transitions and Shannon entropy to quantify chord change rates, harmonic complexities, and tonal stability.
- **P4: Counterfactual "What-If" Fixes:** The scoring engine dynamically computes quantified delta fixes (e.g., exactly how many points you gain by speeding up the tempo) to eliminate AI hallucination.
- **P5: Structural Segmentation:** Recurrence matrix and agglomerative clustering map the song out entirely by sections: intro, verse, bridge, chorus, and outro.
- **P6: MERT LoRA Fine-Tuning:** Architecture ready for Peft/LoRA-based fine-tuning of the base MERT model utilizing raw benchmark engagement ratings (Colab/GPU script included).

## Local Usage

1. **Install Python dependencies:**
   ```bash
   python -m venv venv
   source venv/Scripts/activate # Windows
   pip install -r requirements.txt
   ```

2. **Configure Environment:**
   Create a `.env` file (copy from `.env.example`) and insert your API keys. Without keys, the app gracefully falls back to `DEMO_MODE=true`.
   ```env
   ANTHROPIC_API_KEY=your_key
   PINECONE_API_KEY=your_key
   ```

3. **Start the Engine UI:**
   Launch the FastAPI and interactive dashboard:
   ```bash
   bash run.sh
   # Or double click RUN.bat
   ```
   Navigate to `http://localhost:8000` to upload a track.

## Machine Learning & GPU Training 

ComposerIQ ships with scripts to let you fine-tune the brain of the project.

- **Fast Retrain (Local CPU):**
  If you tweak `scoring_engine.py` scoring formulas, update the XGBoost model locally:
  ```bash
  python train_calibrator.py
  ```

- **MERT Fine-Tuning (Colab GPU):**
  To achieve deep audio-to-engagement vector mapping, upload `colab_bundle.zip` (which strips the bloated venv) to Google Colab with a T4 GPU.
  ```python
  !unzip -q colab_bundle.zip
  !pip install peft transformers librosa torch xgboost scikit-learn
  !python train_mert_lora.py
  ```
  Move the resulting `models/mert_lora` folder back to your local repo!

## License
MIT License.
