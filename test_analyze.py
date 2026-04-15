import traceback
from agent.visibility_agent import run_analysis
from pipelines.audio_analysis import AudioAnalyzer
import numpy as np
import librosa
import os

# Create a dummy 5-second mp3 or just use a dummy audio path that librosa can parse
# We will create a dummy sine wave and save as wav to use it
import soundfile as sf
y = np.sin(2 * np.pi * 440 * np.linspace(0, 5, 22050 * 5))
sf.write("dummy.wav", y, 22050)

print("Starting analysis...")
try:
    res = run_analysis("dummy.wav", "ambient", "neutral", "test")
    print(res)
except Exception as e:
    traceback.print_exc()

os.remove("dummy.wav")
