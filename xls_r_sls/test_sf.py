import soundfile as sf
import numpy as np

try:
    data, samplerate = sf.read('/home/lab2208/Documents/deepfake_models/data/asvspoof/ASVspoof2021_LA_eval/flac/LA_E_9332881.flac')
    print(f"Shape: {data.shape}, SR: {samplerate}, Dtype: {data.dtype}")
    print(f"Min: {data.min()}, Max: {data.max()}")
except Exception as e:
    print(f"Error: {e}")
