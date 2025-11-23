import soundfile as sf
import os
from tqdm import tqdm
import glob

data_dir = "/home/lab2208/Documents/deepfake_models/data/asvspoof/ASVspoof2021_LA_eval/flac/"
files = glob.glob(os.path.join(data_dir, "*.flac"))

print(f"Checking {len(files)} files...")

for f in tqdm(files):
    try:
        sf.read(f)
    except Exception as e:
        print(f"Error reading {f}: {e}")
