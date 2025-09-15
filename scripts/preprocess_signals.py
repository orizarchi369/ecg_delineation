# scripts/preprocess_signals.py (your version without normalization)

import os
import numpy as np
import pandas as pd
import pywt
from glob import glob
from tqdm import tqdm

# Paths
input_dir = '/home/orizarchi/projects/ecg_delineation/data/raw/dat_csv'
output_dir = '/home/orizarchi/projects/ecg_delineation/data/cleaned/dat_csv_cleaned'
os.makedirs(output_dir, exist_ok=True)

# Preprocessing function (denoising only)
def preprocess_signal(signal, wavelet='db4', level=4):
    # Signal: 1D array (5000 samples for one lead)
    
    # Denoise with wavelet transform
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(signal)))
    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    denoised_signal = pywt.waverec(coeffs, wavelet=wavelet)
    
    # Ensure output length matches input
    if len(denoised_signal) > len(signal):
        denoised_signal = denoised_signal[:len(signal)]
    elif len(denoised_signal) < len(signal):
        denoised_signal = np.pad(denoised_signal, (0, len(signal) - len(denoised_signal)), mode='constant')
    
    return denoised_signal

# Process all CSV files
csv_files = glob(os.path.join(input_dir, '*.csv'))
for csv_file in tqdm(csv_files, desc='Preprocessing ECG signals'):
    record_id = os.path.basename(csv_file).split('.csv')[0]
    
    # Load signal (5000 x 12)
    sig = pd.read_csv(csv_file, header=None).values  # np.array [5000, 12]
    
    # Process each lead
    processed_sig = np.zeros_like(sig)
    for lead_idx in range(sig.shape[1]):  # 12 leads
        processed_sig[:, lead_idx] = preprocess_signal(sig[:, lead_idx])
    
    # Save to output directory
    output_path = os.path.join(output_dir, f'{record_id}.csv')
    pd.DataFrame(processed_sig).to_csv(output_path, header=False, index=False)

print(f'Processed {len(csv_files)} files. Saved to {output_dir}')