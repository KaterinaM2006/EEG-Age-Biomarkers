import mne
import numpy as np
import pandas as pd

raw_path = "data/sub-032301/sub-032301_EC.set"
raw = mne.io.read_raw_eeglab(raw_path, preload=True)
raw.filter(l_freq=1.0, h_freq=45.0)

epochs = mne.make_fixed_length_epochs(
    raw, duration=2.0, overlap=0.0, preload=True, reject_by_annotation=True
)

# Compute power spectral density (Welch's method) per epoch, per channel
psd = epochs.compute_psd(method="welch", fmin=1, fmax=45, n_fft=500)
psds, freqs = psd.get_data(return_freqs=True)  # shape: (n_epochs, n_channels, n_freqs)

# Average across all epochs -> one PSD curve per channel for this subject
mean_psd = psds.mean(axis=0)  # shape: (n_channels, n_freqs)

# Standard EEG frequency bands (Hz)
bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

band_power = {}
for band_name, (fmin, fmax) in bands.items():
    freq_mask = (freqs >= fmin) & (freqs < fmax)
    band_power[band_name] = mean_psd[:, freq_mask].mean(axis=1)

# One row per channel, one column per band
df = pd.DataFrame(band_power, index=epochs.ch_names)
print(df)

df.to_csv("results/sub-032301_band_power.csv")
print("Saved to results/sub-032301_band_power.csv")