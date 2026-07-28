import os
import mne
import numpy as np
import pandas as pd

DATA_DIR = "data"
LABELS_CSV = os.path.join(DATA_DIR, "participants_lemon.csv")

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

def age_bin_to_label(age_bin):
    # e.g. "65-70" -> take the lower number, 65
    start = int(str(age_bin).split("-")[0])
    return "young" if start <= 35 else "old"

def extract_features_for_subject(subject_id):
    raw_path = os.path.join(DATA_DIR, subject_id, f"{subject_id}_EC.set")
    if not os.path.exists(raw_path):
        return None

    raw = mne.io.read_raw_eeglab(raw_path, preload=True, verbose=False)
    raw.filter(l_freq=1.0, h_freq=45.0, verbose=False)

    epochs = mne.make_fixed_length_epochs(
        raw, duration=2.0, overlap=0.0, preload=True,
        reject_by_annotation=True, verbose=False,
    )

    n_fft = min(500, len(epochs.times))
    psd = epochs.compute_psd(method="welch", fmin=1, fmax=45, n_fft=n_fft, verbose=False)
    psds, freqs = psd.get_data(return_freqs=True)
    mean_psd = psds.mean(axis=0)  # shape: channels x freqs

    features = {}
    for ch_idx, ch_name in enumerate(epochs.ch_names):
        band_powers = {}
        for band_name, (fmin, fmax) in BANDS.items():
            freq_mask = (freqs >= fmin) & (freqs < fmax)
            band_powers[band_name] = mean_psd[ch_idx, freq_mask].mean()

        total_power = sum(band_powers.values())
        for band_name, power in band_powers.items():
            features[f"{ch_name}_{band_name}"] = power / total_power

    return features

def main():
    labels_df = pd.read_csv(LABELS_CSV)
    labels_df.columns = [c.strip() for c in labels_df.columns]
    id_col = labels_df.columns[0]  # the first column holds the subject ID, even though it has no header name
    age_col = [c for c in labels_df.columns if "age" in c.lower()][0]

    def age_bin_to_group(age_bin):
        start = int(str(age_bin).split("-")[0])
        if start < 40:  # young cluster (20-35), no longer split
            return "20-35"
        else:  # old cluster (59-77), still split in two
            return "59-70" if start < 68 else "70-77"

    label_map = {
        row[id_col]: age_bin_to_group(row[age_col])
        for _, row in labels_df.iterrows()
    }

    subject_ids = sorted(
        d for d in os.listdir(DATA_DIR)
        if d.startswith("sub-") and os.path.isdir(os.path.join(DATA_DIR, d))
    )

    rows = []
    for subject_id in subject_ids:
        if subject_id not in label_map:
            print(f"Skipping {subject_id}: no label found")
            continue

        print(f"Processing {subject_id}...")
        features = extract_features_for_subject(subject_id)
        if features is None:
            print(f"Skipping {subject_id}: EC file not found")
            continue

        features["subject_id"] = subject_id
        features["label"] = label_map[subject_id]
        rows.append(features)

    dataset = pd.DataFrame(rows)
    dataset.to_csv("results/labeled_dataset.csv", index=False)
    print(f"\nSaved dataset with {len(dataset)} subjects to results/labeled_dataset.csv")
    print(dataset[["subject_id", "label"]])

if __name__ == "__main__":
    main()