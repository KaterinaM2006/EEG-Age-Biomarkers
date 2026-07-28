import mne

raw_path = "data/sub-032301/sub-032301_EC.set"
raw = mne.io.read_raw_eeglab(raw_path, preload=True)

# 1. Bandpass filter: keep only 1-45 Hz, the range that covers
# delta through gamma and removes slow drift + high-frequency noise
raw.filter(l_freq=1.0, h_freq=45.0)

# 2. Segment into fixed-length epochs (2-second windows here)
# reject_by_annotation=True means MNE will automatically skip
# any epoch that overlaps a 'boundary' event (the discontinuities
# we saw in the warnings earlier) rather than including corrupted data
epochs = mne.make_fixed_length_epochs(
    raw, duration=2.0, overlap=0.0, preload=True, reject_by_annotation=True
)

print(f"Number of epochs created: {len(epochs)}")
print(f"Each epoch shape (channels x timepoints): {epochs.get_data().shape[1:]}")