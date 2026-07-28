import mne

# Path to the first subject's eyes-closed file
raw_path = "data/sub-032301/sub-032301_EC.set"

# Load the EEG data
raw = mne.io.read_raw_eeglab(raw_path, preload=True)

# Print basic info: channel count, sampling rate, duration
print(raw.info)
print(f"Sampling rate: {raw.info['sfreq']} Hz")
print(f"Number of channels: {len(raw.ch_names)}")
print(f"Duration: {raw.times[-1]:.1f} seconds")

# Plot the raw signal (opens an interactive window)
raw.plot(duration=10, n_channels=20)
input("Press Enter to close...")
