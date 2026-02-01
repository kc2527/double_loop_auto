# ------------------------------------------------------------
# Demo script (runs top-to-bottom):
# Simulate spike trains -> "synaptic current" (spike convolution)
# -> a few cortical sources (VIS, PM_A, PM_B, M1_A, M1_B)
# -> 64-channel scalp EEG via a simple mixing (leadfield) matrix.
#
# Notes:
# - This is NOT biologically accurate. It is a demonstrator.
# - VIS is 100x100 units (10,000). We simulate VIS spikes efficiently
#   as an aggregated population spike count per time bin (Poisson),
#   rather than storing 10,000 individual spike trains.
# - PM and M1 are 2 units each (A/B). We simulate explicit spikes for them.
# - Replace the simulated spikes/currents with your own if desired.
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

np.random.seed(1)

# -----------------------------
# 1) Timing / trials
# -----------------------------
fs = 1000                 # Hz
dt = 1.0 / fs
n_ch = 64                 # EEG channels

n_trials = 60
trial_dur = 1.2           # seconds (demo)
iti = 0.3                 # seconds
T = n_trials * (trial_dur + iti)
t = np.arange(0, T, dt)
n_samples = t.size

# Stimulus window = middle third of each trial
stim_on = (trial_dur / 3.0)
stim_off = (2.0 * trial_dur / 3.0)

# Trial onsets
trial_onsets = np.arange(0, n_trials) * (trial_dur + iti)

# Category per trial (A=0, B=1) just for demonstration
cat = (np.random.rand(n_trials) > 0.5).astype(int)

# Build stimulus indicator over continuous time
stim = np.zeros(n_samples, dtype=float)
for k in range(n_trials):
    t0 = trial_onsets[k] + stim_on
    t1 = trial_onsets[k] + stim_off
    i0 = int(np.floor(t0 * fs))
    i1 = int(np.floor(t1 * fs))
    i0 = max(i0, 0); i1 = min(i1, n_samples)
    stim[i0:i1] = 1.0

# -----------------------------
# 2) Simulate spikes
# -----------------------------
# VIS: 100x100 units, stimulated during stimulus window
n_vis = 100 * 100

# VIS firing rates (Hz) baseline and during stimulus
vis_rate_base = 2.0
vis_rate_stim = 12.0

# Efficient population simulation:
# total VIS spikes per time bin ~ Poisson( N * rate(t) * dt )
vis_rate_t = vis_rate_base + (vis_rate_stim - vis_rate_base) * stim
vis_pop_spike_counts = np.random.poisson(n_vis * vis_rate_t * dt).astype(float)
# This is a population "spike train" as counts per ms bin.

# PM: 2 units (A,B). Make PM respond during stimulus, stronger for its category
pm_rate_base = 3.0
pm_rate_stim_match = 18.0   # if trial category matches the PM unit
pm_rate_stim_mismatch = 8.0

pm_spikes = np.zeros((2, n_samples), dtype=float)  # impulse trains
for k in range(n_trials):
    # time indices for this trial
    trial_start = trial_onsets[k]
    i_start = int(np.floor(trial_start * fs))
    i_end = int(np.floor((trial_start + trial_dur) * fs))
    i_start = max(i_start, 0); i_end = min(i_end, n_samples)

    # stimulus indices within this trial
    i_stim0 = int(np.floor((trial_onsets[k] + stim_on) * fs))
    i_stim1 = int(np.floor((trial_onsets[k] + stim_off) * fs))
    i_stim0 = max(i_stim0, 0); i_stim1 = min(i_stim1, n_samples)

    # baseline bins in trial
    trial_bins = np.arange(i_start, i_end)
    # stimulus bins
    stim_bins = np.arange(i_stim0, i_stim1)

    for u in [0, 1]:  # u=0 -> A, u=1 -> B
        # baseline spikes (Poisson per bin)
        lam_base = pm_rate_base * dt
        base_sp = (np.random.rand(trial_bins.size) < lam_base).astype(float)

        # stimulus spikes (unit depends on trial category)
        if u == cat[k]:
            r_stim = pm_rate_stim_match
        else:
            r_stim = pm_rate_stim_mismatch
        lam_stim = r_stim * dt
        stim_sp = (np.random.rand(stim_bins.size) < lam_stim).astype(float)

        pm_spikes[u, trial_bins] += base_sp
        pm_spikes[u, stim_bins] += stim_sp

# M1: 2 units (A,B). Make M1 ramp later in trial (e.g., decision/output)
m1_rate_base = 4.0
m1_rate_late_match = 22.0
m1_rate_late_mismatch = 10.0

m1_spikes = np.zeros((2, n_samples), dtype=float)
late_on = 0.65 * trial_dur
late_off = 0.95 * trial_dur

for k in range(n_trials):
    i_start = int(np.floor(trial_onsets[k] * fs))
    i_end = int(np.floor((trial_onsets[k] + trial_dur) * fs))
    i_start = max(i_start, 0); i_end = min(i_end, n_samples)

    i_late0 = int(np.floor((trial_onsets[k] + late_on) * fs))
    i_late1 = int(np.floor((trial_onsets[k] + late_off) * fs))
    i_late0 = max(i_late0, 0); i_late1 = min(i_late1, n_samples)

    trial_bins = np.arange(i_start, i_end)
    late_bins = np.arange(i_late0, i_late1)

    for u in [0, 1]:
        lam_base = m1_rate_base * dt
        base_sp = (np.random.rand(trial_bins.size) < lam_base).astype(float)

        if u == cat[k]:
            r_late = m1_rate_late_match
        else:
            r_late = m1_rate_late_mismatch
        lam_late = r_late * dt
        late_sp = (np.random.rand(late_bins.size) < lam_late).astype(float)

        m1_spikes[u, trial_bins] += base_sp
        m1_spikes[u, late_bins] += late_sp

# -----------------------------
# 3) Spikes -> "synaptic currents" via exponential convolution
# -----------------------------
tau_ms = 12.0
tau = tau_ms / 1000.0

k_t = np.arange(0, 0.25, dt)          # 250 ms kernel
kernel = np.exp(-k_t / tau)
kernel = kernel / kernel.sum()         # unit area (demo)

# VIS current: convolve population spike counts with kernel
vis_current = np.convolve(vis_pop_spike_counts, kernel, mode="full")[:n_samples]

# PM currents: convolve each unit's impulse train
pm_current = np.zeros_like(pm_spikes)
pm_current[0, :] = np.convolve(pm_spikes[0, :], kernel, mode="full")[:n_samples]
pm_current[1, :] = np.convolve(pm_spikes[1, :], kernel, mode="full")[:n_samples]

# M1 currents
m1_current = np.zeros_like(m1_spikes)
m1_current[0, :] = np.convolve(m1_spikes[0, :], kernel, mode="full")[:n_samples]
m1_current[1, :] = np.convolve(m1_spikes[1, :], kernel, mode="full")[:n_samples]

# Optional: normalize currents to comparable scales (purely for demo visualization)
vis_current = vis_current / (np.std(vis_current) + 1e-12)
pm_current = pm_current / (np.std(pm_current) + 1e-12)
m1_current = m1_current / (np.std(m1_current) + 1e-12)

# -----------------------------
# 4) Build source time series
#    Sources: [VIS, PM_A, PM_B, M1_A, M1_B]  -> 5 sources
# -----------------------------
sources = np.zeros((5, n_samples), dtype=float)
sources[0, :] = 1.0 * vis_current
sources[1, :] = 0.8 * pm_current[0, :]
sources[2, :] = 0.8 * pm_current[1, :]
sources[3, :] = 1.0 * m1_current[0, :]
sources[4, :] = 1.0 * m1_current[1, :]

# -----------------------------
# 5) Forward model: sources -> 64ch EEG using a simple "leadfield" matrix
#    We'll create a structured mixing so VIS is stronger on "occipital" channels
#    and M1 is stronger on "central" channels. This is a demo convenience.
# -----------------------------
L = 0.15 * np.random.randn(n_ch, 5)

# Define crude "regions" of electrodes (indices are arbitrary placeholders)
occipital = np.arange(0, 12)       # e.g., O* / PO* channels
central = np.arange(24, 40)        # e.g., C* channels
frontal = np.arange(44, 56)        # e.g., F* channels

# Boost VIS on occipital channels
L[occipital, 0] += 1.5
# Boost PM on frontal channels
L[frontal, 1] += 1.0
L[frontal, 2] += 1.0
# Boost M1 on central channels
L[central, 3] += 1.2
L[central, 4] += 1.2

# Make columns comparable magnitude
L = L / (np.linalg.norm(L, axis=0, keepdims=True) + 1e-12)

# Generate clean EEG
eeg_clean = L @ sources  # (64, time)

# Scale to microvolt-ish range for plots (arbitrary)
eeg_clean = 3.0e-6 * eeg_clean

# -----------------------------
# 6) Add EEG-ish noise (white + low-frequency drift)
# -----------------------------
white = 1.5e-6 * np.random.randn(n_ch, n_samples)

# drift: low-pass filtered noise
b_lp, a_lp = signal.butter(2, 1.0/(fs/2), btype="low")  # <1 Hz drift
drift = signal.filtfilt(b_lp, a_lp, np.random.randn(n_ch, n_samples), axis=1)
drift = 2.0e-6 * (drift / (np.std(drift) + 1e-12))

eeg = eeg_clean + white + drift

# -----------------------------
# 7) Simple preprocessing: bandpass 0.5-40 Hz
# -----------------------------
b_bp, a_bp = signal.butter(4, [0.5/(fs/2), 40.0/(fs/2)], btype="band")
eeg_f = signal.filtfilt(b_bp, a_bp, eeg, axis=1)

# Optional re-reference: common average reference (CAR)
eeg_f = eeg_f - eeg_f.mean(axis=0, keepdims=True)

# -----------------------------
# 8) Plot a few things
# -----------------------------
# Plot sources (normalized)
plt.figure(figsize=(12, 5))
for i, name in enumerate(["VIS", "PM_A", "PM_B", "M1_A", "M1_B"]):
    x = sources[i, :]
    x = x / (np.std(x) + 1e-12)
    plt.plot(t, x + 3.0*i, label=name)
plt.title("Source time series (normalized, offset)")
plt.xlabel("Time (s)")
plt.yticks([])
plt.legend(loc="upper right")
plt.tight_layout()

# Plot a subset of EEG channels
plt.figure(figsize=(12, 6))
show_ch = [0, 5, 10, 28, 32, 48]  # some occipital/central/frontal indices
offset = 0.0
for ch in show_ch:
    x = eeg_f[ch, :]
    x = x / (np.std(x) + 1e-12)
    plt.plot(t, x + offset, label=f"Ch {ch}")
    offset += 4.0

# Mark trial stimulus windows lightly
for k in range(n_trials):
    t0 = trial_onsets[k] + stim_on
    t1 = trial_onsets[k] + stim_off
    plt.axvspan(t0, t1, color="k", alpha=0.03)

plt.title("Simulated 64-ch EEG (filtered + CAR), a few channels (normalized, offset)")
plt.xlabel("Time (s)")
plt.yticks([])
plt.legend(loc="upper right")
plt.tight_layout()

# Stimulus-locked ERP average on one example channel
# Epoch: -200 ms to 800 ms around stimulus onset (stim_on within each trial)
tmin, tmax = -0.2, 0.8
n_pre = int(round(-tmin * fs))
n_post = int(round(tmax * fs))
epoch_len = n_pre + n_post
epochs = []

for k in range(n_trials):
    stim_onset_time = trial_onsets[k] + stim_on
    center = int(round(stim_onset_time * fs))
    i0 = center - n_pre
    i1 = center + n_post
    if i0 >= 0 and i1 < n_samples:
        epochs.append(eeg_f[:, i0:i1])

epochs = np.array(epochs)                 # (trials, ch, time)
erp = epochs.mean(axis=0)                 # (ch, time)
te = np.arange(tmin, tmax, dt)

plt.figure(figsize=(12, 4))
ch = 0  # pick an "occipital" channel for demo
plt.plot(te, 1e6 * erp[ch, :])            # microvolts
plt.axvline(0, color="k", alpha=0.3)
plt.title(f"Stimulus-locked ERP (mean), Channel {ch} (µV)")
plt.xlabel("Time (s) relative to stimulus onset")
plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# OUTPUT:
# - eeg_f is your simulated 64-channel EEG array with shape (64, n_samples)
# - t is the matching time vector in seconds
# - sources is the underlying 5-source generator signal (VIS, PM_A/B, M1_A/B)
# ------------------------------------------------------------

