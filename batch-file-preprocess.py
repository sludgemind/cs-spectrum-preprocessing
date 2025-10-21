import numpy as np
import os
from tqdm import tqdm
from scipy.signal import butter, filtfilt, find_peaks, resample

# === Filtering ===
def filter_signal(signal, sample_rate, hp_cutoff=3000, lp_cutoff=300_000,
                  hp_order=4, lp_order=4, pad_ms=1.5):
    pad_samples = int(sample_rate * pad_ms / 1000)
    padded = np.pad(signal, pad_samples, mode='reflect')

    nyq = sample_rate / 2
    b_hp, a_hp = butter(hp_order, hp_cutoff / nyq, btype='high')
    b_lp, a_lp = butter(lp_order, lp_cutoff / nyq, btype='low')

    filtered = filtfilt(b_hp, a_hp, padded)
    filtered = filtfilt(b_lp, a_lp, filtered)

    return filtered[pad_samples:-pad_samples]

# === Peak filtering logic exactly as single-file ===
def filter_peaks(peaks, signal, min_distance=100, max_rel_diff=0.2):
    filtered = []
    last_peak = -min_distance
    last_value = None

    for p in peaks:
        if p - last_peak >= min_distance:
            if last_value is None:
                filtered.append(p)
                last_peak = p
                last_value = signal[p]
            else:
                current_value = signal[p]
                rel_diff = abs(current_value - last_value) / abs(last_value) if last_value != 0 else float('inf')
                if rel_diff <= max_rel_diff:
                    filtered.append(p)
                    last_peak = p
                    last_value = current_value
    return np.array(filtered)

# === Cut signal by peaks ===
def cut_signal_by_peaks(signal, peaks):
    chunks = []
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]
        chunk = signal[start:end]
        chunks.append(chunk)
    return chunks[1:] if len(chunks) > 1 else []

# === Batch processing ===
def batch_process_signals_folder(folder_path,
                                 sample_rate=3_125_000,
                                 target_rate=1_500_000,
                                 trim_ms=10,
                                 min_peak_distance=100,
                                 max_rel_diff=0.2):
    results = {}
    all_chunks = []

    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".npy")]

    for path in tqdm(file_paths, desc="Processing signals"):
        try:
            data = np.load(path, allow_pickle=True).item()
            raw_signal = np.array(data['scp_data'][1])

            # --- Trim ---
            trimmed_signal = raw_signal[:int((trim_ms / 1000) * sample_rate)]

            # --- Filter ---
            filtered_signal = filter_signal(trimmed_signal, sample_rate)

            # --- Resample ---
            num_samples = int(len(filtered_signal) * target_rate / sample_rate)
            resampled_signal = resample(filtered_signal, num_samples)

            # --- Peaks ---
            peaks, _ = find_peaks(resampled_signal)
            filtered_peaks = filter_peaks(peaks, resampled_signal,
                                          min_distance=min_peak_distance,
                                          max_rel_diff=max_rel_diff)

            # --- Cut chunks ---
            chunks = cut_signal_by_peaks(resampled_signal, filtered_peaks)
            if not chunks:
                print(f"{os.path.basename(path)}: No valid chunks found")
                continue

            # --- Store ---
            results[os.path.basename(path)] = {
                "Vstab": data['scan_params'][0],
                "Amplitude": data['scan_params'][-1],
                "chunks": chunks
            }

            all_chunks.extend(chunks)

        except Exception as e:
            print(f"Error processing {path}: {e}")

    # --- Global trim to minimum chunk length ---
    if all_chunks:
        min_len = min(len(c) for c in all_chunks)
        print(f"Global trim length: {min_len}")
        for file in results:
            trimmed_chunks = [c[:min_len] for c in results[file]["chunks"] if len(c) >= min_len]
            results[file]["chunks"] = np.array(trimmed_chunks)

    return results
    def save_results_for_ML(results, folder_path="data_for_ML"):
    os.makedirs(folder_path, exist_ok=True)

    for filename, data in results.items():
        save_name = os.path.splitext(filename)[0] + ".npz"  # e.g., VStab_1.32773.npz
        save_path = os.path.join(folder_path, save_name)

        np.savez(save_path,
                 Vstab=data["Vstab"],
                 Amplitude=data["Amplitude"],
                 chunks=np.array(data["chunks"]))
    print(f"Saved {len(results)} files to '{folder_path}'")

