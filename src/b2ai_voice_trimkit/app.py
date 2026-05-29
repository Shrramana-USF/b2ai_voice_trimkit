import os
import numpy as np
import pandas as pd
import streamlit as st
import parselmouth
import plotly.graph_objects as go
import soundfile as sf
import librosa
from pydub import AudioSegment
from b2ai_voice_trimkit.styles import apply_styles, display_header


# TRIMMING FUNCTIONS

# Convert MP3 to WAV and return the WAV path
def convert_mp3_to_wav(mp3_path):
    wav_path = mp3_path.rsplit('.', 1)[0] + '.wav'
    if not os.path.exists(wav_path):
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format='wav')
    return wav_path


# Pitch-based trimming using Praat/parselmouth. Trims to first and last voiced frames (F0 > 0)
def trim_audio_with_praat(y, sr):
    sound = parselmouth.Sound(y, sampling_frequency=sr)
    pitch = sound.to_pitch(time_step=0.01)
    voiced_flags = pitch.selected_array['frequency'] > 0
    time_stamps = pitch.xs()
    voiced_indices = np.where(voiced_flags)[0]

    if len(voiced_indices) == 0:
        return y  # No voiced frames found, return original

    start_sample = int(time_stamps[voiced_indices[0]] * sr)
    end_sample = int(time_stamps[voiced_indices[-1]] * sr)
    return y[start_sample:end_sample]


# Energy-based trimming. Finds peak energy and trims where energy drops below threshold of peak
def trim_peak_region(y, sr, threshold_ratio=0.05):
    abs_y = np.abs(y)
    # Smooth energy with 50ms window
    window_size = int(0.05 * sr)
    if window_size < 1:
        window_size = 1
    energy = np.convolve(abs_y, np.ones(window_size) / window_size, mode='same')

    peak_index = np.argmax(energy)
    drop_threshold = threshold_ratio * energy[peak_index]

    start, end = 0, len(y)

    # Find start (go backward from peak)
    for i in range(peak_index, 0, -1):
        if energy[i] < drop_threshold:
            start = i
            break

    # Find end (go forward from peak)
    for i in range(peak_index, len(energy)):
        if energy[i] < drop_threshold:
            end = i
            break

    return y[start:end]


# Detect individual cough segments based on energy. Returns list of (start_sample, end_sample) tuples
def detect_cough_segments(y, sr, threshold_ratio=0.10, min_gap_sec=0.15, min_duration_sec=0.05):
    abs_y = np.abs(y)
    # Smooth energy with 30ms window (coughs are short)
    window_size = int(0.03 * sr)
    if window_size < 1:
        window_size = 1
    energy = np.convolve(abs_y, np.ones(window_size) / window_size, mode='same')

    peak_energy = np.max(energy)
    threshold = threshold_ratio * peak_energy

    # Find regions above threshold
    above_threshold = energy > threshold

    # Find segment boundaries
    segments = []
    in_segment = False
    start = 0

    min_gap_samples = int(min_gap_sec * sr)
    min_duration_samples = int(min_duration_sec * sr)

    for i in range(len(above_threshold)):
        if above_threshold[i] and not in_segment:
            # Start of a segment
            in_segment = True
            start = i
        elif not above_threshold[i] and in_segment:
            # Potential end of segment - check if it's a real gap
            # Look ahead to see if there's another peak soon
            end = i
            gap_end = min(i + min_gap_samples, len(above_threshold))

            if not np.any(above_threshold[i:gap_end]):
                # Real gap - close this segment
                if end - start >= min_duration_samples:
                    segments.append((start, end))
                in_segment = False

    # Handle case where audio ends while in a segment
    if in_segment:
        if len(y) - start >= min_duration_samples:
            segments.append((start, len(y)))

    return segments


# Detect start points of all coughs using energy-based method
def detect_cough_starts_energy(y, sr, threshold_ratio=0.10, min_gap_sec=0.15):
    abs_y = np.abs(y)
    window_size = int(0.03 * sr)
    if window_size < 1:
        window_size = 1
    energy = np.convolve(abs_y, np.ones(window_size) / window_size, mode='same')

    peak_energy = np.max(energy)
    threshold = threshold_ratio * peak_energy

    # Find regions above threshold
    above_threshold = energy > threshold

    # Find start points of each cough
    starts = []
    in_segment = False
    min_gap_samples = int(min_gap_sec * sr)

    for i in range(len(above_threshold)):
        if above_threshold[i] and not in_segment:
            # Start of a segment
            in_segment = True
            starts.append(max(0, i - int(0.02 * sr)))  # Small buffer before
        elif not above_threshold[i] and in_segment:
            # Potential end of segment - check if it's a real gap
            gap_end = min(i + min_gap_samples, len(above_threshold))
            if not np.any(above_threshold[i:gap_end]):
                # Real gap - this segment ended
                in_segment = False

    return starts


# Detect cough segments using Parselmouth voiced region detection
def detect_cough_segments_parselmouth(y, sr, min_gap_sec=0.15, min_duration_sec=0.05):
    sound = parselmouth.Sound(y, sampling_frequency=sr)
    pitch = sound.to_pitch(time_step=0.005)  # Fine time step
    voiced_flags = pitch.selected_array['frequency'] > 0
    time_stamps = pitch.xs()

    if len(time_stamps) == 0 or not np.any(voiced_flags):
        return []

    # Find voiced regions and merge nearby ones
    segments = []
    in_segment = False
    start_time = 0
    last_voiced_time = 0

    min_gap_samples = int(min_gap_sec * sr)
    min_duration_samples = int(min_duration_sec * sr)

    for i, (t, voiced) in enumerate(zip(time_stamps, voiced_flags)):
        if voiced and not in_segment:
            # Start of voiced region
            in_segment = True
            start_time = t
            last_voiced_time = t
        elif voiced and in_segment:
            # Continue voiced region
            last_voiced_time = t
        elif not voiced and in_segment:
            # Potential end - check if gap is large enough to split
            gap_duration = t - last_voiced_time
            if gap_duration >= min_gap_sec:
                # Real gap - close this segment
                start_sample = max(0, int(start_time * sr) - int(0.02 * sr))
                end_sample = min(len(y), int(last_voiced_time * sr) + int(0.03 * sr))
                if end_sample - start_sample >= min_duration_samples:
                    segments.append((start_sample, end_sample))
                in_segment = False

    # Handle segment that extends to end
    if in_segment:
        start_sample = max(0, int(start_time * sr) - int(0.02 * sr))
        end_sample = min(len(y), int(last_voiced_time * sr) + int(0.03 * sr))
        if end_sample - start_sample >= min_duration_samples:
            segments.append((start_sample, end_sample))

    return segments


# Extract a specific cough using Parselmouth-only approach
def extract_cough_parselmouth(y, sr, cough_index, min_gap_sec=0.15, min_duration_sec=0.05):
    segments = detect_cough_segments_parselmouth(y, sr, min_gap_sec, min_duration_sec)

    if len(segments) == 0:
        return y  # No segments found, return original

    # Ensure cough_index is valid
    if cough_index > len(segments):
        cough_index = len(segments)

    start, end = segments[cough_index - 1]
    return y[start:end]


# Detect cough segments using hybrid approach (energy for start, Parselmouth for end)
def detect_cough_segments_hybrid(y, sr, threshold_ratio=0.10, min_gap_sec=0.15, min_duration_sec=0.05):
    # Step 1: Find all cough start points using energy
    cough_starts = detect_cough_starts_energy(y, sr, threshold_ratio, min_gap_sec)

    if len(cough_starts) == 0:
        return []

    segments = []
    min_duration_samples = int(min_duration_sec * sr)

    # Step 2: For each cough, find the end using Parselmouth
    for i, start in enumerate(cough_starts):
        # Define window: from this start to next start (or end of file)
        if i < len(cough_starts) - 1:
            window_end = cough_starts[i + 1]
        else:
            window_end = len(y)

        # Extract window and run Parselmouth
        y_window = y[start:window_end]

        sound = parselmouth.Sound(y_window, sampling_frequency=sr)
        pitch = sound.to_pitch(time_step=0.005)
        voiced_flags = pitch.selected_array['frequency'] > 0
        time_stamps = pitch.xs()
        voiced_indices = np.where(voiced_flags)[0]

        if len(voiced_indices) > 0:
            # Found voiced frames - use last one as end
            last_voiced_time = time_stamps[voiced_indices[-1]]
            end_sample = start + int(last_voiced_time * sr) + int(0.03 * sr)  # 30ms buffer
        else:
            # No voiced frames - fall back to energy-based end
            abs_window = np.abs(y_window)
            window_size = int(0.03 * sr)
            if window_size < 1:
                window_size = 1
            energy = np.convolve(abs_window, np.ones(window_size) / window_size, mode='same')
            peak_energy = np.max(energy)
            threshold = threshold_ratio * peak_energy

            end_in_window = len(y_window)
            for j in range(len(energy) - 1, -1, -1):
                if energy[j] > threshold:
                    end_in_window = j + int(0.05 * sr)
                    break
            end_sample = start + min(end_in_window, len(y_window))

        end_sample = min(end_sample, len(y))

        # Only add if meets minimum duration
        if end_sample - start >= min_duration_samples:
            segments.append((start, end_sample))

    return segments


# Extract a specific cough using hybrid approach (energy for start, Parselmouth for end)
def extract_cough_hybrid(y, sr, cough_index, threshold_ratio=0.10, min_gap_sec=0.15):
    # Step 1: Find all cough start points using energy
    cough_starts = detect_cough_starts_energy(y, sr, threshold_ratio, min_gap_sec)

    if len(cough_starts) == 0:
        # No coughs detected, return original
        return y

    # Ensure cough_index is valid
    if cough_index > len(cough_starts):
        cough_index = len(cough_starts)

    # Step 2: Define the search window
    target_start = cough_starts[cough_index - 1]

    # Window end: start of next cough, or end of file for last cough
    if cough_index < len(cough_starts):
        window_end = cough_starts[cough_index]
    else:
        window_end = len(y)

    # Step 3: Extract the window and run Parselmouth
    y_window = y[target_start:window_end]

    sound = parselmouth.Sound(y_window, sampling_frequency=sr)
    pitch = sound.to_pitch(time_step=0.005)  # Finer time step for accuracy
    voiced_flags = pitch.selected_array['frequency'] > 0
    time_stamps = pitch.xs()
    voiced_indices = np.where(voiced_flags)[0]

    if len(voiced_indices) == 0:
        # No voiced frames found, fall back to energy-based end detection
        abs_window = np.abs(y_window)
        window_size = int(0.03 * sr)
        if window_size < 1:
            window_size = 1
        energy = np.convolve(abs_window, np.ones(window_size) / window_size, mode='same')
        peak_energy = np.max(energy)
        threshold = threshold_ratio * peak_energy

        # Find end by scanning backward
        end_sample = len(y_window)
        for i in range(len(energy) - 1, -1, -1):
            if energy[i] > threshold:
                end_sample = min(len(y_window), i + int(0.05 * sr))  # 50ms buffer
                break

        return y_window[:end_sample]

    # Step 4: Find last voiced frame = accurate end of cough
    last_voiced_time = time_stamps[voiced_indices[-1]]
    end_sample = int(last_voiced_time * sr) + int(0.03 * sr)  # 30ms buffer after
    end_sample = min(end_sample, len(y_window))

    return y_window[:end_sample]


# Trim only leading and trailing silence, keep everything in between
def trim_silence_only(y, sr, threshold_ratio=0.05):
    abs_y = np.abs(y)
    window_size = int(0.03 * sr)
    if window_size < 1:
        window_size = 1
    energy = np.convolve(abs_y, np.ones(window_size) / window_size, mode='same')

    peak_energy = np.max(energy)
    threshold = threshold_ratio * peak_energy

    # Find first sample above threshold
    start = 0
    for i in range(len(energy)):
        if energy[i] > threshold:
            start = max(0, i - int(0.02 * sr))
            break

    # Find last sample above threshold
    end = len(y)
    for i in range(len(energy) - 1, -1, -1):
        if energy[i] > threshold:
            end = min(len(y), i + int(0.02 * sr))
            break

    return y[start:end]


# HELPER FUNCTIONS

# Load audio file, converting MP3 to WAV if needed
def load_audio(file_path):
    if file_path.lower().endswith('.mp3'):
        file_path = convert_mp3_to_wav(file_path)

    y, sr = librosa.load(file_path, sr=None)
    return y, sr, file_path


# Create folder if it doesn't exist and return path
def ensure_folder(parent_dir, folder_name):
    folder_path = os.path.join(parent_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


# Get duration in seconds
def get_duration(y, sr):
    return len(y) / sr


# Create a plotly waveform visualization
def plot_waveform(y, sr, title="Waveform", selectable=False):
    times = np.arange(len(y)) / sr
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=y, mode="lines", name="wave", line=dict(width=0.5)))

    layout_options = {
        "title": title,
        "xaxis_title": "Time (s)",
        "yaxis_title": "Amplitude",
        "height": 300 if selectable else 250,
        "margin": dict(l=40, r=40, t=40, b=40),
    }

    if selectable:
        # Enable box selection mode
        layout_options["dragmode"] = "select"
        layout_options["selectdirection"] = "h"  # Horizontal selection only
        # Add instruction
        layout_options["title"] = f"{title} - Click and drag to select region"

    fig.update_layout(**layout_options)
    return fig


# Create a plotly waveform with a highlighted selection region
def plot_waveform_with_selection(y, sr, start_time, end_time, title="Waveform"):
    times = np.arange(len(y)) / sr
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=y, mode="lines", name="wave", line=dict(width=0.5, color='blue')))

    # Add shaded region for selection
    fig.add_vrect(
        x0=start_time, x1=end_time,
        fillcolor="rgba(0, 255, 0, 0.3)",
        layer="below",
        line_width=2,
        line_color="green",
        annotation_text="Selected",
        annotation_position="top left"
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


# Create a plotly waveform with highlighted segment regions
def plot_waveform_with_segments(y, sr, segments, title="Waveform with Segments", labels=None):
    times = np.arange(len(y)) / sr
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=y, mode="lines", name="wave", line=dict(width=0.5, color='blue')))

    # Task-specific colors
    task_colors = {
        'Sustained Vowel': 'rgba(0, 128, 255, 0.4)',      # Blue
        'Cough': 'rgba(255, 100, 100, 0.4)',              # Red
        'Speech': 'rgba(100, 200, 100, 0.4)',             # Green
        'Other': 'rgba(200, 150, 50, 0.4)',               # Orange
    }
    default_colors = ['rgba(255,0,0,0.3)', 'rgba(0,255,0,0.3)', 'rgba(255,165,0,0.3)', 'rgba(128,0,128,0.3)']

    for i, (start, end) in enumerate(segments):
        start_time = start / sr
        end_time = end / sr

        # Get label and color
        if labels and i < len(labels):
            label = labels[i]
            color = task_colors.get(label, default_colors[i % len(default_colors)])
        else:
            label = f"Segment {i+1}"
            color = default_colors[i % len(default_colors)]

        fig.add_vrect(
            x0=start_time, x1=end_time,
            fillcolor=color, opacity=0.5,
            layer="below", line_width=0,
            annotation_text=label,
            annotation_position="top left"
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


# Process a speech audio file using Praat-based trimming
def process_speech(audio_path):
    result = {
        'audio_path': audio_path,
        'status': 'pending',
        'trimmed_path': None,
        'original_duration': None,
        'trimmed_duration': None,
        'final_path': None,
        'final_method': None
    }

    try:
        # Load audio
        y, sr, wav_path = load_audio(audio_path)
        original_duration = get_duration(y, sr)

        # Get parent directory of the audio file
        parent_dir = os.path.dirname(wav_path)
        base_name = os.path.splitext(os.path.basename(wav_path))[0]

        # Create output folders
        final_folder = ensure_folder(parent_dir, "Final_trim")

        # Apply Praat-based trim for speech
        y_trimmed = trim_audio_with_praat(y, sr)
        trimmed_duration = get_duration(y_trimmed, sr)

        # If Praat trim fails (returns same or empty), use energy-based
        if len(y_trimmed) == 0 or trimmed_duration < 0.1:
            y_trimmed = trim_silence_only(y, sr)
            trimmed_duration = get_duration(y_trimmed, sr)
            result['final_method'] = 'energy_fallback'
        else:
            result['final_method'] = 'praat'

        # Save to final folder
        final_path = os.path.join(final_folder, f"{base_name}_final.wav")
        sf.write(final_path, y_trimmed, sr)

        result['trimmed_path'] = final_path
        result['final_path'] = final_path
        result['original_duration'] = original_duration
        result['trimmed_duration'] = trimmed_duration
        result['wav_path'] = wav_path
        result['sr'] = sr
        result['status'] = 'completed'

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)

    return result


# Process a cough audio file
def process_cough(audio_path, num_coughs_expected, cough_to_isolate,
                   threshold_ratio=0.10, min_gap_sec=0.15, min_duration_sec=0.05,
                   trim_method="hybrid", margin_before=0.0, margin_after=0.0):
    result = {
        'audio_path': audio_path,
        'status': 'pending',
        'detected_segments': [],
        'num_detected': 0,
        'num_expected': num_coughs_expected,
        'cough_to_isolate': cough_to_isolate,
        'segments_match': False,
        'needs_manual': False,
        'trimmed_path': None,
        'original_duration': None,
        'trimmed_duration': None,
        'final_path': None,
        'final_method': None
    }

    try:
        # Load audio
        y, sr, wav_path = load_audio(audio_path)
        original_duration = get_duration(y, sr)

        # Get parent directory
        parent_dir = os.path.dirname(wav_path)
        base_name = os.path.splitext(os.path.basename(wav_path))[0]

        # Create output folders
        draft_folder = ensure_folder(parent_dir, "trimmed_draft")
        final_folder = ensure_folder(parent_dir, "Final_trim")

        # Detect cough segments based on selected method
        if trim_method == "parselmouth":
            segments = detect_cough_segments_parselmouth(y, sr, min_gap_sec, min_duration_sec)
        else:
            # Hybrid: energy for start, Parselmouth for end
            segments = detect_cough_segments_hybrid(y, sr, threshold_ratio, min_gap_sec, min_duration_sec)

        # Apply margins to segments
        if margin_before > 0 or margin_after > 0:
            margin_before_samples = int(margin_before * sr)
            margin_after_samples = int(margin_after * sr)
            adjusted_segments = []
            for start_sample, end_sample in segments:
                new_start = max(0, start_sample - margin_before_samples)
                new_end = min(len(y), end_sample + margin_after_samples)
                adjusted_segments.append((new_start, new_end))
            segments = adjusted_segments

        num_detected = len(segments)

        result['detected_segments'] = segments
        result['num_detected'] = num_detected
        result['original_duration'] = original_duration
        result['wav_path'] = wav_path
        result['sr'] = sr
        result['draft_folder'] = draft_folder
        result['final_folder'] = final_folder
        result['base_name'] = base_name
        result['trim_method'] = trim_method

        # Check if segments match expected count
        segments_match = (num_detected == num_coughs_expected)
        result['segments_match'] = segments_match

        # Process based on which cough to isolate
        y_trimmed = None

        if cough_to_isolate == "All":
            # Just trim silence at start/end
            y_trimmed = trim_silence_only(y, sr)
            result['final_method'] = 'all_coughs'

        elif segments_match:
            # Detected segments match expected count - use selected method
            if trim_method == "parselmouth":
                y_trimmed = extract_cough_parselmouth(
                    y, sr,
                    cough_index=cough_to_isolate,
                    min_gap_sec=min_gap_sec,
                    min_duration_sec=min_duration_sec
                )
                result['final_method'] = f'parselmouth_cough_{cough_to_isolate}'
            else:
                y_trimmed = extract_cough_hybrid(
                    y, sr,
                    cough_index=cough_to_isolate,
                    threshold_ratio=threshold_ratio,
                    min_gap_sec=min_gap_sec
                )
                result['final_method'] = f'hybrid_cough_{cough_to_isolate}'

        else:
            # Segment count mismatch - need manual review
            result['needs_manual'] = True
            result['status'] = 'needs_review'

        # Save if we have a result
        if y_trimmed is not None and len(y_trimmed) > 0:
            trimmed_duration = get_duration(y_trimmed, sr)
            final_path = os.path.join(final_folder, f"{base_name}_cough{cough_to_isolate}_final.wav")
            sf.write(final_path, y_trimmed, sr)

            result['trimmed_path'] = final_path
            result['final_path'] = final_path
            result['trimmed_duration'] = trimmed_duration
            result['status'] = 'completed'

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)

    return result


# Process an audio file with multiple fused tasks
def process_fused(audio_path, num_tasks_expected, segment_types=None,
                   threshold_ratio=0.08, min_gap_sec=0.3, min_duration_sec=0.1):
    result = {
        'audio_path': audio_path,
        'status': 'pending',
        'detected_segments': [],
        'num_detected': 0,
        'num_expected': num_tasks_expected,
        'segment_types': segment_types or [],
        'segments_match': False,
        'needs_manual': False,
        'original_duration': None,
        'saved_segments': [],  # List of saved segment paths
        'final_method': None
    }

    try:
        # Load audio
        y, sr, wav_path = load_audio(audio_path)
        original_duration = get_duration(y, sr)

        # Get parent directory
        parent_dir = os.path.dirname(wav_path)
        base_name = os.path.splitext(os.path.basename(wav_path))[0]

        # Create output folders
        draft_folder = ensure_folder(parent_dir, "trimmed_draft")
        final_folder = ensure_folder(parent_dir, "Final_trim")

        # Detect segments using energy-based detection with user-defined thresholds
        segments = detect_cough_segments(y, sr, threshold_ratio, min_gap_sec, min_duration_sec)
        num_detected = len(segments)

        result['detected_segments'] = segments
        result['num_detected'] = num_detected
        result['original_duration'] = original_duration
        result['wav_path'] = wav_path
        result['sr'] = sr
        result['y'] = y  # Store audio data for UI
        result['draft_folder'] = draft_folder
        result['final_folder'] = final_folder
        result['base_name'] = base_name

        # Check if segments match expected count (exact match required)
        segments_match = (num_detected == num_tasks_expected)
        result['segments_match'] = segments_match

        if segments_match:
            # Auto-save all segments with task_1, task_2, etc. naming
            saved_segments = []
            for seg_idx, (start_samp, end_samp) in enumerate(segments):
                y_seg = y[start_samp:end_samp]
                seg_path = os.path.join(final_folder, f"{base_name}_task_{seg_idx + 1}_final.wav")
                sf.write(seg_path, y_seg, sr)
                saved_segments.append({
                    'path': seg_path,
                    'duration': get_duration(y_seg, sr),
                    'task_num': seg_idx + 1
                })
            result['saved_segments'] = saved_segments
            result['final_method'] = 'auto_segments'
            result['status'] = 'completed'
        else:
            # Need manual review (detected != expected)
            result['needs_manual'] = True
            result['status'] = 'needs_review'

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)

    return result


# Process a single sustained vowel audio file
def process_sustained_vowel(audio_path, results_data, energy_threshold=0.05,
                            auto_accept_threshold=1.0, margin_before=0.0, margin_after=0.0):
    result = {
        'audio_path': audio_path,
        'status': 'pending',
        'e_trimmed_path': None,
        'p_trimmed_path': None,
        'e_duration': None,
        'p_duration': None,
        'diff': None,
        'needs_manual': False,
        'final_path': None,
        'final_method': None
    }

    try:
        # Load audio
        y, sr, wav_path = load_audio(audio_path)
        original_duration = get_duration(y, sr)

        # Get parent directory of the audio file
        parent_dir = os.path.dirname(wav_path)
        base_name = os.path.splitext(os.path.basename(wav_path))[0]

        # Create output folders
        draft_folder = ensure_folder(parent_dir, "trimmed_draft")
        final_folder = ensure_folder(parent_dir, "Final_trim")

        # Apply Energy-based trim with configurable threshold
        y_e_trimmed = trim_peak_region(y, sr, threshold_ratio=energy_threshold)
        e_duration = get_duration(y_e_trimmed, sr)
        e_trimmed_path = os.path.join(draft_folder, f"{base_name}_e_trimmed.wav")
        sf.write(e_trimmed_path, y_e_trimmed, sr)

        # Apply Pitch-based trim (Praat)
        y_p_trimmed = trim_audio_with_praat(y, sr)
        p_duration = get_duration(y_p_trimmed, sr)
        p_trimmed_path = os.path.join(draft_folder, f"{base_name}_p_trimmed.wav")
        sf.write(p_trimmed_path, y_p_trimmed, sr)

        # Calculate difference
        diff = e_duration - p_duration

        # Update result
        result['e_trimmed_path'] = e_trimmed_path
        result['p_trimmed_path'] = p_trimmed_path
        result['e_duration'] = e_duration
        result['p_duration'] = p_duration
        result['diff'] = diff
        result['original_duration'] = original_duration
        result['wav_path'] = wav_path
        result['sr'] = sr
        result['draft_folder'] = draft_folder
        result['final_folder'] = final_folder
        result['base_name'] = base_name
        result['margin_before'] = margin_before
        result['margin_after'] = margin_after

        # Check if manual review needed (using configurable threshold)
        if abs(diff) > auto_accept_threshold:
            result['needs_manual'] = True
            result['status'] = 'needs_review'
        else:
            # Auto-accept energy-based trim with margins applied
            y_final = y_e_trimmed
            # Apply margins if set
            if margin_before > 0 or margin_after > 0:
                # Find where e_trimmed starts/ends in original audio
                # Re-detect boundaries to apply margins
                y_with_margin = trim_peak_region(y, sr, threshold_ratio=energy_threshold)
                # For margins, we need to work with the original audio
                abs_y = np.abs(y)
                window_size = int(0.05 * sr)
                if window_size < 1:
                    window_size = 1
                energy = np.convolve(abs_y, np.ones(window_size) / window_size, mode='same')
                peak_index = np.argmax(energy)
                drop_threshold = energy_threshold * energy[peak_index]

                # Find start
                start = 0
                for i in range(peak_index, 0, -1):
                    if energy[i] < drop_threshold:
                        start = i
                        break
                # Find end
                end = len(y)
                for i in range(peak_index, len(energy)):
                    if energy[i] < drop_threshold:
                        end = i
                        break

                # Apply margins
                margin_before_samples = int(margin_before * sr)
                margin_after_samples = int(margin_after * sr)
                new_start = max(0, start - margin_before_samples)
                new_end = min(len(y), end + margin_after_samples)
                y_final = y[new_start:new_end]

            final_path = os.path.join(final_folder, f"{base_name}_final.wav")
            sf.write(final_path, y_final, sr)
            result['final_path'] = final_path
            result['final_method'] = 'e_trimmed (auto)'
            result['trimmed_duration'] = get_duration(y_final, sr)
            result['status'] = 'completed'

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)

    return result


# STREAMLIT APP

apply_styles()
display_header()

# Task Selection
st.sidebar.header("Task Configuration")

task_type = st.sidebar.selectbox(
    "Select Audio Task",
    ["Sustained Vowel", "Cough", "Speech", "Breathing", "Everything Fused", "General-manual"]
)

# Task-specific options
if task_type == "Sustained Vowel":
    st.sidebar.info("Sustained Vowel: Single segment per file")

    with st.sidebar.expander("Detection Settings"):
        # Initialize defaults in session state if not present
        if 'vowel_energy_threshold_val' not in st.session_state:
            st.session_state.vowel_energy_threshold_val = 5
        if 'vowel_auto_accept_val' not in st.session_state:
            st.session_state.vowel_auto_accept_val = 1.0
        if 'vowel_margin_before_val' not in st.session_state:
            st.session_state.vowel_margin_before_val = 0.0
        if 'vowel_margin_after_val' not in st.session_state:
            st.session_state.vowel_margin_after_val = 0.0

        vowel_energy_threshold = st.slider(
            "Energy threshold (%)",
            min_value=1, max_value=20,
            value=st.session_state.vowel_energy_threshold_val,
            key="vowel_energy_slider",
            help="Percentage of peak energy to detect vowel boundaries"
        ) / 100.0
        st.session_state.vowel_energy_threshold_val = int(vowel_energy_threshold * 100)

        vowel_auto_accept = st.slider(
            "Auto-accept threshold (s)",
            min_value=0.1, max_value=3.0,
            value=st.session_state.vowel_auto_accept_val,
            step=0.1,
            key="vowel_auto_accept_slider",
            help="Auto-accept if energy vs Praat difference is below this threshold"
        )
        st.session_state.vowel_auto_accept_val = vowel_auto_accept

        st.markdown("**Segment Margins**")

        vowel_margin_before = st.slider(
            "Margin before segment (s)",
            min_value=0.0, max_value=1.0,
            value=st.session_state.vowel_margin_before_val,
            step=0.05,
            key="vowel_margin_before_slider",
            help="Extra time to include before detected segment"
        )
        st.session_state.vowel_margin_before_val = vowel_margin_before

        vowel_margin_after = st.slider(
            "Margin after segment (s)",
            min_value=0.0, max_value=1.0,
            value=st.session_state.vowel_margin_after_val,
            step=0.05,
            key="vowel_margin_after_slider",
            help="Extra time to include after detected segment"
        )
        st.session_state.vowel_margin_after_val = vowel_margin_after

        if st.button("Reset to Defaults", key="reset_vowel_settings"):
            st.session_state.vowel_energy_threshold_val = 5
            st.session_state.vowel_auto_accept_val = 1.0
            st.session_state.vowel_margin_before_val = 0.0
            st.session_state.vowel_margin_after_val = 0.0
            st.rerun()

elif task_type == "Cough":
    num_coughs = st.sidebar.selectbox(
        "How many cough sounds in 1 file?",
        [1, 2, 3, 4, 5]
    )
    cough_to_isolate = st.sidebar.selectbox(
        "Cough number to be isolated?",
        list(range(1, num_coughs + 1)) + ["All"]
    )

    # Trimming method selector
    cough_trim_method = st.sidebar.radio(
        "Trimming Method",
        ["Hybrid (Recommended)", "Parselmouth Only"],
        help="Hybrid: Energy for start, Parselmouth for end. Parselmouth Only: Uses voiced region detection for both."
    )

    # Detection threshold settings
    with st.sidebar.expander("Detection Settings"):
        # Initialize defaults in session state if not present
        if 'cough_threshold_val' not in st.session_state:
            st.session_state.cough_threshold_val = 10
        if 'cough_min_gap_val' not in st.session_state:
            st.session_state.cough_min_gap_val = 0.15
        if 'cough_min_duration_val' not in st.session_state:
            st.session_state.cough_min_duration_val = 0.05

        cough_threshold = st.slider(
            "Energy threshold (%)",
            min_value=1, max_value=30,
            value=st.session_state.cough_threshold_val,
            key="cough_thresh_slider",
            help="Percentage of peak energy to detect segments"
        ) / 100.0
        st.session_state.cough_threshold_val = int(cough_threshold * 100)

        cough_min_gap = st.slider(
            "Min gap between coughs (s)",
            min_value=0.05, max_value=0.5,
            value=st.session_state.cough_min_gap_val,
            step=0.05,
            key="cough_gap_slider",
            help="Minimum silence duration to separate coughs"
        )
        st.session_state.cough_min_gap_val = cough_min_gap

        cough_min_duration = st.slider(
            "Min cough duration (s)",
            min_value=0.02, max_value=0.2,
            value=st.session_state.cough_min_duration_val,
            step=0.01,
            key="cough_dur_slider",
            help="Minimum duration for a valid cough segment"
        )
        st.session_state.cough_min_duration_val = cough_min_duration

        st.markdown("**Segment Margins**")

        # Initialize margin defaults in session state
        if 'cough_margin_before_val' not in st.session_state:
            st.session_state.cough_margin_before_val = 0.0
        if 'cough_margin_after_val' not in st.session_state:
            st.session_state.cough_margin_after_val = 0.0

        cough_margin_before = st.slider(
            "Margin before segment (s)",
            min_value=0.0, max_value=1.0,
            value=st.session_state.cough_margin_before_val,
            step=0.05,
            key="cough_margin_before_slider",
            help="Extra time to include before detected segment"
        )
        st.session_state.cough_margin_before_val = cough_margin_before

        cough_margin_after = st.slider(
            "Margin after segment (s)",
            min_value=0.0, max_value=1.0,
            value=st.session_state.cough_margin_after_val,
            step=0.05,
            key="cough_margin_after_slider",
            help="Extra time to include after detected segment"
        )
        st.session_state.cough_margin_after_val = cough_margin_after

        if st.button("Reset to Defaults", key="reset_cough_settings"):
            st.session_state.cough_threshold_val = 10
            st.session_state.cough_min_gap_val = 0.15
            st.session_state.cough_min_duration_val = 0.05
            st.session_state.cough_margin_before_val = 0.0
            st.session_state.cough_margin_after_val = 0.0
            st.rerun()

elif task_type == "Speech":
    st.sidebar.info("Speech: Uses Praat-based trimming to isolate voiced region")

elif task_type == "Breathing":
    st.sidebar.info("Breathing: Trim leading and trailing silence")

    # Trimming mode selection
    breathing_mode = st.sidebar.radio(
        "Trimming Mode",
        ["Auto-trim silence", "Manual trim"],
        help="Auto: Automatically trim leading/trailing silence. Manual: Select region yourself."
    )

    # Settings for auto-trim
    if breathing_mode == "Auto-trim silence":
        with st.sidebar.expander("Auto-trim Settings", expanded=True):
            if 'breathing_silence_threshold_val' not in st.session_state:
                st.session_state.breathing_silence_threshold_val = 2

            breathing_silence_threshold = st.slider(
                "Silence threshold (%)",
                min_value=1, max_value=20,
                value=st.session_state.breathing_silence_threshold_val,
                key="breathing_silence_thresh_slider",
                help="Percentage of peak energy below which is considered silence"
            ) / 100.0
            st.session_state.breathing_silence_threshold_val = int(breathing_silence_threshold * 100)

            if st.button("Reset to Default", key="reset_breathing_settings"):
                st.session_state.breathing_silence_threshold_val = 2
                st.rerun()

elif task_type == "Everything Fused":
    num_tasks_expected = st.sidebar.number_input(
        "How many audio tasks in 1 file?",
        min_value=1, max_value=10, value=1
    )
    segment_types = [f"Task {i+1}" for i in range(num_tasks_expected)]
    st.sidebar.info(f"Will segment into: {', '.join(segment_types)}")

    # Detection threshold settings
    with st.sidebar.expander("Detection Settings"):
        # Initialize defaults in session state if not present
        if 'fused_threshold_val' not in st.session_state:
            st.session_state.fused_threshold_val = 8
        if 'fused_min_gap_val' not in st.session_state:
            st.session_state.fused_min_gap_val = 0.3
        if 'fused_min_duration_val' not in st.session_state:
            st.session_state.fused_min_duration_val = 0.1

        fused_threshold = st.slider(
            "Energy threshold (%)",
            min_value=1, max_value=30,
            value=st.session_state.fused_threshold_val,
            key="fused_thresh_slider",
            help="Percentage of peak energy to detect segments"
        ) / 100.0
        st.session_state.fused_threshold_val = int(fused_threshold * 100)

        fused_min_gap = st.slider(
            "Min gap between tasks (s)",
            min_value=0.1, max_value=1.0,
            value=st.session_state.fused_min_gap_val,
            step=0.1,
            key="fused_gap_slider",
            help="Minimum silence duration to separate tasks"
        )
        st.session_state.fused_min_gap_val = fused_min_gap

        fused_min_duration = st.slider(
            "Min task duration (s)",
            min_value=0.05, max_value=0.5,
            value=st.session_state.fused_min_duration_val,
            step=0.05,
            key="fused_dur_slider",
            help="Minimum duration for a valid task segment"
        )
        st.session_state.fused_min_duration_val = fused_min_duration

        if st.button("Reset to Defaults", key="reset_fused_settings"):
            st.session_state.fused_threshold_val = 8
            st.session_state.fused_min_gap_val = 0.3
            st.session_state.fused_min_duration_val = 0.1
            st.rerun()

elif task_type == "General-manual":
    st.sidebar.info("General-manual: Select and trim any region from your audio files")

# Quit button with confirmation
st.sidebar.markdown("---")
if 'confirm_quit' not in st.session_state:
    st.session_state.confirm_quit = False

if not st.session_state.confirm_quit:
    if st.sidebar.button("Quit Application"):
        st.session_state.confirm_quit = True
        st.rerun()
else:
    st.sidebar.warning("Are you sure you want to quit?")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Yes", key="quit_yes"):
            import os
            os._exit(0)
    with col2:
        if st.button("No", key="quit_no"):
            st.session_state.confirm_quit = False
            st.rerun()

# CSV Upload
st.header("Upload Audio File List")

with st.expander("ℹ️ Instructions"):
    st.markdown("""
    **CSV Format:**
    - Upload a CSV file with a column named `audio_file_path`
    - Each row should contain the full path to an audio file

    **Supported Formats:**
    - `.wav` files (processed directly)
    - `.mp3` files (automatically converted to WAV before processing)

    **Detection Settings (Cough & Fused tasks):**
    - For Cough and Everything Fused tasks, you can adjust detection settings in the sidebar
    - Expand "Detection Settings" to modify energy threshold, minimum gap, and minimum duration
    - If not modified, default settings will be used for processing

    **Tips:**
    - Ensure all file paths are valid and accessible
    - Select the appropriate task type from the sidebar before uploading
    - For Cough/Fused tasks, configure the number of segments in the sidebar and which segment needs to be isolated (if applicable)
    - Automated isolation of breathing segments are less reliable than other tasks; manual trimming is recommended for accurate results
    """)

uploaded_csv = st.file_uploader(
    "Upload CSV file",
    type=['csv']
)

if uploaded_csv is not None:
    df = pd.read_csv(uploaded_csv)

    if 'audio_file_path' not in df.columns:
        st.error("CSV must contain 'audio_file_path' column")
        st.stop()

    st.success(f"Loaded {len(df)} audio files")
    st.dataframe(df[['audio_file_path']].head(10), width='stretch')

    # Initialize session state for results
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = []

    if 'current_review_idx' not in st.session_state:
        st.session_state.current_review_idx = 0

    # SUSTAINED VOWEL PROCESSING
    if task_type == "Sustained Vowel":
        review_all = st.checkbox("Review all files before saving", key="vowel_review_all")

        if st.button("Process All Files", type="primary"):
            st.session_state.processing_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, row in df.iterrows():
                audio_path = row['audio_file_path']
                status_text.text(f"Processing {idx + 1}/{len(df)}: {os.path.basename(audio_path)}")

                # Get settings from session state with defaults
                energy_thresh = st.session_state.get('vowel_energy_threshold_val', 5) / 100.0
                auto_accept = st.session_state.get('vowel_auto_accept_val', 1.0)
                margin_before = st.session_state.get('vowel_margin_before_val', 0.0)
                margin_after = st.session_state.get('vowel_margin_after_val', 0.0)

                result = process_sustained_vowel(
                    audio_path, st.session_state.processing_results,
                    energy_threshold=energy_thresh,
                    auto_accept_threshold=auto_accept,
                    margin_before=margin_before,
                    margin_after=margin_after
                )
                st.session_state.processing_results.append(result)

                progress_bar.progress((idx + 1) / len(df))

            # If review_all is checked, force all to needs_review
            if review_all:
                for r in st.session_state.processing_results:
                    if r['status'] == 'completed':
                        r['status'] = 'needs_review'

            status_text.text("Processing complete!")
            st.rerun()

        # Show results summary
        if st.session_state.processing_results:
            results = st.session_state.processing_results

            # Summary stats
            completed = sum(1 for r in results if r['status'] == 'completed')
            needs_review = sum(1 for r in results if r['status'] == 'needs_review')
            errors = sum(1 for r in results if r['status'] == 'error')

            col1, col2, col3 = st.columns(3)
            col1.metric("Auto-Accepted", completed)
            col2.metric("Needs Manual Review", needs_review)
            col3.metric("Errors", errors)

            # Results table
            st.subheader("Processing Results")
            results_df = pd.DataFrame([
                {
                    'File': os.path.basename(r['audio_path']),
                    'E-Trim Duration': f"{r['e_duration']:.2f}s" if r['e_duration'] else '-',
                    'P-Trim Duration': f"{r['p_duration']:.2f}s" if r['p_duration'] else '-',
                    'Diff': f"{r['diff']:.2f}s" if r['diff'] else '-',
                    'Status': r['status'],
                    'Method': r.get('final_method', 'pending')
                }
                for r in results
            ])
            st.dataframe(results_df, width='stretch')

            # MANUAL REVIEW SECTION
            review_results = [r for r in results if r['status'] == 'needs_review']

            if review_results:
                st.header("Manual Review Required")
                st.markdown(f"**{len(review_results)} files** need manual review (diff > ±1 second)")

                # Navigation
                review_idx = st.session_state.current_review_idx
                if review_idx >= len(review_results):
                    review_idx = 0
                    st.session_state.current_review_idx = 0

                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if st.button("← Previous") and review_idx > 0:
                        st.session_state.current_review_idx -= 1
                        st.rerun()
                with col3:
                    if st.button("Next →") and review_idx < len(review_results) - 1:
                        st.session_state.current_review_idx += 1
                        st.rerun()

                st.markdown(f"**Reviewing {review_idx + 1} of {len(review_results)}**")

                # Current file for review
                current = review_results[review_idx]
                st.markdown(f"### File: `{os.path.basename(current['audio_path'])}`")
                st.markdown(f"**Difference:** {current['diff']:.2f}s (E: {current['e_duration']:.2f}s, P: {current['p_duration']:.2f}s)")

                # Load audio data for visualization
                y, sr, _ = load_audio(current['wav_path'])
                y_e, _ = librosa.load(current['e_trimmed_path'], sr=None)
                y_p, _ = librosa.load(current['p_trimmed_path'], sr=None)

                # Display waveforms
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Energy-Based Trim")
                    st.plotly_chart(plot_waveform(y_e, sr, f"E-Trimmed ({current['e_duration']:.2f}s)"), width='stretch')
                    st.audio(current['e_trimmed_path'])
                    if st.button("Accept E-Trim", key=f"accept_e_{review_idx}"):
                        # Copy to final folder
                        final_path = os.path.join(current['final_folder'], f"{current['base_name']}_final.wav")
                        sf.write(final_path, y_e, sr)
                        current['final_path'] = final_path
                        current['final_method'] = 'e_trimmed (manual)'
                        current['status'] = 'completed'
                        st.success("Accepted E-Trim!")
                        st.rerun()

                with col2:
                    st.markdown("#### Pitch-Based Trim (Praat)")
                    st.plotly_chart(plot_waveform(y_p, sr, f"P-Trimmed ({current['p_duration']:.2f}s)"), width='stretch')
                    st.audio(current['p_trimmed_path'])
                    if st.button("Accept P-Trim", key=f"accept_p_{review_idx}"):
                        # Copy to final folder
                        final_path = os.path.join(current['final_folder'], f"{current['base_name']}_final.wav")
                        sf.write(final_path, y_p, sr)
                        current['final_path'] = final_path
                        current['final_method'] = 'p_trimmed (manual)'
                        current['status'] = 'completed'
                        st.success("Accepted P-Trim!")
                        st.rerun()

                # Manual trim section
                st.markdown("---")
                st.markdown("### Manual Trim (Original Waveform)")
                st.markdown("**Click and drag on the waveform to select the region to trim**")

                # Initialize selection state for this file
                selection_key = f"selection_{review_idx}"
                if selection_key not in st.session_state:
                    st.session_state[selection_key] = {'start': 0.0, 'end': float(current['original_duration'])}

                # Create selectable waveform
                fig = plot_waveform(y, sr, f"Original ({current['original_duration']:.2f}s)", selectable=True)

                # Use plotly chart with selection
                event = st.plotly_chart(
                    fig,
                    width='stretch',
                    on_select="rerun",
                    selection_mode="box",
                    key=f"waveform_select_{review_idx}"
                )

                # Process selection if available
                if event and event.selection and event.selection.box:
                    box = event.selection.box[0]
                    if 'x' in box and len(box['x']) >= 2:
                        st.session_state[selection_key]['start'] = max(0, min(box['x']))
                        st.session_state[selection_key]['end'] = min(float(current['original_duration']), max(box['x']))

                start_sec = st.session_state[selection_key]['start']
                end_sec = st.session_state[selection_key]['end']

                # Show current selection
                st.markdown(f"**Selected region:** {start_sec:.2f}s - {end_sec:.2f}s (Duration: {end_sec - start_sec:.2f}s)")

                # Show preview of selection
                if start_sec != end_sec:
                    st.plotly_chart(
                        plot_waveform_with_selection(y, sr, start_sec, end_sec, "Selection Preview"),
                        width='stretch'
                    )

                st.audio(current['wav_path'])

                # Fallback slider for fine-tuning
                with st.expander("Fine-tune with slider"):
                    start_sec, end_sec = st.slider(
                        "Adjust selection",
                        0.0, float(current['original_duration']),
                        (float(start_sec), float(end_sec)),
                        step=0.01,
                        key=f"slider_{review_idx}"
                    )
                    st.session_state[selection_key]['start'] = start_sec
                    st.session_state[selection_key]['end'] = end_sec

                if st.button("Save Manual Trim", key=f"manual_{review_idx}"):
                    try:
                        s_idx = int(start_sec * sr)
                        e_idx = int(end_sec * sr)
                        y_manual = y[s_idx:e_idx]

                        # Apply Praat refinement to manual selection
                        y_refined = trim_audio_with_praat(y_manual, sr)
                        if len(y_refined) == 0:
                            y_refined = y_manual

                        final_path = os.path.join(current['final_folder'], f"{current['base_name']}_final.wav")
                        sf.write(final_path, y_refined, sr)
                        current['final_path'] = final_path
                        current['final_method'] = 'manual'
                        current['status'] = 'completed'
                        st.success(f"Manual trim saved! Duration: {get_duration(y_refined, sr):.2f}s")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error saving: {e}")

    # SPEECH PROCESSING
    elif task_type == "Speech":
        review_all_speech = st.checkbox("Review all files before saving", key="speech_review_all")

        if st.button("Process All Files", type="primary"):
            st.session_state.processing_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, row in df.iterrows():
                audio_path = row['audio_file_path']
                status_text.text(f"Processing {idx + 1}/{len(df)}: {os.path.basename(audio_path)}")

                result = process_speech(audio_path)
                st.session_state.processing_results.append(result)

                progress_bar.progress((idx + 1) / len(df))

            # If review_all is checked, force all to needs_review
            if review_all_speech:
                for r in st.session_state.processing_results:
                    if r['status'] == 'completed':
                        r['status'] = 'needs_review'

            status_text.text("Processing complete!")
            st.rerun()

        # Show results summary
        if st.session_state.processing_results:
            results = st.session_state.processing_results

            # Summary stats
            completed = sum(1 for r in results if r['status'] == 'completed')
            errors = sum(1 for r in results if r['status'] == 'error')

            col1, col2 = st.columns(2)
            col1.metric("Completed", completed)
            col2.metric("Errors", errors)

            # Results table
            st.subheader("Processing Results")
            results_df = pd.DataFrame([
                {
                    'File': os.path.basename(r['audio_path']),
                    'Original Duration': f"{r['original_duration']:.2f}s" if r.get('original_duration') else '-',
                    'Trimmed Duration': f"{r['trimmed_duration']:.2f}s" if r.get('trimmed_duration') else '-',
                    'Status': r['status'],
                    'Method': r.get('final_method', '-')
                }
                for r in results
            ])
            st.dataframe(results_df, width='stretch')

    # BREATHING PROCESSING (Manual)
    elif task_type == "Breathing":
        # AUTO-TRIM SILENCE MODE
        if breathing_mode == "Auto-trim silence":
            st.subheader("Auto-Trim Silence from Breathing Files")
            st.markdown("Automatically trim leading and trailing silence from all files, keeping all breathing sounds.")

            review_all_breathing = st.checkbox("Review all files before saving", key="breathing_review_all")

            if st.button("Process All Files", type="primary"):
                st.session_state.processing_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                silence_thresh = st.session_state.get('breathing_silence_threshold_val', 2) / 100.0

                for idx, row in df.iterrows():
                    audio_path = row['audio_file_path']
                    status_text.text(f"Processing {idx + 1}/{len(df)}: {os.path.basename(audio_path)}")

                    try:
                        y, sr, wav_path = load_audio(audio_path)
                        original_duration = get_duration(y, sr)

                        # Get parent directory
                        parent_dir = os.path.dirname(wav_path)
                        base_name = os.path.splitext(os.path.basename(wav_path))[0]
                        final_folder = ensure_folder(parent_dir, "Final_trim")

                        # Trim silence from start and end
                        y_trimmed = trim_silence_only(y, sr, threshold_ratio=silence_thresh)
                        trimmed_duration = get_duration(y_trimmed, sr)

                        if review_all_breathing:
                            # Don't save yet, mark for review
                            st.session_state.processing_results.append({
                                'audio_path': audio_path,
                                'status': 'needs_review',
                                'original_duration': original_duration,
                                'trimmed_duration': trimmed_duration,
                                'wav_path': wav_path,
                                'y_trimmed': y_trimmed,
                                'sr': sr,
                                'final_folder': final_folder,
                                'base_name': base_name
                            })
                        else:
                            # Save immediately
                            final_path = os.path.join(final_folder, f"{base_name}_breath_final.wav")
                            sf.write(final_path, y_trimmed, sr)

                            st.session_state.processing_results.append({
                                'audio_path': audio_path,
                                'status': 'completed',
                                'original_duration': original_duration,
                                'trimmed_duration': trimmed_duration,
                                'final_path': final_path
                            })

                    except Exception as e:
                        st.session_state.processing_results.append({
                            'audio_path': audio_path,
                            'status': 'error',
                            'error': str(e)
                        })

                    progress_bar.progress((idx + 1) / len(df))

                status_text.text("Processing complete!")
                st.rerun()

            # Show results
            if st.session_state.get('processing_results'):
                results = st.session_state.processing_results

                completed = sum(1 for r in results if r['status'] == 'completed')
                errors = sum(1 for r in results if r['status'] == 'error')

                col1, col2 = st.columns(2)
                col1.metric("Completed", completed)
                col2.metric("Errors", errors)

                st.subheader("Processing Results")
                results_df = pd.DataFrame([
                    {
                        'File': os.path.basename(r['audio_path']),
                        'Original': f"{r.get('original_duration', 0):.2f}s",
                        'Trimmed': f"{r.get('trimmed_duration', 0):.2f}s",
                        'Status': r['status']
                    }
                    for r in results
                ])
                st.dataframe(results_df, width='stretch')

        # MANUAL TRIM MODE
        else:
            st.subheader("Manual Trimming for Breathing")
            st.markdown("Select a file and manually choose the region to keep.")

            # Initialize breathing session state
            if 'breathing_current_idx' not in st.session_state:
                st.session_state.breathing_current_idx = 0
            if 'breathing_completed' not in st.session_state:
                st.session_state.breathing_completed = set()

            # File navigation
            total_files = len(df)
            completed_count = len(st.session_state.breathing_completed)

            st.markdown(f"**Progress: {completed_count} / {total_files} files trimmed**")

            # File selector
            file_idx = st.session_state.breathing_current_idx
            if file_idx >= total_files:
                file_idx = 0
                st.session_state.breathing_current_idx = 0

            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                if st.button("← Previous", key="breathing_prev") and file_idx > 0:
                    st.session_state.breathing_current_idx -= 1
                    st.rerun()
            with col2:
                file_options = [f"{i+1}. {os.path.basename(row['audio_file_path'])}" +
                              (" ✓" if i in st.session_state.breathing_completed else "")
                              for i, row in df.iterrows()]
                selected_file = st.selectbox(
                    "Select file",
                    file_options,
                    index=file_idx,
                    key="breathing_file_select",
                    label_visibility="collapsed"
                )
                new_idx = file_options.index(selected_file)
                if new_idx != file_idx:
                    st.session_state.breathing_current_idx = new_idx
                    st.rerun()
            with col3:
                if st.button("Next →", key="breathing_next") and file_idx < total_files - 1:
                    st.session_state.breathing_current_idx += 1
                    st.rerun()

            # Current file
            current_row = df.iloc[file_idx]
            audio_path = current_row['audio_file_path']

            st.markdown(f"### File: `{os.path.basename(audio_path)}`")

            try:
                y, sr, wav_path = load_audio(audio_path)
                original_duration = get_duration(y, sr)

                parent_dir = os.path.dirname(wav_path)
                base_name = os.path.splitext(os.path.basename(wav_path))[0]
                final_folder = ensure_folder(parent_dir, "Final_trim")

                st.markdown(f"**Duration:** {original_duration:.2f}s")

                # Show waveform
                st.plotly_chart(
                    plot_waveform(y, sr, "Waveform"),
                    width='stretch'
                )

                st.audio(wav_path)

                # Manual trim section
                st.markdown("---")
                st.markdown("### Select Region to Keep")

                breathing_selection_key = f"breathing_selection_{file_idx}"
                if breathing_selection_key not in st.session_state:
                    st.session_state[breathing_selection_key] = {'start': 0.0, 'end': float(original_duration)}

                # Selectable waveform
                fig = plot_waveform(y, sr, "Click and drag to select region", selectable=True)

                event = st.plotly_chart(
                    fig,
                    width='stretch',
                    on_select="rerun",
                    selection_mode="box",
                    key=f"breathing_waveform_select_{file_idx}"
                )

                if event and event.selection and event.selection.box:
                    box = event.selection.box[0]
                    if 'x' in box and len(box['x']) >= 2:
                        st.session_state[breathing_selection_key]['start'] = max(0, min(box['x']))
                        st.session_state[breathing_selection_key]['end'] = min(float(original_duration), max(box['x']))

                start_sec = st.session_state[breathing_selection_key]['start']
                end_sec = st.session_state[breathing_selection_key]['end']

                st.markdown(f"**Selected region:** {start_sec:.2f}s - {end_sec:.2f}s (Duration: {end_sec - start_sec:.2f}s)")

                with st.expander("Fine-tune with slider"):
                    start_sec, end_sec = st.slider(
                        "Adjust selection",
                        0.0, float(original_duration),
                        (float(start_sec), float(end_sec)),
                        step=0.01,
                        key=f"breathing_slider_{file_idx}"
                    )
                    st.session_state[breathing_selection_key]['start'] = start_sec
                    st.session_state[breathing_selection_key]['end'] = end_sec

                # Preview
                if start_sec != end_sec and (end_sec - start_sec) < original_duration:
                    s_idx = int(start_sec * sr)
                    e_idx = int(end_sec * sr)
                    y_preview = y[s_idx:e_idx]
                    draft_folder = ensure_folder(parent_dir, "trimmed_draft")
                    preview_path = os.path.join(draft_folder, f"{base_name}_manual_preview.wav")
                    sf.write(preview_path, y_preview, sr)
                    st.audio(preview_path)

                if st.button("Save", key=f"breathing_manual_save_{file_idx}", type="primary"):
                    try:
                        s_idx = int(start_sec * sr)
                        e_idx = int(end_sec * sr)
                        y_manual = y[s_idx:e_idx]

                        final_path = os.path.join(final_folder, f"{base_name}_breath_final.wav")
                        sf.write(final_path, y_manual, sr)
                        st.session_state.breathing_completed.add(file_idx)
                        st.success(f"Saved! Duration: {get_duration(y_manual, sr):.2f}s")
                        if file_idx < total_files - 1:
                            st.session_state.breathing_current_idx += 1
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error saving: {e}")

            except Exception as e:
                st.error(f"Error loading file: {e}")

    # COUGH PROCESSING
    elif task_type == "Cough":
        review_all_cough = st.checkbox("Review all files before saving", key="cough_review_all")

        if st.button("Process All Files", type="primary"):
            st.session_state.processing_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, row in df.iterrows():
                audio_path = row['audio_file_path']
                status_text.text(f"Processing {idx + 1}/{len(df)}: {os.path.basename(audio_path)}")

                # Determine trim method from selection
                method = "parselmouth" if "Parselmouth" in cough_trim_method else "hybrid"
                # Get margin values from session state with defaults
                margin_before = st.session_state.get('cough_margin_before_val', 0.0)
                margin_after = st.session_state.get('cough_margin_after_val', 0.0)
                result = process_cough(audio_path, num_coughs, cough_to_isolate,
                                       cough_threshold, cough_min_gap, cough_min_duration,
                                       trim_method=method,
                                       margin_before=margin_before,
                                       margin_after=margin_after)
                st.session_state.processing_results.append(result)

                progress_bar.progress((idx + 1) / len(df))

            # If review_all is checked, force all to needs_review
            if review_all_cough:
                for r in st.session_state.processing_results:
                    if r['status'] == 'completed':
                        r['status'] = 'needs_review'

            status_text.text("Processing complete!")
            st.rerun()

        # Show results summary
        if st.session_state.processing_results:
            results = st.session_state.processing_results

            # Summary stats
            completed = sum(1 for r in results if r['status'] == 'completed')
            needs_review = sum(1 for r in results if r['status'] == 'needs_review')
            errors = sum(1 for r in results if r['status'] == 'error')

            col1, col2, col3 = st.columns(3)
            col1.metric("Completed", completed)
            col2.metric("Needs Manual Review", needs_review)
            col3.metric("Errors", errors)

            # Results table
            st.subheader("Processing Results")
            results_df = pd.DataFrame([
                {
                    'File': os.path.basename(r['audio_path']),
                    'Expected Coughs': r.get('num_expected', '-'),
                    'Detected Coughs': r.get('num_detected', '-'),
                    'Segments Match': 'Yes' if r.get('segments_match') else 'No',
                    'Trimmed Duration': f"{r['trimmed_duration']:.2f}s" if r.get('trimmed_duration') else '-',
                    'Status': r['status'],
                    'Method': r.get('final_method', 'pending')
                }
                for r in results
            ])
            st.dataframe(results_df, width='stretch')

            # COUGH MANUAL REVIEW SECTION
            review_results = [r for r in results if r['status'] == 'needs_review']

            if review_results:
                st.header("Manual Review Required")
                st.markdown(f"**{len(review_results)} files** need manual review (segment count mismatch)")

                # Navigation
                review_idx = st.session_state.current_review_idx
                if review_idx >= len(review_results):
                    review_idx = 0
                    st.session_state.current_review_idx = 0

                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if st.button("← Previous", key="cough_prev") and review_idx > 0:
                        st.session_state.current_review_idx -= 1
                        st.rerun()
                with col3:
                    if st.button("Next →", key="cough_next") and review_idx < len(review_results) - 1:
                        st.session_state.current_review_idx += 1
                        st.rerun()

                st.markdown(f"**Reviewing {review_idx + 1} of {len(review_results)}**")

                # Current file for review
                current = review_results[review_idx]
                st.markdown(f"### File: `{os.path.basename(current['audio_path'])}`")
                st.warning(f"**Mismatch:** Expected {current['num_expected']} coughs, detected {current['num_detected']} segments")

                # Load audio data for visualization
                y, sr, _ = load_audio(current['wav_path'])
                segments = current.get('detected_segments', [])

                # Show waveform with detected segments
                st.markdown("#### Detected Segments")
                if segments:
                    st.plotly_chart(
                        plot_waveform_with_segments(y, sr, segments, "Waveform with Detected Segments"),
                        width='stretch'
                    )

                    # Table-like layout: Segment | Player | Accept (each row)
                    for seg_idx, (start_samp, end_samp) in enumerate(segments):
                        seg_start_time = start_samp / sr
                        seg_end_time = end_samp / sr
                        seg_duration = seg_end_time - seg_start_time
                        y_seg = y[start_samp:end_samp]

                        # Save preview audio
                        seg_preview_path = os.path.join(current['draft_folder'], f"{current['base_name']}_seg{seg_idx+1}_preview.wav")
                        sf.write(seg_preview_path, y_seg, sr)

                        # Single row: Segment info | Audio player | Accept button
                        col1, col2, col3, col4 = st.columns([1, 1.2, 0.6, 2], gap="small")
                        col1.markdown(f"**Segment {seg_idx + 1}**<br>({seg_duration:.2f}s)", unsafe_allow_html=True)
                        col2.audio(seg_preview_path)
                        if col3.button("Accept", key=f"accept_seg_{review_idx}_{seg_idx}"):
                                final_path = os.path.join(
                                    current['final_folder'],
                                    f"{current['base_name']}_cough{current['cough_to_isolate']}_final.wav"
                                )
                                sf.write(final_path, y_seg, sr)
                                current['final_path'] = final_path
                                current['final_method'] = f'detected_segment_{seg_idx + 1}'
                                current['trimmed_duration'] = seg_duration
                                current['status'] = 'completed'
                                st.success(f"Segment {seg_idx + 1} saved!")
                                st.rerun()
                else:
                    st.plotly_chart(
                        plot_waveform(y, sr, "Original Waveform (No segments detected)"),
                        width='stretch'
                    )
                    st.audio(current['wav_path'])

                # Manual trim with interactive selection
                st.markdown("---")
                st.markdown("### Or: Manual Trim")
                st.markdown(f"**Click and drag on the waveform below to select Cough {current['cough_to_isolate']}**")

                # Initialize selection state for this file
                cough_selection_key = f"cough_selection_{review_idx}"
                if cough_selection_key not in st.session_state:
                    st.session_state[cough_selection_key] = {'start': 0.0, 'end': float(current['original_duration'])}

                # Create selectable waveform
                fig = plot_waveform(y, sr, f"Select Cough {current['cough_to_isolate']}", selectable=True)

                # Use plotly chart with selection
                event = st.plotly_chart(
                    fig,
                    width='stretch',
                    on_select="rerun",
                    selection_mode="box",
                    key=f"cough_waveform_select_{review_idx}"
                )

                # Process selection if available
                if event and event.selection and event.selection.box:
                    box = event.selection.box[0]
                    if 'x' in box and len(box['x']) >= 2:
                        st.session_state[cough_selection_key]['start'] = max(0, min(box['x']))
                        st.session_state[cough_selection_key]['end'] = min(float(current['original_duration']), max(box['x']))

                start_sec = st.session_state[cough_selection_key]['start']
                end_sec = st.session_state[cough_selection_key]['end']

                # Show current selection
                st.markdown(f"**Selected region:** {start_sec:.2f}s - {end_sec:.2f}s (Duration: {end_sec - start_sec:.2f}s)")

                # Show preview of selection
                if start_sec != end_sec and (end_sec - start_sec) < current['original_duration']:
                    st.plotly_chart(
                        plot_waveform_with_selection(y, sr, start_sec, end_sec, "Selection Preview"),
                        width='stretch'
                    )
                    # Preview audio
                    s_idx = int(start_sec * sr)
                    e_idx = int(end_sec * sr)
                    y_preview = y[s_idx:e_idx]
                    preview_path = os.path.join(current['draft_folder'], f"{current['base_name']}_preview.wav")
                    sf.write(preview_path, y_preview, sr)
                    st.audio(preview_path)

                # Fallback slider for fine-tuning
                with st.expander("Fine-tune with slider"):
                    start_sec, end_sec = st.slider(
                        "Adjust selection",
                        0.0, float(current['original_duration']),
                        (float(start_sec), float(end_sec)),
                        step=0.01,
                        key=f"cough_slider_{review_idx}"
                    )
                    st.session_state[cough_selection_key]['start'] = start_sec
                    st.session_state[cough_selection_key]['end'] = end_sec

                if st.button("Save Manual Trim", key=f"cough_manual_{review_idx}"):
                    try:
                        s_idx = int(start_sec * sr)
                        e_idx = int(end_sec * sr)
                        y_manual = y[s_idx:e_idx]

                        final_path = os.path.join(
                            current['final_folder'],
                            f"{current['base_name']}_cough{current['cough_to_isolate']}_final.wav"
                        )
                        sf.write(final_path, y_manual, sr)
                        current['final_path'] = final_path
                        current['final_method'] = 'manual'
                        current['trimmed_duration'] = get_duration(y_manual, sr)
                        current['status'] = 'completed'
                        st.success(f"Manual trim saved! Duration: {get_duration(y_manual, sr):.2f}s")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error saving: {e}")

    # EVERYTHING FUSED PROCESSING
    elif task_type == "Everything Fused":
        review_all_fused = st.checkbox("Review all files before saving", key="fused_review_all")

        if st.button("Process All Files", type="primary"):
            st.session_state.processing_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, row in df.iterrows():
                audio_path = row['audio_file_path']
                status_text.text(f"Processing {idx + 1}/{len(df)}: {os.path.basename(audio_path)}")

                result = process_fused(audio_path, num_tasks_expected, segment_types,
                                       fused_threshold, fused_min_gap, fused_min_duration)
                st.session_state.processing_results.append(result)

                progress_bar.progress((idx + 1) / len(df))

            # If review_all is checked, force all to needs_review
            if review_all_fused:
                for r in st.session_state.processing_results:
                    if r['status'] == 'completed':
                        r['status'] = 'needs_review'

            status_text.text("Processing complete!")
            st.rerun()

        # Show results summary
        if st.session_state.processing_results:
            results = st.session_state.processing_results

            # Summary stats
            completed = sum(1 for r in results if r['status'] == 'completed')
            needs_review = sum(1 for r in results if r['status'] == 'needs_review')
            errors = sum(1 for r in results if r['status'] == 'error')

            col1, col2, col3 = st.columns(3)
            col1.metric("Completed", completed)
            col2.metric("Needs Manual Review", needs_review)
            col3.metric("Errors", errors)

            # Results table
            st.subheader("Processing Results")
            results_df = pd.DataFrame([
                {
                    'File': os.path.basename(r['audio_path']),
                    'Expected Tasks': r.get('num_expected', '-'),
                    'Detected Segments': r.get('num_detected', '-'),
                    'Match': 'Yes' if r.get('segments_match') else 'No',
                    'Status': r['status'],
                    'Method': r.get('final_method', 'pending')
                }
                for r in results
            ])
            st.dataframe(results_df, width='stretch')

            # FUSED MANUAL REVIEW SECTION
            review_results = [r for r in results if r['status'] == 'needs_review']

            if review_results:
                st.header("Manual Review Required")
                st.markdown(f"**{len(review_results)} files** need manual review (segment count mismatch)")

                # Navigation
                review_idx = st.session_state.current_review_idx
                if review_idx >= len(review_results):
                    review_idx = 0
                    st.session_state.current_review_idx = 0

                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if st.button("← Previous", key="fused_prev") and review_idx > 0:
                        st.session_state.current_review_idx -= 1
                        st.rerun()
                with col3:
                    if st.button("Next →", key="fused_next") and review_idx < len(review_results) - 1:
                        st.session_state.current_review_idx += 1
                        st.rerun()

                st.markdown(f"**Reviewing {review_idx + 1} of {len(review_results)}**")

                # Current file for review
                current = review_results[review_idx]
                st.markdown(f"### File: `{os.path.basename(current['audio_path'])}`")

                st.warning(f"**Mismatch:** Expected {current['num_expected']} tasks, detected {current['num_detected']} segments")

                # Load audio data for visualization
                y, sr, _ = load_audio(current['wav_path'])
                segments = current.get('detected_segments', [])

                # Show waveform with detected segments (use Task 1, Task 2, etc. labels)
                st.markdown("#### Detected Segments")
                if segments:
                    # Use Task 1, Task 2, etc. as labels
                    display_labels = [f"Task {i+1}" for i in range(len(segments))]
                    st.plotly_chart(
                        plot_waveform_with_segments(y, sr, segments, "Waveform with Detected Segments", labels=display_labels),
                        width='stretch'
                    )

                    # Initialize accepted segments tracker
                    accepted_key = f"fused_accepted_{review_idx}"
                    if accepted_key not in st.session_state:
                        st.session_state[accepted_key] = {}

                    # Table-like layout for each segment
                    st.markdown("**Accept detected segments (assign task type):**")
                    for seg_idx, (start_samp, end_samp) in enumerate(segments):
                        seg_start_time = start_samp / sr
                        seg_end_time = end_samp / sr
                        seg_duration = seg_end_time - seg_start_time
                        y_seg = y[start_samp:end_samp]

                        # Save preview audio
                        seg_preview_path = os.path.join(current['draft_folder'], f"{current['base_name']}_seg{seg_idx+1}_preview.wav")
                        sf.write(seg_preview_path, y_seg, sr)

                        # Check if already accepted
                        is_accepted = seg_idx in st.session_state[accepted_key]

                        # Single row: Segment info | Task selector | Audio player | Accept button
                        col1, col2, col3, col4, col5 = st.columns([0.8, 1.2, 1.2, 0.6, 1.5], gap="small")
                        col1.markdown(f"**Seg {seg_idx + 1}**<br>({seg_duration:.2f}s)", unsafe_allow_html=True)

                        # Task number selector (Task 1, Task 2, etc.)
                        task_options = [f"Task {i+1}" for i in range(current['num_expected'])]
                        task_num_for_seg = col2.selectbox(
                            "Task",
                            task_options,
                            key=f"fused_task_select_{review_idx}_{seg_idx}",
                            label_visibility="collapsed"
                        )

                        col3.audio(seg_preview_path)

                        if is_accepted:
                            col4.success("✓")
                        else:
                            if col4.button("Save", key=f"fused_accept_seg_{review_idx}_{seg_idx}"):
                                # Save this segment with task number
                                task_num = task_num_for_seg.lower().replace(" ", "_")
                                final_path = os.path.join(
                                    current['final_folder'],
                                    f"{current['base_name']}_{task_num}_final.wav"
                                )
                                sf.write(final_path, y_seg, sr)
                                st.session_state[accepted_key][seg_idx] = task_num_for_seg
                                st.success(f"{task_num_for_seg} saved!")
                                st.rerun()

                    # Check if all expected tasks are saved
                    if len(st.session_state[accepted_key]) >= current['num_expected']:
                        current['status'] = 'completed'
                        current['final_method'] = 'manual_segments'
                        st.success("All expected tasks saved!")

                else:
                    st.plotly_chart(
                        plot_waveform(y, sr, "Original Waveform (No segments detected)"),
                        width='stretch'
                    )

                st.audio(current['wav_path'])

                # Manual trim section for adding custom segments
                st.markdown("---")
                st.markdown("### Or: Manual Trim (Add Custom Segment)")
                st.markdown("**Click and drag on the waveform to select a segment**")

                # Initialize selection state
                fused_selection_key = f"fused_selection_{review_idx}"
                if fused_selection_key not in st.session_state:
                    st.session_state[fused_selection_key] = {'start': 0.0, 'end': float(current['original_duration'])}

                # Task number selector for manual segment
                manual_task_options = [f"Task {i+1}" for i in range(current['num_expected'])]
                manual_task_num = st.selectbox(
                    "Task number for this segment:",
                    manual_task_options,
                    key=f"fused_manual_task_num_{review_idx}"
                )

                # Create selectable waveform
                fig = plot_waveform(y, sr, "Select segment to trim", selectable=True)

                event = st.plotly_chart(
                    fig,
                    width='stretch',
                    on_select="rerun",
                    selection_mode="box",
                    key=f"fused_waveform_select_{review_idx}"
                )

                # Process selection
                if event and event.selection and event.selection.box:
                    box = event.selection.box[0]
                    if 'x' in box and len(box['x']) >= 2:
                        st.session_state[fused_selection_key]['start'] = max(0, min(box['x']))
                        st.session_state[fused_selection_key]['end'] = min(float(current['original_duration']), max(box['x']))

                start_sec = st.session_state[fused_selection_key]['start']
                end_sec = st.session_state[fused_selection_key]['end']

                st.markdown(f"**Selected region:** {start_sec:.2f}s - {end_sec:.2f}s (Duration: {end_sec - start_sec:.2f}s)")

                # Fine-tune slider
                with st.expander("Fine-tune with slider"):
                    start_sec, end_sec = st.slider(
                        "Adjust selection",
                        0.0, float(current['original_duration']),
                        (float(start_sec), float(end_sec)),
                        step=0.01,
                        key=f"fused_slider_{review_idx}"
                    )
                    st.session_state[fused_selection_key]['start'] = start_sec
                    st.session_state[fused_selection_key]['end'] = end_sec

                if st.button("Save Manual Segment", key=f"fused_manual_{review_idx}"):
                    try:
                        s_idx = int(start_sec * sr)
                        e_idx = int(end_sec * sr)
                        y_manual = y[s_idx:e_idx]

                        task_label = manual_task_num.lower().replace(" ", "_")
                        final_path = os.path.join(
                            current['final_folder'],
                            f"{current['base_name']}_{task_label}_final.wav"
                        )
                        sf.write(final_path, y_manual, sr)
                        st.success(f"{manual_task_num} saved! Duration: {get_duration(y_manual, sr):.2f}s")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error saving: {e}")

    # GENERAL-MANUAL PROCESSING
    elif task_type == "General-manual":
        st.subheader("General Manual Trimming")
        st.markdown("Select a region from your audio file and save the trimmed segment.")

        # Initialize session state
        if 'general_current_idx' not in st.session_state:
            st.session_state.general_current_idx = 0
        if 'general_completed' not in st.session_state:
            st.session_state.general_completed = set()

        # File navigation
        total_files = len(df)
        completed_count = len(st.session_state.general_completed)

        st.markdown(f"**Progress: {completed_count} / {total_files} files trimmed**")

        # File selector
        file_idx = st.session_state.general_current_idx
        if file_idx >= total_files:
            file_idx = 0
            st.session_state.general_current_idx = 0

        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("← Previous", key="general_prev") and file_idx > 0:
                st.session_state.general_current_idx -= 1
                st.rerun()
        with col2:
            file_options = [f"{i+1}. {os.path.basename(row['audio_file_path'])}" +
                          (" ✓" if i in st.session_state.general_completed else "")
                          for i, row in df.iterrows()]
            selected_file = st.selectbox(
                "Select file",
                file_options,
                index=file_idx,
                key="general_file_select",
                label_visibility="collapsed"
            )
            new_idx = file_options.index(selected_file)
            if new_idx != file_idx:
                st.session_state.general_current_idx = new_idx
                st.rerun()
        with col3:
            if st.button("Next →", key="general_next") and file_idx < total_files - 1:
                st.session_state.general_current_idx += 1
                st.rerun()

        # Current file
        current_row = df.iloc[file_idx]
        audio_path = current_row['audio_file_path']

        st.markdown(f"### File: `{os.path.basename(audio_path)}`")

        try:
            y, sr, wav_path = load_audio(audio_path)
            original_duration = get_duration(y, sr)

            parent_dir = os.path.dirname(wav_path)
            base_name = os.path.splitext(os.path.basename(wav_path))[0]
            final_folder = ensure_folder(parent_dir, "Final_trim")

            st.markdown(f"**Duration:** {original_duration:.2f}s")

            # Show waveform
            st.plotly_chart(
                plot_waveform(y, sr, "Waveform"),
                width='stretch'
            )

            st.audio(wav_path)

            # Manual trim section
            st.markdown("---")
            st.markdown("### Select Region to Trim")

            general_selection_key = f"general_selection_{file_idx}"
            if general_selection_key not in st.session_state:
                st.session_state[general_selection_key] = {'start': 0.0, 'end': float(original_duration)}

            # Selectable waveform
            fig = plot_waveform(y, sr, "Click and drag to select region", selectable=True)

            event = st.plotly_chart(
                fig,
                width='stretch',
                on_select="rerun",
                selection_mode="box",
                key=f"general_waveform_select_{file_idx}"
            )

            if event and event.selection and event.selection.box:
                box = event.selection.box[0]
                if 'x' in box and len(box['x']) >= 2:
                    st.session_state[general_selection_key]['start'] = max(0, min(box['x']))
                    st.session_state[general_selection_key]['end'] = min(float(original_duration), max(box['x']))

            start_sec = st.session_state[general_selection_key]['start']
            end_sec = st.session_state[general_selection_key]['end']

            st.markdown(f"**Selected region:** {start_sec:.2f}s - {end_sec:.2f}s (Duration: {end_sec - start_sec:.2f}s)")

            with st.expander("Fine-tune with slider"):
                start_sec, end_sec = st.slider(
                    "Adjust selection",
                    0.0, float(original_duration),
                    (float(start_sec), float(end_sec)),
                    step=0.01,
                    key=f"general_slider_{file_idx}"
                )
                st.session_state[general_selection_key]['start'] = start_sec
                st.session_state[general_selection_key]['end'] = end_sec

            # Preview
            if start_sec != end_sec and (end_sec - start_sec) < original_duration:
                s_idx = int(start_sec * sr)
                e_idx = int(end_sec * sr)
                y_preview = y[s_idx:e_idx]
                draft_folder = ensure_folder(parent_dir, "trimmed_draft")
                preview_path = os.path.join(draft_folder, f"{base_name}_general_preview.wav")
                sf.write(preview_path, y_preview, sr)
                st.audio(preview_path)

            if st.button("Save", key=f"general_manual_save_{file_idx}", type="primary"):
                try:
                    s_idx = int(start_sec * sr)
                    e_idx = int(end_sec * sr)
                    y_manual = y[s_idx:e_idx]

                    final_path = os.path.join(final_folder, f"{base_name}_trimmed.wav")
                    sf.write(final_path, y_manual, sr)
                    st.session_state.general_completed.add(file_idx)
                    st.success(f"Saved! Duration: {get_duration(y_manual, sr):.2f}s")
                    if file_idx < total_files - 1:
                        st.session_state.general_current_idx += 1
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving: {e}")

        except Exception as e:
            st.error(f"Error loading file: {e}")
