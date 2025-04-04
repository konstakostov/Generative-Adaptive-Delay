import numpy as np
import soundfile as sf

from generative_adaptive_delay import process_audio_with_generative_adaptive_delay


# Define input and output file paths
input_filename = "../Audio-Samples/original_recording_01.wav"
output_filename = "../Audio-Outputs/processed_original_recording_01_01.wav"

# Load the input audio file
input_signal, sampling_rate = sf.read(input_filename)

# Convert stereo to mono if necessary
if input_signal.ndim > 1:
    input_signal = input_signal[:, 0]

# Convert to 32-bit float for better precision
input_signal = input_signal.astype(np.float32)

# Process the audio with adaptive delay
output_signal = process_audio_with_generative_adaptive_delay(
    input_signal=input_signal,
    sampling_rate=sampling_rate,
    output_filename=output_filename,
    display_window_start=5,    # Start from window No
    display_window_count=10,    # Number of windows to display
    waveform_plot="../Plots/atg_comparison_processed_original_recording_01_01.png",
)