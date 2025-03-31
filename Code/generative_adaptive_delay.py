import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt, spectrogram
import matplotlib.pyplot as plt


class MarkovChain:
    """
    A class to represent a Markov Chain for adaptive delay times.

    Attributes:
        states (list): List of possible states.
        transition_matrix (ndarray): Matrix representing state transition probabilities.
        delay_times (dict): Dictionary mapping states to possible delay times.
        current_state (str): The current state of the Markov Chain.
    """

    def __init__(self, states, transition_matrix, delay_times):
        """
        Initialize the MarkovChain with states, transition matrix, and delay times.

        Parameters:
            states (list): List of possible states.
            transition_matrix (ndarray): Matrix representing state transition probabilities.
            delay_times (dict): Dictionary mapping states to possible delay times.
        """
        self.states = states
        self.transition_matrix = transition_matrix
        self.delay_times = delay_times
        self.current_state = np.random.choice(self.states)

    def next_state(self, modulation_factor):
        """
        Determine the next state based on the modulation factor and transition probabilities.

        Parameters:
            modulation_factor (float): Factor to modulate transition probabilities.

        Returns:
            str: The next state of the Markov Chain.
        """
        # Determine state based on modulation factor ranges
        if 0 <= modulation_factor <= 0.1:
            self.current_state = "short"
        elif 1 < modulation_factor <= 0.5:
            self.current_state = "medium"
        else:
            self.current_state = "long"

        # Apply original transition matrix logic
        current_index = self.states.index(self.current_state)
        probabilities = self.transition_matrix[current_index]

        # Modulate probabilities based on the modulation factor
        modulated_probabilities = probabilities * (1 - modulation_factor)
        modulated_probabilities /= modulated_probabilities.sum()

        self.current_state = np.random.choice(self.states, p=modulated_probabilities)
        return self.current_state

    def get_delay_time(self):
        """
        Get a random delay time based on the current state.

        Returns:
            float: A delay time in seconds.
        """
        delay_time_set = self.delay_times[self.current_state]
        return np.random.choice(delay_time_set)


def compute_rms_envelope(x, block_size):
    """
    Compute RMS (Root Mean Square) values for non-overlapping blocks.

    Parameters:
        x (ndarray): Input audio signal
        block_size (int): Number of samples per analysis block

    Returns:
        ndarray: Array of RMS values, one per block

    Notes:
        RMS represents the effective power of the signal in each block
    """
    num_blocks = len(x) // block_size
    rms_values = np.zeros(num_blocks)

    for i in range(num_blocks):
        start = i * block_size
        frame = x[start:start + block_size]
        rms_values[i] = np.sqrt(np.mean(frame ** 2))

    return rms_values


def lowpass_filter(data, cutoff_freq, sample_rate, order=4):
    """
    Apply a Butterworth lowpass filter to smooth data.

    Parameters:
        data (ndarray): Signal to be filtered
        cutoff_freq (float): Cutoff frequency in Hz
        sample_rate (float): Sample rate of signal in Hz
        order (int): Filter order, controls steepness of cutoff

    Returns:
        ndarray: Filtered signal

    Notes:
        Higher order values create steeper cutoffs but may introduce ringing
    """
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)[:2]

    return filtfilt(b, a, data)


def adaptive_basic_delay(x, sample_rate, block_size, markov_chain, gain, cutoff_freq=1.0):
    """
    Apply an adaptive delay effect where delay time varies based on signal amplitude using a Markov chain.

    Parameters:
        x (ndarray): Input audio signal (mono)
        sample_rate (float): Sample rate in Hz
        block_size (int): Analysis block size in samples
        markov_chain (MarkovChain): MarkovChain instance to determine delay times
        gain (float): Gain factor for the delayed signal
        cutoff_freq (float): Cutoff frequency for envelope smoothing in Hz

    Returns:
        ndarray: Processed audio with adaptive delay effect
    """
    # Compute the RMS envelope for the whole signal.
    rms_env = compute_rms_envelope(x, block_size)

    # Apply Butterworth low-pass filter to smooth the envelope
    rms_env = lowpass_filter(rms_env, cutoff_freq, sample_rate, order=4)

    rms_min = np.min(rms_env)
    rms_max = (np.max(rms_env) + 1e-6)  # Avoid division by zero

    # Determine the maximum delay in samples for buffer extension.
    max_delay_samples = int(np.ceil(max(markov_chain.delay_times["long"]) * sample_rate))
    output_length = len(x) + max_delay_samples
    y = np.zeros(output_length, dtype=x.dtype)

    # Copy dry signal into output.
    y[:len(x)] = x

    num_blocks = len(rms_env)

    for i in range(num_blocks):
        block_start = i * block_size
        block_end = block_start + block_size

        # Compute normalized RMS for this block:
        norm_rms = (rms_env[i] - rms_min) / (rms_max - rms_min)

        # Determine the delay time using the Markov chain
        markov_chain.next_state(norm_rms)
        delay_time = markov_chain.get_delay_time()
        delay_samples = int(np.ceil(delay_time * sample_rate))

        # Process each sample in this block:
        for n in range(block_start, min(block_end, len(x))):
            if n >= delay_samples:
                y[n + delay_samples] += gain * x[n - delay_samples]

    return y


def plot_waveforms(original, processed, sample_rate, output_filename="waveform_comparison.png"):
    """
    Generate comparison plots of original and processed waveforms.

    Parameters:
        original (ndarray): Original audio signal
        processed (ndarray): Processed audio signal
        sample_rate (float): Sample rate in Hz
        output_filename (str): Path to save the output image

    Outputs:
        Four subplots:
        1. Original signal waveform
        2. Processed signal waveform
        3. Overlay of both signals
        4. Difference between signals

    Notes:
        Automatically saves the plot to the specified file
    """
    time_axis = np.linspace(0, len(original) / sample_rate, num=len(original))

    difference = processed[:len(original)] - original  # Ensure same length for difference calculation

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(time_axis, original, label="Original Signal", color="blue")
    axes[0].set_title("Original Audio Signal")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    axes[1].plot(time_axis, processed[:len(original)], label="Processed Signal", color="green")
    axes[1].set_title("Processed Audio Signal")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True)

    axes[2].plot(time_axis, original, label="Original Signal", color="blue", alpha=0.6)
    axes[2].plot(time_axis, processed[:len(original)], label="Processed Signal", color="gray", alpha=0.6)
    axes[2].set_title("Overlay: Processed vs. Original")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(time_axis, difference, label="Difference (Processed - Original)", color="red")
    axes[3].set_title("Difference Between Signals")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].set_ylabel("Amplitude")
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Waveform plot saved as {output_filename}")


def plot_spectrogram(original, processed, sample_rate, output_filename="spectrogram_comparison.png"):
    """
    Generate comparison spectrogram of original and processed signals.

    Parameters:
        original (ndarray): Original audio signal
        processed (ndarray): Processed audio signal
        sample_rate (float): Sample rate in Hz
        output_filename (str): Path to save the output image

    Outputs:
        Three subplots:
        1. Original signal spectrogram
        2. Processed signal spectrogram
        3. Difference spectrogram

    Notes:
        Uses logarithmic scaling (dB) with small epsilon to avoid log(0)
        Automatically saves the plot to the specified file
    """
    f1, t1, sxx_orig = spectrogram(original, sample_rate)
    f2, t2, sxx_proc = spectrogram(processed[:len(original)], sample_rate)

    difference = sxx_proc - sxx_orig  # Difference spectrogram

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, sharey=True)

    # Add a small constant to avoid log of zero
    epsilon = 1e-10

    im1 = axes[0].pcolormesh(t1, f1, 10 * np.log10(sxx_orig + epsilon), shading="auto")
    axes[0].set_title("Original Signal Spectrogram")
    axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(im1, ax=axes[0])
    axes[0].grid(True)

    im2 = axes[1].pcolormesh(t2, f2, 10 * np.log10(sxx_proc + epsilon), shading="auto")
    axes[1].set_title("Processed Signal Spectrogram")
    axes[1].set_ylabel("Frequency (Hz)")
    fig.colorbar(im2, ax=axes[1])
    axes[1].grid(True)

    im3 = axes[2].pcolormesh(t1, f1, 10 * np.log10(np.abs(difference) + epsilon), shading="auto", cmap="coolwarm")
    axes[2].set_title("Difference Spectrogram (Processed - Original)")
    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].set_xlabel("Time (seconds)")
    fig.colorbar(im3, ax=axes[2])
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

    print(f"Spectrogram plot saved as {output_filename}")


def process_audio_with_generative_adaptive_delay(
        input_signal,
        sampling_rate,
        output_filename,
        frame_size=8192,
        gain_delay=0.4,
        cutoff_frequency=1.0,
        waveform_plot=None,
        spectrogram_plot=None
):
    """
    Process audio signal with adaptive delay effect using a Markov chain and save the result.

    Parameters:
        input_signal (ndarray): Input audio signal
        sampling_rate (float): Sample rate in Hz
        output_filename (str): Path to save the processed audio
        frame_size (int): Block size in samples for RMS analysis
        gain_delay (float): Gain for delayed signal - controls feedback intensity
        cutoff_frequency (float): Cutoff frequency (Hz) for envelope smoothing
        waveform_plot (str, optional): Path to save waveform comparison plot
        spectrogram_plot (str, optional): Path to save spectrogram comparison plot

    Returns:
        ndarray: Processed audio signal
    """
    # Define Markov chain parameters
    states = ["short", "medium", "long"]

    # Probabilities of transitioning from one state to another
    transition_matrix = np.array([
        [0.5, 0.3, 0.2],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5]
    ])
    delay_times = {
        "short": [0.05, 0.1, 0.15, 0.2],
        "medium": [0.3, 0.4, 0.5, 0.6],
        "long": [0.7, 0.8, 0.9, 1.0]
    }

    # Create MarkovChain instance
    markov_chain = MarkovChain(states, transition_matrix, delay_times)

    # Apply the adaptive delay effect
    output_signal = adaptive_basic_delay(
        input_signal,
        sampling_rate,
        frame_size,
        markov_chain,
        gain_delay,
        cutoff_frequency
    )

    # Save the processed output
    sf.write(output_filename, output_signal, int(sampling_rate))
    print(f"Processed signal saved to {output_filename}")

    # Generate visual representations if paths are provided
    if waveform_plot:
        plot_waveforms(input_signal, output_signal, sampling_rate, waveform_plot)

    if spectrogram_plot:
        plot_spectrogram(input_signal, output_signal, sampling_rate, spectrogram_plot)

    return output_signal
