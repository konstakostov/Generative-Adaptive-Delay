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


def adaptive_basic_delay(
            x,
            sample_rate,
            block_size,
            markov_chain,
            gain,
            cutoff_freq=20.0,
            display_window_start=0,
            display_window_count=10,
    ):
        """
        Apply an adaptive delay effect to an audio signal using a Markov chain.

        Parameters:
            x (ndarray): Input audio signal.
            sample_rate (float): Sample rate of the audio signal in Hz.
            block_size (int): Number of samples per analysis block.
            markov_chain (MarkovChain): Instance of the MarkovChain class to determine delay times.
            gain (float): Gain for the delayed signal.
            cutoff_freq (float, optional): Cutoff frequency for lowpass filtering of RMS values. Default is 20.0 Hz.
            display_window_start (int, optional): Index of the first window to display. Default is 0.
            display_window_count (int, optional): Number of windows to display. Default is 10.

        Returns:
            tuple: Processed audio signal (ndarray) and a dictionary with window information.
        """
        # Create lists to store the values
        input_values = []
        rms_values = []
        normalized_rms_values = []
        current_states = []
        next_states = []
        delay_times = []
        output_values = []
        window_times = []

        # Process blocks individually rather than computing the full envelope
        num_blocks = len(x) // block_size
        display_window_end = min(display_window_start + display_window_count, num_blocks)

        # Find the maximum delay for buffer extension
        max_delay_samples = int(np.ceil(max(markov_chain.delay_times["long"]) * sample_rate))
        output_length = len(x) + max_delay_samples
        y = np.zeros(output_length, dtype=x.dtype)

        # Copy dry signal into output
        y[:len(x)] = x

        # Process each block individually
        all_rms_values = []
        for i in range(num_blocks):
            block_start = i * block_size
            block_end = min(block_start + block_size, len(x))

            # Get the audio frame for this block
            frame = x[block_start:block_end]

            # Calculate RMS for this specific block (no global smoothing)
            if len(frame) > 0:
                block_rms = np.sqrt(np.mean(frame ** 2))
            else:
                block_rms = 0

            all_rms_values.append(block_rms)

        # Convert to numpy array for filtering
        all_rms_values = np.array(all_rms_values)

        # Apply lowpass filtering to the individual RMS values
        if cutoff_freq > 0:
            filtered_rms_values = lowpass_filter(all_rms_values, cutoff_freq, sample_rate / block_size, order=4)
        else:
            filtered_rms_values = all_rms_values

        # Calculate min and max RMS across all blocks
        rms_min = min(filtered_rms_values)
        rms_max = max(filtered_rms_values) + 1e-6  # Avoid division by zero

        # Now process each block and add delay effect
        for i in range(num_blocks):
            block_start = i * block_size
            block_end = min(block_start + block_size, len(x))

            # Calculate window time
            window_time = block_start / sample_rate

            # Get input value from middle of block
            block_middle = block_start + (block_end - block_start) // 2
            input_value = x[block_middle] if block_middle < len(x) else 0

            # Use the filtered RMS value for this block
            raw_rms = filtered_rms_values[i]
            norm_rms = (raw_rms - rms_min) / (rms_max - rms_min)

            # Get current state before transition
            current_state = markov_chain.current_state

            # Determine delay time using Markov chain
            next_state = markov_chain.next_state(norm_rms)
            delay_time = markov_chain.get_delay_time()
            delay_samples = int(np.ceil(delay_time * sample_rate))

            # Process samples in this block
            for n in range(block_start, block_end):
                if n >= delay_samples:
                    y[n] += gain * x[n - delay_samples]

            # Get output value
            output_value = y[block_middle] if block_middle < len(y) else 0

            # Store data for display windows
            if display_window_start <= i < display_window_end:
                window_times.append(window_time)
                input_values.append(input_value)
                rms_values.append(raw_rms)
                normalized_rms_values.append(norm_rms)
                current_states.append(current_state)
                next_states.append(next_state)
                delay_times.append(delay_time)
                output_values.append(output_value)

        # Print the specified windows with their data
        window_range = f"{display_window_start} to {display_window_end - 1}"
        print(f"\nWindows {window_range} of processed data:")
        print(
            f"{'Window Time (s)':<15} "
            f"{'Input Value':<15} "
            f"{'RMS Value':<15} "
            f"{'Normalized RMS':<15} "
            f"{'Current State':<15} "
            f"{'Next State':<15} "
            f"{'Delay Time':<15} "
            f"{'Output Value':<15}"
        )
        print("-" * 120)

        def format_number(value, threshold=0.0001):
            """
            Format a number to display in scientific or decimal notation based on its magnitude.

            Parameters:
                value (float): The number to format.
                threshold (float, optional): Threshold to switch between decimal and scientific notation. Default is 0.0001.

            Returns:
                str: Formatted number as a string.
            """
            if abs(value) >= threshold:
                return f"{value:.4f}".rstrip('0').rstrip('.')
            else:
                return f"{value:.4e}"

        for i in range(len(window_times)):
            # Format values using the helper function
            time_str = format_number(window_times[i])
            input_str = format_number(input_values[i])
            rms_str = format_number(rms_values[i])
            norm_rms_str = format_number(normalized_rms_values[i])
            delay_str = format_number(delay_times[i])
            output_str = format_number(output_values[i])

            print(
                f"{time_str:<15} "
                f"{input_str:<15} "
                f"{rms_str:<15} "
                f"{norm_rms_str:<15} "
                f"{current_states[i]:<15} "
                f"{next_states[i]:<15} "
                f"{delay_str:<15} "
                f"{output_str:<15}"
            )

        # Calculate and print the total time represented by the displayed windows
        total_time = window_times[-1] + (block_size / sample_rate) - window_times[0]

        # Get the max delay time used in the displayed windows
        max_display_delay = max(delay_times) if delay_times else 0
        extended_end_time = window_times[-1] + (block_size / sample_rate) + max_display_delay

        print(
            f"\nThe displayed windows represent {total_time:.4f} seconds of the processed signal "
            f"(from {window_times[0]:.4f}s to {window_times[-1] + (block_size / sample_rate):.4f}s)."
        )
        print(f"With delay effects visible until {extended_end_time:.4f}s")

        # Return the processed signal and info needed for plotting
        return y, {
            'start_time': window_times[0] if window_times else 0,
            'end_time': window_times[-1] + (block_size / sample_rate) if window_times else 0,
            'extended_end_time': extended_end_time,
            'block_size': block_size
        }


def plot_waveforms(
        original,
        processed,
        sample_rate,
        window_info=None,
        output_filename="waveform_comparison.png",
):
    """
    Generate two comparison plots of original and processed waveforms.
    The detailed view shows the timeframe as the specified window range plus delay effects.

    Parameters:
        original (ndarray): The original audio signal.
        processed (ndarray): The processed audio signal.
        sample_rate (float): The sample rate of the audio signals in Hz.
        window_info (dict, optional): Dictionary containing start_time, end_time, and extended_end_time for detailed view.
        output_filename (str, optional): Path to save the waveform comparison plot. Default is "waveform_comparison.png".
    """
    time_axis = np.linspace(0, len(original) / sample_rate, num=len(original))

    # Get the output filename base and extension
    output_base = output_filename.rsplit('.', 1)[0]
    output_ext = output_filename.split('.')[-1]

    # Create the first plot (Enhanced Overlay - full signal)
    plt.figure(figsize=(10, 8))
    plt.plot(
        time_axis,
        original,
        label="Original",
        color=(0.0, 0.7, 0.7),
        linestyle="solid",
        linewidth=1.0,
        alpha=1.0)
    plt.plot(
        time_axis,
        processed[:len(original)],
        label="Processed",
        color=(0.7, 0.0, 0.7),
        linestyle="solid",
        linewidth=1.0,
        alpha=0.5)
    # plt.title("Overlay of Input Audio Signal and the Signal with Applied Delay Effect Using Generative Markov Chain")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_base}_overlay.{output_ext}", dpi=300)
    plt.close()

    # Create the second plot (Detailed Window View - using specified windows)
    if window_info:
        start_time = window_info['start_time']
        # Use the extended end time that includes the delay effect
        end_time = window_info.get('extended_end_time', window_info['end_time'])
    else:
        # Default to first 10 windows if no info provided
        block_size = 8192
        start_time = 0
        end_time = (10 * block_size) / sample_rate

    # Find corresponding indices
    start_idx = int(start_time * sample_rate)
    end_idx = min(int(end_time * sample_rate), len(processed))  # Use processed length to include delay effects

    # Create a proper time axis for the detailed view
    window_time = np.linspace(start_time, end_time, end_idx - start_idx)

    plt.figure(figsize=(10, 8))
    # For original signal, only show up to its length
    orig_end_idx = min(end_idx, len(original))
    orig_time = np.linspace(start_time, start_time + (orig_end_idx - start_idx) / sample_rate, orig_end_idx - start_idx)

    plt.plot(
        orig_time,
        original[start_idx:orig_end_idx],
        label="Original",
        color=(0.0, 0.7, 0.7),
        linestyle="solid",
        linewidth=1.0,
        alpha=1.0)
    plt.plot(
        window_time,
        processed[start_idx:end_idx],
        label="Processed",
        color=(0.7, 0.0, 0.7),
        linestyle="solid",
        linewidth=1.0,
        alpha=0.5)

    # Format the title to match the print output format, rounded to 2 decimals
    # plt.title(f"Input Audio Signal with Applied Delay Effect Using Generative Markov Chain ({start_time:.2f}s - {end_time:.2f}s)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_base}_detail.{output_ext}", dpi=300)
    plt.close()

    print(f"Waveform plots saved as {output_base}_overlay.{output_ext} and {output_base}_detail.{output_ext}")


def process_audio_with_generative_adaptive_delay(
        input_signal,
        sampling_rate,
        output_filename,
        frame_size=8192,
        gain_delay=0.4,
        cutoff_frequency=1.0,
        display_window_start=0,
        display_window_count=0,
        waveform_plot=None,
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
        display_window_start (int): First window index to display (default: 0)
        display_window_count (int): Number of windows to display (default: 10)
        waveform_plot (str, optional): Path to save waveform comparison plot

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
    output_signal, window_info = adaptive_basic_delay(
        input_signal,
        sampling_rate,
        frame_size,
        markov_chain,
        gain_delay,
        cutoff_frequency,
        display_window_start,
        display_window_count
    )

    # Save the processed output
    sf.write(output_filename, output_signal, int(sampling_rate))
    print(f"Processed signal saved to {output_filename}")

    # Generate visual representations if paths are provided
    if waveform_plot:
        plot_waveforms(input_signal, output_signal, sampling_rate, window_info, waveform_plot)

    return output_signal
