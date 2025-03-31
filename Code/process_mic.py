import numpy as np
import soundfile as sf
import sounddevice as sd
import threading
import keyboard
import time

from generative_adaptive_delay import process_audio_with_generative_adaptive_delay


class MicRecorder:
    """
    A class to record audio from the microphone.
    """

    def __init__(self, sample_rate=44100):
        """
        Initialize the MicRecorder.

        :param sample_rate: The sample rate for recording audio, defaults to 44100.
        """
        self.sample_rate = sample_rate
        self.recording = False
        self.audio_data = []
        self.thread = None

    def toggle_recording(self):
        """
        Toggle the recording state. Starts recording if not currently recording,
        stops recording if currently recording.
        """
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """
        Start recording audio from the microphone.
        """
        print("Recording started. Press spacebar to stop.")
        self.recording = True
        self.audio_data = []
        self.thread = threading.Thread(target=self._record)
        self.thread.start()

    def stop_recording(self):
        """
        Stop recording audio from the microphone.
        """
        if self.recording:
            self.recording = False
            if self.thread:
                self.thread.join()
            print("Recording stopped")

    def get_recorded_audio(self):
        """
        Get the recorded audio data as a numpy array.

        :return: A numpy array containing the recorded audio data.
        """
        if len(self.audio_data) > 0:
            return np.concatenate(self.audio_data).flatten()
        return np.array([])

    def _record(self):
        """
        Record audio data in a separate thread.
        """
        chunk_size = 1024
        stream = sd.InputStream(samplerate=self.sample_rate, channels=1,
                                blocksize=chunk_size, dtype="float32")
        stream.start()

        while self.recording:
            data, overflowed = stream.read(chunk_size)
            if overflowed:
                print("Audio buffer overflowed")
            self.audio_data.append(data.copy())

        stream.stop()
        stream.close()


def wait_for_spacebar():
    """
    Wait for the spacebar press and return.
    """
    keyboard.wait("space")
    time.sleep(0.1)  # Small delay to avoid double-triggers


# Set up the recorder
sampling_rate = 44100
recorder = MicRecorder(sampling_rate)

# Define output file paths
original_filename = "../Audio-Outputs/original_recording_01.wav"
processed_filename = "../Audio-Outputs/processed_recording_01.wav"

# Wait for spacebar to start recording
print("Press spacebar to start recording")
wait_for_spacebar()

# Start recording
recorder.start_recording()

# Wait for spacebar to stop recording
wait_for_spacebar()

# Stop recording and get the audio data
recorder.stop_recording()
input_signal = recorder.get_recorded_audio()

if len(input_signal) > 0:
    # Save the original signal
    sf.write(original_filename, input_signal, sampling_rate)
    print(f"Original signal saved to {original_filename}")

    # Process the audio with adaptive delay
    output_signal = process_audio_with_generative_adaptive_delay(
        input_signal=input_signal,
        sampling_rate=sampling_rate,
        output_filename=processed_filename,
        waveform_plot="../Plots/atg_comparison_rec_01.png",
        spectrogram_plot="../Plots/spc_comparison_rec_01.png"
    )
