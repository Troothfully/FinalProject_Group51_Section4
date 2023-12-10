import re
import numpy as np
from scipy.io import wavfile
import scipy.io
import matplotlib.pyplot as plt
import re
from pydub import AudioSegment
from scipy.signal import find_peaks
import os
import librosa
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Model:
    def __init__(self, file_name):
        self.file_name = file_name
        self.rt60high = None
        self.rt60mid = None
        self.high_frequency = None
        self.mid_frequency = None
        self.low_frequency = None
        self.resonance_frequency = None
        self.fft_min = None
        self.frequencies = None
        self.fft_result = None
        self.time = None
        self.length = None
        self.data = None
        self.sample_rate = None
        self.rt60low = None

# use property decorator to use getters & setters
# define file an attribute of Model
    @property
    def file(self):
        return self.file

    @file.setter
    def file(self, value):
        if '.wav' in value.lower():
            self.file_name = value

        elif '.mp3' in value.lower():
            wav_file = os.path.splitext(value)[0] + '.wav'
            sound = AudioSegment.from_mp3(value)
            sound.export(wav_file, format='wav')
            self.file_name = wav_file

        else:
            raise ValueError(f'{value} is neither a MP3 nor a WAV file)')

    def laf(self):
        # Merges Audio Channels
        sound = AudioSegment.from_wav(self.file_name)
        sound = sound.set_channels(1)
        sound.export('mono_channel.wav', format="wav")
        self.file_name = 'mono_channel.wav'
        self.sample_rate, self.data = wavfile.read(self.file_name)
        self.length = self.data.shape[0] / self.sample_rate
        self.time = np.linspace(0., self.length, self.data.shape[0])
        # Perform FFT on the audio data
        self.fft_result = abs(np.fft.fft(self.data))
        self.frequencies = abs(np.fft.fftfreq(len(self.fft_result), d=1 / self.sample_rate))
        self.fft_min = np.max(self.fft_result) * .1
        # Find peaks in the frequency domain
        peaks, _ = find_peaks(np.abs(self.fft_result), self.fft_min)
        # Find the highest peak (resonance frequency)
        highest_peak_index = np.argmax(np.abs(self.fft_result[peaks]))
        self.resonance_frequency = self.frequencies[peaks[highest_peak_index]]
        # Find the low frequency
        self.low_frequency = np.min(self.frequencies[peaks])
        # Find the mid frequency
        self.mid_frequency = np.median(np.abs(self.frequencies[peaks]))
        # Find the high frequency
        self.high_frequency = np.max(self.frequencies[peaks])
        # calculate rt60 values
        self.rt60low = self.calculate_low_freq_rt60()
        self.rt60mid = self.calculate_mid_freq_rt60()
        self.rt60high = self.calculate_high_freq_rt60()

    def calculate_low_freq_rt60(self, low_freq_cutoff=250):
        # Load the audio file
        y, sr = librosa.load(self.file_name, sr=None)
        # Calculate the STFT (Short-Time Fourier Transform)
        stft = librosa.stft(y)
        # Calculate the power spectrogram
        power_spectrogram = np.abs(stft) ** 2
        # Sum the power spectrogram across frequency bins
        power_sum = np.sum(power_spectrogram, axis=0)
        # Calculate the cumulative energy
        cumulative_energy = np.cumsum(power_sum) / np.sum(power_sum)
        # Find the index where the cumulative energy exceeds 60%
        rt60_index = np.argmax(cumulative_energy >= 0.6)
        # Convert the index to frequency in Hz
        rt60_freq = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0])[rt60_index]
        # Calculate RT60 using the formula: RT60 = 0.049 * N / (f_hi - f_lo)
        rt60 = 0.049 * stft.shape[1] / (rt60_freq - low_freq_cutoff)
        return rt60

    def calculate_mid_freq_rt60(self, low_freq_cutoff=250, high_freq_cutoff=5000):
        # Load the audio file
        y, sr = librosa.load(self.file_name, sr=None)
        # Calculate the STFT (Short-Time Fourier Transform)
        stft = librosa.stft(y)
        # Calculate the power spectrogram
        power_spectrogram = np.abs(stft) ** 2
        # Define frequency bins
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0])
        # Find the indices corresponding to the specified frequency range
        low_freq_index = np.argmax(frequencies >= low_freq_cutoff)
        high_freq_index = np.argmax(frequencies >= high_freq_cutoff)
        # Sum the power spectrogram in the specified frequency range
        power_sum = np.sum(power_spectrogram[:, low_freq_index:high_freq_index + 1], axis=1)
        # Calculate the cumulative energy
        cumulative_energy = np.cumsum(power_sum) / np.sum(power_sum)
        # Find the index where the cumulative energy exceeds 60%
        rt60_index = np.argmax(cumulative_energy >= 0.6)
        # Convert the index to frequency in Hz
        rt60_freq = frequencies[low_freq_index + rt60_index]
        # Calculate RT60 using the formula: RT60 = 0.049 * N / (f_hi - f_lo)
        rt60 = 0.049 * stft.shape[1] / (rt60_freq - low_freq_cutoff)
        return rt60

    def calculate_high_freq_rt60(self, high_freq_cutoff=5000):
        # Load the audio file
        y, sr = librosa.load(self.file_name, sr=None)
        # Calculate the STFT (Short-Time Fourier Transform)
        stft = librosa.stft(y)
        # Calculate the power spectrogram
        power_spectrogram = np.abs(stft) ** 2
        # Define frequency bins
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0])
        # Find the index corresponding to the specified high frequency cutoff
        high_freq_index = np.argmax(frequencies >= high_freq_cutoff)
        # Sum the power spectrogram in the high-frequency range
        power_sum = np.sum(power_spectrogram[:, high_freq_index:], axis=1)
        # Calculate the cumulative energy
        cumulative_energy = np.cumsum(power_sum) / np.sum(power_sum)
        # Find the index where the cumulative energy exceeds 60%
        rt60_index = np.argmax(cumulative_energy >= 0.6)
        # Convert the index to frequency in Hz
        rt60_freq = frequencies[high_freq_index + rt60_index]
        # Calculate RT60 using the formula: RT60 = 0.049 * N / (f_hi - f_lo)
        rt60 = 0.049 * stft.shape[1] / (rt60_freq - high_freq_cutoff)
        return rt60

        # function to plot waveform of the FFT
    def plotFFT(self):
        fig = plt.Figure(figsize=(5, 4), dpi=100)
        fig.legend(['Channel 1'])
        fig.suptitle('FFT Waveform')
        fig.supxlabel('Frequency (Hz)')
        fig.supylabel('Amplitude')
        fig.add_subplot(111).plot(self.frequencies, self.fft_result)
        return fig

        # Function to plot spectrogram (one additional plot assigned)
    def plotSpectrogram(self):
        fig = plt.Figure(figsize=(5, 4), dpi=100)
        sample_rate, data = wavfile.read(self.file_name)
        fig.suptitle('Spectrogram')
        spectrum, freqs, t, im = plt.specgram(data, Fs=sample_rate, NFFT=1024, cmap=plt.get_cmap('autumn_r'))
        fig.colorbar(im)
        fig.supxlabel('Time (s)')
        fig.supylabel('Frequency (Hz)')
        #fig.cbar.set_label('Intensity (dB)')
        fig.add_subplot(111).specgram(data, Fs=sample_rate, NFFT=1024, cmap=plt.get_cmap('autumn_r'))
        return fig

    def plotTimeAmp(self):
        fig = plt.Figure(figsize=(5, 4), dpi=100)
        fig.legend(['Channel 1'])
        fig.suptitle('Time/Amplitude Waveform')
        fig.supxlabel('Time(s)')
        fig.supylabel('Amplitude')
        fig.add_subplot(111).plot(self.time, self.data)
        return fig
