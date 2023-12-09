import numpy as np
from scipy.io import wavfile
import scipy.io
import matplotlib.pyplot as plt
import re
from pydub import AudioSegment
from scipy.signal import find_peaks
import os

class File:
    # Initializes the file object and converts .mp3 to .wav if necessary
    def __init__(self, file_name):
        self.file_name = file_name

        #def contains_mp3(file_name):
            #return ".mp3" in file_name.lower()

        #def contains_wav(file_name):
            #return ".wav" in file_name.lower()

        #if contains_mp3(file_name):
            #wav_file = os.path.splitext(file_name)[0] + '.wav'
            #sound = AudioSegment.from_mp3(file_name)
            #sound.export(wav_file, format="wav")
            #self.file_name = wav_file

        #elif contains_wav(file_name):
            #self.file_name = file_name

        #else:
            #print(f"{file_name} is neither a MP3 nor a WAV file.")
            #exit()

        # Merges Audio Channels
        self.mergeChannels()

        # Defines variables for Audio Properties
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

    # Getter function for file_name
    @property
    def file(self):
        return self.file_name

    # setter function for file_name
    @file.setter
    def file(self, file_name):
        def contains_mp3(file_name):
            return ".mp3" in file_name.lower()

        def contains_wav(file_name):
            return ".wav" in file_name.lower()

        if contains_mp3(file_name):
            wav_file = os.path.splitext(file_name)[0] + '.wav'
            sound = AudioSegment.from_mp3(file_name)
            sound.export(wav_file, format="wav")
            self.file_name = wav_file
            self.mergeChannels()

        elif contains_wav(file_name):
            self.file_name = file_name
            self.mergeChannels()

        else:
            print(f"{file_name} is neither a MP3 nor a WAV file.")
            exit()

    # function to merge audio channels
    def mergeChannels(self):
        sound = AudioSegment.from_wav(self.file_name)
        sound = sound.set_channels(1)
        sound.export('mono_channel.wav', format="wav")
        self.file_name = 'mono_channel.wav'

    # function to plot waveform of .wav file
    def plotTimeAmp(self):
        plt.plot(self.time, self.data)
        plt.legend(['Channel 1'])
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Time/Amplitude Waveform')
        plt.show()

    # function to plot waveform of the FFT
    def plotFFT(self):
        plt.plot(self.frequencies, self.fft_result)
        plt.legend(['Channel 1'])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title('FFT Waveform')
        plt.show()

    # Function to plot spectrogram (one additional plot assigned)
    def plotSpectrogram(self):
        sample_rate, data = wavfile.read(self.file_name)
        spectrum, freqs, t, im = plt.specgram(data, Fs=sample_rate, \
                                              NFFT=1024, cmap=plt.get_cmap('autumn_r'))
        cbar = plt.colorbar(im)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        cbar.set_label('Intensity (dB)')
        plt.show()

    def calculate_low_frequency_rt60(self, low_freq_cutoff=125):
        # Calculate the energy envelope of the audio
        power = np.cumsum(self.data ** 2)

        # Normalize the energy envelope
        power /= np.max(power)

        # Apply a low-pass filter to focus on low frequencies
        low_pass_cutoff = min(low_freq_cutoff, self.sample_rate // 2)
        low_pass_filter = np.ones(low_pass_cutoff)
        low_pass_filter /= np.sum(low_pass_filter)
        low_frequency_envelope = np.convolve(power, low_pass_filter, mode='valid')

        # Find the time it takes for the low-frequency energy to decay by 60 dB
        rt60_index = np.argmax(low_frequency_envelope < 0.001)
        rt60 = rt60_index / float(self.sample_rate)

        return rt60

    def calculate_mid_frequency_rt60(self, mid_freq_range=(500, 2000)):
        # Calculate the energy envelope of the audio
        power = np.cumsum(self.data ** 2)

        # Normalize the energy envelope
        power /= np.max(power)

        # Apply a band-pass filter to focus on mid frequencies
        mid_freq_min, mid_freq_max = mid_freq_range
        mid_freq_min = max(mid_freq_min, 0)
        mid_freq_max = min(mid_freq_max, self.sample_rate // 2)

        band_pass_filter = np.zeros_like(power)
        band_pass_filter[mid_freq_min:mid_freq_max + 1] = 1
        mid_frequency_envelope = power * band_pass_filter

        # Find the time it takes for the mid-frequency energy to decay by 60 dB
        rt60_index = np.argmax(mid_frequency_envelope < 0.001)
        rt60 = rt60_index / float(power)

        return rt60

    def calculate_high_frequency_rt60(self, high_freq_cutoff=5000):

        # Apply a high-pass filter to focus on high frequencies
        high_pass_cutoff = min(high_freq_cutoff, self.sample_rate // 2)
        high_pass_filter = np.concatenate([np.zeros(high_pass_cutoff), np.ones(self.sample_rate - high_pass_cutoff)])
        high_frequency_data = np.convolve(self.data, high_pass_filter, mode='same')

        # Calculate the energy envelope of the high-frequency data
        energy_envelope = np.cumsum(high_frequency_data ** 2)

        # Normalize the energy envelope
        energy_envelope /= np.max(energy_envelope)

        # Find the time it takes for the high-frequency energy to decay by 60 dB
        rt60_index = np.argmax(energy_envelope < 0.001)
        rt60 = rt60_index / float(self.sample_rate)

        return rt60
