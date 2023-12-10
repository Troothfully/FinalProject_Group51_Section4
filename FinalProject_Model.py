import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.signal import find_peaks
import os
class Model:
    def __init__(self, file_name):
        self.spectrum = None
        self.freqs = None
        self.t = None
        self.im = None
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
        self.spectrum, self.freqs, self.t, self.im = plt.specgram(self.data, Fs=self.sample_rate, NFFT=1024, cmap=plt.get_cmap('autumn_r'))

        # function to plot waveform of the FFT
    def plotFFT(self):
        fig = plt.Figure(figsize=(5, 4), dpi=100)
        fig.suptitle('FFT Waveform')
        fig.supxlabel('Frequency (Hz)')
        fig.supylabel('Amplitude')
        ax = fig.add_subplot(111).plot(self.frequencies, self.fft_result)
        fig.legend(ax, ['Channel 1'])
        return fig

        # Function to plot spectrogram (one additional plot assigned)

    def plotSpectrogram(self):
        fig = plt.Figure(figsize=(5, 4), dpi=100)
        fig.suptitle('Spectrogram')
        fig.supxlabel('Time (s)')
        fig.supylabel('Frequency (Hz)')
        ax = fig.add_subplot(111).specgram(self.data, Fs=self.sample_rate, NFFT=1024, cmap=plt.get_cmap('autumn_r'))
        cbar = fig.colorbar(self.im)
        cbar.set_label('Intensity (dB)')
        return fig

    #waveform graph
    def plotTimeAmp(self):
        fig = plt.Figure(figsize=(5, 4), dpi=100)
        fig.suptitle('Time/Amplitude Waveform')
        fig.supxlabel('Time(s)')
        fig.supylabel('Amplitude')
        ax = fig.add_subplot(111).plot(self.time, self.data)
        fig.legend(ax, ['Channel 1'])
        return fig

    #rt60 mid graph
    def plotRT60mid(self):
        data_in_db_mid = self.frequency_check_mid()
        fig = plt.Figure(figsize=(5, 4), dpi=100)
        fig.suptitle('RT60 Mid Frequency')
        fig.supxlabel('Time (s)')
        fig.supylabel('Power (dB)')
        index_of_max_mid = np.argmax(data_in_db_mid)
        value_of_max_mid = data_in_db_mid[index_of_max_mid]
        sliced_array_mid = data_in_db_mid[index_of_max_mid:]
        value_of_max_less_5_mid = value_of_max_mid - 5
        value_of_max_less_5_mid = self.find_nearest_value(sliced_array_mid, value_of_max_less_5_mid)
        index_of_max_less_5_mid = np.where(data_in_db_mid == value_of_max_less_5_mid)
        value_of_max_less_25_mid = value_of_max_mid - 25
        value_of_max_less_25_mid = self.find_nearest_value(sliced_array_mid,value_of_max_less_25_mid)
        index_of_max_less_25_mid = np.where(data_in_db_mid == value_of_max_less_25_mid)
        rt20 = (self.t[index_of_max_less_5_mid]-self.t[index_of_max_less_25_mid])[0]
        rt60 = 3 * rt20
        self.rt60mid = round(abs(rt60),2)
        ax = fig.add_subplot(111).plot(self.t, data_in_db_mid, '-m',
                                  self.t[index_of_max_mid], data_in_db_mid[index_of_max_mid], 'go',
                                  self.t[index_of_max_less_5_mid], data_in_db_mid[index_of_max_less_5_mid], 'yo',
                                  self.t[index_of_max_less_25_mid], data_in_db_mid[index_of_max_less_25_mid], 'ro',
                                  linewidth=1)
        fig.legend(ax, ['Mid Frequency'])
        return fig

    def plotRT60low(self):
        data_in_db_low = self.frequency_check_low()
        fig = plt.Figure(figsize=(5, 4), dpi=100)
        fig.suptitle('RT60 Low Frequency')
        fig.supxlabel('Time (s)')
        fig.supylabel('Power (dB)')
        index_of_max_low = np.argmax(data_in_db_low)
        value_of_max_low = data_in_db_low[index_of_max_low]
        sliced_array_low = data_in_db_low[index_of_max_low:]
        value_of_max_less_5_low = value_of_max_low - 5
        value_of_max_less_5_low = self.find_nearest_value(sliced_array_low, value_of_max_less_5_low)
        index_of_max_less_5_low = np.where(data_in_db_low == value_of_max_less_5_low)
        value_of_max_less_25_low = value_of_max_low - 25
        value_of_max_less_25_low = self.find_nearest_value(sliced_array_low,value_of_max_less_25_low)
        index_of_max_less_25_low = np.where(data_in_db_low == value_of_max_less_25_low)
        rt20 = (self.t[index_of_max_less_5_low]-self.t[index_of_max_less_25_low])[0]
        rt60 = 3 * rt20
        self.rt60low = round(abs(rt60),2)
        ax = fig.add_subplot(111).plot(self.t, data_in_db_low, '-c',
                                  self.t[index_of_max_low], data_in_db_low[index_of_max_low], 'go',
                                  self.t[index_of_max_less_5_low], data_in_db_low[index_of_max_less_5_low], 'yo',
                                  self.t[index_of_max_less_25_low], data_in_db_low[index_of_max_less_25_low], 'ro',
                                  linewidth=1)
        fig.legend(ax, ['Low Frequency'])
        return fig

    def plotRT60high(self):
        data_in_db_high = self.frequency_check_high()
        fig = plt.Figure(figsize=(5, 4), dpi=100)
        fig.suptitle('RT60 High Frequency')
        fig.supxlabel('Time (s)')
        fig.supylabel('Power (dB)')
        index_of_max_high = np.argmax(data_in_db_high)
        value_of_max_high = data_in_db_high[index_of_max_high]
        sliced_array_high = data_in_db_high[index_of_max_high:]
        value_of_max_less_5_high = value_of_max_high - 5
        value_of_max_less_5_high = self.find_nearest_value(sliced_array_high, value_of_max_less_5_high)
        index_of_max_less_5_high = np.where(data_in_db_high == value_of_max_less_5_high)
        value_of_max_less_25_high = value_of_max_high - 25
        value_of_max_less_25_high = self.find_nearest_value(sliced_array_high,value_of_max_less_25_high)
        index_of_max_less_25_high = np.where(data_in_db_high == value_of_max_less_25_high)
        rt20 = (self.t[index_of_max_less_5_high]-self.t[index_of_max_less_25_high])[0]
        rt60 = 3 * rt20
        self.rt60high = round(abs(rt60),2)
        ax = fig.add_subplot(111).plot(self.t, data_in_db_high, '-k',
                                  self.t[index_of_max_high], data_in_db_high[index_of_max_high], 'go',
                                  self.t[index_of_max_less_5_high], data_in_db_high[index_of_max_less_5_high], 'yo',
                                  self.t[index_of_max_less_25_high], data_in_db_high[index_of_max_less_25_high], 'ro',
                                  linewidth=1)
        fig.legend(ax, ['High Frequency'])
        return fig

    def plotRT60Combined(self):
        #figure setup
        fig = plt.Figure(figsize=(5, 4), dpi=100)
        fig.suptitle('RT60 Combined')
        fig.supxlabel('Time (s)')
        fig.supylabel('Power (dB)')
        #data for low
        data_in_db_low = self.frequency_check_low()
        index_of_max_low = np.argmax(data_in_db_low)
        value_of_max_low = data_in_db_low[index_of_max_low]
        sliced_array_low = data_in_db_low[index_of_max_low:]
        value_of_max_less_5_low = value_of_max_low - 5
        value_of_max_less_5_low = self.find_nearest_value(sliced_array_low, value_of_max_less_5_low)
        index_of_max_less_5_low = np.where(data_in_db_low == value_of_max_less_5_low)
        value_of_max_less_25_low = value_of_max_low - 25
        value_of_max_less_25_low = self.find_nearest_value(sliced_array_low, value_of_max_less_25_low)
        index_of_max_less_25_low = np.where(data_in_db_low == value_of_max_less_25_low)
        #data for mid
        data_in_db_mid = self.frequency_check_mid()
        index_of_max_mid = np.argmax(data_in_db_mid)
        value_of_max_mid = data_in_db_mid[index_of_max_mid]
        sliced_array_mid = data_in_db_mid[index_of_max_mid:]
        value_of_max_less_5_mid = value_of_max_mid - 5
        value_of_max_less_5_mid = self.find_nearest_value(sliced_array_mid, value_of_max_less_5_mid)
        index_of_max_less_5_mid = np.where(data_in_db_mid == value_of_max_less_5_mid)
        value_of_max_less_25_mid = value_of_max_mid - 25
        value_of_max_less_25_mid = self.find_nearest_value(sliced_array_mid, value_of_max_less_25_mid)
        index_of_max_less_25_mid = np.where(data_in_db_mid == value_of_max_less_25_mid)
        #data for high
        data_in_db_high = self.frequency_check_high()
        index_of_max_high = np.argmax(data_in_db_high)
        value_of_max_high = data_in_db_high[index_of_max_high]
        sliced_array_high = data_in_db_high[index_of_max_high:]
        value_of_max_less_5_high = value_of_max_high - 5
        value_of_max_less_5_high = self.find_nearest_value(sliced_array_high, value_of_max_less_5_high)
        index_of_max_less_5_high = np.where(data_in_db_high == value_of_max_less_5_high)
        value_of_max_less_25_high = value_of_max_high - 25
        value_of_max_less_25_high = self.find_nearest_value(sliced_array_high, value_of_max_less_25_high)
        index_of_max_less_25_high = np.where(data_in_db_high == value_of_max_less_25_high)
        #add to fig
        ax = fig.add_subplot(111).plot(self.t, data_in_db_low, '-c',
                                  self.t, data_in_db_mid, '-m',
                                  self.t, data_in_db_high, '-k',
                                  #dots for low
                                  self.t[index_of_max_low], data_in_db_low[index_of_max_low], 'go',
                                  self.t[index_of_max_less_5_low], data_in_db_low[index_of_max_less_5_low], 'yo',
                                  self.t[index_of_max_less_25_low], data_in_db_low[index_of_max_less_25_low], 'ro',
                                  #dots for mid
                                  self.t[index_of_max_mid], data_in_db_mid[index_of_max_mid], 'go',
                                  self.t[index_of_max_less_5_mid], data_in_db_mid[index_of_max_less_5_mid], 'yo',
                                  self.t[index_of_max_less_25_mid], data_in_db_mid[index_of_max_less_25_mid], 'ro',
                                  #dots for high
                                  self.t[index_of_max_high], data_in_db_high[index_of_max_high], 'go',
                                  self.t[index_of_max_less_5_high], data_in_db_high[index_of_max_less_5_high], 'yo',
                                  self.t[index_of_max_less_25_high], data_in_db_high[index_of_max_less_25_high], 'ro',
                                  linewidth = 1)
        fig.legend(ax,['Low Frequency', 'Medium Frequency', 'High Frequency'])
        return fig

    def find_low_frequency(self, freqs):
        for x in freqs:
            if x > self.low_frequency:
                break
        return x

    def find_high_frequency(self, freqs):
        for x in freqs:
            if x > self.high_frequency:
                break
        return x

    def find_mid_frequency(self, freqs):
        for x in freqs:
            if x > self.mid_frequency:
                break
        return x


    def frequency_check_mid(self):
        target_frequency = self.find_mid_frequency(self.freqs)
        index_of_frequency = np.where(self.freqs == target_frequency)[0][0]
        data_for_frequency = self.spectrum[index_of_frequency]
        data_in_db = 10*np.log10(data_for_frequency)
        return data_in_db

    def frequency_check_low(self):
        target_frequency = self.find_low_frequency(self.freqs)
        index_of_frequency = np.where(self.freqs == target_frequency)[0][0]
        data_for_frequency = self.spectrum[index_of_frequency]
        data_in_db = 10*np.log10(data_for_frequency)
        return data_in_db

    def frequency_check_high(self):
        target_frequency = self.find_high_frequency(self.freqs)
        index_of_frequency = np.where(self.freqs == target_frequency)[0][0]
        data_for_frequency = self.spectrum[index_of_frequency]
        data_in_db = 10*np.log10(data_for_frequency)
        return data_in_db

    def find_nearest_value(self,array,value):
        array = np.asarray(array)
        idx = (np.abs(array-value)).argmin()
        return array[idx]




