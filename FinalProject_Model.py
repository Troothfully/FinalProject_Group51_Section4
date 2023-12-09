import numpy as np
from scipy.io import wavfile
import scipy.io
import matplotlib.pyplot as plt
import re
from pydub import AudioSegment
from scipy.signal import find_peaks
import os

class File:
  #Initializes the file object and converts .mp3 to .wav if necessary
  def __init__(self, file_name):

    def contains_mp3(file_name):
      return ".mp3" in file_name.lower()

    def contains_wav(file_name):
      return ".wav" in file_name.lower()
      
    if contains_mp3(file_name):
      wav_file = os.path.splitext(file_name)[0] + '.wav'
      sound = AudioSegment.from_mp3(file_name)
      sound.export(wav_file, format="wav")
      self.file_name = wav_file

    elif contains_wav(file_name):
      self.file_name = file_name
      
    else:
      print(f"{file_name} is neither a MP3 nor a WAV file.")
      exit()

    #Merges Audio Channels
    self.mergeChannels()

    #Defines variables for Audio Properties
    self.sample_rate, self.data = wavfile.read(self.file_name)
    self.length = self.data.shape[0] / self.sample_rate
    self.time = np.linspace(0., self.length, self.data.shape[0])

    # Perform FFT on the audio data
    self.fft_result = abs(np.fft.fft(self.data))
    self.frequencies = abs(np.fft.fftfreq(len(self.fft_result), d=1/self.sample_rate))
    self.fft_min = np.max(self.fft_result)*.1
    
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
    
  #Getter function for file_name
  @property
  def file(self):
      return self.file_name

  #setter function for file_name
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

    elif contains_wav(file_name):
      self.file_name = file_name

    else:
      print(f"{file_name} is neither a MP3 nor a WAV file.")
      exit()


  #function to merge audio channels
  def mergeChannels(self):
      sound = AudioSegment.from_wav(self.file_name)
      sound = sound.set_channels(1)
      sound.export('mono_channel.wav', format="wav")
      self.file_name = 'mono_channel.wav'

  #function to plot waveform of .wav file
  def plotTimeAmp(self):
      plt.plot(self.time, self.data)
      plt.legend(['Channel 1'])
      plt.xlabel('Time (s)')
      plt.ylabel('Amplitude')
      plt.title('Time/Amplitude Waveform')
      plt.show()

  #function to plot waveform of the FFT
  def plotFFT(self):
      plt.plot(self.frequencies, self.fft_result)
      plt.legend(['Channel 1'])
      plt.xlabel('Frequency (Hz)')
      plt.ylabel('Amplitude')
      plt.title('FFT Waveform')
      plt.show()

  #Function to plot spectrogram (one additional plot assigned)
  def plotSpectrogram(self):
    sample_rate, data = wavfile.read(self.file_name)
    spectrum, freqs, t, im = plt.specgram(data, Fs=sample_rate, \
    NFFT=1024, cmap=plt.get_cmap('autumn_r'))
    cbar = plt.colorbar(im)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    cbar.set_label('Intensity (dB)')
    plt.show()
