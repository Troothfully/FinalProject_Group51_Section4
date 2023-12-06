import numpy as np
from scipy.io import wavfile
import scipy.io
import matplotlib.pyplot as plt
import re
from pydub import AudioSegment
from scipy.signal import find_peaks

class File:
  #Initializes the file object and converts .mp3 to .wav if necessary
  def __init__(self, file_name):
    #if file_name.path.endswith('.wav'):
    self.file_name = file_name

    #elif file_name.path.endswith('.mp3'):
      #sound = AudioSegment.from_mp3(file_name)
      #new_file_name = file_name.split('.')[0] + ".wav"
      #sound.export(new_file_name, format='wav')
      #self.file_name = new_file_name

    #else:
      #raise ValueError(f'Invalid file type (must be .mp3, or .wav: {file_name}')

    #Merges Audio Channels
    self.mergeChannels()

    #Defines variables for Audio Properties
    self.sample_rate, self.data = wavfile.read(self.file_name)
    self.length = self.data.shape[0] / self.sample_rate
    self.time = np.linspace(0., self.length, self.data.shape[0])

    # Perform FFT on the audio data
    fft_result = np.fft.fft(self.data)
    frequencies = np.fft.fftfreq(len(fft_result), d=1/self.sample_rate)

    # Find peaks in the frequency domain
    peaks, _ = find_peaks(np.abs(fft_result))

    # Find the highest peak (resonance frequency)
    highest_peak_index = np.argmax(np.abs(fft_result[peaks]))
    self.resonance_frequency = frequencies[peaks[highest_peak_index]]

    # Find the lowest peak (low frequency)
    lowest_peak_index = np.argmin(np.abs(fft_result[peaks]))
    self.low_frequency = frequencies[peaks[lowest_peak_index]]

    # Find the peak in the middle of the frequency range
    middle_index = len(frequencies) // 2
    self.mid_frequency = frequencies[peaks[np.argmin(np.abs(peaks - middle_index))]]

    # Find the peak in the higher range of the frequency spectrum
    high_index = len(frequencies) // 2  # Consider frequencies above the middle index
    self.high_frequency = frequencies[peaks[np.argmin(np.abs(peaks - high_index))]]

  #Getter function for file_name
  @property
  def file(self):
      return self.file_name

  #setter function for file_name
  @file.setter
  def file(self, file_name):
      if file_name.path.endswith('.wav'):
          self.file_name = file_name

      elif file_name.path.endswith('.mp3'):
          sound = AudioSegment.from_mp3(file_name)
          new_file_name = file_name.split('.')[0] + ".wav"
          sound.export(new_file_name, format='wav')
          self.file_name = new_file_name

      else:
          raise ValueError(f'Invalid file type (must be .mp3, .wav, or .flac): {file_name}')


  #function to merge audio channels
  def mergeChannels(self):
      sound = AudioSegment.from_wav(self.file_name)
      sound = sound.set_channels(1)
      sound.export('mono_channel.wav', format="wav")
      self.file_name = 'mono_channel.wav'

  #function to plot waveform of .wav file
  def plot(self):
      plt.plot(self.time, self.data)
      plt.legend(['Channel 1'])
      plt.xlabel('Time (s)')
      plt.ylabel('Amplitude')
      plt.title('Waveform')
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

userFile = File('16bit2chan.wav')
