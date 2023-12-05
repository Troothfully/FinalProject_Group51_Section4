import re
from pydub import AudioSegment

class FileInput:
    
    def __init__(self, file_name):
        self.file_name = file_name

    @property
    def file(self):
        return self.file_name
    
    @file.setter
    def file(self, file_name):
        pattern = r'\b[A-Za-z0-9._%+-]+.wav\b'
        if re.fullmatch(pattern, file_name):
            self.file_name = file_name
        elif file_name.path.endswith('.mp3') or file_name.path.endswith('.flac'):
            file = AudioSegment.from_file(file_name)
            new_file_name = file_name.split('.')[0] + ".wav"
            file.export(new_file_name, format='wav')
            self.file_name = new_file_name
        else:
            raise ValueError(f'Invalid file type (must be .mp3, .wav, or .flac): {file_name}')

    def save(self):
        with open('file_name.txt', 'a') as f:
            f.write(self.file_name + '\n')


'''
USEFUL CODE TO BE USED LATER FROM CH25 LECTURES

import numpy
import numpy as np
from scipy.io import wavfile
import scipy.io
import matplotlib.pyplot as plt

def merge_channels(channel1, channel2):
    # Check if the channels have the same length
    if len(channel1) != len(channel2):
        raise ValueError("Channels must have the same length")
    # Average the corresponding samples in each channel
    merged_channel = np.mean([channel1, channel2], axis=0)
    return merged_channel

wav_fname = '16bitstereoFX.wav'
samplerate, data = wavfile.read(wav_fname)
print(f"number of channels = {data.shape[len(data.shape) - 1]}")
print(f"sample rate = {samplerate}Hz")
length = data.shape[0] / samplerate
print(f"length = {length}s")
time = np.linspace(0., length, data.shape[0])
'''
'''
Return evenly spaced numbers over a specified interval.
Returns num evenly spaced samples, calculated over the interval [start, stop].
The endpoint of the interval can optionally be excluded.
'''
'''
np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None,
axis=0)[source]
'''Return the shape of an array.'''
'''
numpy.shape(a)[source]
'''
'''
To plot data
'''
'''
time = np.linspace(0., length, data.shape[0])
plt.plot(time, data[:, 0], label="Left channel")
plt.plot(time, data[:, 1], label="Right channel")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()

# audioSpectrum mono only
sample_rate, data = wavfile.read('16bitmono.wav')
spectrum, freqs, t, im = plt.specgram(data, Fs=sample_rate, \
NFFT=1024, cmap=plt.get_cmap('autumn_r'))

# audioSpectrum mono only
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
sample_rate, data = wavfile.read('16bitmono.wav')
spectrum, freqs, t, im = plt.specgram(data, Fs=sample_rate, \
NFFT=1024, cmap=plt.get_cmap('autumn_r'))
cbar = plt.colorbar(im)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
cbar.set_label('Intensity (dB)')
plt.show()
'''
