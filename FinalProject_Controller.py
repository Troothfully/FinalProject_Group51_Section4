import matplotlib.pyplot as plt
import numpy as np


class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view

    def plotTimeAmp(self):
        plt.plot(self.model.time, self.model.data)
        plt.legend(['Channel 1'])
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Time/Amplitude Waveform')
        plt.show()

    def display_rt60_plot_low(self):
        # Assuming the model has methods to compute RT60 for low, mid, and high frequencies
        low_rt60 = self.model.calculate_low_freq_rt60()

        # Plot RT60 for Low frequencies
        plt.figure()
        plt.bar([self.model.time], [low_rt60])
        plt.title('RT60 Low-Freq Plot')
        plt.xlabel('Frequency Band')
        plt.ylabel('RT60 (s)')
        plt.show()

    def display_rt60_plot_mid(self):
        # Assuming the model has methods to compute RT60 for low, mid, and high frequencies
        mid_rt60 = self.model.calculate_mid_freq_rt60()

        # Plot RT60 for Mid frequencies
        plt.figure()
        plt.bar([self.model.time], [mid_rt60])
        plt.title('RT60 Mid-Freq Plot')
        plt.xlabel('Frequency Band')
        plt.ylabel('RT60 (s)')
        plt.show()

    def display_rt60_plot_high(self):
        # Assuming the model has methods to compute RT60 for low, mid, and high frequencies
        high_rt60 = self.model.calculate_high_freq_rt60()

        # Plot RT60 for High frequencies
        plt.figure()
        plt.bar([self.model.time], [high_rt60])
        plt.title('RT60 High-Freq Plot')
        plt.xlabel('Frequency Band')
        plt.ylabel('RT60 (s)')
        plt.show()

    def combine_plots(self):
        # Combine the waveform and RT60 plots into a single plot
        plt.figure()

        # Plot the waveform
        plt.subplot(2, 1, 1)
        waveform, sample_rate = self.model.get_waveform()
        time = np.arange(0, len(waveform)) / sample_rate
        plt.plot(time, waveform)
        plt.title('Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        # Plot RT60 for Low, Mid, and High frequencies
        plt.subplot(2, 1, 2)
        low_rt60 = self.model.calculate_low_frequency_rt60()
        mid_rt60 = self.model.calculate_mid_rt60()
        high_rt60 = self.model.calculate_high_rt60()
        plt.bar(self.model.time, [low_rt60, mid_rt60, high_rt60])
        plt.title('RT60 Combined Plot')
        plt.xlabel('Frequency Band')
        plt.ylabel('RT60 (s)')

        plt.tight_layout()
        plt.show()

    # Function to plot spectrogram (one additional plot assigned)
    def plotSpectrogram(self):
        spectrum, freqs, t, im = plt.specgram(self.model.data, Fs=self.model.sample_rate, \
                                              NFFT=1024, cmap=plt.get_cmap('autumn_r'))
        cbar = plt.colorbar(im)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        cbar.set_label('Intensity (dB)')
        plt.show()

    def save(self, file_name):
        """
        Set file name
        :file_name:
        :return:
        """
        def contains_mp3(file_name):
            return ".mp3" in file_name.lower()

        def contains_wav(file_name):
            return ".wav" in file_name.lower()

        if contains_mp3(file_name) or contains_wav(file_name):
            #sets file name to new file_name
            self.model.fileSet(file_name)
            # show a success message
            self.view.show_success(f'File: {file_name} set!')
        else:
            #show an error message
            self.view.show_error(f'{file_name} is neither a MP3 nor a WAV file.')
