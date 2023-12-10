from turtledemo.chaos import f

import matplotlib.pyplot as plt
import numpy as np

class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view

    def display_waveform(self, wave_file):
        # Assuming wave_file is the path to the .wav file
        waveform, sample_rate = self.model.load_waveform(wave_file)
        time = np.arange(0, len(waveform)) / sample_rate

        # Plot the waveform
        plt.figure()
        plt.plot(time, waveform)
        plt.title('Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.show()

    def display_rt60_plot(self):
        # Assuming the model has methods to compute RT60 for low, mid, and high frequencies
        low_rt60 = self.model.calculate_rt60('low')
        mid_rt60 = self.model.calculate_rt60('mid')
        high_rt60 = self.model.calculate_rt60('high')

        # Plot RT60 for Low, Mid, and High frequencies
        plt.figure()
        plt.bar(['Low', 'Mid', 'High'], [low_rt60, mid_rt60, high_rt60])
        plt.title('RT60 Plot')
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
        low_rt60 = self.model.calculate_rt60('low')
        mid_rt60 = self.model.calculate_rt60('mid')
        high_rt60 = self.model.calculate_rt60('high')
        plt.bar(['Low', 'Mid', 'High'], [low_rt60, mid_rt60, high_rt60])
        plt.title('RT60 Plot')
        plt.xlabel('Frequency Band')
        plt.ylabel('RT60 (s)')

        plt.tight_layout()
        plt.show()

    def add_button_and_visual_data(self):
        # Assuming you have a method to retrieve additional visual data from the model
        additional_data = self.model.get_additional_data()

        # Plot the additional visual data
        plt.figure()
        # Plot your additional visual data (replace 'x' and 'y' with actual data)
        plt.plot(additional_data['x'], additional_data['y'])
        plt.title('Additional Visual Data')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()