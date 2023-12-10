from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view

    def laf(self, file):

        try:

            # laf function from the model
            self.model.file = file
            self.model.laf()

            # show a success message
            self.view.show_success(f'The file {file} has been saved as mono_channel.wav!')
            self.view.T.configure(state='normal')
            self.view.T.delete('1.0','end')
            self.view.T.insert('1.0', f'{round(self.model.rt60high,4)} seconds is the RT60 for high frequency.\n')
            self.view.T.insert('2.0', f'{round(self.model.rt60mid,4)} seconds is the RT60 for mid frequency.\n')
            self.view.T.insert('3.0', f'{round(self.model.rt60low,4)} seconds is the RT60 for low frequency.\n')
            self.view.T.insert('4.0', f'{round(self.model.high_frequency,4)} hz is the high frequency.\n')
            self.view.T.insert('5.0', f'{round(self.model.mid_frequency,4)} hz is the mid frequency.\n')
            self.view.T.insert('6.0', f'{round(self.model.low_frequency,4)} hz is the low frequency.\n')
            self.view.T.insert('7.0', f'{round(self.model.resonance_frequency,4)} hz is the resonance frequency.\n')
            self.view.T.insert('8.0', f'{round(self.model.length,4)} seconds is the length of the sound file.\n')
            self.view.T.configure(state='disabled')

            figFFT = self.model.plotFFT()
            chart = FigureCanvasTkAgg(figFFT,self.view.tab3)
            chart.get_tk_widget().grid(row=0,column=0)

            figWave = self.model.plotTimeAmp()
            chart = FigureCanvasTkAgg(figWave, self.view.tab1)
            chart.get_tk_widget().grid(row=0, column=0)

            figSpec = self.model.plotSpectrogram()
            chart = FigureCanvasTkAgg(figSpec, self.view.tab2)
            chart.get_tk_widget().grid(row=0, column=0)


        except ValueError as error:
            # show an error message
            self.view.show_error(error)
