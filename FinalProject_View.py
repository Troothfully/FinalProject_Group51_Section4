import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
class View(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        # create widgets

        #textbox for data
        self.T = tk.Text(self, height=10, width=52)
        self.T.grid(row=1,column=0)
        self.T.configure(state='disabled')

        #tabs for graphs
        self.tabs = ttk.Notebook(self)
        self.tab1 = ttk.Frame(self.tabs)
        self.tab2 = ttk.Frame(self.tabs)
        self.tab3 = ttk.Frame(self.tabs)
        self.tab4 = ttk.Frame(self.tabs)
        self.tab5 = ttk.Frame(self.tabs)
        self.tab6 = ttk.Frame(self.tabs)
        self.tab7 = ttk.Frame(self.tabs)
        #name tabs
        self.tabs.add(self.tab1, text='Waveform')
        self.tabs.add(self.tab2, text='Spectrogram')
        self.tabs.add(self.tab3, text='FFT')
        self.tabs.add(self.tab4, text='RT60-Low')
        self.tabs.add(self.tab5, text='RT60-Mid')
        self.tabs.add(self.tab6, text='RT60-High')
        self.tabs.add(self.tab7, text='RT60-Combined')

        #locate tabs
        self.tabs.grid(row=2, column = 0)

        self.data = ttk.Label(self, text='Key Data:')
        self.data.grid(row=0, column=0, sticky=tk.W)

        # label
        self.label = ttk.Label(self, text='File name:')
        self.label.grid(row=0, column=0, sticky=tk.W)

        #file entry
        self.file_var = tk.StringVar()
        self.file_entry = ttk.Entry(self, textvariable=self.file_var, width=30)
        self.file_entry.grid(row=0, column=0, sticky=tk.W, padx=65)

        # laf button
        self.laf_button = ttk.Button(self, text='Load Audio File', command=self.laf_button_clicked)
        self.laf_button.grid(row=0, column=0, sticky=tk.E)

        # message
        self.message_label = ttk.Label(self, text='', foreground='red')
        self.message_label.grid(row=5, column=0, sticky=tk.S)

        # set the controller
        self.controller = None

    def set_controller(self, controller):
        """
        Set the controller
        :param controller:
        :return:
        """
        self.controller = controller

    def laf_button_clicked(self):
        if self.controller:
            self.controller.laf(self.file_var.get())

    def show_error(self, message):
        self.message_label['text'] = message
        self.message_label['foreground'] = 'red'
        self.message_label.after(3000, self.hide_message)
        self.file_entry['foreground'] = 'red'

    def show_success(self, message):
        self.message_label['text'] = message
        self.message_label['foreground'] = 'green'
        self.message_label.after(10000, self.hide_message)

        # reset the form
        self.file_entry['foreground'] = 'black'
        self.file_var.set('')

    def hide_message(self):
        self.message_label['text'] = ''
