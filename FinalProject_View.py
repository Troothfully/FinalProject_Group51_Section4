import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd



class View(ttk.Frame): # dont know what to call this :(
    def __init__(self, variable):
        super().__init__(variable)


        self.label = ttk.Label(self, text = 'Sound file GUI')
        self.label.grid(row = 1, column = 1)

        #add the button to select the file (file button
        self.Fbutton = ttk.Button(self, text = 'Open a file', command= select_file())
        self.Fbutton.grid(row = 2, column = 1)

        #display the time
        #call a function to dispaly the time
        self.time = ttk.Label(self, text = time_of())
        self.time.grid(row = 2, column = 1)
        #display a chart here
        #dont know exactly how many charts but this can be duplicated
        self.charts = ttk.mainframe(self, listvariable = get_chart(), height = 6, width = 30)
        self.charts.grid(row=3, column=1)

        self.Morecharts = ttk.mainframe(self, listvariable=get_chart(), height=6, width=30)
        self.Morecharts.grid(row=4, column=1)


        #place holders
        self.changeButton = ttk.Button(self, text = 'High Med or Low', command= get_chart())
        self.changeButton.grid(row = 5, column = 1)




#this can be formated easily

    def set_controller(self, controller):
        self.controller = controller






def select_file():
    filetypes = (
        ('sound files', '*.wav'),
    )

    filename = fd.askopenfilename(
        title='Open a sound file',
        initialdir='/',
        filetypes=filetypes)

    gfile = filename





#place holder functions
def time_of():
    return 0


def get_chart():
    return 0
