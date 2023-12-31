from FinalProject_Model import Model
from FinalProject_View import View
from FinalProject_Controller import Controller
import tkinter as tk

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title('Final Project COP2080')

        # create a model
        model = Model('test.wav')

        # create a view and place it on the root window
        view = View(self)
        view.grid(row=0, column=0, padx=10, pady=10)

        # create a controller
        controller = Controller(model, view)

        # set the controller to view
        view.set_controller(controller)


if __name__ == '__main__':
    app = App()
    app.geometry('525x650')
    app.mainloop()
