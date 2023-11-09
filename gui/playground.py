from tkinter import ttk
from tkinter import *
import time

root = Tk()
root.title("window")
root.geometry("600x400")

def step():
    my_progress['value'] = 20
    my_progress.start(10)

my_progress = ttk.Progressbar(
    root, 
    orient=HORIZONTAL, 
    length=200, 
    mode='determinate'
)
my_progress.pack(pady=20)

my_button = Button(root, text="Progress", command=step)
my_button.pack(pady=20)

root.mainloop()