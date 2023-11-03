import tkinter as tk

root = tk.Tk()
root.title("Flight Delay Predictor")
root.geometry("600x400")

flightNum = tk.Entry(root, width=50)
flightNum.pack()
flightNum.insert(0, "Enter Flight Number: ")

flightDate = tk.Entry(root, width=50)
flightDate.pack()
flightDate.insert(0, "Enter Date of Flight: ")

def myClick():
    hello = "Hello " + flightNum.get()
    myLabel = tk.Label(root, text=hello)
    myLabel.pack()

myButton = tk.Button(root, text="Submit", command=myClick)
myButton.pack()

root.mainloop()