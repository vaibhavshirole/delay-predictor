import tkinter as tk

root = tk.Tk()
root.title("Flight Delay Predictor")
root.geometry("600x400")

num_var = tk.StringVar()
date_var = tk.StringVar() 

def submit():

    flight_number = num_entry.get()
    hello = "Selected flight: " + flight_number

    output = tk.Label(root, text=hello)
    output.grid(row=2,column=2)

# creating label and text entry for flight_number
num_label = tk.Label(root, text = 'Enter Flight Number: ')
num_entry = tk.Entry(root, textvariable = num_var)

# creating label and date entry for flight_date
date_label = tk.Label(root, text = 'Enter Flight Date: ')
date_entry = tk.Entry(root, textvariable = date_var)

# creating a button to call submit func
sub_btn=tk.Button(root,text = 'ESTIMATE', command = submit, font=('calibre',14, 'bold'))

# placing initial labels and entry boxes on grid
num_label.grid(row=0, column=0, padx = 5, pady = 10)
num_entry.grid(row=0, column=1, padx = 5, pady = 10)
date_label.grid(row=1, column=0, padx = 20, pady = 10)
date_entry.grid(row=1, column=1, padx = 20, pady = 10)
sub_btn.grid(row=2, column=1, padx = 20, pady = 10)

root.mainloop()