import tkinter as tk
import threading
import delay_predict as dp

root = tk.Tk()
root.title("Flight Delay Predictor")
root.geometry("500x300")


def predictor(flight_number):
    global delay_label # label
    global delay       # value
    label_text = ""

    delay_label.config(text="Working...")

    # give estimated delay
    delay = dp.predict(flight_number)
    if(delay < 0):
        label_text += "Early by " + str( round(abs(delay/60), 2) ) + " min"
    else:
        label_text += "Late by " + str( round(abs(delay/60), 2) ) + " min"
    delay_label.config(text=label_text)

    # give estimated time of arrival
    # arrival = "Arrival time: 19:35"
    # estimated_arrival = tk.Label(root, text=arrival, font=('calibre',14, 'bold'))
    # estimated_arrival.grid(row=5, column=0, padx=10, pady=10, sticky="w")


def submit():
    global delay_label # label
    global error_label # label
    global delay       # value

    #display prediction on gui
    if(flight_number_entry.get()):
        error_label.grid_remove()   # remove input error
        delay_label.grid() # reinstate label

        threading.Thread(target=predictor, daemon=True, args=(flight_number_entry.get(),)).start()
   
    else:
        delay_label.grid_remove()   # remove prediction
        error_label.grid() # reinstate label

        error_label.config(text="Input a valid flight number")


# output label : predicted delay
delay_label = tk.Label(root, text="", font=('calibre',14,'bold'))
delay_label.grid(row=4, column=0, padx=10, pady=10, sticky="w")
delay = " "

# output label : user error for flight number submission
error_label = tk.Label(root, text="", font=('calibre',14,'bold'))
error_label.grid(row=4, column=1, padx=10, pady=10, sticky="w")
error = " "

# entry box : flight number
flight_num = tk.StringVar() # entry data
flight_number_label = tk.Label(root, text = 'Enter Flight Number (ex. UAL1): ')
flight_number_entry = tk.Entry(root, textvariable = flight_num)

# entry box : flight date (unused!)
flight_date = tk.StringVar()   # entry data
flight_number_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
flight_number_entry.grid(row=0, column=1, padx=50, pady=10, sticky="e")

# button : submit user input data
sub_btn=tk.Button(root,text = 'ESTIMATE', command = submit, font=('calibre',14, 'bold'))
sub_btn.grid(row=2, column=1, padx=10, pady=10)


root.mainloop()