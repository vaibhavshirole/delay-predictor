import tkinter as tk
#import delay_predict as dp

root = tk.Tk()
root.title("Flight Delay Predictor")
root.geometry("500x300")

# entry data
flight_number = tk.StringVar()
flight_date = tk.StringVar() 

# init output var
estimated_delay = tk.Label(root, text="", font=('calibre',14, 'bold'))

def submit():
    global estimated_delay

    #display prediction on gui
    if(flight_number_entry.get()):
        estimated_delay.grid_remove()
        
        # TO DO: put loading bar

        # TO DO: run prediction in separate thread
        #delay = "Delay: " + str(dp.predict(flight_number_entry.get()))
        delay = flight_number_entry.get()

        # TO DO: remove loading bar

        estimated_delay = tk.Label(root, text=delay, font=('calibre',14, 'bold'))
        estimated_delay.grid(row=4, column=0, padx=20, pady=10, sticky="w")
        
        arrival = "Arrival time: 19:35"
        estimated_arrival = tk.Label(root, text=arrival, font=('calibre',14, 'bold'))
        estimated_arrival.grid(row=5, column=0, padx=20, pady=10, sticky="w")
    else:
        delay = "Input valid flight number"
        estimated_delay = tk.Label(root, text=delay, font=('calibre',14, 'bold'))
        estimated_delay.grid(row=4, column=1, padx=20, pady=10, sticky="w")

# creating label and text entry for flight_number
flight_number_label = tk.Label(root, text = 'Enter Flight Number: ')
flight_number_entry = tk.Entry(root, textvariable = flight_number)
flight_number_label.grid(row=0, column=0, padx=20, pady=10, sticky="w")
flight_number_entry.grid(row=0, column=1, padx=50, pady=10, sticky="e")

# creating a button to call submit func
sub_btn=tk.Button(root,text = 'ESTIMATE', command = submit, font=('calibre',14, 'bold'))
sub_btn.grid(row=2, column=1, padx=20, pady=10)

root.mainloop()