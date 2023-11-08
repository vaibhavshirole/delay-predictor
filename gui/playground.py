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
    # display user inputs on gui
    # flight_number = flight_number_entry.get()
    # user_flight_input = "Selected flight: " + flight_number
    # chosen_flight = tk.Label(root, text=user_flight_input)
    # chosen_flight.grid(row=3,column=0)

    # flight_date = date_entry.get()
    # user_date_input = "Selected flight date: " + flight_date
    # chosen_date = tk.Label(root, text=user_date_input)
    # chosen_date.grid(row=3,column=1)

    #display prediction on gui
    global estimated_delay
    if(flight_number_entry.get()):
        estimated_delay.grid_remove()
        
        # TO DO: put loading gif

        delay = "Delay: " + str(dp.predict(flight_number_entry.get()))

        # TO DO: remove loading gif

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

# creating label and date entry for flight_date
# date_label = tk.Label(root, text = 'Enter Flight Date: ')
# date_entry = tk.Entry(root, textvariable = flight_date)
# date_label.grid(row=1, column=0, padx=20, pady=10, sticky="w")
# date_entry.grid(row=1, column=1, padx=50, pady=10, sticky='e')

# creating a button to call submit func
sub_btn=tk.Button(root,text = 'ESTIMATE', command = submit, font=('calibre',14, 'bold'))
sub_btn.grid(row=2, column=1, padx=20, pady=10)

root.mainloop()