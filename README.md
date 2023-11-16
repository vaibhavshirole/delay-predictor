# delay-predictor

## Interface: 
<p align="center">
  <img src="/gui/interface.png" width="75%" alt="Preview">
</p>

## How to use: 
* navigate to /releases
* download latest release
* set python 3.11 as environment
* install the following libraries with pip3.11
~~~
pip install requests
pip install pandas
pip install json
pip install numpy
pip install matplotlib
pip install tensorflow
pip install scikit-learn
pip install tkinter
~~~
* run app.py
~~~
python app.py
~~~
* get flight number to test from FlightAware - [linked arrivals to LHR](https://www.flightaware.com/live/airport/EGLL)
* input flight number without spaces
* wait ~90 seconds for result
* validate result with what FlightAware says

## Results: 
* RMSE = 0.707
* Avg Accuracy = 80.80%
<p align="center">
  <img src="/gui/results.png" width="75%" alt="Preview">
</p>
