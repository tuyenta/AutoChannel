# SOME GUIDELINES FOR COLLECTING DATA WITH WIRELESS INSITE

## RECEIVER
There are many way to set up Receiver Set​:

Moving point, Route, and Trajectory require left-click for inputting segments and right-click for ending. But
- Moving point:​ allow us to select time spacing between adjacent position and total simulation time.
The number of positions of receiver is automatically generated (evenly for each segment but not for
all segments) and can be compute by total simulation time/time spacing.
- Route: ​enables us to select spacing sample in metter and the moving velocity.
- Trajectory​ is similar to Route except that A spline is fit to the control points and then individual
transmitter or receiver points are located along the spline according to the spacing provided by the
user in the set’s property window.


In summary, we choice “Route”​ mode because it is easy for us to control.
- Step A.1:​ select segments by left-clicking. Note that:
    - The total length of the route should be larger than 1024*wavelength​ so that we can have more than 2048 samples.
    - The segment should not cross​ the building. If not, the received powers at the hidden point would be very bad and this makes difficulties for training our future CNNs.
    - The minimum distance to transmitter should be larger than 15m so that the effect of fading is clear

- Step A.2: ​Go to Tx/Rx LAYOUT PROPERTIES
    - Select:

- Step A.3:​ Go to Receiver PROPERTIES and configure

## TRANSMITTER
It is possible to add many transmitters/receiver routes and running at the same time. But it is
time-consuming and a little bit difficult for getting out the results. So at first: 1 Tx and 1 Rx route​.
Steps for setting up transmitter:
- Step B.1:​ Select Points​ mode and left click one point for Tx (then right click for ending).
    - Note that, for simple, at first we select the position of Tx outside of buildings
- Step B.2​: Similar to step A.2 and A.3 (Right-click on Tx instead). Go to TRANSMITTER PROPERTIES and LAYOUT PROPERTIES to configure Tx

## STUDY AREA
Should not be too large but it should cover all the Rx Route

## OUTPUT AND SAVING RESULT

Please sure that you have at least checked “Propagation paths​” and “Received Power​” for Output
Requests before running.
- After finishing the calculation, please go to the working directory of the project (saving place). You
will see a folder with the name of the Study Area that you have defined.
- Please save at least two files: “*power*.p2m​” and “*paths*.p2m​”

- In case that you have available hard drive, additionally saving “*.sqlite​” file will be a very good option for future use.