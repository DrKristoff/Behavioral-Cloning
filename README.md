# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Introduction
The purpose of this project was to introduce principle of deep learning by training a convolutional neural network to train a car to navigate around a simulated track.  Udacity provided an application developed with the Unity game engine as the simulator.  It was developed using TensorFlow and Keras as a framework.  

## Simulator
The Udacity simulator provides two modes, a training mode and an autonomous mode.  In training mode the simulator records driving data based on the manual inputs of the user.  This data includes steering angle, throttle, brake, and speed aligned with visual data from three cameras located on the front left, front center, and front right of the car.  The user controls the car with the arrow keys or WASD, and can also steer using a click-and-drag with the mouse to control precise steering angle.  

<img src="https://github.com/DrKristoff/Behavioral-Cloning/blob/master/examples/Udacity_Simulator.JPG?raw=true" width="400px" alt="Udacity Simulator">
Udacity Simulator

<img src="https://github.com/DrKristoff/Behavioral-Cloning/blob/master/examples/Simulator_Training _View.JPG?raw=true" width="400px" alt="Simulator Training View">
Simulator Training View

## Model Architecture
The architecture I used was based on the NVIDIA model.  I doubled the stride length of some of the early layers to reduce the parameter count.  Early on in the development phase I noticed that the accuracy on the training data was very high, but actual performance was very poor.  It was apparent that the powerful architecture was suffering from overfitting to the training data.  To combat this, I experimented with some pooling layers, sub-sampling, and different levels of dropout.  

<img src="https://github.com/DrKristoff/Behavioral-Cloning/blob/master/examples/NVIDIA.jpg?raw=true" width="400px" alt="NVIDIA architecture">
NVIDIA architecture

The architecture that resulted was the following:

Layer (type)                 Output Shape             		Param count

lambda_1 (Lambda)            (None, 160, 320, 3)       		0

cropping2d_1 (Cropping2D)    (None, 160, 225, 3)       		0

conv2d_1 (Conv2D)            (None, 78, 111, 24)      		1824

dropout_1 (Dropout)          (None, 78, 111, 24)       		0

conv2d_2 (Conv2D)            (None, 37, 54, 36)        		21636

conv2d_3 (Conv2D)            (None, 17, 25, 48)        		43248

dropout_2 (Dropout)          (None, 17, 25, 48)        		0

conv2d_4 (Conv2D)            (None, 8, 12, 64)         		27712

conv2d_5 (Conv2D)        	   (None, 6, 10, 64)         		36928

flatten_1 (Flatten)          (None, 3840)              	  0

dense_1 (Dense)              (None, 100)              		384100

dropout_3 (Dropout)          (None, 100)               		0

dense_2 (Dense)              (None, 50)                		5050

dropout_4 (Dropout)          (None, 50)                		0

dense_3 (Dense)              (None, 10)                		510

dense_4 (Dense)              (None, 1)                    11

Total params: 521,019
Trainable params: 521,019
Non-trainable params: 0

## Python Generator
Since I was using tens of thousands of images for the training and validation, memory management quickly became an issue.  I used the Python generator pattern to more efficiently handle such large amounts of images.   The generator pattern allows you to create a function that acts like an iterator.  This allowed me to shuffle and batch portions of the images directory without requiring to load all of them into memory, only what was immediately needed for the current batch operation.  

## Training
To train the model I had to record myself driving around the simulated track manually.  The recording process involved choosing a directory to save images and then pressing the “R” key to start and stop recording.  I also used the mouse rather than the keyboard for steering, that way I would record a much higher level of steering resolution.  For the overall strategy, I took a few different strategies to add to the dataset.  I recorded two full laps of center driving, which involved trying to stay in the very center of the track as much as possible.  I also recorded a few recovery laps, which were laps where I purposely drove to the edge of the track, hit record, and then recovered to the center of the track.

<img src="https://github.com/DrKristoff/Behavioral-Cloning/blob/master/examples/left_2017_11_11_07_47_08_177.jpg?raw=true" width="133px">
Example Left Image
<img src="https://github.com/DrKristoff/Behavioral-Cloning/blob/master/examples/center_2017_11_11_07_47_08_177.jpg?raw=true" width="133px">
Example Center Image
<img src="https://github.com/DrKristoff/Behavioral-Cloning/blob/master/examples/right_2017_11_11_07_47_08_177.jpg?raw=true" width="133px">
Example Right Image

angle = -0.1886793	  throttle = 0	 brake = 0	 speed = 6.361172

Of all the steps of the development process, the training and data collection seemed to have the greatest impact on success.  An important lesson I learned is that your neural network’s performance is very dependent on the data that you feed into it.  I spent a significant amount of time working with training data, both on data collection strategies and data preparation and feature engineering.  

## Data Processing
To split the data into test and training data, I used sklearn function train_test_split with a ratio of 80/20, returning 80% of the data as training data and 20% as validation data.  
To augment the data I collected through training, I used numpy to flip the images horizontally, while reporting the inverse steering angle.  The result of this is effectively a duplication of driving training distance, as well as providing left/right turn data symmetry.   I also found that the dataset was filled with many steering angles of 0, which introduced a lot of error into the overall model, and caused severe understeer on the corners.  I initially added a check during preprocessing to eliminate all images and data where the steering angle was 0, which made the car drive significantly better around corners.  Conversely, its performance on straight sections tanked.  It was lost and had low confidence on optimal steering angle.  I experimented with a rolling average to eliminate the zeros, and ultimately landed on eliminating a random 50% of the data.  Not a very elegant solution, but it worked for this project1!

## Running the Files
To begin training the neural network run the following command:

python clone.py

As the neural network is trained on the images, you get status updates on the fly like the following line:

22/235 [=>............................] - ETA: 2:58 - loss: 0.0792

The 22 indicates the current step of the epoch and 235 indicates the number of steps in each epoch.  

After the model finishes training, the file saves the model file in the local directory as an h5 file.  You can then run the file from the command line using the following command:

Python drive.py model.h5

This create a Socket IO server that waits for the simulator to be running and in autonomous mode.  Once that is detected, it sends the throttle and steering angle predictions from the Keras model that was saved.  

To create the video of the model running, I added an extra argument to the command in the command line.  

Python drive.py model.h5 run1

I then ran the provided python file video.py to create the mp4 file.  


