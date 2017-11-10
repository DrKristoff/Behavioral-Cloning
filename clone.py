import csv
import cv2
import numpy as np
import sklearn

correction = 0.1
batch = 32
lines = []
with open('driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=batch):
    num_samples = len(samples)
    rolling_ave_list = []
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                if(len(batch_sample)==0):
                  continue
                  
                center_name = 'IMG/'+batch_sample[0].split('\\')[-1]
                center_image = cv2.imread(center_name)
                np.fliplr(center_image)
                
                left_name = 'IMG/'+batch_sample[1].split('\\')[-1]
                left_image = cv2.imread(center_name)
                np.fliplr(left_image)
                right_name = 'IMG/'+batch_sample[2].split('\\')[-1]
                right_image = cv2.imread(center_name)
                np.fliplr(right_image)
                
                center_angle = float(batch_sample[3])
                                                
                rolling_ave_list.append(center_angle)
                if(len(rolling_ave_list)>3):
                  rolling_ave_list.pop(0)
                  
                center_angle = sum(rolling_ave_list)/len(rolling_ave_list)
                left_angle = float(batch_sample[3])+correction
                right_angle = float(batch_sample[3])-correction
                
                images.extend([center_image, left_image, right_image])
                angles.extend([center_angle, left_angle, right_angle])

            augmented_images, aug_angles = [], []
            
            for image,angle in zip(images, angles):
              augmented_images.append(image)
              aug_angles.append(angle)
              augmented_images.append(np.fliplr(image))
              aug_angles.append(-angle)

            X_train = np.array(augmented_images)
            y_train = np.array(aug_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=batch)
validation_generator = generator(validation_samples, batch_size=batch)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Cropping2D

from keras.models import Model

print("setting up the model")
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((0,0), (70,25))))
model.add(Conv2D(24,(5,5),activation="relu", strides=(2,2)))
model.add(Conv2D(36,(5,5),activation="relu", strides=(2,2)))
model.add(Conv2D(48,(5,5),activation="relu", strides=(2,2)))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer = 'adam', metrics=['accuracy'])

from keras.models import load_model
model = load_model('solution.h5')
print(model.summary())


model.fit_generator(train_generator, steps_per_epoch= len(train_samples)/batch, validation_data=validation_generator, validation_steps=len(validation_samples)/batch, epochs=1, verbose=1)
            
model.save('model.h5')
