# Rock Paper Scissors Image Classification Project

This project demonstrates a simple image classification model to play the game of Rock, Paper, Scissors. The model is trained to recognize hand gestures representing rock, paper, and scissors using images captured from a webcam. The trained model then plays the game against the user.

## Requirements

To run this project, you'll need the following libraries and tools installed:

- Python
- Google Colab
- TensorFlow 2.x
- NumPy
- Matplotlib
- tqdm
- ffmpeg
- pycocotools
- colab_utils (for image and video processing)
- Keras (for image preprocessing and data augmentation)

You can install the required libraries using the following commands:
```bash
!pip install pycocotools 
!pip install ffmpeg

#Project Structure
capture_images: Function to capture images using the webcam.
Data Augmentation: Using Keras' ImageDataGenerator to augment the captured images.
Model Definition: Building a Convolutional Neural Network (CNN) using TensorFlow and Keras.
Training: Training the CNN on the augmented images.
Prediction: Using the trained model to predict the user's hand gesture and play Rock, Paper, Scissors.
How to Run
Mount Google Drive:

python
Copy code
from google.colab import drive
drive.mount('/content/drive')
Set Working Directory:

python
Copy code
import os
work_dir = "/content/drive/MyDrive/SKILLIT Courses/AI Level 2/Final Project"
os.chdir(work_dir)
Install Required Libraries:

bash
Copy code
!pip install pycocotools 
!pip install ffmpeg
Import Necessary Modules:

python
Copy code
from colab_utils import imshow, videoGrabber
import numpy as np
import matplotlib.pyplot as plt
from google.colab import output
%tensorflow_version 2.X
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tqdm.auto import tqdm
import time
import random
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
Capture and Label Images:

python
Copy code
paper_images, paper_label = capture_images(numImage=100, label=0)
rock_images, rock_label = capture_images(numImage=100, label=1)
scissor_images, scissor_label = capture_images(numImage=100, label=2)
train_images = np.concatenate((paper_images, rock_images, scissor_images))
train_images = train_images / 255
train_labels = np.concatenate((paper_label, rock_label, scissor_label))
Data Augmentation:

python
Copy code
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
# Train the Model:

python
Copy code
model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(40, 60, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.Flatten()) 
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3))
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(new_train_images, new_train_labels, epochs=10)
Make Predictions:

python
Copy code
test_image, _ = capture_images(1)
test_image = test_image / 255
prediction = model.predict(test_image)
plt.imshow(test_image[0])
plt.title(CLASS_NAME[np.argmax(prediction[0])])
model.save('self-made_model.h5')
Play the Game:

python
Copy code
output.clear()
print("****************************************************************")
print("Welcome to Rock Paper Scissor!")
print("****************************************************************")

for i in range(3):
    time.sleep(1)
    output.clear()
    print("**********************************************************************")
    print("Welcome to Rock Paper Scissor!")
    print("****************************************************************")
    print(3 - i)

output.clear()
print("ROCK - PAPER- SCISSOR--------- SHOOT!")

user_image, _ = capture_images(1)
user_image = user_image / 255
prediction = model.predict(user_image)
output.clear()
user_selection = CLASS_NAME[np.argmax(prediction[0])]
print(f'You have selected: {user_selection}')

random_int = random.randint(0, 2)
computer_selection = CLASS_NAME[random_int]
print(f'The Computer has selected: {computer_selection}')

if user_selection == 'Rock':
    if computer_selection == 'Rock':
        print("We both selected Rock... It's a TIE")
    if computer_selection == 'Paper':
        print("Paper beats Rock! COMPUTER WINS")
    if computer_selection == 'Scissor':
        print("Rock beats Scissor! YOU WIN")

if user_selection == 'Paper':
    if computer_selection == 'Rock':
        print("Paper beats Rock! You WIN")
    if computer_selection == 'Paper':
        print("We both selected Paper... It's a TIE")
    if computer_selection == 'Scissor':
        print("Scissor beats Paper! COMPUTER WINS")

if user_selection == 'Scissor':
    if computer_selection == 'Rock':
        print("Rock beats Scissor! COMPUTER WINS")
    if computer_selection == 'Paper':
        print("Scissor beats Paper! You WIN")
    if computer_selection == 'Scissor':
        print("We both selected Scissor... It's a TIE")


## Conclusion
This project demonstrates how to build a simple image classification model using TensorFlow and Keras, and use it to play the game of Rock, Paper, Scissors with real-time image capture. The model is trained on custom-captured images and can predict hand gestures in real-time to play the game.

Feel free to explore and improve the model for better accuracy and performance!
