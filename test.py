import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

def Computer_makemove():
   computer_move = random.choice(('rock', 'paper', 'scissors'))
   print(computer_move)
   return (computer_move)

def Result(user_move, computer_move):
  
  if (user_move == 'rock' and computer_move == 'paper') or (user_move == 'scissors' and computer_move == 'rock') or (user_move == 'paper' and computer_move == 'scissors'): 
        winner = "Computer"
        print ('The winner is: ' + winner) 
         
  elif (user_move == 'rock' and computer_move == 'scissors') or (user_move == 'scissors' and computer_move == 'paper') or (user_move == 'paper' and computer_move == 'rock'): 
        winner = "User"
        print ('The winner is: ' + winner)
        
  else:
    winner = "It is a TIE"
    print(winner)
     
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open('Fotografi den 13.04.2021 kl. 13.58.jpg')

#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
image_array = np.asarray(image)

# display the resized image
image.show()

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)
print(prediction)

moves= ['nothing','rock', 'paper','scissors']
max = 0
for x in range (0, len(prediction[0])):
    if (((prediction[0])[max]) < ((prediction[0])[x])):
        max = ((prediction[0])[x]) 
user_move = moves[max]
if (user_move == 'nothing'):
     make_move()
else:
    Result(user_move,Computer_makemove)
    

