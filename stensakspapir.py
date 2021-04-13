import numpy as np
import cv2
import time
import tensorflow.keras 
from PIL import Image, ImageOps
import random

cap = cv2.VideoCapture(0)
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)
# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')
# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


def make_move():
    timeStart = time.time()
    timePassed = time.time()
    
    # capture video and count down for 5 seconds while showing the count
    while(timePassed - timeStart < 3):
        count = str(int(5 - (timePassed-timeStart)))
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Display the resulting frame
        font = cv2.FONT_HERSHEY_SIMPLEX  
        cv2.putText(frame, count,(10,500),font,4,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow('frame',frame)
        timePassed=time.time()
        #print (count)
        image = frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    print(image.shape)
    size = (224, 224)
    #image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image = ImageOps.fit(Image.fromarray(image), size, Image.ANTIALIAS)

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
   
    moves= ['Nothing','Stone','Paper','Scissors'] 
    pred = list(prediction[0])
    
    biggest = pred.index(max(pred))
    #predindex = pred.index(biggest)
    user_move = moves[biggest]
    
    
    #user_move = 'Paper'
    print(user_move)
    if (user_move == 'Nothing'):
        make_move()
    else:
        Result(user_move,Computer_makemove())
    

def Computer_makemove():
   computer_move = random.choice(('Stone', 'Paper', 'Scissors'))
   print(computer_move)
   return (computer_move)

def Result(user_move, computer_move):
  
  if (user_move == 'Stone' and computer_move == 'Paper') or (user_move == 'Scissors' and computer_move == 'Stone') or (user_move == 'Paper' and computer_move == 'Scissors'): 
        winner = "Computer"
        print ('The winner is: ' + winner) 
         
  elif (user_move == 'Stone' and computer_move == 'Scissors') or (user_move == 'Scissors' and computer_move == 'Paper') or (user_move == 'Paper' and computer_move == 'Stone'): 
        winner = "User"
        print ('The winner is: ' + winner)
        
  else:
    winner = "It is a TIE"
    print(winner)
     


make_move()
