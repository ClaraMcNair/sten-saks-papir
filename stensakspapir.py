import numpy as np
import cv2
import time
import tensorflow.keras 
from PIL import Image, ImageOps
import random


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)
# Load the model
model = tensorflow.keras.models.load_model('keras_model2.h5')
# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

#method for the user to make a move
def make_move():
    cap = cv2.VideoCapture(0)
    timeStart = time.time()
    timePassed = time.time()
    
    # capture video and count down for 5 seconds while showing the count
    while(timePassed - timeStart < 5):
        count = str(int(5 - (timePassed-timeStart)))
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)
        # Display the resulting frame
        font = cv2.FONT_HERSHEY_SIMPLEX  
        cv2.putText(frame, count,(10,500),font,4,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow('Rock, paper, scissor',frame)
        timePassed=time.time()
        
        image = frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    
    size = (224, 224)
    image = ImageOps.fit(Image.fromarray(image), size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)
   
    moves= ['Nothing','Rock','Scissors','Paper'] #String-list with the moves' names placed at the same index, as in the model
    pred = list(prediction[0]) #list of all the values of the prediction
    
    biggest = pred.index(max(pred)) #find the index of largest value
    
    user_move = moves[biggest]#cast the users move into a string, by finding the item at the same index in our String-list of moves  
   
    print(user_move)
    if (user_move == 'Nothing'): #try again if no move is detected
        cv2.putText(frame, "No move detected!",(10,150),font,2,(255,0,0),2,cv2.LINE_AA)
        cv2.putText(frame, "Press key 0 to try again!",(10,200),font,2,(255,0,0),2,cv2.LINE_AA)
        cv2.imshow('Rock, paper, scissor',frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        make_move()
    else:
        Result(user_move,Computer_makemove(),frame) #if a move is detected, run the Result()-method with the user-move and a call to Computer_makemove as its arguments.
    
#method for the computer to make a random move
def Computer_makemove():
   computer_move = random.choice(('Rock', 'Paper', 'Scissors'))
   print(computer_move)
   return (computer_move)

#method for getting the result
def Result(user_move, computer_move, frame):
    winner = ''
#all possible outcomes where the computer wins
    if (user_move == 'Rock' and computer_move == 'Paper') or (user_move == 'Scissors' and computer_move == 'Rock') or (user_move == 'Paper' and computer_move == 'Scissors'): 
            winner = "the Computer won!"
            
        #all possible outcomes where the user wins     
    elif (user_move == 'Rock' and computer_move == 'Scissors') or (user_move == 'Scissors' and computer_move == 'Paper') or (user_move == 'Paper' and computer_move == 'Rock'): 
            winner = "You won!"
            
    #if neither of those otcomes occur, the game is tied.        
    else:
        winner = "It is a TIE"
        
    font = cv2.FONT_HERSHEY_SIMPLEX  
    cv2.putText(frame, "Computers move: " + computer_move,(10,100),font,2,(255,0,0),2,cv2.LINE_AA)
    cv2.putText(frame, "Your move: " + user_move,(10,150),font,2,(255,0,0),2,cv2.LINE_AA)
    cv2.putText(frame, winner,(10,280),font,4,(255,0,0),2,cv2.LINE_AA)
    cv2.putText(frame, "Play again press key A ",(10,650),font,1,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame, "To quit press key Q ",(10,700),font,1,(0,0,255),2,cv2.LINE_AA)
    cv2.imshow('Rock, paper, scissor', frame)

    k = cv2.waitKey(0)
    if k == ord('a'):
        cv2.destroyAllWindows()
        make_move()
    elif k == ord('q'):
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()
    
        
make_move()

