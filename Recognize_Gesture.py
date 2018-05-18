
import cv2
import numpy as np
from keras.models import load_model

#Loading the model that was saved while training
model = load_model('model.h5')


#Initilizing dictionary 
gesture = {
    0: "1",
    1: "2",
    2: "3",
    3: "4"
}

#Predict function which get image as parameters and predict it's label and return the label
def predict(hand):
    img = cv2.resize(hand, (50,50) )
    img = np.array(img)
    img = img.reshape( (1,50,50,1) )
    img = img/255.0
    res = model.predict( img )
    max_ind = res.argmax()
    return gesture[ max_ind ]


#Creating Window named image using Opencv
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
#Creating Window named hand using Opencv
cv2.namedWindow("hand", cv2.WINDOW_NORMAL)

#Setting image capturing device id using Opencv
vc = cv2.VideoCapture(0)
#Reading frames from live video stream
rval, frame = vc.read()

#Setting dimension of box in window
image_x = 350
image_y = 125
image_w = 200
image_h = 200

old_txt_pred = ""
txt_pred = ""
count_frames = 0
tot_string = ""

# A loop where each captured image is applied number of functions like:
#1)Converting image into Grayscale
#2)Applying Gaussian Blur and Median blurr for reducing noise and image details
#3)Applying Threshold to extract image from background
#4)And passing image frames to predict function to predict its label
#5)Creating a blackboard to print the image predicted label on it

while True:
    
    if frame is not None: 
        
        frame = cv2.flip(frame, 1)
        frame = cv2.resize( frame, (640,480) )
        
        cv2.rectangle(frame, (image_x,image_y), (image_x + image_w,image_y + image_h), (0,255,0), 2)
        
        hand = frame[image_y:image_y+image_h, image_x:image_x+image_w]
        hand = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(hand, (11,11), 0)
        blur = cv2.medianBlur(blur, 15)
        thresh = cv2.threshold(blur,210,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        thresh = cv2.bitwise_not(thresh)
        
        old_txt_pred = txt_pred
        
        txt_pred = predict(thresh)
        
        if old_txt_pred == txt_pred:
            count_frames += 1
        else:
            count_frames = 0
        
        
        blackboard = np.zeros(frame.shape, dtype=np.uint8)
        cv2.putText(blackboard, "Text Prediction - ", (30, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
        if count_frames > 20 and txt_pred != "":
            tot_string += txt_pred + " "
            count_frames = 0
            
        cv2.putText(blackboard, tot_string, (30, 80), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 127))
        res = np.hstack((frame, blackboard))
        
        cv2.imshow("image", res)
        cv2.imshow("hand", thresh)
        
    rval, frame = vc.read()
    keypress = cv2.waitKey(1)
    
    if keypress == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

