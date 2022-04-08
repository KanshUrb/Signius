import uuid
import cv2

"""short script created to collect photos easily"""
cap = cv2.VideoCapture(0) #if video capturing is not working, change the value for 1/2/3 etc. basically it should work with id 0 which is default id for camera
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

"""In an infinite loop the program processes the image from the camera, under the variable 'frame' there is the current frame from the camera," \
" to take a picture press "q". In the variable 'imgname' enter the path to the folder where you want to save your pictures"""

while True:
    ret, frame = cap.read()
    """We are using uuid library, to make for each image unique identifier"""
    imgname = '/home/kansh_dev/PycharmProjects/object_decetion/Images/Test.{}.jpg'.format(str(uuid.uuid1()))
    cv2.imwrite(imgname, frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
