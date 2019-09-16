import cv2
import pandas as pd
import os


facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
v_capture = cv2.VideoCapture(0)

name = input("Name:",)

path = os.getcwd()
path = path+'\\dataset\\'+name
print(path)

collect_data = False

#Create a Directory
try:  
    os.mkdir(path)
except OSError:  
    print ("Creation of the directory %s failed" % path)
else:  
    print ("Successfully created the directory %s " % path)

count = 0

while True:
    ref_point = []
    ret, frame = v_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Detect Face
    faces = facecascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    #print(faces)
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 1)
        ref_point = [(x, y), (x+w, y+h)]    
   
#    if cv2.waitKey(1) & 0xFF == ord('c'):
#        collect_data = True
    if len(ref_point) == 2:
        image_name = "face_frame{}.png".format(count)
        #image = cv2.imread(ret ,frame)
        clone = gray.copy()
        crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
        cv2.imwrite(os.path.join(path , image_name), crop_img)
    
    cv2.imshow('Collecting Data...', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    count+=1
# When everything is done, release the capture
v_capture.release()
cv2.destroyAllWindows()