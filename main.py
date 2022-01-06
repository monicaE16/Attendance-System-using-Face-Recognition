from face_detection import *
from face_recognition import *
import joblib
from matplotlib import cm
import json
import numpy as np
import time


filename = 'finalized_model.sav'
finalized_model_1 = joblib.load(filename)

f= open("pca.json","r")
data = json.load(f)
print(data)
f.close()
mu=np.array(data['mu'])
eigenvectors=np.array(data['eigenvectors'])

[X, y] = read_images(image_path='DataBase')
X_rows = as_row_matrix(X)
projections = np.dot(X_rows - mu, eigenvectors)

vid = cv2.VideoCapture(0)
#(1.1,1.7),(1.5,3),(2,4),
scales=[(2,3),(3,4)]
resize=(150,150)
rate=30
counter=-1
while(True):
    start = time.time()
    partitions = []
    coordinates=[]
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    print(frame.shape)
    rescaling_factor=(frame.shape[0]/resize[0],frame.shape[1]/resize[1])
    print(rescaling_factor)
    coordinates=[]
    counter+=1
    for i,j in scales:
        new_partitions,new_coordinates=search_for_face(cv2.resize(rgb2gray(frame), resize, interpolation = cv2.INTER_AREA),i,j,8)

        for coordinate in new_coordinates:
            coordinates.append(coordinate)
            print(coordinate)
            cv2.rectangle(frame,(round(coordinate[1]*rescaling_factor[1]),round(coordinate[0]*rescaling_factor[0])),(round(coordinate[3]*rescaling_factor[1]),round(coordinate[2]*rescaling_factor[0])),color=(0,255,0),thickness=2)
        partitions = partitions + new_partitions
    i=0
    for partition in partitions:
        coordinate=coordinates[i]
        i=i+1
        image = Image.fromarray(np.uint8(cm.gist_earth(partition)*255))
        image = image.convert ("L")

        image = image.resize ((250,250), Image.ANTIALIAS )
        test_image = np. asarray (image , dtype =np. uint8 )
        predicted = predict (eigenvectors, mu , projections, y, test_image)
        cv2.putText(img=frame, text=y[predicted], org=((round(coordinate[1]*rescaling_factor[1])), round(coordinate[0]*rescaling_factor[0])), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 0),thickness=2)
        print(y[predicted])
    print("hello")
    end = time.time()
    print(end - start)
    # Display the resulting frame
    #image = Image.open("test/ranime.10.jpeg")

    cv2.imshow('frame', frame)
    # test_image = np. asarray (image , dtype =np. uint8 )
    # predicted = predict (eigenvectors, mu , projections, y, test_imageq)
    # print(y[predicted])
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
