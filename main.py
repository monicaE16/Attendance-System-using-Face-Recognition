from face_detection import *
from face_recognition import *
import joblib
from matplotlib import cm
import json
import numpy as np
import time

scales=[(2,3),(3,4)]
resize=(150,150)

def prepare_Data():
    print("before loading")
    f= open("pca.json","r")
    print("openning")
    data = json.load(f)
    print("loading")
    f.close()
    print("terminate")
    mu=np.array(data['mu'])
    eigenvectors=np.array(data['eigenvectors'])
    
    [X, y] = read_images(image_path='Dataset')
    X_rows = as_row_matrix(X)
    projections = np.dot(X_rows - mu, eigenvectors)
    return projections,y,eigenvectors,mu
    


     
        
        
def start_Reco(frame,projections,y,eigenvectors,mu):
     
    
        start = time.time()
        partitions = []
        coordinates=[]
        # Capture the video frame
        # by frame
        print(frame.shape)
        rescaling_factor=(frame.shape[0]/resize[0],frame.shape[1]/resize[1])
        print(rescaling_factor)
        coordinates=[]
        points = []
        
        #trying different scales
        for i,j in scales:
            
            #for detection
            new_partitions,new_coordinates=search_for_face(cv2.resize(rgb2gray(frame), resize, interpolation = cv2.INTER_AREA),i,j,8)
            coors = np.asarray(new_coordinates)
            points = non_max_suppression_fast(coors)
            
           
            for coordinate in new_coordinates:
                coordinates.append(coordinate)
                #print(coordinate)
            partitions = partitions + new_partitions
        i=0
        for partition in partitions:
            coordinate=coordinates[i]
            i=i+1
            image = Image.fromarray(np.uint8(cm.gist_earth(partition)*255))
            image = image.convert ("L")

            image = image.resize ((200,200), Image.ANTIALIAS )
            test_image = np. asarray (image , dtype =np. uint8 )
            predicted = predict (eigenvectors, mu , projections, y, test_image)
            print("my prediction",y[predicted])
        end = time.time()
        print("time taken",end-start)
        return y[predicted],points,rescaling_factor
         
    






















