from face_detection import *
from face_recognition import *
import joblib
from matplotlib import cm

filename = 'finalized_model.sav'
finalized_model_1 = joblib.load(filename)

[X, y] = read_images() 
X_rows = as_row_matrix(X)
[eigenvalues, eigenvectors, mu] = pca (as_row_matrix(X), y, 15)
print(y)

projections = np.dot (X_rows - mu , eigenvectors)
# define a video capture object
vid = cv2.VideoCapture(0)
#(1.1,1.7),(1.5,3),(2,4),
scales=[(1.1,1.7),(2,3),(3,4),(4,6)]
while(True):
    partitions = []  
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    cv2.imshow('frame', frame)
    for i,j in scales:
        partitions = partitions + search_for_face(cv2.resize(rgb2gray(frame), (150,150), interpolation = cv2.INTER_AREA),i,j,8)
    for partition in partitions:
        image = Image.fromarray(np.uint8(cm.gist_earth(partition)*255))
        image = image.convert ("L")

        image = image.resize ((250,250), Image.ANTIALIAS )
        test_image = np. asarray (image , dtype =np. uint8 )
        predicted = predict (eigenvectors, mu , projections, y, test_image)
        print(y[predicted])
    # Display the resulting frame
    image = Image.open("test/ranime.10.jpeg")

    
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
