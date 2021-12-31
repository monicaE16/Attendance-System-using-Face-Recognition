from face_detection import *
import joblib
filename = 'finalized_model.sav'
finalized_model_1 = joblib.load(filename)

# define a video capture object
vid = cv2.VideoCapture(0)
scales = [2, 3, 4]
while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    for scale in scales:
        search_for_face_cv(frame, scale)
    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()