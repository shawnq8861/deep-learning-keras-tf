#
#	Load the pretrained OpenFace model
#	Build feature vector for stored image
#	Acquire live image from camera
#       Detect faces in acquired image
#       Crop ROI image from acquired image
#	Build feature vector for live image
#	Compare feature vectors using L2 norm
#	Determine if live face matches stored face
#

from load_open_face_model import load_model
import utils
import cv2 as cv
import os
import numpy as np
import time

window_name = "Shawn"
test_file = window_name + ".jpg"
match_threshold = 1.0

#
# define gstreamer launch command
#
def open_onboard_camera():   
    return cv.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw,format=(string)BGRx !                               videoconvert ! video/x-raw, format=(string)BGR ! appsink")

def generate_feature_vector(model, image):
    #
    # resize the image
    #
    image = cv.resize(image, (96, 96))
    #
    # convert to RGB
    #
    rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    #
    # convert to 4D array of 3D tensors for Keras
    # dim 1 is index of image tensor
    # dim 2 is height of image
    # dim 3 is width of image
    # dim 4 is number of color channels
    #
    np_img = np.array([rgb_img])
    #
    # generate embedding vector
    #
    embed_vec = model.predict_on_batch(np_img)
    
    return embed_vec

def main():
    #
    # initilaize cascade classifier for face detection
    #
    path = os.path.abspath(
            "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")
    face_cascade = cv.CascadeClassifier(path) 

    #
    print("loading model...\n")
    #
    model = load_model()
    print("model loaded, loading and setting weights...")
    #
    # Load weights from csv files
    #
    weights = utils.weights
    weights_dict = utils.load_weights()
    #
    # Set layer weights of the model
    #
    for name in weights:
        if model.get_layer(name) != None:
            model.get_layer(name).set_weights(weights_dict[name])
        elif model.get_layer(name) != None:
            model.get_layer(name).set_weights(weights_dict[name])
    print("weights set...")
    #
    # read reference image from file
    #
    img = cv.imread(test_file)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image=gray, scaleFactor=1.2, 
            minNeighbors=4, minSize=(180, 180))
    #
    # if we detected at least one face in the reference image
    #
    if len(faces) > 0:
        #
        # crop first image at bounding box corners
        #
        (x, y, width, height) = faces[0]
        crop_img = img[y:(y + height), x:(x + width)]
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.moveWindow(window_name, 500, 250)
        print("press space bar to continue...")
        cv.imshow(window_name, crop_img)
        key = cv.waitKey(-1)
        #
        # calculate feature vector
        #
        feature_vec = generate_feature_vector(model, crop_img)
        print("printing feature vector...")
        print("")
        print(feature_vec)
    #
    # start video capture
    # enter a loop
    # grab a live image
    # detect faces
    # crop
    # compute feature vector
    # calculate L2 norm(s)
    #
    video_capture = open_onboard_camera()
    if video_capture.isOpened():
        cropWindow = "Crop Image"
        while True:
            #
            # grab a frame from the stream
            #
            ret_val, frame = video_capture.read();
            if frame is None:
                #
                # null frame, keep looping until a frame is read
                #
                continue
            #
            # detect faces
            #
            img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.2, 
                    minNeighbors=4, minSize=(180, 180))
            #
            # if we detected at least one face in the reference image
            #
            if len(faces) > 0:
                #
                # crop first image at bounding box corners
                #
                (x, y, width, height) = faces[0]
                top_left = (x, y)
                bottom = y + height
                right = x + width
                bottom_right = (right, bottom)
                small_img = frame[y:bottom, x:right]
                #
                # calculate feature vector
                #
                live_feature_vec = generate_feature_vector(model, small_img)
                delta_feature_vec = feature_vec - live_feature_vec
                print("printing delta feature vector...")
                print("")
                print(delta_feature_vec)
                #
                # calculate L2 norm
                #
                l2_norm = np.linalg.norm(delta_feature_vec, ord=2)
                print("L2 Norm...")
                print(l2_norm)
                rows, cols, colors = small_img.shape
                print("rows: {0}".format(rows))
                print("cols: {0}".format(cols))
                text_x = int(cols/2) - 20
                print("text_x: {0}".format(text_x))
                text_y = int(rows - 10)
                print("text_y: {0}".format(text_y))
                if l2_norm < match_threshold:
                    small_img = cv.putText(small_img, 
                            window_name, 
                            (text_x, text_y), 
                            cv.FONT_HERSHEY_SIMPLEX, 
                            1.5, 
                            (0, 255, 0), 
                            3, 
                            cv.LINE_AA)
                #
                # display the images
                #
                cv.namedWindow(cropWindow, cv.WINDOW_NORMAL)
                cv.imshow(cropWindow, small_img)
            key = -1
            key = cv.waitKey(100)
            if key > 0:
                #
                # break on any key
                #
                break
    else:
        print ("camera open failed")

    #
    # release the stream and close the window
    #
    print("releasing stream")
    video_capture.release()
    time.sleep(1.0)
    print("destroying windows...")
    cv.destroyAllWindows()

if __name__ == '__main__':
    #
    # run main as standalone application
    #
    main()
