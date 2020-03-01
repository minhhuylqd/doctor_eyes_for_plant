import cv2
import os
import numpy as np

#image_folder = 'video_test'
image_folder = 'baohq'
video_name = 'testbaohq.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

#height, width, layers = (500, 500, 3)

#print(frame.shape)


video = cv2.VideoWriter(video_name, 0, 1, (500,500))

count = 0
for image in images:
    """ # Create new image white 500x500
    whiteImage = np.zeros([500,500,3],dtype=np.uint8)
    whiteImage.fill(255)
    originImage = cv2.imread(os.path.join(image_folder, image))
    #Paste origin image at center
    _width = 500
    _height = 500
    x_offset = int((_width - originImage.shape[1])/2)
    y_offset = int((_height - originImage.shape[0])/2)
    whiteImage[ y_offset:y_offset+originImage.shape[0], x_offset:x_offset+originImage.shape[1]] = originImage
    cv2.imwrite("baohq/baohq" + str(count) + ".jpg", whiteImage)
    count += 1 """
    #newImage = cv2.imread(os.path.join(image_folder, image))
    #whiteImage[256:256, 0:0] = newImage
    #cv2.imshow('white',whiteImage)
    #print(type(whiteImage))
    video.write(cv2.imread(os.path.join(image_folder, image)))
    #video.write(whiteImage)

cv2.destroyAllWindows()
video.release()
