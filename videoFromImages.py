import cv2
import os

image_folder = '/home/ashwin/intel-oneAPI/videoCreator/'
video_name = 'output_video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 30, (width,height))

# for image in images:
for i in range(len(images)):
    print("image = ", i+1,"/",len(images))
    video.write(cv2.imread(os.path.join(image_folder, str(i+1)+".png")))

cv2.destroyAllWindows()
video.release()