# given the input folder of images, sort these images regard their name and create a video file
import cv2
import os
def create_video(input_folder, output_file):
    images = [img for img in os.listdir(input_folder) if img.endswith(".jpg")]
    images.sort()
    frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_file, 0, 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(input_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    return video.isOpened()

if __name__ == "__main__":
    f __name__ == '__main__':
    create_video("images", "video.mp4")
    print("Video Created")