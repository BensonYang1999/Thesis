import os
import numpy as np
import cv2

input_dir = "./test_all_frames/edges_old"
output_dir = "./test_all_frames/edge_count.csv"

video_folders = os.listdir(input_dir)
video_folders.sort()

# read all frame images under all video_folders and compute the percentage of white pixels
# and in the result file edge_count.csv, the first column is video_id, the rest columns are the percentage of white pixels for each frame
with open(output_dir, "w") as f:
    for video_folder in video_folders:
        video_dir = os.path.join(input_dir, video_folder)
        frame_list = os.listdir(video_dir)
        frame_list.sort()
        count_list = []
        for frame in frame_list:
            frame_dir = os.path.join(video_dir, frame)
            frame_image = cv2.imread(frame_dir)
            frame_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
            white_pixel = np.count_nonzero(frame_image)
            total_pixel = frame_image.shape[0] * frame_image.shape[1]
            percentage = white_pixel / total_pixel
            # times 100 to get percentage and get the round 3
            percentage = round(percentage * 100, 3)
            count_list.append(percentage)

        # write the count_list to csv file with the first column as video_id
        f.write(video_folder)
        for count in count_list:
            f.write(f",{count}")
        f.write("\n")

