# input_dir = "./YouTubeVOS/test_all_frames/count.csv"

# # each line in the input_dir is the information of a video, the first element is the video_id, the rest are the edge count of each frame
# # summarize the average edge count of each video and save in a pandas dataframe
# # the first column is the video_id, the second column is the average edge count of this video
# # but the number of frames in each video is different, so we need to compute the average edge count of each video first

# import pandas as pd
# import numpy as np

# # read the csv with line without pandas
# video_id_list = []
# count_list = []
# with open(input_dir, "r") as f:
#     lines = f.readlines()
#     for line in lines:
#         line = line.strip()
#         line = line.split(",")
#         video_id = line[0]
#         count = line[1:]
#         count = [float(i) for i in count]
#         avg_count = sum(count)/len(count)
#         video_id_list.append(video_id)
#         count_list.append(avg_count)

# # create a pandas dataframe
# df = pd.DataFrame()
# df["video_id"] = video_id_list
# df["count"] = count_list

# print(df.head()) 

# # print the first quantile, median, third quantile, max of the edge count
# print(df.describe())

# # random select 10 videos from the most 25% edge count videos
# # random select 10 videos from the most 50% edge count videos, but exclude the videos selected in the previous step
# # random select 10 videos from the most 75% edge count videos, but exclude the videos selected in the previous steps
# # random select 10 videos from the most 100% edge count videos, but exclude the videos selected in the previous steps
# # print the id of the selected videos

# # sort the dataframe by the edge count
# df = df.sort_values(by="count", ascending=False)

# # select 10 videos from the most 25% edge count videos
# df_25 = df.iloc[:int(len(df)*0.25)]
# df_25 = df_25.sample(n=10)
# print(df_25)

# # select 10 videos from the most 50% edge count videos, but exclude the videos selected in the previous step
# df_50 = df.iloc[int(len(df)*0.25):int(len(df)*0.5)]
# df_50 = df_50.sample(n=10)
# print(df_50)

# # select 10 videos from the most 75% edge count videos, but exclude the videos selected in the previous steps
# df_75 = df.iloc[int(len(df)*0.5):int(len(df)*0.75)]
# df_75 = df_75.sample(n=10)
# print(df_75)

# # select 10 videos from the most 100% edge count videos, but exclude the videos selected in the previous steps
# df_100 = df.iloc[int(len(df)*0.75):]
# df_100 = df_100.sample(n=10)
# print(df_100)

import pandas as pd
import random
import os
import shutil

def select_videos(input_dir):
    # read the csv with line without pandas
    video_id_list = []
    count_list = []
    with open(input_dir, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split(",")
            video_id = line[0]
            count = line[1:]
            count = [float(i) for i in count]
            avg_count = sum(count)/len(count)
            video_id_list.append(video_id)
            count_list.append(avg_count)

    # create a pandas dataframe
    df = pd.DataFrame()
    df["video_id"] = video_id_list
    df["count"] = count_list

    # sort the dataframe by the edge count
    df = df.sort_values(by="count", ascending=False)

    # select 10 videos from each quantile
    df_25 = df.iloc[:int(len(df)*0.25)].sample(n=10)
    df_50 = df.iloc[int(len(df)*0.25):int(len(df)*0.5)].sample(n=10)
    df_75 = df.iloc[int(len(df)*0.5):int(len(df)*0.75)].sample(n=10)
    df_100 = df.iloc[int(len(df)*0.75):].sample(n=10)

    # testing, print the above dataframes
    print(df.describe())
    print(f"df_25: {df_25}")
    print(f"df_50: {df_50}")
    print(f"df_75: {df_75}")
    print(f"df_100: {df_100}")

    # get the video_id lists
    id_list_25 = df_25['video_id'].tolist()
    id_list_50 = df_50['video_id'].tolist()
    id_list_75 = df_75['video_id'].tolist()
    id_list_100 = df_100['video_id'].tolist()

    return id_list_25, id_list_50, id_list_75, id_list_100

# a function let take a list of video ids and given the input folder and output folder
# the function should copy the corresponding video frames to the output folder
# the input folder is the root folder that contains all the videos
# the output folder is also a root folder that will store the selected videos
# the output folder should have the same structure as the input folder
def copy_videos(video_id_list, input_dir, output_dir):
    for video_id in video_id_list:
        # get the video folder
        video_folder = os.path.join(input_dir, video_id)
        # get the output folder
        output_folder = os.path.join(output_dir, video_id)
        # copy the video folder to the output folder
        shutil.copytree(video_folder, output_folder)

if __name__=="__main__":
    # root_input = "./YouTubeVOS/test_all_frames/"
    # root_output = "./YouTubeVOS_small/test/set_1/"
    root_input = "./DAVIS/JPEGImages"
    root_output = "./DAVIS_small/test_set_1/"

    edge_input_dir = os.path.join(root_input, "edge_count.csv")
    line_input_dir = os.path.join(root_input, "line_count.csv")

    print(f"Edge results:")
    edge_25, edge_50, edge_75, edge_100 = select_videos(edge_input_dir)
    
    print(edge_25)
    print(edge_50)
    print(edge_75)
    print(edge_100)

    print(f"Line results:")
    line_25, line_50, line_75, line_100 = select_videos(line_input_dir)
    print(line_25)
    print(line_50)
    print(line_75)
    print(line_100)

    if "YouTubeVOS" in root_input:
        input_jpeg_dir = os.path.join(root_input, "JPEGImages")
        input_line_dir = os.path.join(root_input, "wireframes")
        input_edge_dir = os.path.join(root_input, "edges_old")
        input_mask_random = os.path.join(root_input, "mask_random")
        input_mask_brush = os.path.join(root_input, "mask_brush")
    else: # DAVIS
        input_jpeg_dir = os.path.join(root_input, "Full-Resolution")
        input_line_dir = os.path.join(root_input, "Full-Resolution_wireframes")
        input_edge_dir = os.path.join(root_input, "Full-Resolution_edges")
        input_mask_random = os.path.join(root_input, "mask_random")
        input_mask_brush = os.path.join(root_input, "mask_brush")

    names = ["25", "50", "75", "100"]
    for name, list in zip(names, [edge_25, edge_50, edge_75, edge_100]):
        output_jpeg_dir = os.path.join(root_output, f"edge_{name}percent/JPEGImages")
        output_line_dir = os.path.join(root_output, f"edge_{name}percent/wireframes")
        output_edge_dir = os.path.join(root_output, f"edge_{name}percent/edges")
        output_mask_random = os.path.join(root_output, f"edge_{name}percent/mask_random")
        # output_mask_brush = os.path.join(root_output, f"edge_{name}percent/mask_brush")

        # copy the videos
        copy_videos(list, input_jpeg_dir, output_jpeg_dir)
        copy_videos(list, input_line_dir, output_line_dir)
        copy_videos(list, input_edge_dir, output_edge_dir)
        copy_videos(list, input_mask_random, output_mask_random)
        # copy_videos(list, input_mask_brush, output_mask_brush)

    # copy the videos with the line condition
    for name, list in zip(names, [line_25, line_50, line_75, line_100]):
        output_jpeg_dir = os.path.join(root_output, f"line_{name}percent/JPEGImages")
        output_line_dir = os.path.join(root_output, f"line_{name}percent/wireframes")
        output_edge_dir = os.path.join(root_output, f"line_{name}percent/edges")
        output_mask_random = os.path.join(root_output, f"line_{name}percent/mask_random")
        # output_mask_brush = os.path.join(root_output, f"line_{name}percent/mask_brush")

        # copy the videos
        copy_videos(list, input_jpeg_dir, output_jpeg_dir)
        copy_videos(list, input_line_dir, output_line_dir)
        copy_videos(list, input_edge_dir, output_edge_dir)
        copy_videos(list, input_mask_random, output_mask_random)
        # copy_videos(list, input_mask_brush, output_mask_brush)


    # print all the selected videos with all the conditions
    print(f"Edge results:")
    print(f"edge_25: {edge_25}")
    print(f"edge_50: {edge_50}")
    print(f"edge_75: {edge_75}")
    print(f"edge_100: {edge_100}")
    
    print(f"Line results:")
    print(f"line_25: {line_25}")
    print(f"line_50: {line_50}")
    print(f"line_75: {line_75}")
    print(f"line_100: {line_100}")

    # check whether there are videos that are selected in both edge and line, find the union
    edge_set = set(edge_25 + edge_50 + edge_75 + edge_100)
    line_set = set(line_25 + line_50 + line_75 + line_100)
    print(f"edge_set: {edge_set}")
    print(f"line_set: {line_set}")
    print(f"edge_set & line_set: {edge_set & line_set}")

    # write all the selected video name under different conditions to a txt file
    # with open(f"./YouTubeVOS_small/{split}_all_frames/selected_videos.txt", "w") as f:
    with open(os.path.join(root_output, "selected_videos.txt"), "w") as f:
        f.write(f"Edge results:\n")
        f.write(f"edge_25: {edge_25}\n")
        f.write(f"edge_50: {edge_50}\n")
        f.write(f"edge_75: {edge_75}\n")
        f.write(f"edge_100: {edge_100}\n")
        f.write(f"Line results:\n")
        f.write(f"line_25: {line_25}\n")
        f.write(f"line_50: {line_50}\n")
        f.write(f"line_75: {line_75}\n")
        f.write(f"line_100: {line_100}\n")
        f.write(f"edge_set: {edge_set}\n")
        f.write(f"line_set: {line_set}\n")
        f.write(f"edge_set & line_set: {edge_set & line_set}\n")