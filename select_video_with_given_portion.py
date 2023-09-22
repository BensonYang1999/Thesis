import cv2
import numpy as np
import os
import csv
import argparse
import itertools
"""
4723b82c19,1,2,0,0,1,1,2,0,1,0,0,3,1,0,0,0,1,2,2,0,0,0,2,0,0,0,1,0,2,3,2,0,2,0,0,1,0,1,0,1,1,0,1,0,1,1,0,0,0,0,1,0,2,2,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,3,0,0,0,0,0,0,1,1,1,0,2,0,1,0,1,1,0,0,1,0,4,0
b3b92781d9,22,19,8,11,31,12,11,35,20,33,14,12,9,3,23,8,10,20,19,13,16,27,12,18,14,23,16,20,19,12,10,16,34,20,21,15,25,10,12,16,22,8,12,12,14,16,15,18,13,9,16,7,4,14,13,4,5,12,18,15,8,12,24,8,16,18,33,17,9,15,28,30,25,14,8,16,9,18,19,14,26,11,13,11,29,14,18,3,31,18,20,25,17,22,27,9,19,9,19,27,15,8,10,21,11,11,7,15,26,16,19,23,7,17,16,33,6,21,31,11,6,10,23,24,18,37,17,36,8,22,9,20,12,11,20,25,23,10,29,25,33,12,21,4,32,33,6,36,25,40,6,17,23,8,32,28,14,21,12,15,19,15,15,33,25,24,3,10,20,23,12,8,17,6,22,12,15,21,15,9
a806e58451,22,22,22,21,19,23,23,28,26,18,20,18,18,20,22,21,26,17,41,11,23,21,27,29,18,29,19,15,22,24,21,14,23,34,17,19,17,21,28,22,21,34,27,26,26,23
ab9a7583f1,9,5,0,9,7,9,3,2,4,5,0,5,7,6,5,10,3,0,6,8,8,9,8,7,1,6,12,7,6,5,6,4,4,4,6,3,6,7,7,6,10,7,3,7,7,2,8,6,8,7,8,10,1,8,3,9,6,5,6,7,2,5,1,9,2,7,6,4,8,5,7,8,7,8,5,2,6,7,4,7,6,7,8,2,8,6,8,5,0,3,7,8,10,9,9,0
369919ef49,10,11,1,17,0,82,29,65,12,22,12,0,8,6,20,22,16,22,8,4,8,47,69,4,5,21,88,57,14,13,9,6,5,0,3,1,1,6,0,9,6,3,7,8,13,4,2,2,6,3,11,76,2,0,19,87,9,38,1,8,5,4,2,67,104,8,4,12,7,11,7,64,8,7,26,3,6,17,6,17,19,4,2,3,14,4,4,5,16,6,22,80,7,38,90,47,12,86,3,11,68,7,10,82,11,13,12,20,1,4,19,30,0,17,37,84,52,0,11,83,50,2,62,90,109,69,3,8,42,55,7,13,24,2,0,4,7,9,6,69,0,0,0,1,12,0,1,32,26,14
0620b43a31,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0
d1dc5a71e1,78,63,76,72,77,61,60,63,54,60,71,63,90,55,73,78,54,59,87,60,75,62,78,63,68,82,49,74,69,66,88,67,58,73,67,77,74,71,66,67,78,59,76,69,66,40,53,61,86,59,68,68,68,65,93,74,64,51,57,91,67,70,65,85,58,35,61,57,79,61,55,78,61,72,67,92,67,67,76,53,67,60,67,60,68,43,44,78,76,87,62
c16d9a4ade,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
6cb5b08d93,18,14,29,20,11,29,1,5,15,7,29,26,24,10,19,16,7,11,12,26,8,25,32,9,14,10,26,10,6,9,30,21,9,9,8,10,5,23,8,9,9,11,8,31,5,8,10,5,14,6,32,27,11,10,17,11,31,16,15,15,5,32,14,23,9,32,7,13,0,25,10,10,10,24,29,10,13,15,9,8,4,18,32,8,16,17,12,16,4,11,8,13,8,10,19,7,14,26,15,11,10,21,6,11,12,12,11,21,9,9,11,7,29,19,25,6,11,10,8,15,8,12,27,6,23,15,17,4,19,13,19,10,10,11,8,11,5,0,10,11,4,5,11,16,5,7,23,4,6,6,14,9,25,14,10,7,8,9,11,25,8,2,24,8,12,7,12,12,11,3,8,8,11,5,10,15,18,26,16,22
eac28d985f,5,5,4,7,4,6,6,7,6,0,5,7,4,0,5,2,4,1,6,1,3,5,7,0,4,7,3,6,6,8,0,8,7,5,7,2,4,8,7,2,7,5,4,8,4,6,7,7,5,8,1,4,7,3,6,6,6,4,2,0,4,1,4,5,0,6,6,2,8,7,6,6,12,7,9,7,4,7,5,8,1,0,4,7,1,1,7,3,0,4,4,8,1,5,0,5
"""

# given above input txt that each line refer to a video, the first column is the video id, and the rest of the columns are the count of line detected in each frame
# write a function that input the threshold of line/edge count, and filter out the average line/edge count of each video that is above the threshold
# output is a list of video id that is above the threshold

def filter_line_count(input_dir:str, threshold):
    output = []

    with open(input_dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split(',')
            video_id = line[0]
            line_count = line[1:]
            line_count = [int(i) for i in line_count]
            avg_line_count = sum(line_count)/len(line_count)
            if avg_line_count >= threshold:
                output.append(video_id)

    return output

def count_white_pixels(input_dir:str):
    # read image in gray scale
    img = cv2.imread(input_dir, cv2.IMREAD_GRAYSCALE)

    total_pixels = img.shape[0] * img.shape[1]
    
    white_pixels = np.sum(img > 0)
    white_ratio = white_pixels / total_pixels    

    return white_ratio

# there is a folder that store videos and sub folders and in each video subfolders store the jpg files of each frame
# write a function that input the path of the folder, and read all the jpg files in the subfolders
# count the white pixels in each jpg file and calculate the percentage of white pixels
# then write a txt file that each line refer to a video, the first column is the video id, and the rest of the columns are the percentage of white pixels in each frame
# output is a txt file
dir = "./datasets/YouTubeVOS/valid_all_frames/edges_old"
def count_white_pixels_in_folder(input_dir:str):
    output = []
    output_all = [] # store the results of all the videos

    # get all the subfolders
    subfolders = [f.path for f in os.scandir(input_dir) if f.is_dir()]

    # loop through each subfolder
    for subfolder in subfolders:
        # get the video id
        video_id = subfolder.split('/')[-1]
        # get all the jpg files in the subfolder
        jpg_files = [f.path for f in os.scandir(subfolder) if f.is_file()]
        # loop through each jpg file
        for jpg_file in jpg_files:
            # count the white pixels in the jpg file
            white_ratio = count_white_pixels(jpg_file)
            # white_ratio to percentage with three floating points
            white_ratio = round(white_ratio * 100, 3)
            # append the white ratio to the output
            output.append(white_ratio)

        # append the video id to the output in the first column
        output.insert(0, video_id)
        # append the output to the output_all
        output_all.append(output)
        # clear the output
        output = []
        
    # write the output_all to a csv file with name 'edge_cout.csv' under input_dir's parent folder
    output_dir = os.path.join(os.path.dirname(input_dir), 'edge_count.csv')
    with open(output_dir, 'w') as f:
        writer = csv.writer(f)
        # the first column is video name as string and the rest are float



# a edge version of filter_edge_count
def filter_edge_count(input_dir:str, threshold):
    output = []

    with open(input_dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split(',')
            video_id = line[0]
            edge_count = line[1:]
            edge_count = [float(i) for i in edge_count]
            avg_edge_count = sum(edge_count)/len(edge_count)
            if avg_edge_count >= threshold:
                output.append(video_id)

    return output

def read_csv_results(input_dir):
    output = []
    with open(input_dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split(',')
            output.append(line)
    return output

def calculate_avereage_with_th(result_dir, line_th, edge_th):

    pass_line_list = filter_line_count("./datasets/YouTubeVOS/valid_all_frames/line_count.csv", line_th)
    pass_edge_list = filter_edge_count("./datasets/YouTubeVOS/valid_all_frames/edge_count.csv", edge_th)

    # get the intersection of pass_line_list and pass_edge_list
    pass_list = list(set(pass_line_list) & set(pass_edge_list))
    # calculate the average of the result if the video is in the pass_list
    result = read_csv_results(result_dir)
    result_filted = [i for i in result if i[0] in pass_list]
    result_filted = [i[1:] for i in result_filted]
    result_filted = [[float(j) for j in i if j.strip()] for i in result_filted]
    for i, res in enumerate(result_filted):
        result_filted[i] = np.mean(np.array(res), axis=0)
        # three floating points
        result_filted[i] = round(result_filted[i], 3)

    # compute the overall average
    print(f"len of result: {len(result_filted)}")
    # print(f"pass list: {pass_list}")
    overall_avg = np.mean(result_filted)
    overall_avg = round(overall_avg, 3)

    print(f"overall average: {overall_avg}")
    return overall_avg



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_dir', type=str, default='./ckpt/0530_ZITS_video_YoutubeVOS_max500k_mix458k_turn470k/validation/')
    parser.add_argument('--input_dir', type=str, default='../FuseFormer/2023-06-30_gen_00050_scratch_youtube_results/')
    parser.add_argument('--line_th', type=float, default=0)
    parser.add_argument('--edge_th', type=float, default=0)
    args = parser.parse_args()
    # test the calculate_avereage_with_th function
    # line_th = args.line_th
    # edge_th = args.edge_th
    
    # print(f"\nline threshold: {line_th}, edge threshold: {edge_th}")

    # print(f"\n=== PSNR ===")
    # result_dir = os.path.join(args.input_dir, 'PSNR.csv')
    # calculate_avereage_with_th(result_dir, line_th, edge_th)

    # print(f"\n=== SSIM ===")
    # result_dir = os.path.join(args.input_dir, 'SSIM.csv')
    # calculate_avereage_with_th(result_dir, line_th, edge_th)
    psnr_dir = os.path.join(args.input_dir, "PSNR.csv")
    ssim_dir = os.path.join(args.input_dir, "SSIM.csv")

    line_th = [0, 50, 100, 150, 200]
    edge_th = [0, 2, 4, 6, 8, 10]

    import pandas as pd
    # create a dataframe to store the results of different thresholds
    df = pd.DataFrame(columns=["line_th", "edge_th", "psnr", "ssim"])

    # go through each combination of thresholds
    for (line_th, edge_th) in itertools.product(line_th, edge_th):
        psnr_avg = calculate_avereage_with_th(psnr_dir, line_th, edge_th)
        ssim_avg = calculate_avereage_with_th(ssim_dir, line_th, edge_th)
        df = df.append({"line_th": line_th, "edge_th": edge_th, "psnr": psnr_avg, "ssim": ssim_avg}, ignore_index=True)


    # save the results with the header
    df.to_csv(os.path.join(args.input_dir, "metrics.csv"), index=False)

    # print the results
    print(df)