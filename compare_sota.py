import pandas as pd
import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--onlyMask', action='store_true', help='whether to use onlyMask')
parser.add_argument('--dataset', type=str, default='YouTubeVOS', help='dataset name')
args = parser.parse_args()

onlyMask = args.onlyMask

### whole dataset
# fuse_dir = f"../FuseFormer/fuseformer_5frames_youtube_results/2023-07-07_gen_00050_youtubevos_split_test/metrics_onlyMask_{onlyMask}.csv"
# sttn_dir = f"../STTN/results/youtube/metrics_onlyMask_{onlyMask}.csv"
# ours_old_dir = f"./results/0530_ZITS_video_YoutubeVOS_max500k_mix458k_turn470k/2023-07-07_youtubevos/metrics_onlyMask_{onlyMask}.csv"
# ours_new_dir = f"./results/0710_ZITS_video_YoutubeVOS_max500k_mix458k_turn470k_frame-1_1_ReFFC_removed_last/2023-07-13_youtubevos/metrics_onlyMask_{onlyMask}.csv"
### part of dataset
fuse_dir = f"../FuseFormer/fuseformer_5frames_youtube_results/2023-07-06_{args.dataset}/metrics_onlyMask_{onlyMask}.csv"
# sttn_dir = f"../STTN/results/youtube/metrics_onlyMask_{onlyMask}.csv"
ours_old_dir = f"./results/0530_ZITS_video_YoutubeVOS_max500k_mix458k_turn470k/test/2023-07-06_{args.dataset}/metrics_onlyMask_{onlyMask}.csv"
ours_new_dir = f"./results/0710_ZITS_video_YoutubeVOS_max500k_mix458k_turn470k_frame-1_1_ReFFC_removed_last/2023-07-13_{args.dataset}/metrics_onlyMask_{onlyMask}.csv"

gt_dir = "./datasets/YouTubeVOS/test_all_frames/JPEGImages"
mask_dir = "./datasets/YouTubeVOS/test_all_frames/mask_random"

output_dir = f"./results/compare_sota/{args.dataset}_onlyMask_{onlyMask}"
# get the root folder of the above csv files
fuse_root = os.path.dirname(fuse_dir)
# sttn_toot = os.path.dirname(sttn_dir)
ours_old_root = os.path.dirname(ours_old_dir)
ours_new_root = os.path.dirname(ours_new_dir)

# check if the output dir exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def save_as_video(img_dir, video_path):
    img_list = os.listdir(img_dir)
    img_list.sort()
    img_list = [os.path.join(img_dir, img) for img in img_list]
    img_list = [cv2.imread(img) for img in img_list]
    height, width, _ = img_list[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(video_path, fourcc, 10, (width, height))
    for img in img_list:
        videoWriter.write(img)
    videoWriter.release()

def save_as_video_with_mask(img_dir, mask_dir, video_path):
    # save each frame with the mask covered
    img_list = os.listdir(img_dir)
    img_list.sort()
    img_list = [os.path.join(img_dir, img) for img in img_list]
    mask_list = os.listdir(mask_dir)
    mask_list.sort()
    mask_list = [os.path.join(mask_dir, mask) for mask in mask_list]
    img_list = [cv2.imread(img) for img in img_list]
    mask_list = [cv2.imread(mask) for mask in mask_list]

    # resize image to 432*240
    img_list = [cv2.resize(img, (432, 240)) for img in img_list]
    mask_list = [cv2.resize(mask, (432, 240)) for mask in mask_list]
    # conver the mask to a binary mask
    mask_list = [cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) for mask in mask_list]
    mask_list = [np.array(m > 0).astype(np.uint8) for m in mask_list]
    mask_list = [cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4) for m in mask_list]
    # expand the mask to 3 channels
    mask_list = [np.expand_dims(m, axis=2) for m in mask_list]
    
    height, width, _ = img_list[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(video_path, fourcc, 10, (width, height))
    for img, mask in zip(img_list, mask_list):
        img = img * (1 - mask) + mask * 255
        videoWriter.write(img)
    videoWriter.release()


def load_dfs(csv_file_paths):
    return [pd.read_csv(file) for file in csv_file_paths]

def analyze_and_save_videos(dataframes, scoring_methods, root_directories, output_dir, ascending_flags, best=True):
    assert len(dataframes) == len(root_directories)

    for method, asc in zip(scoring_methods, ascending_flags):
        print(f"Analyzing for method: {method}")
        # get the video name of highest/lowest score in ours_new
        ours_new_df = dataframes[-1].sort_values(by=[method], ascending=asc)
        video_id = ours_new_df.iloc[0]['video_id']

        for i, df in enumerate(dataframes):
            video_path = os.path.join(root_directories[i], video_id)
            if best:
                save_as_video(video_path, os.path.join(output_dir, f"{method}_best_ours_new_{video_id}.mp4"))
                # save fuseformer video
                save_as_video(os.path.join(fuse_root, video_id), os.path.join(output_dir, f"{method}_fuse_{video_id}.mp4"))
                # save sttn video
                # save_as_video(os.path.join(sttn_root, video_id), os.path.join(output_dir, f"{method}_sttn_{video_id}.mp4"))
                # save old video
                save_as_video(os.path.join(ours_old_root, video_id), os.path.join(output_dir, f"{method}_ours_old_{video_id}.mp4"))
            else:
                save_as_video(video_path, os.path.join(output_dir, f"{method}_worst_ours_new_{video_id}.mp4"))
                # save fuseformer video
                save_as_video(os.path.join(fuse_root, video_id), os.path.join(output_dir, f"{method}_fuse_{video_id}.mp4"))
                # save sttn video
                # save_as_video(os.path.join(sttn_root, video_id), os.path.join(output_dir, f"{method}_sttn_{video_id}.mp4"))
                # save old video
                save_as_video(os.path.join(ours_old_root, video_id), os.path.join(output_dir, f"{method}_ours_old_{video_id}.mp4"))
            print(f"Scores for video_id {video_id}: {df[df['video_id'] == video_id]}")
        print()

        # Save ground truth video
        gt_path = os.path.join(gt_dir, video_id)
        save_as_video(gt_path, os.path.join(output_dir, f"gt_{video_id}.mp4"))

def save_results():
    # fuse_df, sttn_df, ours_old_df, ours_new_df = load_dfs([fuse_dir, sttn_dir, ours_old_dir, ours_new_dir])

    # dataframes = [fuse_df, sttn_df, ours_old_df, ours_new_df]
    # root_directories = [fuse_root, sttn_root, ours_old_root, ours_new_root]
    # remove the sttn columns
    fuse_df, ours_old_df, ours_new_df = load_dfs([fuse_dir, ours_old_dir, ours_new_dir])
    dataframes = [fuse_df, ours_old_df, ours_new_df]
    root_directories = [fuse_root, ours_old_root, ours_new_root]

    scoring_methods = ['psnr', 'ssim', 'lpips', 'mse', 'vif', 'uqi']
    ascending_flags = [False, False, True, True, False, False]

    analyze_and_save_videos(dataframes, scoring_methods, root_directories, output_dir, ascending_flags, best=True)

    # For 'worst' scores
    scoring_methods = ['psnr', 'ssim', 'lpips', 'mse']
    ascending_flags = [True, True, False, False]
    
    analyze_and_save_videos(dataframes, scoring_methods, root_directories, output_dir, ascending_flags, best=False)



def quantize(input_dir):
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
    df = df.sort_values(by="count", ascending=True)

    df_25 = df.iloc[:int(len(df)*0.25)]
    df_50 = df.iloc[int(len(df)*0.25):int(len(df)*0.5)]
    df_75 = df.iloc[int(len(df)*0.5):int(len(df)*0.75)]
    df_100 = df.iloc[int(len(df)*0.75):]

    # get the video_id lists
    id_list_25 = df_25['video_id'].tolist()
    id_list_50 = df_50['video_id'].tolist()
    id_list_75 = df_75['video_id'].tolist()
    id_list_100 = df_100['video_id'].tolist()

    return id_list_25, id_list_50, id_list_75, id_list_100

def load_dataframe(file_dir):
    return pd.read_csv(file_dir)

def filter_dataframe(df, filter_list):
    return df[df['video_id'].isin(filter_list)]

def compute_statistics(df):
    psnr_mean = df['psnr'].mean()
    ssim_mean = df['ssim'].mean()
    lpips_mean = df['lpips'].mean()
    mse_mean = df['mse'].mean()
    vif_mean = df['vif'].mean()
    uqi_mean = df['uqi'].mean()
    return psnr_mean, ssim_mean, lpips_mean, mse_mean, vif_mean, uqi_mean

def print_statistics(df, model_name, percentage):
    psnr, ssim, lpips, mse, vif, uqi = compute_statistics(df)
    print(f"{model_name} ({percentage}%)    {psnr:.4f}  {ssim:.4f}  {lpips:.4f}    {mse:.4f}  {vif:.4f}  {uqi:.4f}")


if __name__=="__main__":

    # read dataframe from csv from fuse_dir
    df_fuse = load_dataframe(fuse_dir)
    # df_sttn = load_dataframe(sttn_dir)
    df_ours_old =load_dataframe(ours_old_dir)
    df_ours_new = load_dataframe(ours_new_dir)
    
    line_csv = "./datasets/YouTubeVOS/test_all_frames/line_count.csv"
    edge_csv = "./datasets/YouTubeVOS/test_all_frames/edge_count.csv"
    line_25, line_50, line_75, line_100 = quantize(line_csv)
    edge_25, edge_50, edge_75, edge_100 = quantize(edge_csv)

    # filter dataframes and print statistics
    line_filter_lists = [line_25, line_50, line_75, line_100]
    edge_filter_lists = [edge_25, edge_50, edge_75, edge_100]
    # model_dataframes = [df_fuse, df_sttn, df_ours_old, df_ours_new]
    # model_names = ["Fuse", "STTN", "Ours_old", "Ours_new"]
    model_dataframes = [df_fuse, df_ours_old, df_ours_new]
    model_names = ["Fuse", "Ours_old", "Ours_new"]
    percentage = [25, 50, 75, 100]

    for percent, filter_list in zip(percentage, line_filter_lists):
        print(f"===== Line onlyMask({onlyMask}) =====")
        print(f"Model   PSNR    SSIM    LPIPS   MSE VIF UQI")
        
        for df, model_name in zip(model_dataframes, model_names):
            df_filtered = filter_dataframe(df, filter_list)
            print_statistics(df_filtered, model_name, percent)

    for percent, filter_list in zip(percentage, edge_filter_lists):
        print(f"===== Edge onlyMask({onlyMask}) =====")
        print(f"Model   PSNR    SSIM    LPIPS   MSE VIF UQI")
        
        for df, model_name in zip(model_dataframes, model_names):
            df_filtered = filter_dataframe(df, filter_list)
            print_statistics(df_filtered, model_name, percent)

    # combine three dataframes by the video_id
    df = pd.merge(df_ours_new, df_ours_old, on='video_id', suffixes=('_ours_new', '_ours_old'))
    df = pd.merge(df, df_fuse, on='video_id')
    df = df.rename(columns={'psnr': 'psnr_fuse', 'ssim': 'ssim_fuse', 'lpips': 'lpips_fuse', 'mse': 'mse_fuse', 'vif': 'vif_fuse', 'uqi': 'uqi_fuse'})
    # df = pd.merge(df, df_sttn, on='video_id')
    # df = df.rename(columns={'psnr': 'psnr_sttn', 'ssim': 'ssim_sttn', 'lpips': 'lpips_sttn', 'mse': 'mse_sttn', 'vif': 'vif_sttn', 'uqi': 'uqi_sttn'})

    # reorganize the columns
    # df = df[['video_id', 'psnr_ours_new', 'psnr_ours_old', 'psnr_fuse', 'psnr_sttn', 'ssim_ours_new', 'ssim_ours_old', 'ssim_fuse', 'ssim_sttn', 'lpips_ours_new', 'lpips_ours_old', 'lpips_fuse', 'lpips_sttn', 'mse_ours_new', 'mse_ours_old', 'mse_fuse', 'mse_sttn', 'vif_ours_new', 'vif_ours_old', 'vif_fuse', 'vif_sttn', 'uqi_ours_new', 'uqi_ours_old', 'uqi_fuse', 'uqi_sttn']]
    # remove the sttn columns
    df = df[['video_id', 'psnr_ours_new', 'psnr_ours_old', 'psnr_fuse', 'ssim_ours_new', 'ssim_ours_old', 'ssim_fuse', 'lpips_ours_new', 'lpips_ours_old', 'lpips_fuse', 'mse_ours_new', 'mse_ours_old', 'mse_fuse', 'vif_ours_new', 'vif_ours_old', 'vif_fuse', 'uqi_ours_new', 'uqi_ours_old', 'uqi_fuse']]

    # round the values to 4 decimal places
    df = df.round(4)
    # save the dataframe to csv file and separate the columns by tab
    # df.to_csv(os.path.join(output_dir, 'result.csv'), index=False, sep='\t')
    df.to_csv(os.path.join(output_dir, 'result.csv'), index=False)

    # count psnr_ours_old > psnr_fuse
    print(f"psnr_ours_old > psnr_fuse: {df['psnr_ours_old'].gt(df['psnr_fuse']).sum()}")
    print(f"psnr_ours_new > psnr_fuse: {df['psnr_ours_new'].gt(df['psnr_fuse']).sum()}")
    # print(f"psnr_ours_old > psnr_sttn: {df['psnr_ours_old'].gt(df['psnr_sttn']).sum()}")

    # count ssim_ours_old > ssim_fuse
    print(f"ssim_ours_old > ssim_fuse: {df['ssim_ours_old'].gt(df['ssim_fuse']).sum()}")
    print(f"ssim_ours_new > ssim_fuse: {df['ssim_ours_new'].gt(df['ssim_fuse']).sum()}")
    # print(f"ssim_ours_old > ssim_sttn: {df['ssim_ours_old'].gt(df['ssim_sttn']).sum()}")

    # count lpips_ours_old > lpips_fuse
    print(f"lpips_ours_old < lpips_fuse: {df['lpips_ours_old'].le(df['lpips_fuse']).sum()}")
    print(f"lpips_ours_new < lpips_fuse: {df['lpips_ours_new'].le(df['lpips_fuse']).sum()}")
    # print(f"lpips_ours_old < lpips_sttn: {df['lpips_ours_old'].le(df['lpips_sttn']).sum()}")


    save_results()

    