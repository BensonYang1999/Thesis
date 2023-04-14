import os
import time
import cv2
from glob import glob
from skimage import feature
from skimage.color import rgb2gray
from scipy import ndimage as ndi
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

dataset_roots = {'YouTubeVOS': './datasets/YouTubeVOS/test_all_frames/JPEGImages'}
edge_roots = {'YouTubeVOS': './datasets/YouTubeVOS/test_all_frames/edges_gau2_canny2'}

def process_video(args):
    video_path, output_edge_path = args
    for img in glob(video_path + '/*'):
        image = rgb2gray(cv2.imread(img))
        edge = feature.canny(ndi.gaussian_filter(image, 2), sigma=2).astype(float)
        cv2.imwrite(os.path.join(output_edge_path, img.split('/')[-2], img.split('/')[-1]), edge * 255)

def process_videos(video_paths, output_edge_path):
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap(process_video, [(video_path, output_edge_path) for video_path in video_paths]), total=len(video_paths)))

if __name__ == '__main__':
    time_costs = {}
    for dataset, root in dataset_roots.items():
        start_t = time.time()
        video_list = glob(root + "/*")
        output_edge_path = edge_roots[dataset]
        for video_path in video_list:
            video_name = video_path.split("/")[-1]
            os.makedirs(os.path.join(output_edge_path, video_name), exist_ok=True)
        process_videos(video_list, output_edge_path)
        time_costs[dataset] = round(time.time() - start_t, 4)

    print("[Time costs]:")
    for dataset, time in time_costs.items():
        print(f"{dataset} -> {time} (sec.)")