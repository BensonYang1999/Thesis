from os import walk
from os.path import join

# path = "/home/ZITS_inpainting/datasets/DAVIS/JPEGImages/Full-Resolution"
path = "/home/ZITS_inpainting/datasets/YouTubeVOS/train_all_frames/JPEGImages"
# path = "/home/ZITS_inpainting/datasets/irregular_mask"

with open("./data_list/youtube_train_list.txt", "w") as out_file:    
    count = 0
    for root, dirs, files in walk(path):
        video_name = root.split("/")[-1]
        # print(f"video {count}: {video_name}")
        if video_name != path.split("/")[-1]:
            out_file.write(f"video {count}: {video_name}\n")

        files.sort()
        for f in files:
            fullpath = join(root, f)
            if ".jpg" in fullpath or ".png" in fullpath:
                out_file.write(f"{fullpath}\n")
        count += 1

# with open("./data_list/irregular_mask_list.txt", "w") as out_file:    
#     for root, dirs, files in walk(path):
#         files.sort()
#         for f in files:
#             fullpath = join(root, f)
#             if ".jpg" in fullpath or ".png" in fullpath:
#                 out_file.write(f"{fullpath}\n")