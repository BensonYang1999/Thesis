# import cv2
# import os

# def create_video(input_folder, output_file):
#     images = [img for img in os.listdir(input_folder) if img.endswith(".png") or img.endswith(".jpg")]
#     images.sort()
    
#     # Read the first image to get the shape of each frame
#     frame = cv2.imread(os.path.join(input_folder, images[0]))
#     height, width, layers = frame.shape
    
#     # Create a VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video = cv2.VideoWriter(output_file, fourcc, 24, (width, height))

#     for image in images:
#         video.write(cv2.imread(os.path.join(input_folder, image)))

#     video.release()
#     cv2.destroyAllWindows()
    
#     return video.isOpened()

# if __name__ == "__main__":
#     dir = "./datasets/YouTubeVOS_small/valid/edge_50percent/wireframes/d1dd586cfd"
#     create_video(dir, "video_line.mp4")
#     print("Video Created")


import os
import imageio
from PIL import Image
import numpy as np

def create_gif(input_folder, output_file):
    images = [img for img in os.listdir(input_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort()
    
    frames = []
    for image in images:
        frame = imageio.imread(os.path.join(input_folder, image))
        frames.append(frame)
    
    imageio.mimsave(output_file, frames, duration=1/24.0)  # 24 frames per second

# given two folders of images, one is for the original images and the other is for the mask, create a gif that shows the masked images
def create_gif_with_mask(input_folder, mask_folder, output_file):
    images = [img for img in os.listdir(input_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort()
    
    frames = []
    for image in images:
        frame = imageio.imread(os.path.join(input_folder, image))
        mask = imageio.imread(os.path.join(mask_folder, image))
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = mask[:, :, None]
        mask = mask.repeat(3, axis=2)
        # if the value is > 0, than it is 255, otherwise it is 0
        mask = mask > 125

        # mask = mask.astype(int) / 255
        mask = 1 - mask  # invert the mask
        mask = mask.astype('uint8')
        # interpolate the mask to the same size as the frame
        mask = Image.fromarray(mask)
        mask = mask.resize((frame.shape[1], frame.shape[0]), Image.BILINEAR)

        frame = frame * mask
        frames.append(frame)
    
    imageio.mimsave(output_file, frames, duration=1/24.0)  # 24 frames per second

# given three folders, one is the original images, one is the mask, and the other is the inpainted images, create a gif that shows the inpainted images
# in mask image, 1 means the pixel is masked, 0 means the pixel is not masked
# when combining the result, used the original image if the mask is 0, otherwise use the inpainted image, and the color of the inpaintind result is blue green 
def create_gif_with_inpainting(input_folder, mask_folder, inpainted_folder, output_file):
    images = [img for img in os.listdir(input_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort()
    masks = [m for m in os.listdir(mask_folder) if m.endswith(".png") or m.endswith(".jpg")]
    masks.sort()
    
    frames = []
    for image, m in zip(images, masks):
        frame = imageio.imread(os.path.join(input_folder, image))
        mask = imageio.imread(os.path.join(mask_folder, m))
        inpainted = imageio.imread(os.path.join(inpainted_folder, image))
        # if the frame is a gray scale image, convert it to a 3 channel image
        if len(frame.shape) == 2:
            frame = frame[:, :, None]
            frame = frame.repeat(3, axis=2)

        # if the inpainted image is a gray scale image, convert it to a 3 channel image
        if len(inpainted.shape) == 2:
            inpainted = inpainted[:, :, None]
            inpainted = inpainted.repeat(3, axis=2)

        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = mask[:, :, None]
        mask = mask.repeat(3, axis=2)
        # if the value is > 0, than it is 255, otherwise it is 0
        mask = mask > 125

        # mask = mask.astype(int) / 255
        mask = 1 - mask  # invert the mask
        mask = mask.astype('uint8')
        # interpolate the mask to the same size as the frame
        mask = Image.fromarray(mask)
        mask = mask.resize((frame.shape[1], frame.shape[0]), Image.BILINEAR)
        mask = np.array(mask)

        # convert the (255, 255, 255) to (0, 255, 255)
        # Convert inpainted to a NumPy array
        inpainted = np.array(inpainted)

        # Find all white pixels and change them to [0, 255, 255]
        indices = np.where((inpainted == [255, 255, 255]).all(axis=2))
        inpainted[indices] = [0, 255, 255]

        frame = frame * mask + inpainted * (1 - mask)
        frames.append(frame)
    
    imageio.mimsave(output_file, frames, duration=1/24.0)  # 24 frames per second

def main():
    base_path = "./datasets/YouTubeVOS_small/valid/edge_50percent/"
    common_folder = "d1dd586cfd"
    
    directories = {
        'img': f"{base_path}JPEGImages/{common_folder}",
        'line': f"{base_path}wireframes/{common_folder}",
        'edge': f"{base_path}edges_old/{common_folder}",
        'mask': f"{base_path}mask_random/{common_folder}"
    }
    
    # Uncomment the lines below to create GIFs as needed
    # create_gif(directories['img'], "video.gif")
    # create_gif(directories['line'], "video_line.gif")
    # create_gif(directories['edge'], "video_edge.gif")
    # create_gif_with_mask(directories['img'], directories['mask'], "video_masked.gif")
    # create_gif_with_inpainting(directories['edge'], directories['mask'], directories['edge'], "video_edge_inpainting.gif")

    result_dir = "../STTN/results/youtube/valid/2023-08-28_edge_50percent/d1dd586cfd"
    create_gif(result_dir, "video_STTN_inpainted_result.gif")

    print("GIF Created")

if __name__ == "__main__":
    main()

