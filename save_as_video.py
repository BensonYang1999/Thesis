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
        
        # If the mask has color channels, just keep one (e.g., the red channel)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        # Convert mask values to range [0, 1] and invert it
        mask_alpha = mask / 255.0
        mask_alpha = 0.5 * mask_alpha  # Adjust the transparency level by changing this value
        
        # Resize mask to the size of the frame
        mask_alpha = Image.fromarray((mask_alpha * 255).astype('uint8'))
        mask_alpha = mask_alpha.resize((frame.shape[1], frame.shape[0]), Image.BILINEAR)
        mask_alpha = np.array(mask_alpha) / 255.0

        # Create a black image for the masked area
        black_img = np.zeros_like(frame)
        
        # Alpha blending
        blended_frame = frame * (1 - mask_alpha[:, :, None]) + black_img * mask_alpha[:, :, None]
        
        frames.append(blended_frame.astype('uint8'))
    
    imageio.mimsave(output_file, frames, duration=1/24.0)  # 24 frames per second


# given three folders, one is the original images, one is the mask, and the other is the inpainted images, create a gif that shows the inpainted images
# in mask image, 1 means the pixel is masked, 0 means the pixel is not masked
# when combining the result, used the original image if the mask is 0, otherwise use the inpainted image, and the color of the inpaintind result is blue green 
def create_gif_with_inpainting(input_folder, mask_folder, inpainted_folder, output_file):
    images = [img for img in os.listdir(input_folder) if img.endswith(".png") or img.endswith(".jpg")] 
    images.sort()
    masks = [m for m in os.listdir(mask_folder) if m.endswith(".png") or m.endswith(".jpg")]
    masks.sort()
    inpaints = [i for i in os.listdir(inpainted_folder) if i.endswith(".png") or i.endswith(".jpg")]
    inpaints.sort()

    frames = []
    for image, m, inpaint in zip(images, masks, inpaints):
        frame = imageio.imread(os.path.join(input_folder, image))
        mask = imageio.imread(os.path.join(mask_folder, m))
        inpainted = imageio.imread(os.path.join(inpainted_folder, inpaint))
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
        mask = mask.resize((inpainted.shape[1], inpainted.shape[0]), Image.BILINEAR)
        mask = np.array(mask)

        # convert the (255, 255, 255) to (0, 255, 255)
        # Convert inpainted to a NumPy array
        inpainted = np.array(inpainted)

        # Find all white pixels and change them to [0, 255, 255]
        indices = np.where((inpainted == [255, 255, 255]).all(axis=2))
        inpainted[indices] = [0, 255, 255]

        # resize frame to the same size as inpainted
        frame = Image.fromarray(frame)
        frame = frame.resize((inpainted.shape[1], inpainted.shape[0]), Image.BILINEAR)
        frame = np.array(frame)

        frame = frame * mask + inpainted * (1 - mask)
        frames.append(frame)
    
    imageio.mimsave(output_file, frames, duration=1/24.0)  # 24 frames per second

def main():
    gt_base_path = "./datasets/YouTubeVOS_small/test/set_1/line_25percent"
    # gt_base_path = "./datasets/DAVIS_small/test_set_1/line_25percent"

    result_base_path = "./results/1024_SERVI_finetune0926_l1HoleWeight/Final_result_thesis/Youtube_test_set1_smooth_mask/2023-10-26_line_25percent/"
    # result_base_path = "./results/1024_SERVI_finetune0926_l1HoleWeight/Final_result_thesis/DAVIS_test_set1_smooth_mask/2023-10-26_line_25percent"
    common_folder = "8d55a5aebb"
    
    gt_directories = {
        'img': f"{gt_base_path}/JPEGImages/{common_folder}",
        'line': f"{gt_base_path}/wireframes/{common_folder}",
        'edge': f"{gt_base_path}/edges/{common_folder}",
        'mask': f"{gt_base_path}/mask_random/{common_folder}"
    }

    result_directories = {
        'img': f"{result_base_path}/{common_folder}",
        'line': f"{result_base_path}/{common_folder}/pred_lines",
        'edge': f"{result_base_path}/{common_folder}/pred_edges"
    }
    
    # Uncomment the lines below to create GIFs as needed
    # create_gif(directories['img'], "video.gif")
    # create_gif(directories['line'], "video_line.gif")
    # create_gif(directories['edge'], "video_edge.gif")

    create_gif_with_mask(gt_directories['img'], gt_directories['mask'], f"thesis_pic/{common_folder}_gt_masked.gif")
    # create_gif(result_directories['img'], f"thesis_pic/{common_folder}_pred_masked.gif")
    # create_gif_with_inpainting(gt_directories['edge'], gt_directories['mask'], result_directories['edge'], f"thesis_pic/{common_folder}_pred_edge_inpainting.gif")
    # create_gif_with_inpainting(gt_directories['line'], gt_directories['mask'], result_directories['line'], f"thesis_pic/{common_folder}_pred_line_inpainting.gif")

    # result_dir = "./results/1024_SERVI_finetune0926_l1HoleWeight/Final_result_thesis/Youtube_test_set1_smooth_mask/2023-10-26_line_25percent/8d55a5aebb"
    # create_gif(result_dir, "video_ours_8d55a5aebb_inpainted_result.gif")

    print("GIF Created")

if __name__ == "__main__":
    main()

