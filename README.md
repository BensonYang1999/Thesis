# SERVI: Structure Enhanced Regional Video Inpainting
by [MuXi Chen](https://github.com/ChenMuHsi), YuChee Tseng, YenAnn Chen

> [!IMPORTANT]  
> This work is referencing to [ZITS](https://github.com/DQiaole/ZITS_inpainting) and [FuseFormer](https://github.com/ruiliu-ai/FuseFormer)

## Dataset
### 1. YoutubeVOS
[Official Website](https://youtube-vos.org)

### 2. DAVIS
[Official Website](https://davischallenge.org)

## Pretrain model
The downloaded files should be organized as a folder under ./ckpt
OneDrive Link: [1024_SERVI](https://1drv.ms/f/s!AuoSU7-7YWU1hbAOOs1G8sgpSJTpSQ?e=v6djZ7)

## Train
### TSR model
* Template
    ```
    python3 TSR_train_video.py --name <model_name> --dataset_root ./datasets --dataset_name <YouTubeVOS/DAVIS> --batch_size 4 --train_epoch 100 --loss_hole_valid_weight 0.8 0.2 --GPU_ids 0 --loss_choice bce
    ```
* Example:
    ```
    python3 TSR_train_video.py --name 1019_ZITS_video_YouTubeVOS_256_256_08hole_02valid_1edge_1line_minMaxNorm_oldEdge_bs2_bce   --dataset_root ./datasets --dataset_name YouTubeVOS --batch_size 4 --train_epoch 100 --loss_hole_valid_weight 0.8 0.2 --GPU_ids 0 --loss_choice bce
    ```

### FTR model
* Template
    ```
    python3 FTR_train_video.py --model_name <model_name> --DDP
    ```
* Example:
    ```
    python3 FTR_train_video.py --model_name 1024_SERVI_finetune0926_l1HoleWeight --DDPe
    ```

## Inference

### TSR model
* Template
    ```
    python3 TSR_inference_video.py --GPU_ids 0 --ckpt_path <ckpt_dir> --dataset_root ./ --dataset_name <data_foler_name> --iterations 1 --save_url <save_folder>
    ```
* Example:
    ```
    python3 TSR_inference_video.py --GPU_ids 0 --ckpt_path ./ckpt/0521_ZITS_video_YouTubeVOS_08hole_02valid_1edge_1line_minMaxNorm_oldEdge_bs2_bce/best.pth --dataset_root ./ --dataset_name 1002_pic --save_url 1002_pic
    ```

### FTR model
* Template
    ```
    python3 FTR_inference_video.py --path <ckpt_folder> --input <data_folder_name> --output <save_folder>
    ```

* Example:
    ```
    python3 FTR_inference_video.py --path ckpt/1024_SERVI_finetune0926_l1HoleWeight --input davis --output DAVIS_all
    ```
> [!NOTE]  
> The testing data reading procedure is wrote in src/utils.py with the function get_frame_mask_edge_line_list (It's Still Ugly)

## Evaluation
### TSR model

### FTR model
Evaluate the metrics (PSNR/SSIM/LPIPS/VFID/VIF) between inpainted video and GT video
* Template
    ```
    python3 compute_score_summarize.py --input_dir <inpainting_result_folder>
    ```

* Example:
    ```
    python3 compute_score_summarize.py --input_dir ./results/0819_ZITS_video_YoutubeVOS_max100k_mix91k_turn94k_prev_and_fixModelForward749_fixEvalLineEdge_fixMaskEdgeLine/2023-08-20_edge_75percent
    ```