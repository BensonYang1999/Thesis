# python3 evaluate_fiveFrames.py -c checkpoints/fuseformer_youtube-vos/gen_00050.pth -v ../ZITS_inpainting/datasets/YouTubeVOS_small/valid/JPEGImages/ -m ../ZITS_inpainting/datasets/YouTubeVOS_small/valid/mask_random/ --dataset line_gt0_sample10 --dump_results
# python3 evaluate_fiveFrames.py -c checkpoints/fuseformer_youtube-vos/gen_00050.pth -v ../ZITS_inpainting/datasets/YouTubeVOS_small/valid/JPEGImages/ -m ../ZITS_inpainting/datasets/YouTubeVOS_small/valid/mask_random/ --dataset line_gt50_sample10 --dump_results
# python3 evaluate_fiveFrames.py -c checkpoints/fuseformer_youtube-vos/gen_00050.pth -v ../ZITS_inpainting/datasets/YouTubeVOS_small/valid/JPEGImages/ -m ../ZITS_inpainting/datasets/YouTubeVOS_small/valid/mask_random/ --dataset line_gt100_sample10 --dump_results
# python3 evaluate_fiveFrames.py -c checkpoints/fuseformer_youtube-vos/gen_00050.pth -v ../ZITS_inpainting/datasets/YouTubeVOS_small/valid/JPEGImages/ -m ../ZITS_inpainting/datasets/YouTubeVOS_small/valid/mask_random/ --dataset line_gt150_sample10 --dump_results
# python3 evaluate_fiveFrames.py -c checkpoints/fuseformer_youtube-vos/gen_00050.pth -v ../ZITS_inpainting/datasets/YouTubeVOS_small/valid/JPEGImages/ -m ../ZITS_inpainting/datasets/YouTubeVOS_small/valid/mask_random/ --dataset line_gt200_sample10 --dump_results
# python3 evaluate_fiveFrames.py -c checkpoints/fuseformer_youtube-vos/gen_00050.pth -v ../ZITS_inpainting/datasets/YouTubeVOS_small/valid/JPEGImages/ -m ../ZITS_inpainting/datasets/YouTubeVOS_small/valid/mask_random/ --dataset edge_gt0_sample10 --dump_results
# python3 evaluate_fiveFrames.py -c checkpoints/fuseformer_youtube-vos/gen_00050.pth -v ../ZITS_inpainting/datasets/YouTubeVOS_small/valid/JPEGImages/ -m ../ZITS_inpainting/datasets/YouTubeVOS_small/valid/mask_random/ --dataset edge_gt2_sample10 --dump_results
# python3 evaluate_fiveFrames.py -c checkpoints/fuseformer_youtube-vos/gen_00050.pth -v ../ZITS_inpainting/datasets/YouTubeVOS_small/valid/JPEGImages/ -m ../ZITS_inpainting/datasets/YouTubeVOS_small/valid/mask_random/ --dataset edge_gt4_sample10 --dump_results
# python3 evaluate_fiveFrames.py -c checkpoints/fuseformer_youtube-vos/gen_00050.pth -v ../ZITS_inpainting/datasets/YouTubeVOS_small/valid/JPEGImages/ -m ../ZITS_inpainting/datasets/YouTubeVOS_small/valid/mask_random/ --dataset edge_gt6_sample10 --dump_results
# python3 evaluate_fiveFrames.py -c checkpoints/fuseformer_youtube-vos/gen_00050.pth -v ../ZITS_inpainting/datasets/YouTubeVOS_small/valid/JPEGImages/ -m ../ZITS_inpainting/datasets/YouTubeVOS_small/valid/mask_random/ --dataset edge_gt8_sample10 --dump_results
# python3 evaluate_fiveFrames.py -c checkpoints/fuseformer_youtube-vos/gen_00050.pth -v ../ZITS_inpainting/datasets/YouTubeVOS_small/valid/JPEGImages/ -m ../ZITS_inpainting/datasets/YouTubeVOS_small/valid/mask_random/ --dataset edge_gt10_sample10 --dump_results

python3 compute_score.py --input_dir ../FuseFormer/2023-07-03_gen_00050_scratch_edge_gt0_sample10_results
python3 compute_score.py --input_dir ../FuseFormer/2023-07-03_gen_00050_scratch_edge_gt2_sample10_results
python3 compute_score.py --input_dir ../FuseFormer/2023-07-03_gen_00050_scratch_edge_gt4_sample10_results
python3 compute_score.py --input_dir ../FuseFormer/2023-07-03_gen_00050_scratch_edge_gt6_sample10_results
python3 compute_score.py --input_dir ../FuseFormer/2023-07-03_gen_00050_scratch_edge_gt8_sample10_results
python3 compute_score.py --input_dir ../FuseFormer/2023-07-03_gen_00050_scratch_edge_gt10_sample10_results
python3 compute_score.py --input_dir ../FuseFormer/2023-07-03_gen_00050_scratch_line_gt50_sample10_results
python3 compute_score.py --input_dir ../FuseFormer/2023-07-03_gen_00050_scratch_line_gt100_sample10_results
python3 compute_score.py --input_dir ../FuseFormer/2023-07-03_gen_00050_scratch_line_gt150_sample10_results
python3 compute_score.py --input_dir ../FuseFormer/2023-07-03_gen_00050_scratch_line_gt200_sample10_results