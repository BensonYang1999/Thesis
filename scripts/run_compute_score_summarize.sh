
# python3 compute_score_summarize.py --input_dir ./results/2023-07-13_edge_75percent/2023-07-13_edge_25percent
# python3 compute_score_summarize.py --input_dir ./results/2023-07-13_edge_75percent/2023-07-13_edge_50percent
# python3 compute_score_summarize.py --input_dir ./results/2023-07-13_edge_75percent/2023-07-13_edge_25percent
# python3 compute_score_summarize.py --input_dir ./results/2023-07-13_edge_75percent/2023-07-13_edge_50percent

# python3 compute_score_summarize.py --root "../FuseFormer" --model fuseformer_5frames_youtube_results --date 2023-07-06
# python3 compute_score_summarize.py --root "../FuseFormer" --model fuseformer_5frames_youtube_results --date 2023-07-06 --onlyMask

# python3 compute_score_summarize.py --model 0717_ZITS_video_YoutubeVOS_max500k_mix458k_turn470k_prev_and_fixModelForward749 --date 2023-07-25 --cuda
# python3 compute_score_summarize.py --model 0710_ZITS_video_YoutubeVOS_max500k_mix458k_turn470k_frame-1_1_ReFFC_removed_last --date 2023-07-13

python3 compute_score_summarize.py --model 0717_ZITS_video_YoutubeVOS_max500k_mix458k_turn470k_prev_and_fixModelForward749/GT_line_edge --date 2023-08-28 --split valid
python3 compute_score_summarize.py --model 0717_ZITS_video_YoutubeVOS_max500k_mix458k_turn470k_prev_and_fixModelForward749/Inpainted_line_edge --date 2023-08-28 --split valid