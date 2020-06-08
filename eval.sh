python -m src.run --train_test test \
                  --data_root  data_1M \
                  --logging_root log \
                  --checkpoint_dir checkpoints \
                  --test_output_dir res_out \
                  --experiment_name 1M \
                  --sf 1000 \
                  --vf 1000 \
                  --checkpoint_enc checkpoint/1M/encoder-epoch_49_iter_20300.pth \
                  --checkpoint_dec checkpoint/1M/decoder-epoch_49_iter_20300.pth \
                  --batch_size 1

# 1. run re_process.sh if corresponding data is not generated yet
# 2. generate residual data to generate corresponding residuals (python)
#  for dir in test_videos:
#         p = os.path.join(<raw_path>, dir)
#         if os.path.isdir(p):
#             kittiResidual(os.path.join(p, 'uncomp'), 
#                           os.path.join(p, <decomp>), <data_path>, dir, 'png')
# 3. run this script. remember to change experiment name. this will save network residuals to npy
# 4. run eval.py
# 5. run huffman test, get correponding bitrate
# 5. generate corresponding comp.h264 --> decomp.mp4 --> .png
# 6. get ssim using eval.py (change mode)
