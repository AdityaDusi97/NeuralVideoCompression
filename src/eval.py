import os
from .utils import metricCompute
from .HuffmanCompression import getHuffmanBin, getHuffmanStat
from .dataset import test_video

mode       = 'first'
# mode       = 'second'
decomp_dir = 'decomp_1M'
out_root   = 'out'
res_root   = 'res_out'
exp_name   = '1M'
out_dir = os.path.join(out_root, exp_name)

comp_dir = ''

def main():
    # run decomp SSIM
    # run decomp with residual SSIM
   
    if mode == 'first':
        for v_dir in test_video:
            print('run {}'.format(v_dir))
            uncomp = os.path.join('raw', v_dir, 'uncomp')
            decomp = os.path.join('raw', v_dir, decomp_dir)
            res_dir = os.path.join(res_root, exp_name, v_dir)
            metricCompute(
                uncomp, 
                decomp, 
                out_dir, 
                res_dir, 
                mode='other',
                info=v_dir+'_decomp')
            
            metricCompute(
                uncomp, 
                decomp, 
                out_dir, 
                res_dir, 
                mode='residual',
                info=v_dir+'_res')

    # run huffman stat
    # then run correponding bitrate test
    elif mode == 'second':
        for v_dir in test_video:
            metricCompute(
                uncomp, 
                comp_dir, 
                out_root, 
                res_dir, 
                mode='other',
                info=v_dir+'_comp')
        
        

if __name__ == "__main__":
    main()

