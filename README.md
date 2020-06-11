# Neural Video Compression
Our implementation of the paper <em>Learning Binary Residual Representations for Domain-specific Video Streaming by Yi-Hsuan et. al</em>. This work was done as a course project for CS348K: Visual Computing Systems at Stanford University, Spring 2020. Our report can be found here: https://drive.google.com/file/d/1sUxpa0e8DkQEYD929XsKOIJyHnqAE3f4/view?usp=sharing.

![Cover Pic](https://github.com/AdityaDusi97/NeuralVideoCompression/blob/master/Final_Pic.png)

## Requirement
- Python 3.6 +
- PyTorch 1.4

## Directory structure
- src: source code
- log: all tensorboard logs, organized by experiment names. Sub directory will be train/test
- data: main directory with pre-processed data. Sub directory will be data name with will further have train and test as sub directory
- checkpoints: organized experiment name wise, has both enc and dec checkpoints in the experiment subdirectory

## Checkpoint for quick evaluation on KITTI Vision Dataset
Our pre-trained checkpoints for the encoder and decoder can be found here: https://drive.google.com/drive/folders/1i_jg47WPfiH1pZkAWkQVk3-75qk_dW5h?usp=sharing

## Preparing dataset
There are useful functions in utils.py. Place the video frames in data/. Use the following commands in python:
```
import utils
import os
x = os.listdir('../raw/')
p = '../raw/'
for dir in x:
	utils.kittiResidual(p + dir + '/uncomp',p + dir + '/decomp', '../data/<name>', dir, 'png')
```

## Train command
```
./train.sh
```

## Test command
```
./eval.sh
```

## Command to evaluate MSE and SSIM
Functions are again provided in utils.py.
```
import utils
utils.metricCompute(<path to uncomp>, <path to decomp>, '../<output_dir name>', '<path to test output>')
```

The example command I use is:
```
utils.metricCompute('/home/ubuntu/NeuralVideoCompression/raw/kitti_2011_09_26_drive_0014_sync/uncomp', '/home/ubuntu/NeuralVideoCompression/raw/kitti_2011_09_26_drive_0014_sync/decomp', '../eval', '../test_out/0_01M/kitti_2011_09_26_drive_0014_sync')
```

### Contact
Feel free to write to use with questions, comments and feedback.
- Fang-Yu Lin (fangyuln@stanford.edu)
- Aditya Dusi (adusi@stanford.edu)


