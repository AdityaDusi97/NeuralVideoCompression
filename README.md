# Neural Video Compression

## Requirement
- Python 3.7
- PyTorch 1.4

## Directory structure
- src: source code
- log: all tensorboard logs, organized by experiment names. Sub directory will be train/test
- data: main directory with pre-processed data. Sub directory will be data name with will further have train and test as sub directory
- checkpoints: organized experiment name wise, has both enc and dec checkpoints in the experiment subdirectory

## Train command
```
./train.sh
```

## Test command
```
./eval.sh
```


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

