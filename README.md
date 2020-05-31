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
python -m src.run --train_test train --data_root data/<data_name>/train --logging_root log --checkpoint_dir checkpoints --experiment_name <experiment_name>
```

## Test command
```
python -m src.run --train_test test --data_root data/<data_name>/test --logging_root log --checkpoint_enc checkpoints/<exp_name>/encoder-epoch_0_iter_0.pth --checkpoint_dec checkpoints/<exp_name>/decoder-epoch_0_iter_0.pth  --experiment_name oo  
```
There could be a better way to parse the checkpoints but I will fix it later

## Preparing dataset
There are useful functions in utils.py. Use the following commands in python:
```
import utils
utils.kittiResidual(<path_to_uncompressed>, <path_to_decompressed>, "NeuralVideoCompression/data", <dataset_name>)
```

The example command I used is:
```
utils.kittiResidual("/Users/aditya/Documents/cs348k/Project/NeuralVideoCompression/raw_data/2011_09_26/2011_09_26_drive_0001_sync/image_00/data", "/Users/aditya/Documents/cs348k/Project/NeuralVideoCompression/raw_data/2011_09_26/2011_09_26_drive_0001_sync/image_00/data/decomp", "/Users/aditya/Documents/cs348k/Project/NeuralVideoCompression/data", "kitti1")
```
