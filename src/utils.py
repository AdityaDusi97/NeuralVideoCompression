import numpy as np
import cv2
import os
import pdb
import random
import torch 
import torch.nn as nn
import torchvision
from glob import glob
from datetime import datetime
from skimage.metrics import structural_similarity as ssim # perhaps do our own later on lol
from skimage.measure import compare_ssim, compare_mse # familiar with this lol

def genResidualDataset(raw_path, data_path, uncomp='uncomp', decomp='decomp'):
    for dir in os.listdir(raw_path):
        p = os.path.join(raw_path, dir)
        if os.path.isdir(p):
            kittiResidual(os.path.join(p, uncomp), 
                          os.path.join(p, decomp), data_path, dir, 'png')


def kittiResidual(uncomp: str, decomp: str, out_dir:str, exp_name:str, fmt='npy')->None:
    """
    Function to go to a file and save residual
    params:
        uncomp: path to original frames
        decomp: path to decompressed frames
        out_dir: direcotry to save results in 
        exp_name: experiment name
    """
    out_dir = os.path.join(out_dir, exp_name)
    if not os.path.exists(os.path.join(out_dir, 'train')):
        os.makedirs(os.path.join(out_dir, 'train'))
    if not os.path.exists(os.path.join(out_dir, 'test')):
        os.makedirs(os.path.join(out_dir, 'test'))

    uncomp_frames = sorted(glob(os.path.join(uncomp, '*.png')))
    decomp_frames = sorted(glob(os.path.join(decomp, '*.png')))

    # need some check to make sure we are only calculating between correct frames
    limit = min(len(uncomp_frames), len(decomp_frames))
    test_idx = random.sample(list(range(limit)), int(limit*0.2))

    for idx in range(limit):
        if idx in test_idx:
            makeResidual(uncomp_frames[idx], decomp_frames[idx], os.path.join(out_dir,'test'), fmt=fmt)
        else:
            makeResidual(uncomp_frames[idx], decomp_frames[idx], os.path.join(out_dir,'train'), fmt=fmt)
    
    print("Written all frames")
    

def makeResidual(uncomp: str, decomp: str, out_dir: str, img_size = (1200,360), fmt='npy'):


    uc = cv2.imread(uncomp).astype(np.float32)
    dc = cv2.imread(decomp).astype(np.float32)
    # TODO: crop data first so that we don't need to resize
    # uc = cv2.resize(uc, img_size).astype(np.int16)
    # dc = cv2.resize(dc, img_size).astype(np.int16) # to save space
    res = (uc - dc)

    name = (decomp.split('/')[-1]).split('.')[0]
    if fmt == 'npy':
        res = np.transpose(res, axes=(2,0,1)) # make dim=(C,H,W)
        np.save(os.path.join(out_dir, "{}.npz".format(name)), res) # TODO: save to pos/neg files seperately???
        print("{}.npy saved to {}".format(name, out_dir))
    elif fmt == 'png':
        pos = np.zeros_like(res)
        neg = np.zeros_like(res)
        pos[res >= 0] = (res[res >= 0])
        neg[res < 0]  = -res[res < 0]
        cv2.imwrite(os.path.join(out_dir, "{}_pos.png".format(name)), pos.astype(np.uint8))
        cv2.imwrite(os.path.join(out_dir, "{}_neg.png".format(name)), neg.astype(np.uint8))
        print("{}_pos/neg.png saved to {}".format(name, out_dir))
    else:
        raise ValueError('wrong format: {}'.foramt(fmt))


def getSSIMfromTensor(tensor1, tensor2):
    """
        tensor1 and tensor2 are 1) 3 dim tensors or 2) 4 dim tensors with batchSize=1
    """
    img1 = np.transpose(np.squeeze(tensor1.cpu().numpy()), axes=(1,2,0))
    img2 = np.transpose(np.squeeze(tensor2.cpu().numpy()), axes=(1,2,0))

    return ssim(img1, img2,
                data_range=img2.max() - img2.min(),
                multichannel=True)

def saveTensorToNpy(tensor, filename):
    npy = np.clip(tensor.cpu().numpy(), -1, 1) # -1 ~ 1
    npy = np.squeeze(npy)
    #npy = np.transpose(np.squeeze(npy)#, axes=(1,2,0)) 
    np.save(filename+'.npy', npy)

    # cv2.imwrite(filename+'.png', (npy*128 + 128).astype(int))

def metricCompute(uncomp: str, decomp: str, out_dir: str, residual_dir: str, mode="residual", info=None)->None:
    """
    mode: residual: to use residual, else does on raw frames
    """
    if info is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = os.path.join(out_dir, timestamp)
    else:
        out_dir = os.path.join(out_dir, info)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("Making output directory")
    
    filePath = os.path.join(out_dir, 'log.txt')
    file = open(filePath, 'w+')
    file.write("Video SSIM and MSE\n")
    # file.write("Quality: %d" %(quality))
    
    #pdb.set_trace()
    uncomp_frames = sorted(glob(os.path.join(uncomp, '*.png')))
    decomp_frames = sorted(glob(os.path.join(decomp, '*.png')))
    residual_frames = None
    if mode=='residual':
        residual_frames = sorted(glob(os.path.join(residual_dir, '*.npy')))
        # incase different things for each were used
        limit = min(len(uncomp_frames), len(decomp_frames), len(residual_frames))
    else:
        limit = min(len(uncomp_frames), len(decomp_frames))

    residual_im = 0
    saveName = os.path.join(out_dir, "Frame")

    avg_ssim = 0
    avg_mse = 0
    for i in range(limit):
        uncomp_im = cv2.imread(uncomp_frames[i])
        decomp_im = cv2.imread(decomp_frames[i])

        if mode == 'residual':
            #pdb.set_trace()
            residual_im = np.load(residual_frames[i]) * 255.0
            residual_im = np.transpose(residual_im.astype(np.int32), axes=(1,2,0))
            
        rec_im = np.clip(decomp_im + residual_im, 0, 255)
        ssim_value = compare_ssim(uncomp_im, rec_im, multichannel=True, full=False, gradient=False)
        mse_value = compare_mse(uncomp_im, rec_im)

        avg_ssim += ssim_value
        avg_mse += ssim_value

        if mode == 'residual':
            cv2.imwrite(saveName + str(i) + ".png", rec_im)

        file.write("Frame {}: SSIM {} , MSE {} \n".format(i, ssim_value, mse_value))

    file.write("Average: SSIM {} , MSE {} \n".format(avg_ssim/limit, avg_mse/limit))
    file.close()

####### Function Archive #########

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample

def FrameLoader(data_path, batch_size=4):
    ## For image data
    # data_path = 'data/train/'
    # dataloader = torchvision.datasets.ImageFolder(
    #     root=data_path,
    #     transform=torchvision.transforms.ToTensor()
    # )

    ## For non-image data
    dataloader = torchvision.datasets.DatasetFolder(
        root=data_path, loader=npy_loader, 
        extensions=('.npy')
        # transform=torchvision.transforms.ToTensor()
    )
    ### 
    dataset = torch.utils.data.DataLoader(
        dataloader,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        drop_last=True
    )
    return dataset

# TODO: Some random prints. Need to fix that
def Video2Frames(readfile: str, savefile: str) -> None:
    video = cv2.VideoCapture(readfile)
    status = True
    count = 0
    while status:
        status, frame = video.read()
        frame = np.transpose(frame, (1, 0, -1))
        saveName = os.path.join(savefile, "frame" + str(count) + ".png")
        cv2.imwrite(saveName, frame)
        count += 1
    print("Wrote Frames\n")

def Video2Residual(readpath: str, savePath: str, quality: int = 1) -> None:
    """
    params: 
    	readpath: path to video
    	savePath: path to save a directory with video and logs
    	qualityL 0-100, 100 being best quality
    """
    video = cv2.VideoCapture(readpath)
    status = True
    count = 0
    videoPath = os.path.join(savePath, "data")
    if not os.path.exists(videoPath):
        os.makedirs(videoPath)
        print('Creating output directory')
    else:
        print('Rewriting output directory')
    filePath = os.path.join(savePath, 'log.txt')
    file = open(filePath, 'w+')
    file.write("Source Video: " + readpath + "\n")
    file.write("Quality: %d" %(quality))
    file.close()

    while status:
        #pdb.set_trace()
        status, frame = video.read()
        encoded, encodedFrame = cv2.imencode('.jpg', frame, [quality, 0])
        decodedFrame = cv2.imdecode(encodedFrame, cv2.IMREAD_COLOR)
        saveName = os.path.join(videoPath, "Frame" + str(count) + '.png')
        residual = frame - decodedFrame
        count +=1
        # format in in BGR
        cv2.imwrite(saveName, residual)
    print('Wrote Frames')


