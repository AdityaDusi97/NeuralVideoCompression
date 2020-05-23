import numpy as np
import cv2
import os
import pdb
import torch 
import torch.nn as nn
import torchvision

from skimage.metrics import structural_similarity as ssim # perhaps do our own later on lol


# no labels

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample

def FrameLoader(data_path, batch_size=4):
    # For image data
    #data_path = 'data/train/'
    # dataloader = torchvision.datasets.ImageFolder(
    #     root=data_path,
    #     transform=torchvision.transforms.ToTensor()
    # )

    ### TODO: this is only for non-image data
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


def makeResidual(uncomp: str, decomp: str, out_dir: str, img_size = (1200,360)):
    uc = cv2.imread(uncomp).astype(np.float32)
    dc = cv2.imread(decomp).astype(np.float32)
    uc = cv2.resize(uc, img_size).astype(np.int16)
    dc = cv2.resize(dc, img_size).astype(np.int16) # to save space...
    res = np.transpose((uc - dc), axes=(2,0,1))
    name = (decomp.split('/')[-1]).split('.')[0]
    np.save(os.path.join(out_dir, "{}.npz".format(name)), res) # TODO: save to pos/neg files seperately???
    print("{}.npy saved to {}".format(name, out_dir))


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
    npy = tensor.cpu().numpy()
    npy = np.transpose(np.squeeze(npy), axes=(1,2,0))
    np.save(filename+'.npy', npy)

    cv2.imwrite(filename+'.png', (npy*128 + 128).astype(int))
