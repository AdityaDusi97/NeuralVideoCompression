import argparse
import os, time, datetime
import torch
from torch import nn
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from .model import Encoder, Decoder
from .utils import FrameLoader, saveTensorToNpy
from .dataset import  ResidualDataset
import pdb

from tensorboardX import SummaryWriter
from .HuffmanCompression import HuffmanCoding

# params
parser = argparse.ArgumentParser()

# path and modes
parser.add_argument('--train_test', type=str, required=True, help='mode')
parser.add_argument('--data_root', required=True, help='path to directory of images')
parser.add_argument('--logging_root', type=str, required=True,
                    help='path to save checkpoints')
parser.add_argument('--checkpoint_dir', type=str, default="checkpoint", 
                    help='path to directory with experiments checkpoints ')
parser.add_argument('--test_output_dir', type=str, help='path to save test output')

# train params
parser.add_argument('--experiment_name', type=str, required=True, help='name of experiment')
parser.add_argument('--max_epoch', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--lf', type=int, default=50, help='logging frequency')
parser.add_argument('--sf', type=int, default=200, help='checkpoints saving frequency')
parser.add_argument('-vf', type=int, default=10, help='val during train frequency')

# for retraining
parser.add_argument('--checkpoint_enc', default=None, help='model to load')
parser.add_argument('--checkpoint_dec', default=None, help='model to load')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--batch_size', type=int, default=4, help='start epoch')

opt = parser.parse_args()

# device = torch.device('cpu')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train(models, dataset_train, dataset_val):
    """
    As this is similar to an autoencoder, data just comprises of images
    params:
        models: a tuple with encoder and decoder
        dataset: some class that has an iterator for elements
    """
    enc, dec = models
    # for retraining
    if opt.checkpoint_enc is not None and opt.checkpoint_dec is not None :
        enc.load_state_dict(torch.load(opt.checkpoint_enc))
        dec.load_state_dict(torch.load(opt.checkpoint_dec))  

    dir_name = opt.experiment_name #+ 'train' # TODO: Add directory names
    
    # place to save tensorboard logs
    log_dir = os.path.join(opt.logging_root, dir_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(os.path.join(log_dir, "train"))

    # place to save checkpoints
    ckpt_dir = os.path.join(opt.checkpoint_dir, opt.experiment_name)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    ####  Deep learning part now
    enc.train()
    dec.train()
    optimizerE = torch.optim.Adam(enc.parameters(), lr=opt.lr)
    optimizerD = torch.optim.Adam(dec.parameters(), lr=opt.lr)
    criterion = nn.MSELoss() # from paper
    iter = opt.start_epoch * len(dataset_train)

    with torch.autograd.set_detect_anomaly(True):
        print('Beginning training...')
        for epoch in range(opt.start_epoch, opt.max_epoch):
            # for idx, (sample, _) in enumerate(dataset):
            for idx, sample in enumerate(dataset_train):
                optimizerE.zero_grad()
                optimizerD.zero_grad()
                # ### TODO: fix training data shape:
                # model_input = model_input.type(torch.FloatTensor) / 255.0
                # #############
                model_input = sample['image']
                bin_out = enc(model_input)
                rec_img = dec(bin_out)
                loss = criterion(model_input, rec_img)
                loss.backward()
                optimizerD.step()
                optimizerE.step()
                if iter % opt.lf == 0:
                    print("Iter %07d   Epoch %03d   loss %0.4f" % (iter, epoch, loss))
                    writer.add_scalar('train_loss', loss.detach().numpy(), global_step=iter)

                if iter % opt.sf ==0:
                    torch.save(enc.state_dict(), os.path.join(ckpt_dir, 'encoder-epoch_%d_iter_%s.pth' % (epoch, iter)))
                    torch.save(dec.state_dict(), os.path.join(ckpt_dir, 'decoder-epoch_%d_iter_%s.pth' % (epoch, iter)))

                if iter % opt.vf == 0:
                    print("Evaluating")
                    test((enc, dec), dataset_val, mode="val")

                iter +=1

    #TODO: Need to complete the code here
    print("Finished Training")
    torch.save(enc.state_dict(), os.path.join(ckpt_dir, 'encoder-epoch_%d_iter_%s.pth' % (epoch, iter)))
    torch.save(dec.state_dict(), os.path.join(ckpt_dir, 'decoder-epoch_%d_iter_%s.pth' % (epoch, iter)))


def test(models, dataset, mode='val'):
    """
    As this is similar to an autoencoder, data just comprises of images
    params:
        models: a tuple with encoder and decoder
        dataset: some class that has an iterator for elements
    """
    #### Setup
    enc, dec = models
    #pdb.set_trace()
    if mode == 'val':
        print("Eval in train")
    elif opt.checkpoint_enc is not None and opt.checkpoint_dec is not None:
        enc.load_state_dict(torch.load(opt.checkpoint_enc))
        dec.load_state_dict(torch.load(opt.checkpoint_dec))
    else:
        print("Have to give checkpoint!")
        return

    enc.eval()
    dec.eval()

    dir_name = 'test_result' #TODO: Fill in directory name

    print('Beginning evaluation...')
    criterion = nn.MSELoss() # from paper


    out_dir = os.path.join(opt.test_output_dir, opt.experiment_name)
    if mode != 'val':
        if not os.path.exists(out_dir):
        	os.makedirs(out_dir)
    out_name = os.path.join(out_dir, "Frame_")
    iter =0
    with torch.no_grad():
        for idx, sample in enumerate(dataset):
            ### TODO: fix training data shape:
            model_input = sample['image']
            #############
            bin_out = enc(model_input)
            rec_img = dec(bin_out)
            
            #ssim = getSSIMfromTensor(rec_img, model_input)
            loss = criterion(model_input, rec_img)
            if not idx%10:
                print(idx)
                print("Iter %07d  loss %0.4f ssim %0.6f" % (idx, loss, ssim))

                if mode == 'val':
                    # only called in the train loop
                    writer.add_scalar('val_loss', loss.detach().numpy(), global_step=iter)

            if mode != 'val':
                saveTensorToNpy(rec_img, out_name + str(idx))

            iter +=1

                


"""
At test time, out pipeline should actually be split into two stages:
 1. 
    a) Encode frames with h246 and get residual
    b) Huffman compress residual
 
 2.
    a) huffman decompress residual
    b) Decode h246 frame and add residual back
"""
## VERY IMPORTANT ##
# at test time, batch size should be one!
def test_encode(enc, dataset):
    if opt.checkpoint_enc is not None:
        enc.load_state_dict(torch.load(opt.checkpoint_enc))
    else:
        print("Have to give checkpoint!")
        return
    enc.eval()

    dir_name = 'test_encoded_frames'
    # TODO: look into the path thing and all

    print("Test Time encoding frames and getting residuals")
    #criterion = nn.MSELoss()
    pdb.set_trace()
    residuals_path = os.path.join(dir_name, "residuals")
    if not os.path.exists(residuals_path):
        os.makedirs(residuals_path)
    
    residual_file = os.path.join(residuals_path, "residuals.txt")
    output_file = open(residual_file, "w")

    hCompressor = HuffmanCoding()
    with torch.no_grad():
        for idx, (model_input, _) in enumerate(dataset):
            if idx==1:
                break
            model_input = model_input.type(torch.FloatTensor) / 255.0
            bin_out = enc(model_input).squeeze().cpu().numpy() # assuming batch dimension will get squeezed
            
            # convert it .txt file and then .bin for huffman
            for row in bin_out:
                np.savetxt(output_file, row)
            print("wrote ", idx , " to file")
        # now, let's do huffman coding
        hCompressor.compress(residual_file, residuals_path)
        # saves things as residual.bin
    print(bin_out.shape)
    print("Finished Huffman Coding residuals")
    # I will also return dimensions as that will be useful at decode time
    return bin_out.shape #tuple

def test_decode(dec, dataset, bin_out_shape):
    pdb.set_trace()
    if opt.checkpoint_enc is not None:
        dec.load_state_dict(torch.load(opt.checkpoint_dec))
    else:
        print("Have to give checkpoint!")
        return
    dec.eval()

    dir_name = 'test_encoded_frames'
    residuals_path = os.path.join(dir_name, "residuals")
    if not os.path.exists(residuals_path):
        print("File hasn't been encoded")
        return
        
    hCompressor = HuffmanCoding()
    residual_file = os.path.join(residuals_path, "residuals.bin")
    hCompressor.decompress(residual_file, residuals_path)
    residual_rec = os.path.join(residuals_path, "rec_residuals.txt")
    rec_bin_out = torch.tensor(np.loadtxt(residual_file).reshape(-1, *bin_out_shape)) # all frames

    dir_name = 'test_result'
    print("At deocoder end")
    criterion = nn.MSELoss() # from paper
    with torch.no_grad():
        for idx, (model_input, _) in enumerate(dataset):
            ### TODO: fix training data shape:
            if idx==1:
                break
            model_input = model_input.type(torch.FloatTensor) / 255.0
            #############
            #bin_out = enc(model_input)
            rec_img = dec(rec_bin_out[idx])
            print("decoded ", idx)
            ssim = getSSIMfromTensor(rec_img, model_input)
            saveTensorToNpy(rec_img, 'test_rec')

            loss = criterion(model_input, rec_img)
            if not idx%10:
                print(idx)
                print("Iter %07d  loss %0.4f ssim %0.6f" % (idx, loss, ssim))



def main():

    # dataset = FrameLoader(opt.data_root, opt.batch_size)
    dSet = ResidualDataset(opt.data_root, opt.train_test, device)
    # TODO: FU: add test, val, train modes etc for data
    # Test and val same here
    dataset = torch.utils.data.DataLoader( dSet,
                                           batch_size=opt.batch_size, shuffle=True,
                                           num_workers=0)

    if opt.train_test == 'train':
        enc = Encoder()
        dec = Decoder()
        enc.to_device(device)
        dec.to_device(device)
        train((enc, dec), dataset_train, dataset_test)
    elif opt.train_test == 'test':
        enc = Encoder()
        dec = Decoder()
        enc.to_device(device)
        dec.to_device(device)
        test((enc, dec), dataset_test, mode="test")
        ## This is the new function, will uncomment later
        # bin_out_shape = test_encode(enc, dataset)
        # pdb.set_trace()
        # test_decode(dec, dataset, bin_out_shape)
        

    else:
        print('Unknown Mode')
        return None

if __name__ == "__main__":
    main()


