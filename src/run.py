import argparse
import os, time, datetime
import torch
from torch import nn
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from .model import Encoder, Decoder
from .utils import FrameLoader, getSSIMfromTensor, saveTensorToNpy

from tensorboardX import SummaryWriter

# params
parser = argparse.ArgumentParser()

# path and modes
parser.add_argument('--train_test', type=str, required=True, help='mode')
parser.add_argument('--data_root', required=True, help='path to directory of images')
parser.add_argument('--logging_root', type=str, required=True,
                    help='path to save checkpoints')

# train params
parser.add_argument('--experiment_name', type=str, default='', help='name of experiment')
parser.add_argument('--max_epoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')

# for retraining
parser.add_argument('--checkpoint_enc', default=None, help='model to load')
parser.add_argument('--checkpoint_dec', default=None, help='model to load')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--batch_size', type=int, default=4, help='start epoch')

opt = parser.parse_args()

def train(models, dataset):
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

    dir_name = opt.experiment_name + 'testlol' # TODO: Add directory names
    
    # place to save checkpoints
    log_dir = os.path.join(opt.logging_root, 'logs', dir_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    #TODO: Need to add stuff for tensorboard X
    writer = SummaryWriter()

    ####  Deep learning part now
    enc.train()
    dec.train()
    optimizerE = torch.optim.Adam(enc.parameters(), lr=opt.lr)
    optimizerD = torch.optim.Adam(dec.parameters(), lr=opt.lr)
    criterion = nn.MSELoss() # from paper
    iter = opt.start_epoch * len(dataset)

    with torch.autograd.set_detect_anomaly(True):
        print('Beginning training...')
        for epoch in range(opt.start_epoch, opt.max_epoch):
            for idx, (model_input, _) in enumerate(dataset):
                optimizerE.zero_grad()
                optimizerD.zero_grad()
                ### TODO: fix training data shape:
                model_input = model_input.permute(0,3,1,2).type(torch.FloatTensor) / 256.0
                #############
                bin_out = enc(model_input)
                rec_img = dec(bin_out)
                loss = criterion(model_input, rec_img)
                loss.backward()
                optimizerD.step()
                optimizerE.step()
                print("Iter %07d   Epoch %03d   loss %0.4f" % (iter, epoch, loss))
                writer.add_scalar('train_loss', loss.cpu().numpy(), global_step=iter)
                iter +=1
                if iter % 500 ==0:
                    torch.save(enc.state_dict(), os.path.join(log_dir, 'encoder-epoch_%d_iter_%s.pth' % (epoch, iter)))
                    torch.save(dec.state_dict(), os.path.join(log_dir, 'decoder-epoch_%d_iter_%s.pth' % (epoch, iter)))  

    #TODO: Need to complete the code here
    print("Finished Training")
    torch.save(enc.state_dict(), os.path.join(log_dir, 'encoder-epoch_%d_iter_%s.pth' % (epoch, iter)))
    torch.save(dec.state_dict(), os.path.join(log_dir, 'decoder-epoch_%d_iter_%s.pth' % (epoch, iter)))


def test(models, dataset):
    """
    As this is similar to an autoencoder, data just comprises of images
    params:
        models: a tuple with encoder and decoder
        dataset: some class that has an iterator for elements
    """
    #### Setup
    enc, dec = models
    if opt.checkpoint_enc is not None and opt.checkpoint_dec is not None :
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
    with torch.no_grad():
        for idx, (model_input, _) in enumerate(dataset):
            ### TODO: fix training data shape:
            model_input = model_input.permute(0,3,1,2).type(torch.FloatTensor) / 256.0
            #############
            bin_out = enc(model_input)
            rec_img = dec(bin_out)

            ssim = getSSIMfromTensor(rec_img, model_input)
            saveTensorToNpy(rec_img, 'test_rec')

            loss = criterion(model_input, rec_img)
            if not idx%10:
                print(idx)
                print("Iter %07d  loss %0.4f ssim %0.6f" % (idx, loss, ssim))


def main():
    dataset = FrameLoader(opt.data_root, opt.batch_size)
    if opt.train_test == 'train':
        enc = Encoder()
        dec = Decoder()
        train((enc, dec), dataset)
    elif opt.train_test == 'test':
        enc = Encoder()
        dec = Decoder()
        test((enc, dec), dataset)
    else:
        print('Unknown Mode')
        return None

if __name__ == "__main__":
    main()


