import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import network
import morphology
import os
import math
import torch.nn as nn
import torch.nn.init as init
import torchvision
from tqdm import tqdm
from PIL import ImageFont
from PIL import ImageDraw 
import json
import torchvision.transforms as transforms
from torchvision.utils import make_grid


idx = 0
EPS = 1e-1

    
PAD_INFO = {
    1: (0, 256),
    2: (0, 128), 
    3: (4, 252//3), 
    4: (0, 256//4), 
    5: (6, 250//5),
    6: (4, 252//6)
}    

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

class BetaVAE_B_256(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=10, nc=1):
        super(BetaVAE_B_256, self).__init__()
        self.nc = nc
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            View((-1, 32*4*4)),                  # B, 512
            nn.Linear(32*4*4, 256),              # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32*4*4),              # B, 512
            nn.ReLU(True),
            View((-1, 32, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1), # B,  nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, mu, logvar
    
    def sample(self,z):
        x_recon = self._decode(z)
        x_recon = torch.sigmoid(x_recon)
        return x_recon

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)
    



class BetaVAE_B_256_Conditional_Path(nn.Module):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim=5, c_dim = 5, nc=1, encoder = '1DCNN', decoder='1DCNN', n_features = 8, seq_length = 300, mask_gen = '1DCNN', path_type = 'jointpath', device = 'cpu'):
        super(BetaVAE_B_256_Conditional_Path, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.path_type = path_type
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        ### IMAGE ENCODER
        print('loaded 2DCNN img encoder')
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(nc+1, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            View((-1, 32*4*4)),                  # B, 512
            nn.Linear(32*4*4, 256),              # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 128),             # B, z_dim*2
        )

        ### PATH ENCODER
        if encoder=='1DCNN':
            print('loaded 1DCNN path encoder')
            self.path_encoder = nn.Sequential(
                nn.Conv1d(n_features,32,4,2,1),
                nn.ReLU(True),
                nn.Conv1d(32,32,4,2,1),
                nn.ReLU(True),
                nn.Conv1d(32,32,4,2,1),
                nn.ReLU(True),
                nn.Conv1d(32,32,4,2,1),
                nn.ReLU(True),
                nn.Conv1d(32,32,4,2,1),
                nn.ReLU(True),
                nn.Conv1d(32,32,4,2,1),
                nn.ReLU(True),
                View((-1,32*4))
            )

        elif encoder=='RNN':
            print('loaded RNN path encoder')
            self.path_encoder = nn.Sequential(Encoder(number_of_features = n_features,
                                hidden_size=128,
                                hidden_layer_depth=3,
                                latent_length=30,
                                dropout=0.05,
                                block='LSTM')
            )

        ### IMAGE-PATH TO LATENT ENCODER
        self.encoder_linear = nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU(True),
            nn.Linear(256,z_dim*2)
        )

        ### GRID ENCODER
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            View((-1, 32*4*4)),                  # B, 512
            nn.Linear(32*4*4, 256),              # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, c_dim),             # B, z_dim*2
        )

        self.img_decoder = nn.Sequential(
            nn.Linear(z_dim+c_dim, 256),               # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32*4*4),              # B, 512
            nn.ReLU(True),
            View((-1, 32, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1), # B,  nc, 64, 64
        )
        
        ### PATH DECODER
        if decoder=='1DCNN':
            print('loaded 1DCNN path decoder')
            self.path_decoder = nn.Sequential(
                nn.Linear(z_dim+c_dim, 256),               # B, 256
                nn.ReLU(True),
                nn.Linear(256, 256),                 # B, 256
                nn.ReLU(True),
                nn.Linear(256, 32*4),              # B, 512
                nn.ReLU(True),
                View((-1, 32, 4)),                # B,  32,  4,  4
                nn.ConvTranspose1d(32, 32, 4, 3, 3), # B,  32,  8,  8
                nn.ReLU(True),
                nn.ConvTranspose1d(32, 32, 4, 3, 2), # B,  32, 32, 32
                nn.ReLU(True),
                nn.ConvTranspose1d(32, 32, 4, 2, 1), # B,  32, 32, 32
                nn.ReLU(True),
                nn.ConvTranspose1d(32, 32, 4, 2, 0), # B,  32, 32, 32
                nn.ReLU(True),
                nn.ConvTranspose1d(32, 32, 4, 2, 0), # B,  32, 32, 32
                nn.ReLU(True),
                nn.ConvTranspose1d(32, n_features, 4, 2, 1), # B,  nc, 64, 64
            )

        elif decoder=='RNN':
            print('loaded RNN path decoder')
            self.path_decoder = nn.Sequential(Decoder(sequence_length=seq_length,
                                batch_size =32,
                                hidden_size=128,
                                hidden_layer_depth=3,
                                latent_length=z_dim + c_dim,
                                output_size=n_features,
                                block='LSTM',
                                dtype=torch.cuda.FloatTensor)
            )

        ### MASK GENERATOR
        if mask_gen == 'MLP':
            print('loaded MLP mask decoder')
            self.mask_decoder = nn.Sequential(
                nn.Linear(z_dim+c_dim,256),
                nn.ReLU(True),
                nn.Linear(256,300),
                nn.Tanh()
            )  
        elif mask_gen == '1DCNN':
            print('loaded 1DCNN mask decoder')
            self.mask_decoder = nn.Sequential(
                nn.Linear(z_dim+c_dim, 256),               # B, 256
                nn.ReLU(True),
                nn.Linear(256, 256),                 # B, 256
                nn.ReLU(True),
                nn.Linear(256, 32*4),              # B, 512
                nn.ReLU(True),
                View((-1, 32, 4)),                # B,  32,  4,  4
                nn.ConvTranspose1d(32, 32, 4, 3, 3), # B,  32,  8,  8
                nn.ReLU(True),
                nn.ConvTranspose1d(32, 32, 4, 3, 2), # B,  32, 32, 32
                nn.ReLU(True),
                nn.ConvTranspose1d(32, 32, 4, 2, 1), # B,  32, 32, 32
                nn.ReLU(True),
                nn.ConvTranspose1d(32, 32, 4, 2, 0), # B,  32, 32, 32
                nn.ReLU(True),
                nn.ConvTranspose1d(32, 32, 4, 2, 0), # B,  32, 32, 32
                nn.ReLU(True),
                nn.ConvTranspose1d(32, 1, 4, 2, 1), # B,  nc, 64, 64
                View((-1,300))
            )
        elif mask_gen == 'RNN':
            print('loaded RNN mask decoder')
            self.mask_decoder = nn.Sequential(Decoder(sequence_length=seq_length,
                                batch_size =32,
                                hidden_size=128,
                                hidden_layer_depth=3,
                                latent_length=z_dim+c_dim,
                                output_size=1,
                                block='LSTM',
                                dtype=torch.cuda.FloatTensor),
                                nn.Tanh(),
                                View((-1,300))
            )
        
        self.weight_init()     
        
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)
                
    def forward(self, img, path, grid):
        # img_feature = self.conv_encoder(x)
        # t = torch.cat((img_feature, c),dim=1)
        distributions = self._encode(img, path, grid)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        # t2 = torch.cat((z,c),dim=1)
        img_recon, path_recon, mask_recon = self._decode(z,grid)

        return img_recon, path_recon, mask_recon, mu, logvar
    
    def sample(self,z,grid):
        img_recon, path_recon, mask_recon = self._decode(z,grid)
        img_recon = torch.sigmoid(img_recon)
        return img_recon, path_recon, mask_recon

    def _encode(self, img, path, grid):
        concat = torch.cat([img,grid],dim=1)
        img_feature = self.encoder_conv(concat)
        if self.encoder=='RNN':
            path = path.permute(2,0,1)
        path_feature = self.path_encoder(path)
        t = torch.cat([img_feature, path_feature],dim=1)
        distributions = self.encoder_linear(t)
        return distributions

    def _decode(self, z,grid):
        grid_feature = self.grid_encoder(grid)
        print(grid_feature.shape, z.shape)
        t = torch.cat([z, grid_feature],dim=1)

        recon_img = self.img_decoder(t)
        # print(z.shape)
        recon_path = self.path_decoder(t)
        if self.decoder=='RNN':
            recon_path = recon_path.permute(1,2,0)
        
        recon_mask = self.mask_decoder(t)
        # print(recon_mask.shape, recon_path.shape)
        recon_path = (recon_mask>0).float().unsqueeze(1) * recon_path
        if self.path_type =='jointpath':
            temp = -torch.ones(recon_path.shape)*2
        elif self.path_type=='strokepath':
            temp = -torch.ones(recon_path.shape)*1
        else:
            raise ValueError('A very specific bad thing happened.')
        recon_path = recon_path + (recon_mask<=0).float().unsqueeze(1) * temp.to(self.device)
        # recon_length = self.length_decoder(z)
        # print(recon_img.shape, recon_path.shape)
        # print(recon_path.shape, recon_length.shape)
        # mask = torch.zeros(recon_path.shape[0],9,300).cuda()
        # idxs = torch.argmax(recon_length, dim=1)
        # for i in range(recon_path.shape[0]):
        #     mask [i,:,:int(recon_length[i]*300)] = 1
        # print(recon_path.shape, mask.shape)
        # recon_path = recon_path
        return recon_img, recon_path, recon_mask

class Encoder(nn.Module):
    """
    Encoder network containing enrolled LSTM/GRU

    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param dropout: percentage of nodes to dropout
    :param block: LSTM/GRU block
    """
    def __init__(self, number_of_features, hidden_size, hidden_layer_depth, latent_length, dropout, block = 'LSTM'):

        super(Encoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length

        if block == 'LSTM':
            self.model = nn.LSTM(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout)
        elif block == 'GRU':
            self.model = nn.GRU(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout)
        else:
            raise NotImplementedError

    def forward(self, x):
        """Forward propagation of encoder. Given input, outputs the last hidden state of encoder

        :param x: input to the encoder, of shape (sequence_length, batch_size, number_of_features)
        :return: last hidden state of encoder, of shape (batch_size, hidden_size)
        """

        _, (h_end, c_end) = self.model(x)

        h_end = h_end[-1, :, :]
        return h_end


class Lambda(nn.Module):
    """Lambda module converts output of encoder to latent vector

    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size, latent_length):
        super(Lambda, self).__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance

        :param cell_output: last hidden state of encoder
        :return: latent vector
        """

        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)

        # if self.training:
        #     std = torch.exp(0.5 * self.latent_logvar)
        #     eps = torch.randn_like(std)
        #     return eps.mul(std).add_(self.latent_mean)
        # else:
        #     return self.latent_mean
        return self.latent_mean, self.latent_logvar

class Decoder(nn.Module):
    """Converts latent vector into output

    :param sequence_length: length of the input sequence
    :param batch_size: batch size of the input sequence
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param output_size: 2, one representing the mean, other log std dev of the output
    :param block: GRU/LSTM - use the same which you've used in the encoder
    :param dtype: Depending on cuda enabled/disabled, create the tensor
    """
    def __init__(self, sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length, output_size, dtype, block='LSTM'):

        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.output_size = output_size
        self.dtype = dtype

        if block == 'LSTM':
            self.model = nn.LSTM(1, self.hidden_size, self.hidden_layer_depth)
        elif block == 'GRU':
            self.model = nn.GRU(1, self.hidden_size, self.hidden_layer_depth)
        else:
            raise NotImplementedError

        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)

        self.decoder_inputs = torch.zeros(self.sequence_length, self.batch_size, 1, requires_grad=True).type(self.dtype)
        self.c_0 = torch.zeros(self.hidden_layer_depth, self.batch_size, self.hidden_size, requires_grad=True).type(self.dtype)

        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, latent):
        """Converts latent to hidden to output

        :param latent: latent vector
        :return: outputs consisting of mean and std dev of vector
        """
        h_state = self.latent_to_hidden(latent)

        if isinstance(self.model, nn.LSTM):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(self.decoder_inputs, (h_0, self.c_0))
        elif isinstance(self.model, nn.GRU):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(self.decoder_inputs, h_0)
        else:
            raise NotImplementedError

        out = self.hidden_to_output(decoder_output)
        return out

    
    
def save_img(img, output_path, name):
    # print((img.data.cpu().numpy().transpose((1, 2, 0))).shape)
    result = Image.fromarray((img.data.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8), mode='CMYK').convert('RGB')
    draw = ImageDraw.Draw(result)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    font = ImageFont.truetype("./arial.ttf", 16)
    # draw.text((x, y),"Sample Text",(r,g,b))
    draw.text((0, 0),name,(0,0,0),font=font)
    # img.save('sample-out.jpg')
    result.save(os.path.join(output_path, name))

def resize_stroke(img, gridnum, stroke_num, device, frame_dir, img_size = 256, visualize_grid=False):
    if gridnum==1:
        return img
    else:
        padding, gridsize = PAD_INFO[gridnum]
        total_gridnum = gridnum**2
        
        transformed = []
        
        for i in range(img.shape[0]): 
            gridindex = i//stroke_num
            temp = torch.zeros(total_gridnum,1,gridsize,gridsize).to(device)
            TrnPad = transforms.Pad(padding//2)
            TrnGray = transforms.Grayscale(num_output_channels=1)
            TrnResize = transforms.Resize([gridsize, gridsize])
            resized_stroke = TrnResize(img[i])
            temp[gridindex] = resized_stroke

            final = make_grid(temp, nrow=gridnum, padding=int(visualize_grid),pad_value=1)
            final = TrnPad(TrnGray(final))
            save_img(final.repeat(4,1,1), frame_dir+"/validation", "gridnum_{}_gridindex_{:02d}.png".format(gridnum, gridindex))
            transformed.append(final)
        
        transformed = torch.stack(transformed, 0)

        return transformed
    

def latent2stroke_bg(param, H,W, background_vae, background_info):

    # param: b, 10 (latent) + 3 (RGB)
    trn_resize = transforms.Resize([H,W])
    trn_ToPIL = transforms.ToPILImage(mode='CMYK')
    trn_ToTensor = transforms.ToTensor()

    ## param : grid x stroke_num, 10 (5 determining the direction of unit vector, 1 for z norm (0~2), 4 for CMYK color)
    if background_info['painter_nparam']==9:
        sigma = 2
        param_latent = (param[:,:5] / torch.norm(param[:,:5],dim=1).unsqueeze(1))*sigma
    elif background_info['painter_nparam']==10:
        param_latent = (param[:,:5] / torch.norm(param[:,:5],dim=1).unsqueeze(1))*(1+param[:,5:6])
    else:
        raise ValueError("unknown painter nparam.")

    orig_img = background_vae.sample(param_latent) ### this outputs bx1xHxW image
    orig_img = trn_resize(orig_img)

    matte = (orig_img>EPS).float()
    cmyk = matte.repeat(1,4,1,1)
    alpha = orig_img
    binary = (orig_img>EPS).float()

    color = (1+param[:,-4:]).unsqueeze(2).unsqueeze(3)/2
    matte_color = cmyk*color*alpha
    
    return matte_color, binary


def latent2stroke_grid(param, H,W, gridbased_vae, gridnum, gridbased_info):

    # param: b, 10 (latent) + 3 (RGB)
    trn_resize = transforms.Resize([H,W])
    trn_ToPIL = transforms.ToPILImage(mode='CMYK')
    trn_ToTensor = transforms.ToTensor()
        
    ## param : grid x stroke_num, 10 (5 determining the direction of unit vector, 1 for z norm (0~2), 4 for CMYK color)
    if gridbased_info['painter_nparam']==9:
        sigma = 2
        param_latent = (param[:,:5] / torch.norm(param[:,:5],dim=1).unsqueeze(1))*sigma
    elif gridbased_info['painter_nparam']==10:
        param_latent = (param[:,:5] / torch.norm(param[:,:5],dim=1).unsqueeze(1))*(1+param[:,5:6])
    else:
        raise ValueError("unknown painter nparam.")

    grid_class = grid_generator(gridnum, gridbased_info)
    grid_class = grid_class.to(gridbased_info['device'])

    orig_img, _, _ = gridbased_vae.sample(param_latent, grid_class) ### this outputs bx1xHxW image
    orig_img = trn_resize(orig_img)

    matte = (orig_img>EPS).float()
    cmyk = matte.repeat(1,4,1,1)
    alpha = orig_img
    binary = (orig_img>EPS).float()

    color = (1+param[:,-4:]).unsqueeze(2).unsqueeze(3)/2
    matte_color = cmyk*color*alpha
    
    return matte_color, binary


def grid_generator(grid_num, gridbased_info, for_painter = False, H=256, W=256):

    padding, grid_size = PAD_INFO[grid_num]
    trnPad = transforms.Pad(padding//2)
    trnGrayScale = transforms.Grayscale()
    
    grids = []
    for i in range(grid_num**2):
        temp = torch.zeros(grid_num**2, 1, grid_size, grid_size)
        temp[i] = torch.ones(1, grid_size, grid_size)

        grid = make_grid(temp, padding=0, nrow=grid_num)
        grid = trnGrayScale(trnPad(grid))
        if for_painter:
            grids.append(grid.unsqueeze(0))
        else:
            grids.append(grid.unsqueeze(0).repeat(gridbased_info['stroke_num'],1,1,1))
    
    return torch.cat(grids, dim=0)


def read_img(img_path, img_type='RGB', h=None, w=None, boundary=20):
    img = Image.open(img_path).convert(img_type)
    if h is not None and w is not None:
        img = img.resize((w-boundary, h-boundary), resample=Image.NEAREST)
    img = add_margin(img, boundary//2, boundary//2, boundary//2, boundary//2, (0,0,0,0))
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float() / 255.
    return img


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def pad(img, H, W):
    b, c, h, w = img.shape
    pad_h = (H - h) // 2
    pad_w = (W - w) // 2
    remainder_h = (H - h) % 2
    remainder_w = (W - w) % 2
    img = torch.cat([torch.zeros((b, c, pad_h, w), device=img.device), img,
                     torch.zeros((b, c, pad_h + remainder_h, w), device=img.device)], dim=-2)
    img = torch.cat([torch.zeros((b, c, H, pad_w), device=img.device), img,
                     torch.zeros((b, c, H, pad_w + remainder_w), device=img.device)], dim=-1)
    return img


def crop(img, h, w):
    H, W = img.shape[-2:]
    pad_h = (H - h) // 2
    pad_w = (W - w) // 2
    remainder_h = (H - h) % 2
    remainder_w = (W - w) % 2
    img = img[:, :, pad_h:H - pad_h - remainder_h, pad_w:W - pad_w - remainder_w]
    return img


def main(input_path,
            draw_bg, 
            boundary, 
            background_info, 
            gridbased_info,
            output_dir,
            resize_h = 256,         # resize original input to this size. None means do not resize.
            resize_w = 256,         # resize original input to this size. None means do not resize.
            grid_nums = [1,2,3,4,5], 
            detail = False,
            device = "cuda:7"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    input_name = os.path.basename(input_path)
    output_path = os.path.join(output_dir, input_name)

    frame_dir = os.path.join(output_dir, input_name[:input_name.find('.')])
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir+"/cropped")
        os.makedirs(frame_dir+"/validation")

    json.dump(locals(), open(os.path.join(frame_dir, '../arguments.json'), 'w'))


    ### MODEL LOADING

    if draw_bg: 
        print('Loaded background drawing painter model.')
        painter_bg = network.Painter_Original(8, background_info['stroke_num'], 256, 8, 3, 3).to(device)
        painter_bg.load_state_dict(torch.load(background_info['painter_path']))
        painter_bg.eval()
        
        for param in painter_bg.parameters():
            param.requires_grad = False

        print('Loaded generative model for background drawing.')
        model = BetaVAE_B_256(z_dim=background_info['vae_zdim'], nc=1)
        state = torch.load(os.path.join(background_info['vae_path']),map_location='cpu')  
        model.load_state_dict(state['model_states']['net'])
            
        for param in model.parameters():
            param.requires_grad = False
            
        background_vae = model.to(device)


    print('Loaded conditional grid-based painter model. Requires grid information for inference.')
    painter_grid = network.Painter_Conditional(8, gridbased_info['stroke_num'], 256, 8, 3, 3).to(device)
    painter_grid.load_state_dict(torch.load(gridbased_info['painter_path']))
    painter_grid.eval()
    
    for param in painter_grid.parameters():
        param.requires_grad = False


    print('Loaded generative model for grid-based drawing.')
    model = BetaVAE_B_256_Conditional_Path(z_dim=gridbased_info['vae_zdim'], c_dim = gridbased_info['vae_cdim'], nc=1, encoder = gridbased_info['vae_encoder'], 
                                             decoder=gridbased_info['vae_decoder'], mask_gen = gridbased_info['vae_maskgen'], path_type= 'jointpath', device =device)
    state = torch.load(os.path.join(gridbased_info['vae_path']),map_location='cpu')  
    model.load_state_dict(state['model_states']['net'])
        
    for param in model.parameters():
        param.requires_grad = False
        
    gridbased_vae = model.to(device)


    ### INFERNCE
    
    with torch.no_grad():
        original_img = read_img(input_path, 'CMYK', resize_h, resize_w, boundary).to(device)  # 이미지 읽어옴
        final_result = torch.zeros_like(original_img).to(device)
        # final_result = final_result.unsqueeze(0)
        save_img(original_img[0], frame_dir, "target.png")
        previous_step = torch.zeros_like(original_img).to(device)
        
        img_size = 256
        idx=-1

        if draw_bg:

            for repeat in tqdm(range(background_info['repeat'])):
                
                param, decisions = painter_bg(original_img, final_result)
                param = param.view(-1, background_info['painter_nparam']).contiguous()
                foregrounds, alphas = latent2stroke_bg(param, resize_h, resize_w, background_vae, background_info)

                foregrounds = foregrounds.view(-1, background_info['stroke_num'], 4, resize_h, resize_w)
                alphas = alphas.view(-1, background_info['stroke_num'], 1, resize_h, resize_w)
                decisions = network.SignWithSigmoidGrad.apply(decisions.view(-1, background_info['stroke_num'], 1, 1, 1).contiguous())
                for j in range(foregrounds.shape[1]):
                    foreground = foregrounds[:, j, :, :, :]
                    alpha = alphas[:, j, :, :, :]
                    decision = decisions[:, j, :, :, :]
                    # if decision_switch:
                    final_result = torch.clip(foreground*decision + final_result, torch.min(foreground, final_result), torch.max(foreground, final_result))
                    # else:
                    #     final_result = torch.clip(foreground + final_result, torch.min(foreground, final_result), torch.max(foreground, final_result))
                if repeat%1 ==0:
                    save_img(final_result[0], frame_dir, "background_{:05d}.png".format(repeat))


        for grid_num in grid_nums:

            idx= idx+1
            
            padding, grid_size = PAD_INFO[grid_num]
            trn_centercrop = transforms.CenterCrop([img_size-padding, img_size-padding])
            trn_padding = transforms.Pad(padding//2)
            
            # if img_size%grid_num==0:
            print('divided the grid with {}x{}, with the padding {} '.format(grid_num, grid_num, padding))
            
            grid_target = trn_centercrop(original_img)
            grid_target = grid_target.reshape(-1,4,grid_num,grid_size,grid_num,grid_size)  # 1 4 2 128 2 128
            grid_target = grid_target.permute(0,2,4,1,3,5)  # 1 2 2 4 128 128 
            grid_target = grid_target.reshape(-1, 4, grid_size, grid_size) # 4 4 128 128 
            
            grid_current = trn_centercrop(final_result)
            grid_current = grid_current.reshape(-1,4,grid_num,grid_size,grid_num,grid_size)
            grid_current = grid_current.permute(0,2,4,1,3,5)
            grid_current = grid_current.reshape(-1,4,grid_size,grid_size)
            
            grid_target = transforms.Resize([256,256])(grid_target)
            
            for j in range(grid_current.shape[0]):
                save_img(grid_current[j], frame_dir+"/cropped", "{:02d}_{}_current.png".format(idx,j))
                save_img(grid_target[j], frame_dir+"/cropped", "{:02d}_{}_target.png".format(idx,j))

            for kk in range(gridbased_info['repeat']):
                grid_current = trn_centercrop(final_result)
                grid_current = grid_current.reshape(-1,4,grid_num,grid_size,grid_num,grid_size)
                grid_current = grid_current.permute(0,2,4,1,3,5)
                grid_current = grid_current.reshape(-1,4,grid_size,grid_size)
                
                grid_current = transforms.Resize([256,256])(grid_current)

                grid_class = grid_generator(grid_num, gridbased_info, for_painter = True)
                grid_class = grid_class.to(device)
                
                # print(grid_target.shape, grid_current.shape)
                # print(grid_target.shape, grid_current.shape, grid_class.shape)
                param, decisions = painter_grid(grid_target, grid_current, grid_class)

                param = param.view(-1, gridbased_info['painter_nparam']).contiguous()
                
                foregrounds, alphas = latent2stroke_grid(param, resize_h, resize_w, gridbased_vae, grid_num, gridbased_info)
                
                foregrounds = foregrounds.view(-1, 4, resize_h, resize_w)

                alphas = alphas.view(-1, 1, resize_h, resize_w)
                decisions = network.SignWithSigmoidGrad.apply(decisions.view(-1, 1, 1, 1).contiguous())
                for j in range(foregrounds.shape[0]):
                    foreground = foregrounds[j, :, :, :]
                    alpha = alphas[j, :, :, :]
                    decision = decisions[j, :, :, :]
                    # if decision_switch:
                    final_result = torch.clip(foreground*decision + final_result, torch.min(foreground, final_result), torch.max(foreground, final_result))
                    # else:
                    #     final_result = torch.clip(foreground + final_result, torch.min(foreground, final_result), torch.max(foreground, final_result))
                            
            save_img(final_result[0], frame_dir, "{}_{:02d}_{}x{}.png".format(input_name[:-4],idx,grid_num,grid_num))
            print('    saved its result to {:02d}_final.png'.format(idx))
                     
        command = 'convert {}/*.png {}/{}.gif'.format(frame_dir, frame_dir, input_name[:input_name.find('.')])
        print("converting to {}.gif".format(input_name[:input_name.find('.')]))
        os.system(command)


if __name__ == '__main__':
    
    pic_list = ['sangok','labs','naver','1','2','3','starry_night','gradient','ocean','jennifer','face']
    # pic_list = ['']
    
    output_dir = './output/testing_gridbased_idea_wobackground_new'
        
    for i in pic_list:
        main(input_path = '../picture/{}.jpg'.format(i),
            draw_bg = False, 
            boundary = 22, 
            background_info = {
                'vae_zdim' : 5, 
                'vae_cdim' : 0, 
                'vae_encoder' : None, 
                'vae_decoder' : None, 
                'vae_maskgen' : None, 
                'vae_path' : '../train/gen1.pt', 
                'painter_path' : '../train/checkpoints/painter_model_gen1_pix50gt100dec10_lr1e5_300epoch/latest_net_g.pth', 
                'painter_nparam' : 9, 
                'stroke_num' : 32, 
                'repeat' : 2, 
            }, 
            gridbased_info = {
                'vae_zdim' : 5, 
                'vae_cdim' : 5, 
                'vae_encoder' : '1DCNN', 
                'vae_decoder' : '1DCNN', 
                'vae_maskgen' : '1DCNN', 
                'vae_path' : '../train/grid_classified_z5c5_gamma100_last.pt', 
                'painter_path' : '../train/checkpoints/grid_aware_stroke20_bg30_grid2345_bs20_gridcrop_pix50gt100dec10_lr1e5_300epoch/latest_net_g.pth', 
                'painter_nparam' : 10, 
                'stroke_num' : 32, 
                'repeat' : 3, 
                'device' : 'cuda:0'
            },
            output_dir = output_dir,
            resize_h = 256,         # resize original input to this size. None means do not resize.
            resize_w = 256,         # resize original input to this size. None means do not resize.
            grid_nums = [2,3,4,5], 
            detail = False,
            device = "cuda:0")          # if need animation, serial must be True.

    command = 'zip -r {}.zip {}'.format(output_dir, output_dir)
    os.system(command)

## delete 618th