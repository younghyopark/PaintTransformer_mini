import torch
import numpy as np
from .base_model import BaseModel
from . import networks
from util import morphology
from scipy.optimize import linear_sum_assignment
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os
import torchvision
import random
from torchvision.utils import make_grid
import torchvision.transforms as transforms

EPS1 = 1e-1
EPS2 = 1e-1

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


class BetaVAE_B_256_Conditional(nn.Module):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim=3, c_dim = 4, nc=1):
        super(BetaVAE_B_256_Conditional, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        self.c_dim = c_dim
        
        self.embed_label = nn.Sequential(
            nn.Linear(c_dim, 512),              # B, 256
            nn.ReLU(True),
            nn.Linear(512, 2048),                 # B, 256
            nn.ReLU(True),
            nn.Linear(2048, 256 *256),             # B, z_dim*2
        )
        self.embed_data = nn.Sequential(
            nn.Conv2d(self.nc, self.nc, kernel_size=1)
        )

        self.conv_encoder = nn.Sequential(
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
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )


        self.decoder = nn.Sequential(
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
        self.weight_init()     
        
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)
                
    def forward(self, x, c):
        # img_feature = self.conv_encoder(x)
        # t = torch.cat((img_feature, c),dim=1)
        distributions = self._encode(x,c)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        # t2 = torch.cat((z,c),dim=1)
        x_recon = self._decode(z,c)

        return x_recon, mu, logvar
    
    def sample(self,z,c):
        x_recon = self._decode(z,c)
        x_recon = torch.sigmoid(x_recon)
        return x_recon

    def _encode(self, x, c):
        c = F.one_hot(c.to(torch.int64), self.c_dim).float()
        class_feature = self.embed_label(c)
        class_feature = class_feature.reshape(-1,1,256,256)
        img_feature = self.embed_data(x)
        t = torch.cat((img_feature,class_feature),dim=1)
        distributions = self.conv_encoder(t)
        return distributions

    def _decode(self, z,c):
        c = F.one_hot(c.to(torch.int64), self.c_dim).float()
#         print(z.shape,c.shape)
        t = torch.cat((z,c),dim=1)
        return self.decoder(t)
    

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

    def __init__(self, z_dim=5, c_dim = 5, nc=1, encoder = '1DCNN', decoder='1DCNN', n_features = 8, seq_length = 300, mask_gen = '1DCNN', path_type = 'jointpath'):
        super(BetaVAE_B_256_Conditional_Path, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.path_type = path_type
        self.encoder = encoder
        self.decoder = decoder
        
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
        recon_path = recon_path + (recon_mask<=0).float().unsqueeze(1) * temp.cuda()
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
    

    
    
class PainterModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='null')
        parser.add_argument('--used_strokes', type=int, default=8,
                            help='actually generated strokes number')
        parser.add_argument('--num_blocks', type=int, default=3,
                            help='number of transformer blocks for stroke generator')
        parser.add_argument('--lambda_w', type=float, default=10.0, help='weight for w loss of stroke shape')
        parser.add_argument('--lambda_pixel', type=float, default=10.0, help='weight for pixel-level L1 loss')
        parser.add_argument('--lambda_gt', type=float, default=50.0, help='weight for ground-truth loss')
        parser.add_argument('--lambda_decision', type=float, default=10.0, help='weight for stroke decision loss')
        parser.add_argument('--lambda_recall', type=float, default=10.0, help='weight of recall for stroke decision loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['pixel', 'gt', 'decision']
        if self.opt.long_horizon:
            self.visual_names = ['old', 'render','immediate_next','rec']
        elif self.opt.grid_class:
            if self.opt.grid_cropout:
                self.visual_names = ['old', 'render','rec','grid_class_visualize',
                                    'old_grid_crop_resize','render_grid_crop_resize','rec_grid_crop_resize','grid_class_visualize']
            else:
                self.visual_names = ['real_bg', 'old', 'render','rec']
        else:
            self.visual_names = ['old', 'render','rec']
        self.model_names = ['g']
        
        self.d = self.opt.generative_zdim + 4 + int(self.opt.decide_largesmall or self.opt.grid_class)
        self.d_shape = self.d
            
        def read_img(img_path, img_type='RGB'):
            img = Image.open(img_path).convert(img_type)
            img = np.array(img)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
            img = img.transpose((2, 0, 1))
            img = torch.from_numpy(img).unsqueeze(0).float() / 255.
            return img

        if not self.opt.generative:
            brush_large_vertical = read_img('brush/brush_small_vertical.png', 'L').to(self.device)
            brush_large_horizontal = read_img('brush/brush_small_horizontal.png', 'L').to(self.device)
            self.meta_brushes = torch.cat(
                [brush_large_vertical, brush_large_horizontal], dim=0)
        else:
            if self.opt.generative_cdim >0 : 
                model = BetaVAE_B_256_Conditional_Path(z_dim=5, c_dim = 5, nc=1)       
                state = torch.load(os.path.join('./{}.pt'.format(self.opt.generative_path)),map_location='cpu')        
                model.load_state_dict(state['model_states']['net'])
                for param in model.parameters():
                    param.requires_grad = False
            else:
                model = BetaVAE_B_256(z_dim=5, nc=1)       
                state = torch.load(os.path.join('./{}.pt'.format(self.opt.generative_path)),map_location='cpu')        
                model.load_state_dict(state['model_states']['net'])
                for param in model.parameters():
                    param.requires_grad = False
            self.generative_model = model.to(self.device)        

            if self.opt.grid_class:
                model = BetaVAE_B_256(z_dim=5, nc=1)       
                state = torch.load(os.path.join('./gen1.pt'.format(self.opt.generative_path)),map_location='cpu')        
                model.load_state_dict(state['model_states']['net'])
                for param in model.parameters():
                    param.requires_grad = False
                self.background_generative_model = model.to(self.device)
        
        if 'DEEPER' not in self.opt.name:
            if not self.opt.grid_class:
                net_g = networks.Painter_Original(self.d_shape, opt.used_strokes, 256,
                                        n_enc_layers=opt.num_blocks, n_dec_layers=opt.num_blocks)
            else:
                print('loaded conditional model. original depth.')
                net_g = networks.Painter_Conditional(self.d_shape, opt.used_strokes, 256,
                                        n_enc_layers=opt.num_blocks, n_dec_layers=opt.num_blocks)
        else:
            net_g = networks.Painter(self.d_shape, opt.used_strokes, 512,
                                    n_enc_layers=opt.num_blocks, n_dec_layers=opt.num_blocks, largesmall=int(self.opt.decide_largesmall))
                
            
        self.net_g = networks.init_net(net_g, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.opt.continue_training:
            self.net_g.load_state_dict(torch.load(self.opt.continue_path))

        self.old = None
        self.render = None
        self.rec = None
        self.real_bg = None
        self.gt_param = None
        self.pred_param = None
        self.gt_decision = None
        self.pred_decision = None
        self.grid_class = None
        self.gt_grid_class = None
        self.patch_size = 256
        self.loss_pixel = torch.tensor(0., device=self.device)
        self.loss_gt = torch.tensor(0., device=self.device)
        self.loss_w = torch.tensor(0., device=self.device)
        self.loss_decision = torch.tensor(0., device=self.device)
        self.criterion_pixel = torch.nn.L1Loss().to(self.device)
        self.criterion_decision = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(opt.lambda_recall)).to(self.device)
        if self.isTrain:
            self.optimizer = torch.optim.Adam(self.net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)

    def latent2stroke2(self, param, H,W, select_c = 0):
        ## define transformations
        trn_resize = torchvision.transforms.Resize([H+40,W+40])
        trn_crop= torchvision.transforms.CenterCrop([H,W])
        trn_resize2 = torchvision.transforms.Resize([H,W])
        
        param_latent = (param[:,:self.opt.generative_zdim] / torch.norm(param[:,:self.opt.generative_zdim],dim=1).unsqueeze(1))*self.opt.sigma
        if self.opt.generative_cdim>0: 
            if self.opt.decide_largesmall:
                c = networks.SignWithSigmoidGrad.apply(param[:,-5])  
                orig_img = self.generative_model.sample(param_latent, c) 
            else:
                if select_c==1:
                    c = torch.ones(param.shape[0]).cuda()
                elif select_c==0:
                    c = torch.zeros(param.shape[0]).cuda()
                orig_img = self.generative_model.sample(param_latent, c) 
        else:
            orig_img = self.generative_model.sample(param_latent) 
            
        if self.opt.no_crop:
            orig_img = trn_resize2(orig_img)
        else:
            orig_img = trn_crop(trn_resize(orig_img))
        
        EPS = EPS1
        binary = (orig_img>EPS).float()
        cmyk = binary.repeat(1,4,1,1)
        alpha = orig_img
        
        color = (1+param[:,-4:]).unsqueeze(2).unsqueeze(3)/2
        matte_color = cmyk*color*alpha
        return matte_color, binary

    def latent2stroke2_bg(self, param, H,W):
        ## define transformations
        trn_resize = torchvision.transforms.Resize([H,W])
        
        param_latent = (param[:,:self.opt.generative_zdim] / torch.norm(param[:,:self.opt.generative_zdim],dim=1).unsqueeze(1))\
            * (1+param[:,self.opt.generative_zdim:self.opt.generative_zdim+1])

        orig_img = self.background_generative_model.sample(param_latent) 
            
        orig_img = trn_resize(orig_img)
        
        EPS = EPS1
        binary = (orig_img>EPS).float()
        cmyk = binary.repeat(1,4,1,1)
        alpha = orig_img
        
        color = (1+param[:,-4:]).unsqueeze(2).unsqueeze(3)/2
        matte_color = cmyk*color*alpha
        return matte_color, binary

    def generate_gridclass(self, grid_num, grid_index):
        GRID_INFO = {
            2: (0, 256//2), 
            3: (4, 252//3), 
            4: (0, 256//4),
            5: (6, 250//5),
            6: (4, 252//6)
        }
        
        grid_imgs = []

        padding, grid_size = GRID_INFO[grid_num]
        
        temp = torch.zeros(grid_num**2, 1, grid_size, grid_size)
        temp[grid_index] = torch.ones(1, grid_size, grid_size)
        
        grid = make_grid(temp, padding=0, nrow=grid_num)
        
        TPad = transforms.Pad(padding//2)
        TGrayScale = transforms.Grayscale(num_output_channels=1)
        grid = TGrayScale(TPad(grid))
        
        return grid

    def crop_out_grid(self, imgs, grid, H=256, W=256, resize = True):
        if not (imgs.dim()==4 and grid.dim()==4):
            raise ValueError('only accept batch')
        TResize = transforms.Resize([H,W])
        cropped = []
        for i in range(imgs.shape[0]):
            nonzero = grid[i].nonzero()
            
            x_min = torch.min(nonzero[:,1])
            x_max = torch.max(nonzero[:,1])
            y_min = torch.min(nonzero[:,2])
            y_max = torch.max(nonzero[:,2])
            temp = imgs[i][:,x_min:x_max, y_min:y_max]
            if resize:
                temp = TResize(temp)
            cropped.append(temp)
        result = torch.stack(cropped)
        return result


    def generate_gridclasses(self, grid_nums, grid_indexs):
        GRID_INFO = {
            2: (0, 256//2), 
            3: (4, 252//3), 
            4: (0, 256//4),
            5: (6, 250//5),
            6: (4, 252//6)
        }
        
        grid_imgs = []
        for i, grid_num in enumerate(grid_nums):
            padding, grid_size = GRID_INFO[grid_num]
            grid_index =  grid_indexs[i]
            
            temp = torch.zeros(grid_num**2, 1, grid_size, grid_size)
            temp[grid_index] = torch.ones(1, grid_size, grid_size)
            
            grid = make_grid(temp, padding=0, nrow=grid_num)
            
            TPad = transforms.Pad(padding//2)
            TGrayScale = transforms.Grayscale(num_output_channels=1)
            grid = TGrayScale(TPad(grid))

            grid_imgs.append(grid)

        grid_imgs = torch.stack(grid_imgs,dim=0)
        
        return grid_imgs

    def randomly_generate_gridclasses(self, batch_size, grid_num_choices = [2,3,4,5]):
        GRID_INFO = {
            2: (0, 256//2), 
            3: (4, 252//3), 
            4: (0, 256//4),
            5: (6, 250//5),
            6: (4, 252//6)
        }
        grid_nums = []
        grid_indexs = []
        for i in range(batch_size):
            random_grid_num = np.random.choice(grid_num_choices)
            random_grid_index = np.random.randint(random_grid_num**2)
            grid_nums.append(random_grid_num)
            grid_indexs.append(random_grid_index)
        
        grid_imgs = []
        for i, grid_num in enumerate(grid_nums):
            padding, grid_size = GRID_INFO[grid_num]
            grid_index =  grid_indexs[i]
            
            temp = torch.zeros(grid_num**2, 1, grid_size, grid_size)
            temp[grid_index] = torch.ones(1, grid_size, grid_size)
            
            grid = make_grid(temp, padding=0, nrow=grid_num)
            
            TPad = transforms.Pad(padding//2)
            TGrayScale = transforms.Grayscale(num_output_channels=1)
            grid = TGrayScale(TPad(grid))

            grid_imgs.append(grid)

        grid_imgs = torch.stack(grid_imgs,dim=0)
        
        return grid_imgs        

    def latent2stroke3(self, param, H,W, grid_classes):
        ## define transformations
        # trn_crop= torchvision.transforms.CenterCrop([H,W])
        trn_resize = torchvision.transforms.Resize([H,W])
        
        param_latent = (param[:,:self.opt.generative_zdim] / torch.norm(param[:,:self.opt.generative_zdim],dim=1).unsqueeze(1))\
            * (1+param[:,self.opt.generative_zdim:self.opt.generative_zdim+1])

        # print(param_latent.shape, grid_classes.shape)
        orig_img, _, _ = self.generative_model.sample(param_latent, grid_classes) 
            
        orig_img =  trn_resize(orig_img)
        
        EPS = EPS1
        binary = (orig_img>EPS).float()
        cmyk = binary.repeat(1,4,1,1)
        alpha = orig_img
        
        color = (1+param[:,-4:]).unsqueeze(2).unsqueeze(3)/2
        matte_color = cmyk*color*alpha
        return matte_color, binary

    def latent2stroke_randomgrid(self, param, H,W):
        ## define transformations
        # trn_crop= torchvision.transforms.CenterCrop([H,W])
        trn_resize = torchvision.transforms.Resize([H,W])

        grid_class = self.randomly_generate_gridclasses(batch_size = param.shape[0])
        grid_class = grid_class.cuda()

        param_latent = (param[:,:self.opt.generative_zdim] / torch.norm(param[:,:self.opt.generative_zdim],dim=1).unsqueeze(1))\
            * (1+param[:,self.opt.generative_zdim:self.opt.generative_zdim+1])

        # print(param_latent.shape, grid_class.shape)
        orig_img, _, _ = self.generative_model.sample(param_latent, grid_class) 
            
        orig_img =  trn_resize(orig_img)
        
        EPS = EPS1
        binary = (orig_img>EPS).float()
        cmyk = binary.repeat(1,4,1,1)
        alpha = orig_img
        
        color = (1+param[:,-4:]).unsqueeze(2).unsqueeze(3)/2
        matte_color = cmyk*color*alpha
        return matte_color, binary
        

    def set_input2(self, input_dict, background_stroke_times = None):
        """
        Training data creation code for PaintTransformer_mini_v3 (revised version).
        """
        self.image_paths = input_dict['A_paths']
        if self.opt.strategy is not None:
            if 'linear_CMYKmax' in self.opt.strategy:
                largestroke_CMYKsplit = self.opt.coarse_CMYKmax / self.opt.coarse_num
                smallstroke_CMYKsplit = (self.opt.fine_CMYKmax - self.opt.fine_CMYKmin) / self.opt.fine_num
                
        with torch.no_grad():
            if not self.opt.generative:
                gt_param = torch.rand(self.opt.fake_batch_size, self.opt.inference_repeat_num, self.opt.used_strokes, self.d, device=self.device)
                gt_param[:, :, :4] = gt_param[:, :, :4] * 0.5 + 0.2
                gt_param[:, :, -4:-1] = gt_param[:, :, -7:-4]
            else:
                gt_param = -1 + torch.rand(self.opt.fake_batch_size, self.opt.inference_repeat_num, self.opt.used_strokes, self.d, device=self.device) * 2
                if self.opt.strategy is not None:
                    if 'linear_CMYKmax' in self.opt.strategy:
                        for i in range(self.opt.coarse_num):
                            gt_param[:,i,:,-4:] = -1 + torch.rand(self.opt.fake_batch_size,self.opt.used_strokes,4)*largestroke_CMYKsplit*(i+1)* 2
                        for i in range(self.opt.fine_num):
                            gt_param[:,i+self.opt.coarse_num,:,-4:] = -1 + torch.rand(self.opt.fake_batch_size,self.opt.used_strokes,4)*(self.opt.fine_CMYKmin + smallstroke_CMYKsplit*(i+1))* 2
                            
                    if self.opt.CMYK_maxclip>0:
                        for i in range(self.opt.coarse_num):
                            gt_param[:,i,:,-4:] = torch.clip(gt_param[:,i,:,-4:], 0, self.opt.CMYK_maxclip)
                        for i in range(self.opt.fine_num):
                            gt_param[:,i+self.opt.coarse_num,:,-4:] = torch.clip(gt_param[:,i+self.opt.coarse_num,:,-4:], 0, self.opt.CMYK_maxclip)
                            
                    if self.opt.decide_largesmall:
                        for i in range(self.opt.coarse_num):
                            gt_param[:,i,:,-5] = -torch.ones(self.opt.fake_batch_size, self.opt.used_strokes)
                        for i in range(self.opt.fine_num):
                            gt_param[:,i+self.opt.coarse_num,:,-5] = torch.ones(self.opt.fake_batch_size, self.opt.used_strokes)
                        
            gt_param = gt_param.view(-1, self.d).contiguous() ## b x used_strokes x repeat_num  strokes will be rendered. 

            if not self.opt.generative:
                foregrounds, alphas = self.param2stroke(gt_param, self.patch_size, self.patch_size)
                foregrounds = morphology.Dilation2d(m=1)(foregrounds)
                alphas = morphology.Erosion2d(m=1)(alphas)
            else:
                foregrounds, alphas = self.latent2stroke2(gt_param, self.patch_size, self.patch_size)

                foregrounds = foregrounds.view(self.opt.fake_batch_size, self.opt.inference_repeat_num, self.opt.used_strokes, 4, self.patch_size,
                                           self.patch_size).contiguous()
                
            alphas = alphas.view(self.opt.fake_batch_size, self.opt.inference_repeat_num, self.opt.used_strokes, 1, self.patch_size,
                                 self.patch_size).contiguous()
            result_content_wc = torch.zeros(self.opt.fake_batch_size, 1+self.opt.inference_repeat_num, 4, self.patch_size, self.patch_size, device=self.device)
            gt_decision = torch.ones(self.opt.fake_batch_size, self.opt.inference_repeat_num, self.opt.used_strokes, device=self.device)
            # gt_decision = torch.empty(self.opt.fake_batch_size, self.opt.inference_repeat_num, self.opt.used_strokes, device=self.device).uniform_(0, 1)
            # gt_decision = torch.bernoulli(gt_decision)
            
            for j in range(1,self.opt.inference_repeat_num+1):
                for i in range(self.opt.used_strokes):
                    content_wc = foregrounds[:, j-1,i, :, :, :]  
                    alpha = alphas[:, j-1,i, :, :, :]  
                    decision = gt_decision[:, j-1,i].view(self.opt.fake_batch_size, 1, 1, 1).contiguous() # shape b x 1 x 1 x 1
                    for k in range(i):
                        iou = (torch.sum(alpha * alphas[:, j-1,k, :, :, :], dim=(-3, -2, -1)) + 1e-5) / (
                                torch.sum(alphas[:, j-1,k, :, :, :], dim=(-3, -2, -1)) + 1e-5)
                        gt_decision[:, j-1,i] = ((iou < 0.8) | (~gt_decision[:, j-1,k].bool())).float() * gt_decision[:, j-1,i]
                    result_content_wc [:,j,:,:,:] = torch.clip(content_wc*decision+ result_content_wc[:,j,:,:,:], torch.min(content_wc,result_content_wc[:,j,:,:,:]),torch.max(content_wc,result_content_wc[:,j,:,:,:]))
                if not j == self.opt.inference_repeat_num:
                    result_content_wc[:,j+1,:,:,:] = result_content_wc[:,j,:,:,:]
            
            gt_param = gt_param.view(self.opt.fake_batch_size, self.opt.inference_repeat_num, self.opt.used_strokes, self.d).contiguous()
            
            if not self.opt.simpler_long_horizon:
                new_batch_size = self.opt.fake_batch_size * (self.opt.inference_repeat_num)*(self.opt.inference_repeat_num+1)//2
            else:
                new_batch_size = self.opt.fake_batch_size * (self.opt.inference_repeat_num)
                
            self.gt_param = torch.zeros(new_batch_size, self.opt.used_strokes, self.d, device=self.device)
            self.old = torch.zeros(new_batch_size, 4, self.patch_size, self.patch_size, device=self.device)
            self.render = torch.zeros(new_batch_size, 4, self.patch_size, self.patch_size, device=self.device)
            self.immediate_next = torch.zeros(new_batch_size, 4, self.patch_size, self.patch_size, device=self.device)
            self.gt_decision = torch.ones(new_batch_size, self.opt.used_strokes, device=self.device)


            if not self.opt.simpler_long_horizon:
                _idx = 0
                for b in range(self.opt.fake_batch_size):
                    for i in range(self.opt.inference_repeat_num):
                        for j in range(i+1, self.opt.inference_repeat_num+1):
                            self.gt_param[_idx, :,:] = gt_param[b, i,:,:]
                            self.gt_decision[_idx,:] = gt_decision[b, i, :]
                            # if 'coarse_to_fine' in self.opt.strategy:
                            #     self.gt_largesmall[_idx,:] = gt_largesmall[b,i,:]
                            self.old[_idx,:,:,:] = result_content_wc[b,i,:,:,:]
                            self.render[_idx,:,:,:] = result_content_wc[b,j,:,:,:]
                            self.immediate_next[_idx,:,:,:] = result_content_wc[b,i+1,:,:,:]
                            _idx+=1 
            else:
                _idx = 0
                for b in range(self.opt.fake_batch_size):
                    for i in range(self.opt.inference_repeat_num):
                        self.gt_param[_idx, :,:] = gt_param[b, i,:,:]
                        self.gt_decision[_idx,:] = gt_decision[b, i, :]
                        self.old[_idx,:,:,:] = result_content_wc[b,i,:,:,:]
                        self.render[_idx,:,:,:] = result_content_wc[b,self.opt.inference_repeat_num,:,:,:]
                        self.immediate_next[_idx,:,:,:] = result_content_wc[b,i+1,:,:,:]
                        _idx+=1 

            ## shuffle the batch
            idx = torch.randperm(self.gt_param.shape[0])
            self.gt_param = self.gt_param[idx].view(self.gt_param.size())
            self.gt_decision = self.gt_decision[idx].view(self.gt_decision.size())
            self.old = self.old[idx].view(self.old.size())
            self.render = self.render[idx].view(self.render.size())
            self.immediate_next = self.immediate_next[idx].view(self.immediate_next.size())

            if self.opt.debug:
                torch.save(result_content_wc, './strategy_check.pth')
                torch.save(self.gt_param, './gt_param_debug.pth')
                torch.save(self.old, './old.pth')
                torch.save(self.render, './render.pth')

            """
            render-bf = b x T x img   0, 1, ... , T-1 
            gt param = b x T x n x param    0,1, ..., T-1

            old    =  b x ( 0, 0, ...,  0, 1, 1, ...,   1, 2, 2, ...,   2, ) x img         = (bx(Tx(T-1))//2) x img
            render =  b x ( 1, 2, ... T-1, 2, 3, ..., T-1, 3, 4, ..., T-1, ) x img         = (bx(Tx(T-1))//2) x img
            gt_param= b x ( 0, 0, ...,  0, 1, 1, ...,   1, 2, 2, ...,   2, ) x n x param   = (bx(Tx(T-1))//2) x n x param
            """


    def set_input(self, input_dict, background_stroke_times):
        self.image_paths = input_dict['A_paths']
        stroke_num = np.random.randint(1,self.opt.used_strokes * background_stroke_times)
        with torch.no_grad():
            if not self.opt.generative:
                old_param = torch.rand(self.opt.batch_size, stroke_num, self.d, device=self.device)
                old_param[:, :, :4] = old_param[:, :, :4] * 0.5 + 0.2
                old_param[:, :, -4:-1] = old_param[:, :, -7:-4]
            else:
                if self.opt.background_tile:
                    old_param = -1 + torch.rand(self.opt.batch_size//4, stroke_num, self.d, device=self.device) * 2
                elif self.opt.tbt_background_tile:
                    old_param = -1 + torch.rand(self.opt.batch_size//9, stroke_num, self.d, device=self.device) * 2
                else:
                    old_param = -1 + torch.rand(self.opt.batch_size, stroke_num, self.d, device=self.device) * 2

            old_param = old_param.view(-1, self.d).contiguous()
            if not self.opt.generative:
                foregrounds, alphas = self.param2stroke(old_param, self.patch_size, self.patch_size)
                foregrounds = morphology.Dilation2d(m=1)(foregrounds)
                alphas = morphology.Erosion2d(m=1)(alphas)
            else:
                if self.opt.background_tile:
                    foregrounds, alphas = self.latent2stroke2(old_param, self.patch_size*2, self.patch_size*2, int(self.opt.latent2stroke_cvalues[0]))
                else:
                    foregrounds, alphas = self.latent2stroke2(old_param, self.patch_size, self.patch_size, int(self.opt.latent2stroke_cvalues[0]))

                # original_img = original_img.reshape(-1,4,2**repeat,img_size,2**repeat,img_size)  # 1 4 2 128 2 128
                # original_img = original_img.permute(0,2,4,1,3,5)  # 1 2 2 4 128 128 
                # original_img = original_img.reshape(-1, 4, img_size, img_size) # 4 4 128 128 
            
            if self.opt.background_tile:
                foregrounds = foregrounds.view(self.opt.batch_size // 4, stroke_num, 4, self.patch_size * 2,
                                               self.patch_size * 2).contiguous()
                alphas = alphas.view(self.opt.batch_size // 4, stroke_num, 1, self.patch_size * 2,
                                     self.patch_size * 2).contiguous()
                result_content_wc = torch.zeros(self.opt.batch_size // 4, 4, self.patch_size * 2, self.patch_size * 2, device=self.device)
                old_decision = torch.ones(self.opt.batch_size//4, stroke_num, device=self.device)
            elif self.opt.tbt_background_tile:
                foregrounds = foregrounds.view(self.opt.batch_size // 9, stroke_num, 4, self.patch_size,
                                               self.patch_size).contiguous()
                alphas = alphas.view(self.opt.batch_size //9, stroke_num, 1, self.patch_size,
                                     self.patch_size).contiguous()
                result_content_wc = torch.zeros(self.opt.batch_size // 9, 4, self.patch_size, self.patch_size, device=self.device)
                old_decision = torch.ones(self.opt.batch_size//9, stroke_num, device=self.device)
            else:
                foregrounds = foregrounds.view(self.opt.batch_size, stroke_num, 4, self.patch_size,
                                               self.patch_size).contiguous()
                alphas = alphas.view(self.opt.batch_size, stroke_num, 1, self.patch_size,
                                     self.patch_size).contiguous()
                result_content_wc = torch.zeros(self.opt.batch_size, 4, self.patch_size, self.patch_size, device=self.device)
                old_decision = torch.ones(self.opt.batch_size, stroke_num, device=self.device)
            
            for i in range(stroke_num):
                content_wc = foregrounds[:, i, :, :, :]
                alpha = alphas[:, i, :, :, :]
                for j in range(i):
                    iou = (torch.sum(alpha * alphas[:, j, :, :, :], dim=(-3, -2, -1)) + 1e-5) / (
                            torch.sum(alphas[:, j, :, :, :], dim=(-3, -2, -1)) + 1e-5)
                    old_decision[:, i] = ((iou < 0.8) | (~old_decision[:, j].bool())).float() * old_decision[:, i]
                if self.opt.background_tile:
                    decision = old_decision[:, i].view(self.opt.batch_size//4, 1, 1, 1).contiguous()
                elif self.opt.tbt_background_tile:
                    decision = old_decision[:, i].view(self.opt.batch_size//9, 1, 1, 1).contiguous()
                else:
                    decision = old_decision[:, i].view(self.opt.batch_size, 1, 1, 1).contiguous()
                result_content_wc = torch.clip(content_wc*decision+ result_content_wc, torch.min(content_wc,result_content_wc),torch.max(content_wc,result_content_wc))
                old = result_content_wc
                
            if self.opt.background_tile:
                self.old = old.view(self.opt.batch_size // 4, 4, 2, self.patch_size, 2, self.patch_size).contiguous()
                self.old = self.old.permute(0, 2, 4, 1, 3, 5).contiguous()
                self.old = self.old.view(self.opt.batch_size, 4, self.patch_size, self.patch_size).contiguous()
                result_content_wc = self.old
                self.old_content_wc = result_content_wc.clone()

            if self.opt.tbt_background_tile:
                _temp_trn1 = torchvision.transforms.CenterCrop([256-22, 256-22])
                old= _temp_trn1(old)
                # print(old.shape)
                _grid_size = 234//3
                old = old.view(self.opt.batch_size//9, 4, 3, _grid_size, 3, _grid_size).contiguous()
                old = old.permute(0, 2, 4, 1, 3, 5).contiguous()
                old = old.view(self.opt.batch_size, 4, _grid_size, _grid_size).contiguous()
                _temp_trn2 = torchvision.transforms.Resize([256,256])
                self.old = _temp_trn2(old)
                result_content_wc = self.old
                self.old_content_wc = result_content_wc.clone()
                
                
            else:
                self.old = old.view(self.opt.batch_size, 4, self.patch_size, self.patch_size).contiguous()
                self.old_content_wc = result_content_wc.clone()

            if not self.opt.generative:
                gt_param = torch.rand(self.opt.batch_size, self.opt.used_strokes, self.d, device=self.device)
                gt_param[:, :, :4] = gt_param[:, :, :4] * 0.5 + 0.2
                gt_param[:, :, -4:-1] = gt_param[:, :, -7:-4]
            else:
                gt_param = -1 + torch.rand(self.opt.batch_size, self.opt.used_strokes, self.d, device=self.device) * 2

            self.gt_param = gt_param[:, :, :self.d_shape]
            gt_param = gt_param.view(-1, self.d).contiguous()
            if not self.opt.generative:
                foregrounds, alphas = self.param2stroke(gt_param, self.patch_size, self.patch_size)
                foregrounds = morphology.Dilation2d(m=1)(foregrounds)
                alphas = morphology.Erosion2d(m=1)(alphas)
            else:
                foregrounds, alphas = self.latent2stroke2(gt_param, self.patch_size, self.patch_size, int(self.opt.latent2stroke_cvalues[1]))
            foregrounds = foregrounds.view(self.opt.batch_size, self.opt.used_strokes, 4, self.patch_size,
                                           self.patch_size).contiguous()
            alphas = alphas.view(self.opt.batch_size, self.opt.used_strokes, 1, self.patch_size,
                                 self.patch_size).contiguous()
            self.render = self.old.clone()
            self.content_wc = self.old_content_wc.clone()
            gt_decision = torch.ones(self.opt.batch_size, self.opt.used_strokes, device=self.device)

            for i in range(self.opt.used_strokes):
                content_wc = foregrounds[:, i, :, :, :]
                alpha = alphas[:, i, :, :, :]
                for j in range(i):
                    iou = (torch.sum(alpha * alphas[:, j, :, :, :], dim=(-3, -2, -1)) + 1e-5) / (
                            torch.sum(alphas[:, j, :, :, :], dim=(-3, -2, -1)) + 1e-5)
                    gt_decision[:, i] = ((iou < 0.8) | (~gt_decision[:, j].bool())).float() * gt_decision[:, i]
                decision = gt_decision[:, i].view(self.opt.batch_size, 1, 1, 1).contiguous()
                self.content_wc = torch.clip(content_wc*decision+ self.content_wc, torch.min(content_wc,self.content_wc),torch.max(content_wc,self.content_wc))
                self.render = self.content_wc
                
            self.gt_decision = gt_decision
            
            
            ## shuffle the batch
            idx = torch.randperm(self.gt_param.shape[0])
            self.gt_param = self.gt_param[idx].view(self.gt_param.size())
            self.gt_decision = self.gt_decision[idx].view(self.gt_decision.size())
            self.old = self.old[idx].view(self.old.size())
            self.render = self.render[idx].view(self.render.size())
            self.content_wc = self.content_wc[idx].view(self.content_wc.size())
            
    def set_input_grid(self, input_dict):
        self.image_paths = input_dict['A_paths']
        bg_stroke_num = int(self.opt.used_strokes * self.opt.multiply)
        with torch.no_grad():

            old_param = -1 + torch.rand(self.opt.batch_size, bg_stroke_num, self.d, device=self.device) * 2
            old_param = old_param.view(-1, self.d).contiguous()

            foregrounds, alphas = self.latent2stroke2_bg(old_param, self.patch_size, self.patch_size)

            foregrounds = foregrounds.view(self.opt.batch_size, bg_stroke_num, 4, self.patch_size,
                                            self.patch_size).contiguous()
            alphas = alphas.view(self.opt.batch_size, bg_stroke_num, 1, self.patch_size,
                                    self.patch_size).contiguous()
            result_content_wc = torch.zeros(self.opt.batch_size, 4, self.patch_size, self.patch_size, device=self.device)
            old_decision = torch.ones(self.opt.batch_size, bg_stroke_num, device=self.device)
            
            for i in range(bg_stroke_num):
                content_wc = foregrounds[:, i, :, :, :]
                alpha = alphas[:, i, :, :, :]
                # for j in range(i):
                #     iou = (torch.sum(alpha * alphas[:, j, :, :, :], dim=(-3, -2, -1)) + 1e-5) / (
                #             torch.sum(alphas[:, j, :, :, :], dim=(-3, -2, -1)) + 1e-5)
                #     old_decision[:, i] = ((iou < 0.95) | (~old_decision[:, j].bool())).float() * old_decision[:, i]
                decision = old_decision[:, i].view(self.opt.batch_size, 1, 1, 1).contiguous()
                result_content_wc = torch.clip(content_wc*decision+ result_content_wc, torch.min(content_wc,result_content_wc),torch.max(content_wc,result_content_wc))
                # old = result_content_wc


            old_param = -1 + torch.rand(self.opt.batch_size, bg_stroke_num, self.d, device=self.device) * 2
            old_param = old_param.view(-1, self.d).contiguous()

            foregrounds, alphas = self.latent2stroke_randomgrid(old_param, self.patch_size, self.patch_size)

            foregrounds = foregrounds.view(self.opt.batch_size, bg_stroke_num, 4, self.patch_size,
                                            self.patch_size).contiguous()
            alphas = alphas.view(self.opt.batch_size, bg_stroke_num, 1, self.patch_size,
                                    self.patch_size).contiguous()
            # result_content_wc = torch.zeros(self.opt.batch_size, 4, self.patch_size, self.patch_size, device=self.device)
            old_decision = torch.ones(self.opt.batch_size, bg_stroke_num, device=self.device)
            
            for i in range(bg_stroke_num):
                content_wc = foregrounds[:, i, :, :, :]
                alpha = alphas[:, i, :, :, :]
                # for j in range(i):
                #     iou = (torch.sum(alpha * alphas[:, j, :, :, :], dim=(-3, -2, -1)) + 1e-5) / (
                #             torch.sum(alphas[:, j, :, :, :], dim=(-3, -2, -1)) + 1e-5)
                #     old_decision[:, i] = ((iou < 0.8) | (~old_decision[:, j].bool())).float() * old_decision[:, i]
                decision = old_decision[:, i].view(self.opt.batch_size, 1, 1, 1).contiguous()
                result_content_wc = torch.clip(content_wc*decision+ result_content_wc, torch.min(content_wc,result_content_wc),torch.max(content_wc,result_content_wc))
                old = result_content_wc
                
            self.grid_class = self.randomly_generate_gridclasses(self.opt.batch_size)   # batch_size, 1, 256, 256
            self.grid_class = self.grid_class.to(self.device)
            
            self.old = old.view(self.opt.batch_size, 4, self.patch_size, self.patch_size).contiguous()
            self.real_bg = self.old
            self.old = self.old #* self.grid_class #+ (1-self.grid_class).repeat(1,4,1,1)
            self.old_content_wc = self.old.clone()

            gt_param = -1 + torch.rand(self.opt.batch_size, self.opt.used_strokes, self.d, device=self.device) * 2 # batch_size, stroke_num, param_dim
            gt_grid_class = self.grid_class.unsqueeze(1).repeat(1, self.opt.used_strokes, 1, 1, 1)  # batch_sizez, stroke_num, 1, 256, 256

            self.gt_param = gt_param[:, :, :self.d_shape]
            gt_param = gt_param.view(-1, self.d).contiguous()
            gt_grid_class = gt_grid_class.view(-1, 1, 256, 256).contiguous()
            self.gt_grid_class = gt_grid_class

            foregrounds, alphas = self.latent2stroke3(gt_param, self.patch_size, self.patch_size, gt_grid_class)
            foregrounds = foregrounds.view(self.opt.batch_size, self.opt.used_strokes, 4, self.patch_size,
                                           self.patch_size).contiguous()
            alphas = alphas.view(self.opt.batch_size, self.opt.used_strokes, 1, self.patch_size,
                                 self.patch_size).contiguous()
            self.render = self.old.clone()
            self.content_wc = self.old_content_wc.clone()
            gt_decision = torch.ones(self.opt.batch_size, self.opt.used_strokes, device=self.device)

            for i in range(self.opt.used_strokes):
                content_wc = foregrounds[:, i, :, :, :]
                alpha = alphas[:, i, :, :, :]
                for j in range(i):
                    iou = (torch.sum(alpha * alphas[:, j, :, :, :], dim=(-3, -2, -1)) + 1e-5) / (
                            torch.sum(alphas[:, j, :, :, :], dim=(-3, -2, -1)) + 1e-5)
                    gt_decision[:, i] = ((iou < 0.8) | (~gt_decision[:, j].bool())).float() * gt_decision[:, i]
                decision = gt_decision[:, i].view(self.opt.batch_size, 1, 1, 1).contiguous()
                self.content_wc = torch.clip(content_wc*decision+ self.content_wc, torch.min(content_wc,self.content_wc),torch.max(content_wc,self.content_wc))
                self.render = self.content_wc

            if self.opt.grid_cropout:
                self.old_grid_crop_resize = self.crop_out_grid(self.old, self.grid_class)
                self.render_grid_crop_resize = self.crop_out_grid(self.render, self.grid_class)
                self.grid_class_visualize = 1 - self.grid_class.repeat(1,4,1,1)
            self.gt_decision = gt_decision
            
            ## shuffle the batch
            # idx = torch.randperm(self.gt_param.shape[0])
            # self.gt_param = self.gt_param[idx].view(self.gt_param.size())
            # self.gt_decision = self.gt_decision[idx].view(self.gt_decision.size())
            # self.old = self.old[idx].view(self.old.size())
            # self.render = self.render[idx].view(self.render.size())
            # self.content_wc = self.content_wc[idx].view(self.content_wc.size())


    def forward(self):
        # param : batch_size, stroke_num, param_dim
        # decision : batch_size, stroke_num
        # render : batch_size, 4, 256, 256
        # old : batch_size, 4, 256, 256
        # grid_class : batch_size, 1, 256, 256
        if not self.opt.grid_class:
            param, decisions = self.net_g(self.render, self.old)
        else:
            if self.opt.grid_cropout:
                param, decisions = self.net_g(self.render_grid_crop_resize, self.old_grid_crop_resize, self.grid_class)
            else:
                param, decisions = self.net_g(self.render, self.old, self.grid_class)
        
        self.pred_decision = decisions.view(-1, self.opt.used_strokes).contiguous()
        self.pred_param = param[:, :, :self.d_shape] 
        param = param.view(-1, self.d).contiguous()   
        if not self.opt.generative:
            foregrounds, alphas = self.param2stroke(param, self.patch_size, self.patch_size)
            foregrounds = morphology.Dilation2d(m=1)(foregrounds)
            alphas = morphology.Erosion2d(m=1)(alphas)
        else:
            if self.opt.grid_class:
                foregrounds, alphas = self.latent2stroke3(param, self.patch_size, self.patch_size, self.gt_grid_class)
            else:
                foregrounds, alphas = self.latent2stroke2(param, self.patch_size, self.patch_size, int(self.opt.latent2stroke_cvalues[1]))

        foregrounds = foregrounds.view(-1, self.opt.used_strokes, 4, self.patch_size, self.patch_size)
        alphas = alphas.view(-1, self.opt.used_strokes, 1, self.patch_size, self.patch_size)
        decisions = networks.SignWithSigmoidGrad.apply(decisions.view(-1, self.opt.used_strokes, 1, 1, 1).contiguous())
        self.rec = self.old.clone()
        self.rec_content_wc = self.old.clone()
        
        for i in range(self.opt.used_strokes):
            content_wc = foregrounds[:, i, :, :, :]
            alpha = alphas[:, i, :, :, :]
            decision =decisions[:,i,:,:,:]
            self.rec_content_wc = torch.clip(content_wc*decision+ self.rec_content_wc, torch.min(content_wc,self.rec_content_wc),torch.max(content_wc,self.rec_content_wc))
            self.rec = self.rec_content_wc

        self.rec_grid_crop_resize = self.crop_out_grid(self.rec, self.grid_class)

    def optimize_parameters(self):
        self.forward()
        if not self.opt.long_horizon:
            if self.opt.grid_cropout:
                self.loss_pixel = self.criterion_pixel(self.rec_grid_crop_resize, self.render_grid_crop_resize) * self.opt.lambda_pixel
            else:
                self.loss_pixel = self.criterion_pixel(self.rec, self.render) * self.opt.lambda_pixel
        else:
            self.loss_pixel = self.criterion_pixel(self.rec, self.immediate_next) * self.opt.lambda_pixel
        cur_valid_gt_size = 0
        with torch.no_grad():
            r_idx = []
            c_idx = []
            for i in range(self.gt_param.shape[0]):  ## iterate over the batch
                is_valid_gt = self.gt_decision[i].bool()    # this is the boolean of 8 strokes in single image (8)
                valid_gt_param = self.gt_param[i, is_valid_gt] # this only contains the ground truth stroke parameters that are actually drawn on the canvas. (?,5)
                cost_matrix_l1 = torch.cdist(self.pred_param[i], valid_gt_param, p=1)  
                decision = self.pred_decision[i]  # 8
                cost_matrix_decision = (1 - decision).unsqueeze(-1).repeat(1, valid_gt_param.shape[0]) #  (8,?)
                r, c = linear_sum_assignment((cost_matrix_l1 + cost_matrix_decision).cpu()) # + cost_matrix_w + 
                r_idx.append(torch.tensor(r + self.pred_param.shape[1] * i, device=self.device))
                c_idx.append(torch.tensor(c + cur_valid_gt_size, device=self.device))
                cur_valid_gt_size += valid_gt_param.shape[0]
            r_idx = torch.cat(r_idx, dim=0)
            c_idx = torch.cat(c_idx, dim=0)
            paired_gt_decision = torch.zeros(self.gt_decision.shape[0] * self.gt_decision.shape[1], device=self.device)
            paired_gt_decision[r_idx] = 1.
        all_valid_gt_param = self.gt_param[self.gt_decision.bool(), :]
        all_pred_param = self.pred_param.view(-1, self.pred_param.shape[2]).contiguous()
        all_pred_decision = self.pred_decision.view(-1).contiguous()
        paired_gt_param = all_valid_gt_param[c_idx, :]
        paired_pred_param = all_pred_param[r_idx, :]
        self.loss_gt = self.criterion_pixel(paired_pred_param, paired_gt_param) * self.opt.lambda_gt
        self.loss_decision = self.criterion_decision(all_pred_decision, paired_gt_decision) * self.opt.lambda_decision
        loss = self.loss_pixel + self.loss_gt + self.loss_decision 
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        