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

EPS = 1e-1

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
        self.visual_names = ['old', 'render', 'rec']
        self.model_names = ['g']
        self.d = 9  # latent 5 + rgb 3
        self.d_shape = 9

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
            model = BetaVAE_B_256_Conditional(z_dim=5, c_dim = 2, nc=1)       
            state = torch.load(os.path.join('./strokes_alpha_gamma100_z5_c2_size256_last.pt'),map_location='cpu')        
            model.load_state_dict(state['model_states']['net'])
    #         model = model.detach()
            for param in model.parameters():
                print(param, param.requires_grad)
                param.requires_grad = False
                
            self.generative_model = model.to(self.device)
        
        
        net_g = networks.Painter(self.d_shape, opt.used_strokes, opt.ngf,
                                 n_enc_layers=opt.num_blocks, n_dec_layers=opt.num_blocks)
        self.net_g = networks.init_net(net_g, opt.init_type, opt.init_gain, self.gpu_ids)
        self.old = None
        self.render = None
        self.rec = None
        self.gt_param = None
        self.pred_param = None
        self.gt_decision = None
        self.pred_decision = None
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

    def param2stroke(self, param, H, W):
        # param: b, 12
        b = param.shape[0]
        param_list = torch.split(param, 1, dim=1)
        x0, y0, w, h, theta = [item.squeeze(-1) for item in param_list[:5]]
        R0, G0, B0, R2, G2, B2, _ = param_list[5:]
        sin_theta = torch.sin(torch.acos(torch.tensor(-1., device=param.device)) * theta)
        cos_theta = torch.cos(torch.acos(torch.tensor(-1., device=param.device)) * theta)
        index = torch.full((b,), -1, device=param.device)
        index[h > w] = 0
        index[h <= w] = 1
        brush = self.meta_brushes[index.long()]
        alphas = torch.cat([brush, brush, brush], dim=1)
        alphas = (alphas > 0).float()
        t = torch.arange(0, brush.shape[2], device=param.device).unsqueeze(0) / brush.shape[2]
        color_map = torch.stack([R0 * (1 - t) + R2 * t, G0 * (1 - t) + G2 * t, B0 * (1 - t) + B2 * t], dim=1)
        color_map = color_map.unsqueeze(-1).repeat(1, 1, 1, brush.shape[3])
        brush = brush * color_map
#         print('1',alphas)
        warp_00 = cos_theta / w
        warp_01 = sin_theta * H / (W * w)
        warp_02 = (1 - 2 * x0) * cos_theta / w + (1 - 2 * y0) * sin_theta * H / (W * w)
        warp_10 = -sin_theta * W / (H * h)
        warp_11 = cos_theta / h
        warp_12 = (1 - 2 * y0) * cos_theta / h - (1 - 2 * x0) * sin_theta * W / (H * h)
        warp_0 = torch.stack([warp_00, warp_01, warp_02], dim=1)
        warp_1 = torch.stack([warp_10, warp_11, warp_12], dim=1)
        warp = torch.stack([warp_0, warp_1], dim=1)
#         print(warp.shape)
        grid = torch.nn.functional.affine_grid(warp, torch.Size((b, 3, H, W)), align_corners=False)
        brush = torch.nn.functional.grid_sample(brush, grid, align_corners=False)
        alphas = torch.nn.functional.grid_sample(alphas, grid, align_corners=False)
#         print('2',alphas)
        return brush, alphas
    
    
    def latent2stroke(self, param, H,W):
        # param: b, 10 (latent) + 3 (RGB)
        T = torchvision.transforms.Resize([H,W])
        b = param.shape[0]
#         print(param[:,:-3].shape)
#         with torch.no_grad():
        param_latent = (param[:,:5] / torch.norm(param[:,:5],dim=1).unsqueeze(1))*self.opt.sigma
        img = self.generative_model.sample(param_latent) ### this outputs bx3xHxW image
        # print(param[:,:5])
        img = T(img)
        img = img.repeat(1,3,1,1)
        # print(img.shape)
        alphas = (img>0.1).float()
#         img[img<0.3] = 0
        img = alphas*img
        rgb = (1+param[:,5:8]).unsqueeze(2).unsqueeze(3)/2
        # print('rgb',rgb)
        # if alpha
#         print(img.device, rgb.device)
        # print(rgb)
        brush = img*rgb
    
        return brush, alphas
    
    def latent2stroke2(self, param, H,W):
        # param: b, 10 (latent) + 3 (RGB)
        trn_resize = torchvision.transforms.Resize([H+20,W+20])
        trn_crop= torchvision.transforms.CenterCrop([H,W])
        trn_ToPIL = torchvision.transforms.ToPILImage(mode='CMYK')
        trn_ToTensor = torchvision.transforms.ToTensor()
        b = param.shape[0]
#         print(param[:,:-3].shape)
#         with torch.no_grad():
        param_latent = (param[:,:5] / torch.norm(param[:,:5],dim=1).unsqueeze(1))*self.opt.sigma
        c = torch.zeros(param.shape[0]).cuda()
        orig_img = self.generative_model.sample(param_latent, c) ### this outputs bx1xHxW image
        orig_img = trn_crop(trn_resize(orig_img))
        matte = (orig_img>EPS).float()
        cmyk = matte.repeat(1,4,1,1)
        # print(param[:,:5])
        # batch = []
        # for i in range(img.shape[0]):
        #     cmyk = trn_resize(trn_ToTensor(trn_ToPIL(img[0]))).unsqueeze(0)
        #     batch.append(cmyk)
        # cmyk = torch.cat(batch,0).cuda()
        # print(cmyk.shape)
        # cmyk = trn_resize(trn_ToTensor((trn_ToPIL(img))))
        
        # img = img.repeat(1,3,1,1)
        # content = (img>1e-1).float()
        # content = content.repeat(1,3,1,1)
        alpha = orig_img
        binary = (orig_img>EPS).float()

        # print(img.shape)
        # alphas = (img>0.1).float()
#         img[img<0.3] = 0
        # img = alphas*img
        color = (1+param[:,5:9]).unsqueeze(2).unsqueeze(3)/2
        # aug_color = torch.ones(orig_img.shape[0],1,1,1).cuda()
        # color = torch.cat([color, aug_color],1)
        # print(color.shape)
        # print('rgb',rgb)
        # if alpha
#         print(img.device, rgb.device)
        matte_color = cmyk*color*alpha
        # print(rgb)
    
        return matte_color, alpha


    # def set_input(self, input_dict):
    #     self.image_paths = input_dict['A_paths']
    #     with torch.no_grad():
    #         old_param = torch.rand(self.opt.batch_size // 4, self.opt.used_strokes, self.d, device=self.device)
    #              # batch_size //4 because we are gonna create a background by drawing 4x larger images and splitting it to 4
    #         old_param[:, :, :4] = old_param[:, :, :4] * 0.5 + 0.2
    #         old_param[:, :, -4:-1] = old_param[:, :, -7:-4]
    #         old_param = old_param.view(-1, self.d).contiguous()
    #         foregrounds, alphas = self.param2stroke(old_param, self.patch_size * 2, self.patch_size * 2)
    #         foregrounds = morphology.Dilation2d(m=1)(foregrounds)
    #         alphas = morphology.Erosion2d(m=1)(alphas)
    #         foregrounds = foregrounds.view(self.opt.batch_size // 4, self.opt.used_strokes, 3, self.patch_size * 2,
    #                                        self.patch_size * 2).cparam[:,:5]
    #             alpha = alphas[:, i, :, :, :]
    #             old = foreground * alpha + old * (1 - alpha)
    #         old = old.view(self.opt.batch_size // 4, 3, 2, self.patch_size, 2, self.patch_size).contiguous()
    #         old = old.permute(0, 2, 4, 1, 3, 5).contiguous()
    #         self.old = old.view(self.opt.batch_size, 3, self.patch_size, self.patch_size).contiguous()

    #         gt_param = torch.rand(self.opt.batch_size, self.opt.used_strokes, self.d, device=self.device)
    #         gt_param[:, :, :4] = gt_param[:, :, :4] * 0.5 + 0.2
    #         gt_param[:, :, -4:-1] = gt_param[:, :, -7:-4]
    #         self.gt_param = gt_param[:, :, :self.d_shape]
    #         gt_param = gt_param.view(-1, self.d).contiguous()
    #         foregrounds, alphas = self.param2stroke(gt_param, self.patch_size, self.patch_size)
    #         foregrounds = morphology.Dilation2d(m=1)(foregrounds)
    #         alphas = morphology.Erosion2d(m=1)(alphas)
    #         foregrounds = foregrounds.view(self.opt.batch_size, self.opt.used_strokes, 3, self.patch_size,
    #                                        self.patch_size).contiguous()
    #         alphas = alphas.view(self.opt.batch_size, self.opt.used_strokes, 3, self.patch_size,
    #                              self.patch_size).contiguous()
    #         self.render = self.old.clone()
    #         gt_decision = torch.ones(self.opt.batch_size, self.opt.used_strokes, device=self.device)

    #         for i in range(self.opt.used_strokes):
    #             foreground = foregrounds[:, i, :, :, :]
    #             alpha = alphas[:, i, :, :, :]
    #             for j in range(i):
    #                 iou = (torch.sum(alpha * alphas[:, j, :, :, :], dim=(-3, -2, -1)) + 1e-5) / (
    #                         torch.sum(alphas[:, j, :, :, :], dim=(-3, -2, -1)) + 1e-5)
    #                 gt_decision[:, i] = ((iou < 0.75) | (~gt_decision[:, j].bool())).float() * gt_decision[:, i]
    #             decision = gt_decision[:, i].view(self.opt.batch_size, 1, 1, 1).contiguous()
    #             self.render = foreground * alpha * decision + self.render * (1 - alpha * decision)
    #         self.gt_decision = gt_decision


    def set_input(self, input_dict, background_stroke_times):
        self.image_paths = input_dict['A_paths']
        stroke_num = np.random.randint(8,self.opt.used_strokes * background_stroke_times)
        with torch.no_grad():
            if not self.opt.generative:
                old_param = torch.rand(self.opt.batch_size, stroke_num, self.d, device=self.device)
                     # batch_size //4 because we are gonna create a background by drawing 4x larger images and splitting it to 4
                old_param[:, :, :4] = old_param[:, :, :4] * 0.5 + 0.2
                old_param[:, :, -4:-1] = old_param[:, :, -7:-4]
            else:
                old_param = -1 + torch.rand(self.opt.batch_size, stroke_num, self.d, device=self.device) * 2
                # old_param[:,:,:11] = -3 + torch.rand(self.opt.batch_size, self.opt.used_strokes, 11, device=self.device) * 6
#                 old_param[:,:,:5] = self.opt.sigma * (old_param[:,:,:5] / torch.norm(old_param[:,:,:5], dim=2).unsqueeze(2))
                     # batch_size //4 because we are gonna create a background by drawing 4x larger images and splitting it to 4
#                 old_param[:, :, :4] = old_param[:, :, :4] * 0.5 + 0.2
#                 old_param[:, :, -4:-1] = old_param[:, :, -7:-4]

            old_param = old_param.view(-1, self.d).contiguous()
            if not self.opt.generative:
                foregrounds, alphas = self.param2stroke(old_param, self.patch_size, self.patch_size)
                foregrounds = morphology.Dilation2d(m=1)(foregrounds)
                alphas = morphology.Erosion2d(m=1)(alphas)
            else:
                foregrounds, alphas = self.latent2stroke2(old_param, self.patch_size, self.patch_size)
            # elif : 
            foregrounds = foregrounds.view(self.opt.batch_size, stroke_num, 4, self.patch_size,
                                           self.patch_size).contiguous()
            alphas = alphas.view(self.opt.batch_size, stroke_num, 1, self.patch_size,
                                 self.patch_size).contiguous()
            result_content_wc = torch.zeros(self.opt.batch_size, 4, self.patch_size, self.patch_size, device=self.device)
            result_alpha = torch.zeros(self.opt.batch_size, 1, self.patch_size, self.patch_size, device=self.device)
            for i in range(stroke_num):
                content_wc = foregrounds[:, i, :, :, :]
                alpha = alphas[:, i, :, :, :]
                # binary = binaries[:,i,:,:,:]
                # result_content_wc = (result_content_wc>1e-1).float()*(content_wc>1e-1).float()*content_wc * result_content_wc + content_wc * (1-(result_content_wc>1e-1).float()) + (1-(content_wc>1e-1).float()) * result_content_wc
                result_content_wc = torch.clip(content_wc+ result_content_wc, torch.min(content_wc,result_content_wc),torch.max(content_wc,result_content_wc))
                # result_alpha = 1 - (1-result_alpha)* (1-alpha)
                old = result_content_wc
            # old = old.view(self.opt.batch_size, 3, self.patch_size, self.patch_size).contiguous()
            # old = old.permute(0, 2, 4, 1, 3, 5).contiguous()
#             del old_param
#             del foregrounds
#             del alphas
#             torch.cuda.empty_cache() 
            self.old = old.view(self.opt.batch_size, 4, self.patch_size, self.patch_size).contiguous()
            self.old_content_wc = result_content_wc.clone()
            self.old_alpha = result_alpha.clone()

            if not self.opt.generative:
                gt_param = torch.rand(self.opt.batch_size, self.opt.used_strokes, self.d, device=self.device)
                gt_param[:, :, :4] = gt_param[:, :, :4] * 0.5 + 0.2
                gt_param[:, :, -4:-1] = gt_param[:, :, -7:-4]
            else:
                gt_param = -1 + torch.rand(self.opt.batch_size, self.opt.used_strokes, self.d, device=self.device) * 2
                # gt_param[:,:,:11] = -3 + torch.rand(self.opt.batch_size, self.opt.used_strokes, 11, device=self.device) * 6
#                 gt_param[:,:,:5] = self.opt.sigma * (gt_param[:,:,:5] / torch.norm(gt_param[:,:,:5], dim=2).unsqueeze(2))

            self.gt_param = gt_param[:, :, :self.d_shape]
            gt_param = gt_param.view(-1, self.d).contiguous()
            if not self.opt.generative:
                foregrounds, alphas = self.param2stroke(gt_param, self.patch_size, self.patch_size)
                foregrounds = morphology.Dilation2d(m=1)(foregrounds)
                alphas = morphology.Erosion2d(m=1)(alphas)
            else:
                foregrounds, alphas = self.latent2stroke2(gt_param, self.patch_size, self.patch_size)
            foregrounds = foregrounds.view(self.opt.batch_size, self.opt.used_strokes, 4, self.patch_size,
                                           self.patch_size).contiguous()
            alphas = alphas.view(self.opt.batch_size, self.opt.used_strokes, 1, self.patch_size,
                                 self.patch_size).contiguous()
            self.render = self.old.clone()
            self.content_wc = self.old_content_wc.clone()
            self.alpha = self.old_alpha.clone()
            gt_decision = torch.ones(self.opt.batch_size, self.opt.used_strokes, device=self.device)

            for i in range(self.opt.used_strokes):
                content_wc = foregrounds[:, i, :, :, :]
                alpha = alphas[:, i, :, :, :]
                for j in range(i):
                    iou = (torch.sum(alpha * alphas[:, j, :, :, :], dim=(-3, -2, -1)) + 1e-5) / (
                            torch.sum(alphas[:, j, :, :, :], dim=(-3, -2, -1)) + 1e-5)
                    gt_decision[:, i] = ((iou < 0.8) | (~gt_decision[:, j].bool())).float() * gt_decision[:, i]
                decision = gt_decision[:, i].view(self.opt.batch_size, 1, 1, 1).contiguous()
                # print(decision.shape)
                # if decision ==1 : 
                self.content_wc = torch.clip(content_wc*decision+ self.content_wc, torch.min(content_wc,self.content_wc),torch.max(content_wc,self.content_wc))
                # self.alpha = 1 - (1-self.alpha)* (1-alpha*decision)
                self.render = self.content_wc

                # self.render = foreground * alpha * decision + self.render * (1 - alpha * decision)
            self.gt_decision = gt_decision
#             print('self.gt_decision :', self.gt_decision.shape)
#             print('self.gt_param : ', self.gt_param.shape)
            



    def forward(self):
        param, decisions = self.net_g(self.render, self.old)
        # print(self.net_g.linear_param[0].weight)
        # print('latent', param[:,:5])
        # print('color', param[:,5:8])
#         print(param.shape)
        # stroke_param: b, stroke_per_patch, param_per_stroke
        # decision: b, stroke_per_patch, 1
        self.pred_decision = decisions.view(-1, self.opt.used_strokes).contiguous()
        self.pred_param = param[:, :, :self.d_shape] ## 3,8,5
        param = param.view(-1, self.d).contiguous()   # 24,12
        if not self.opt.generative:
            foregrounds, alphas = self.param2stroke(param, self.patch_size, self.patch_size)
            foregrounds = morphology.Dilation2d(m=1)(foregrounds)
            alphas = morphology.Erosion2d(m=1)(alphas)
        else:
            foregrounds, alphas = self.latent2stroke2(param, self.patch_size, self.patch_size)

        # foreground, alpha: b * stroke_per_patch, 3, output_size, output_size
        foregrounds = foregrounds.view(-1, self.opt.used_strokes, 4, self.patch_size, self.patch_size)
        alphas = alphas.view(-1, self.opt.used_strokes, 1, self.patch_size, self.patch_size)
        # foreground, alpha: b, stroke_per_patch, 3, output_size, output_size
        decisions = networks.SignWithSigmoidGrad.apply(decisions.view(-1, self.opt.used_strokes, 1, 1, 1).contiguous())
        # print('decisions',decisions)
        self.rec = self.old.clone()
        self.rec_content_wc = self.old_content_wc.clone()
        self.rec_alpha = self.old_alpha.clone()
        
        # for j in range(foregrounds.shape[1]):
        #     foreground = foregrounds[:, j, :, :, :]
        #     alpha = alphas[:, j, :, :, :]
        #     decision = decisions[:, j, :, :, :]
        #     # print((alpha==0).all())
        #     # print(foreground.shape, decision.shape, alpha.shape)
        #     self.rec = foreground * alpha * decision + self.rec * (1 - alpha * decision)
        for i in range(self.opt.used_strokes):
            content_wc = foregrounds[:, i, :, :, :]
            alpha = alphas[:, i, :, :, :]
            decision =decisions[:,i,:,:,:]
            # for j in range(i):
            #     iou = (torch.sum(alpha * alphas[:, j, :, :, :], dim=(-3, -2, -1)) + 1e-5) / (
            #             torch.sum(alphas[:, j, :, :, :], dim=(-3, -2, -1)) + 1e-5)
            #     gt_decision[:, i] = ((iou < 0.8) | (~gt_decision[:, j].bool())).float() * gt_decision[:, i]
            # decision = gt_decision[:, i].view(self.opt.batch_size, 1, 1, 1).contiguous()
            # print(decision.shape)
            # if decision ==1 : 
            self.rec_content_wc = torch.clip(content_wc*decision+ self.rec_content_wc, torch.min(content_wc,self.rec_content_wc),torch.max(content_wc,self.rec_content_wc))
            # self.rec_alpha = 1 - (1-self.rec_alpha)* (1-alpha*decision)
            self.rec = self.rec_content_wc

    @staticmethod
    def get_sigma_sqrt(w, h, theta):
        sigma_00 = w * (torch.cos(theta) ** 2) / 2 + h * (torch.sin(theta) ** 2) / 2
        sigma_01 = (w - h) * torch.cos(theta) * torch.sin(theta) / 2
        sigma_11 = h * (torch.cos(theta) ** 2) / 2 + w * (torch.sin(theta) ** 2) / 2
        sigma_0 = torch.stack([sigma_00, sigma_01], dim=-1)
        sigma_1 = torch.stack([sigma_01, sigma_11], dim=-1)
        sigma = torch.stack([sigma_0, sigma_1], dim=-2)
        return sigma

    @staticmethod
    def get_sigma(w, h, theta):
        sigma_00 = w * w * (torch.cos(theta) ** 2) / 4 + h * h * (torch.sin(theta) ** 2) / 4
        sigma_01 = (w * w - h * h) * torch.cos(theta) * torch.sin(theta) / 4
        sigma_11 = h * h * (torch.cos(theta) ** 2) / 4 + w * w * (torch.sin(theta) ** 2) / 4
        sigma_0 = torch.stack([sigma_00, sigma_01], dim=-1)
        sigma_1 = torch.stack([sigma_01, sigma_11], dim=-1)
        sigma = torch.stack([sigma_0, sigma_1], dim=-2)
        return sigma

    def gaussian_w_distance(self, param_1, param_2):
        mu_1, w_1, h_1, theta_1 = torch.split(param_1, (2, 1, 1, 1), dim=-1)
        w_1 = w_1.squeeze(-1)
        h_1 = h_1.squeeze(-1)
        theta_1 = torch.acos(torch.tensor(-1., device=param_1.device)) * theta_1.squeeze(-1)
        trace_1 = (w_1 ** 2 + h_1 ** 2) / 4
        mu_2, w_2, h_2, theta_2 = torch.split(param_2, (2, 1, 1, 1), dim=-1)
        w_2 = w_2.squeeze(-1)
        h_2 = h_2.squeeze(-1)
        theta_2 = torch.acos(torch.tensor(-1., device=param_2.device)) * theta_2.squeeze(-1)
        trace_2 = (w_2 ** 2 + h_2 ** 2) / 4
        sigma_1_sqrt = self.get_sigma_sqrt(w_1, h_1, theta_1)
        sigma_2 = self.get_sigma(w_2, h_2, theta_2)
        trace_12 = torch.matmul(torch.matmul(sigma_1_sqrt, sigma_2), sigma_1_sqrt)
        trace_12 = torch.sqrt(trace_12[..., 0, 0] + trace_12[..., 1, 1] + 2 * torch.sqrt(
            trace_12[..., 0, 0] * trace_12[..., 1, 1] - trace_12[..., 0, 1] * trace_12[..., 1, 0]))
        return torch.sum((mu_1 - mu_2) ** 2, dim=-1) + trace_1 + trace_2 - 2 * trace_12

    def optimize_parameters(self):
        self.forward()
        self.loss_pixel = self.criterion_pixel(self.rec, self.render) * self.opt.lambda_pixel
        cur_valid_gt_size = 0
        with torch.no_grad():
            r_idx = []
            c_idx = []
            for i in range(self.gt_param.shape[0]):  ## iterate over the batch
                is_valid_gt = self.gt_decision[i].bool()    # this is the boolean of 8 strokes in single image (8)
                valid_gt_param = self.gt_param[i, is_valid_gt] # this only contains the ground truth stroke parameters that are actually drawn on the canvas. (?,5)
                cost_matrix_l1 = torch.cdist(self.pred_param[i], valid_gt_param, p=1)   # calculate the cdist between the pred_param (shape 8,5) and valid_gt_param (shape ?, 5)
                  # this shape is (?,5)
#                 pred_param_broad = self.pred_param[i].unsqueeze(1).contiguous().repeat(  
#                     1, valid_gt_param.shape[0], 1)    # self.pred_param[i] shape is (8,5)
                # then it becomes (8,1,5)  then it becomes (8,?,5)
#                 valid_gt_param_broad = valid_gt_param.unsqueeze(0).contiguous().repeat(
#                     self.pred_param.shape[1], 1, 1)
                        # it becomes (1,?,5) -> (8,?,5)
                
#                 cost_matrix_w = self.gaussian_w_distance(pred_param_broad, valid_gt_param_broad)
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
        # print(paired_pred_param.shape)
        self.loss_gt = self.criterion_pixel(paired_pred_param, paired_gt_param) * self.opt.lambda_gt
#         self.loss_w = self.gaussian_w_distance(paired_pred_param, paired_gt_param).mean() * self.opt.lambda_w
        self.loss_decision = self.criterion_decision(all_pred_decision, paired_gt_decision) * self.opt.lambda_decision
        loss = self.loss_pixel + self.loss_gt + self.loss_decision # + self.loss_w 
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        