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


def latent2stroke2(param, H,W, model, device, decide_largesmall, choice=None):
    # param: b, 10 (latent) + 3 (RGB)
    trn_resize = torchvision.transforms.Resize([H+40,W+40])
    trn_crop= torchvision.transforms.CenterCrop([H,W])
    trn_ToPIL = torchvision.transforms.ToPILImage(mode='CMYK')
    trn_ToTensor = torchvision.transforms.ToTensor()
    b = param.shape[0]
#         print(param[:,:-3].shape)
#         with torch.no_grad():
    param_latent = (param[:,:5] / torch.norm(param[:,:5],dim=1).unsqueeze(1))*2
    if decide_largesmall==1:
        c = network.SignWithSigmoidGrad.apply(param[:,-5])  
        # print(param_latent.shape, c.shape)
        orig_img = model.sample(param_latent, c) ### this outputs bx1xHxW image
    else:
        if choice is not None:
            c = torch.ones(param.shape[0]) * choice
            orig_img = model.sample(param_latent, c)
        else:
            orig_img = model.sample(param_latent) ### this outputs bx1xHxW image
    orig_img = trn_crop(trn_resize(orig_img))
    matte = (orig_img>EPS).float()
    cmyk = matte.repeat(1,4,1,1)
    alpha = orig_img
    binary = (orig_img>EPS).float()

    color = (1+param[:,-4:]).unsqueeze(2).unsqueeze(3)/2
    # if not details:
    matte_color = cmyk*color*alpha
    # else:
    #     matte_color = cmyk*color
    return matte_color, binary



def read_img(img_path, img_type='RGB', h=None, w=None):
    img = Image.open(img_path).convert(img_type)
    if h is not None and w is not None:
        img = img.resize((w, h), resample=Image.NEAREST)
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float() / 255.
    return img


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


def main(input_path, model_path, model_path2, c_dim, generative_path, detail_c_dim, detail_generative_path, output_dir, generative = True, need_animation=False, resize_h=None, resize_w=None, repeat_num = 10,repeat_num_detail=10, stroke_num = 10, stroke_num_detail = 10, serial=False, decision_switch=True, decide_largesmall=0,detail=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    input_name = os.path.basename(input_path)
    output_path = os.path.join(output_dir, input_name)
    frame_dir = None
    if need_animation:
        if not serial:
            print('It must be under serial mode if animation results are required, so serial flag is set to True!')
            serial = True
        frame_dir = os.path.join(output_dir, input_name[:input_name.find('.')])
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir+"/cropped")
    # device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    json.dump(locals(), open(os.path.join(frame_dir, '../arguments.json'), 'w'))
    # device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    try:
        net_g = network.Painter(8, stroke_num, 512, 8, 3, 3,largesmall=decide_largesmall).to(device)
        net_g.load_state_dict(torch.load(model_path))
    except:
        print('loaded original smaller model')
        net_g = network.Painter_Original(8, stroke_num, 256, 8, 3, 3).to(device)
        net_g.load_state_dict(torch.load(model_path))
    net_g.eval()
    
    for param in net_g.parameters():
        param.requires_grad = False


    if not generative:
        brush_large_vertical = read_img('brush/brush_large_vertical.png', 'L').to(device)
        brush_large_horizontal = read_img('brush/brush_large_horizontal.png', 'L').to(device)
        meta_brushes = torch.cat(
            [brush_large_vertical, brush_large_horizontal], dim=0)
    else:
        if c_dim>0:
            model = BetaVAE_B_256_Conditional(z_dim=5, c_dim = c_dim, nc=1)       
            state = torch.load(os.path.join(generative_path),map_location='cpu')        
            model.load_state_dict(state['model_states']['net'])
        else:
            model = BetaVAE_B_256(z_dim=5, nc=1)       
            state = torch.load(os.path.join(generative_path),map_location='cpu')        
            model.load_state_dict(state['model_states']['net'])
            
        for param in model.parameters():
            param.requires_grad = False
            
        generative_model = model.to(device)
        

    with torch.no_grad():
        original_img = read_img(input_path, 'CMYK', resize_h, resize_w).to(device)  # 이미지 읽어옴
        final_result = torch.zeros_like(original_img).to(device)
        save_img(original_img[0], frame_dir, "target.png")
        
        img_size = 256
        idx=-1
        ### Background Drawing   (256 -> 128 -> 64 -> 32 -> 16)
        for repeat in tqdm(range(repeat_num)):
            # for j in range(2):
                # print(original_img.shape)
            idx= idx+1
            if repeat>0:
                img_size = img_size//2
                original_img = original_img.reshape(-1,4,2**repeat,img_size,2**repeat,img_size)  # 1 4 2 128 2 128
                original_img = original_img.permute(0,2,4,1,3,5)  # 1 2 2 4 128 128 
                original_img = original_img.reshape(-1, 4, img_size, img_size) # 4 4 128 128 

                final_result = final_result.reshape(-1,4,2**repeat,img_size, 2**repeat,img_size)
                final_result = final_result.permute(0,2,4,1,3,5)
                final_result = final_result.reshape(-1, 4, img_size, img_size)

            original_img = transforms.Resize([256,256])(original_img)
            final_result = transforms.Resize([256,256])(final_result)
            for j in range(original_img.shape[0]):
                save_img(final_result[j], frame_dir+"/cropped", "{:02d}_{}_current.png".format(idx,j))
                save_img(original_img[j], frame_dir+"/cropped", "{:02d}_{}_target.png".format(idx,j))

            for kk in range(2):
                param, decisions = net_g(original_img, final_result)
                if decide_largesmall==1:
                    param = param.view(-1, 10).contiguous()
                else:
                    param = param.view(-1, 9).contiguous()
                
                foregrounds, alphas = latent2stroke2(param, resize_h, resize_w, generative_model, device, decide_largesmall, choice=0)
                foregrounds = foregrounds.view(-1, stroke_num, 4, resize_h, resize_w)
                alphas = alphas.view(-1, stroke_num, 1, resize_h, resize_w)
                decisions = network.SignWithSigmoidGrad.apply(decisions.view(-1, stroke_num, 1, 1, 1).contiguous())
                for j in range(foregrounds.shape[1]):
                    foreground = foregrounds[:, j, :, :, :]
                    alpha = alphas[:, j, :, :, :]
                    decision = decisions[:, j, :, :, :]
                    if decision_switch:
                        final_result = torch.clip(foreground*decision + final_result, torch.min(foreground, final_result), torch.max(foreground, final_result))
                    else:
                        final_result = torch.clip(foreground + final_result, torch.min(foreground, final_result), torch.max(foreground, final_result))
            final_result = transforms.Resize([img_size,img_size])(final_result)
            original_img = transforms.Resize([img_size,img_size])(original_img)
            for j in range(original_img.shape[0]):
                save_img(final_result[j], frame_dir+"/cropped", "{:02d}_{}_predicted.png".format(idx,j))

            # print(final_result.shape)   # 16 4 64 64
            #                             # 4 4 4 64 64 
                                          # 4 4 64 4 64 
                                          # 4 256 256 
            # final_result = final_result.reshape(2**repeat,2**repeat,-1,img_size, img_size)
            # final_result = final_result.permute(2,0,3,1,4)
            # final_result = final_result.reshape(4,256,2**repeat,img_size)
            # final_result = final_result.reshape(4,256,256).unsqueeze(0)
            # final_result = final_result.reshape(4, img_size * 2**repeat * 2**repeat, img_size )
            final_result = make_grid(final_result, nrow=2**repeat, padding=0).unsqueeze(0)
            print(final_result.shape)
            save_img(final_result[0], frame_dir+"/cropped", "{:02d}_predicted_resized_intermediate.png".format(idx))
            # final_result = final_result.reshape(1, 4, 256, 256)
            original_img = make_grid(original_img, nrow=2**repeat, padding=0).unsqueeze(0)

            if repeat%1==0:
                save_img(final_result[0], frame_dir, "{:02d}_final.png".format(idx))
                     
        command = 'convert {}/*.png {}/{}.gif'.format(frame_dir, frame_dir, input_name[:input_name.find('.')])
        print("converting to {}.gif".format(input_name[:input_name.find('.')]))
        os.system(command)






if __name__ == '__main__':
    pic_list = ['sangok','1','2','3','starry_night','gradient','ocean','jennifer','face']
    # pic_list = ['sangok']
    model_path = "../train/checkpoints/REVISITING_fore32_back1to96_gt100_pix30_dec10_300epoch_genz5c2_lr1e-5/latest_net_g.pth"
    detail_model_path = '../train/checkpoints/DEEPER_DETAILS_ONLY_stroke200_gt100_pix50_dec10_600epoch_genz5c2_600epoch_lr1e-5/latest_net_g.pth'
    output_dir = './output/cropped_testing_real7'
    generative_path = '../train/strokes_alpha_gamma100_z5_c2_size256_iter_750000.pt'
    generative_path2 = '../train/markers_large_small_gamma100_z5_c2_size256_last.pt'
    for i in pic_list:
        main(input_path='../picture/{}.jpg'.format(i),
            model_path=model_path,
            model_path2=detail_model_path,
            c_dim = 2, 
            generative_path = generative_path,
            detail_c_dim = 2, 
            detail_generative_path = generative_path2,
            output_dir=output_dir,
            generative=True,
            need_animation=True,  # whether need intermediate results for animation.
            resize_h=256,         # resize original input to this size. None means do not resize.
            resize_w=256,         # resize original input to this size. None means do not resize.
            repeat_num = 4,
            repeat_num_detail = 20,
            stroke_num = 32, 
            stroke_num_detail = 200,
            serial=True,
            decision_switch = True,
            decide_largesmall = 0,
            detail=False)          # if need animation, serial must be True.
    command = 'zip -r {}.zip {}'.format(output_dir, output_dir)
    os.system(command)

## delete 618th