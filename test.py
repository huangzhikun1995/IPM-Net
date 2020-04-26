# -*- coding: utf-8 -*-
from __future__ import print_function
from utils import get_config, pytorch03_to_pytorch04
from trainer import IPMNet_Trainer 
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
import random
from torchvision import transforms
from PIL import Image
import time

# number of iterations
num = '01000000'
# config name 
name = 'config'

if not os.path.isdir('./outputs/%s'%name):
    assert 0, "please changse the name to your model name"
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./outputs/%s/config.yaml'%name, help="net configuration")
parser.add_argument('--checkpoint', type=str, default="./outputs/%s/checkpoints/gen_%s.pt"%(name, num), help="checkpoint of autoencoders")
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and 0 for b2a")
parser.add_argument('--sty_num', type=int, default=30, help="the number of randomly selected reference images.")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--num_style',type=int, default=10, help="number of styles to sample")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_path', type=str, default="./test_results/%s/%s"%(name, num), help="output image path")
parser.add_argument('--trainer', type=str, default='IPMNet')
opts = parser.parse_args()


opts.input = '~/IPM-Net/dataset/makeup/testB/'
opts.input_mask ='~/IPM-Net/dataset/makeup/testB_mask/'
opts.input_texture ='~/IPM-Net/dataset/makeup/testB_highcontract/'
opts.style = '~/IPM-Net/dataset/makeup/testA/'
opts.style_mask ='~/IPM-Net/dataset/makeup/testA_mask/'
opts.style_texture ='~/IPM-Net/dataset/makeup/testA_highcontract/'

if not os.path.exists(opts.output_path):
    os.makedirs(opts.output_path)

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
# Load experiment setting
config = get_config(opts.config)
opts.num_style = 1 if opts.style != '' else opts.num_style

# Setup model and data loader
# config['vgg_model_path'] = opts.output_path
if opts.trainer == 'MUNIT':
    style_dim = config['gen']['style_dim']
    trainer = IPMNet_Trainer(config)

else:
    sys.exit("Only support MUNIT|UNIT")

try:
    state_dict = torch.load(opts.checkpoint)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
except:
    state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint), opts.trainer)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])

trainer.cuda()
trainer.eval()
encode = trainer.gen_a.encode #if opts.a2b else trainer.gen_b.encode # encode function
style_encode = trainer.gen_b.encode #if opts.a2b else trainer.gen_a.encode # encode function
decode = trainer.gen_b.decode #if opts.a2b else trainer.gen_a.decode # decode function

if 'new_size' in config:
    new_size = config['new_size']
else:
    new_size = config['new_size_a']

if 'crop_image_height' and 'crop_image_width'  in config:
    height = config['crop_image_height']
    weight = config['crop_image_width']


# get files lists
image_files = os.listdir(opts.input)
style_files = os.listdir(opts.style)

img_transform = transforms.Compose([transforms.Resize(new_size),
                                        transforms.CenterCrop((height, weight)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
mask_transform = transforms.Compose([transforms.Resize(new_size),
                                        transforms.CenterCrop((height, weight)),
                                        transforms.ToTensor()])
with torch.no_grad():
    for i in range(len(image_files)):
        # time_start = time.time()
        img = str(i) + '.png'
        source = Variable(img_transform(Image.open(opts.input + img).convert('RGB')).unsqueeze(0).cuda())
        source_mask = Variable(mask_transform(Image.open(opts.input_mask + img)).unsqueeze(0).cuda())
        source_texture = Variable(mask_transform(Image.open(opts.input_texture + img)).unsqueeze(0).cuda())
        
        subpath = os.path.join(opts.output_path, str(i))
        if not os.path.exists(subpath):
            os.makedirs(subpath)
        # save source image
        path = os.path.join(subpath, '%d_0.png'%i)
        image_save = vutils.make_grid(source, nrow=source.size(0), padding=0, normalize=True, scale_each=True)
        vutils.save_image(image_save, path, nrow=1)

        for n,sty in enumerate(style_files): 
            style_image = Variable(img_transform(Image.open(opts.style + sty).convert('RGB')).unsqueeze(0).cuda()) if opts.style != '' else None
            style_mask = Variable(mask_transform(Image.open(opts.style_mask + sty)).unsqueeze(0).cuda())
            style_texture = Variable(mask_transform(Image.open(opts.style_texture + sty)).unsqueeze(0).cuda())
            if i == 0:
                style_savepath = os.path.join(opts.output_path, 'style')
                if not os.path.exists(style_savepath):
                    os.makedirs(style_savepath)
                style_save = vutils.make_grid(style_image, nrow=source.size(0), padding=0, normalize=True, scale_each=True)
                vutils.save_image(style_save, os.path.join(style_savepath, 'style_%d.png'%n), nrow=1)

            content, _, c_facial_mask = encode(source, source_mask, source_texture)
            _, source_sty, _ = style_encode(source, source_mask, style_texture)
            _, style, _ = style_encode(style_image, style_mask, style_texture)
            # make the makeup controllable
            level = 0
            new_sty = level*source_sty.unsqueeze(0) + (1-level)*(style.unsqueeze(0))
            outputs = decode(content, new_sty, c_facial_mask)
            # save transfer image
            path = os.path.join(subpath, '%d_%s.png'%(i+1,sty))
            image_save = vutils.make_grid(outputs, nrow=source.size(0), padding=0, normalize=True, scale_each=True)
            vutils.save_image(image_save, path, nrow=1)
        
        print(i)
