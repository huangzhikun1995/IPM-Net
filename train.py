# -*- coding: utf-8 -*-
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer, randomflip, randomcrop
import argparse
from torch.autograd import Variable
from trainer import IPMNet_Trainer 
import torch.backends.cudnn as cudnn
import torch
import numpy.random as random
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='IPMNet')
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path

# Setup model and data loader
if opts.trainer == 'IMPNet':
    trainer = IPMNet_Trainer(config)
    trainer.cuda()

random.seed(7) # fix random result
train_loader_a, train_loader_b, test_loader_a, test_loader_b, train_mask_loader_a, train_mask_loader_b,\
test_mask_loader_a, test_mask_loader_b, train_texture_loader_a, train_texture_loader_b,test_texture_loader_a,\
test_texture_loader_b = get_all_data_loaders(config)

train_a_rand = random.permutation(len(train_loader_a.dataset))[0:display_size] 
train_b_rand = random.permutation(len(train_loader_b.dataset))[0:display_size] 
test_a_rand = random.permutation(len(test_loader_a.dataset))[0:display_size] 
test_b_rand = random.permutation(len(test_loader_b.dataset))[0:display_size] 

train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in train_a_rand]).cuda()
train_display_mask_a = torch.stack([train_mask_loader_a.dataset[i] for i in train_a_rand]).cuda()
train_display_texture_a = torch.stack([train_texture_loader_a.dataset[i] for i in train_a_rand]).cuda()
train_display_images_a, train_display_mask_a, train_display_texture_a = randomcrop(train_display_images_a, train_display_mask_a, train_display_texture_a, config['crop_image_height'], config['crop_image_width'])

train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in train_b_rand]).cuda()
train_display_mask_b = torch.stack([train_mask_loader_b.dataset[i] for i in train_b_rand]).cuda()
train_display_texture_b = torch.stack([train_texture_loader_b.dataset[i] for i in train_b_rand]).cuda()
train_display_images_b, train_display_mask_b, train_display_texture_b = randomcrop(train_display_images_b, train_display_mask_b, train_display_texture_b, config['crop_image_height'], config['crop_image_width'])

test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in test_a_rand]).cuda()
test_display_mask_a = torch.stack([test_mask_loader_a.dataset[i] for i in test_a_rand]).cuda()
test_display_texture_a = torch.stack([test_texture_loader_a.dataset[i] for i in test_a_rand]).cuda()
test_display_images_a, test_display_mask_a, test_display_texture_a = randomcrop(test_display_images_a, test_display_mask_a, test_display_texture_a, config['crop_image_height'], config['crop_image_width'])

test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in test_b_rand]).cuda()
test_display_mask_b = torch.stack([test_mask_loader_b.dataset[i] for i in test_b_rand]).cuda()
test_display_texture_b = torch.stack([test_texture_loader_b.dataset[i] for i in test_b_rand]).cuda()
test_display_images_b, test_display_mask_b, test_display_texture_b = randomcrop(test_display_images_b, test_display_mask_b, test_display_texture_b, config['crop_image_height'], config['crop_image_width'])

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
while True:
    for it, (images_a, images_b, mask_a, mask_b, texture_a, texture_b)in enumerate(zip(train_loader_a,train_loader_b,
                                                                            train_mask_loader_a,
                                                                            train_mask_loader_b,
                                                                            train_texture_loader_a,
                                                                            train_texture_loader_b)):
        trainer.update_learning_rate()
        images_a, mask_a, texture_a = randomflip(images_a, mask_a, texture_a)
        images_b, mask_b, texture_b = randomflip(images_b, mask_b, texture_b)

        # randflip
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()
        mask_a, mask_b = mask_a.cuda().detach(), mask_b.cuda().detach()
        texture_a, texture_b = texture_a.cuda().detach(), texture_b.cuda().detach()
        # randcrop
        images_a, mask_a, texture_a = randomcrop(images_a, mask_a, texture_a, config['crop_image_height'], config['crop_image_width'])
        images_b, mask_b, texture_b = randomcrop(images_b, mask_b, texture_b, config['crop_image_height'], config['crop_image_width'])


        with Timer("Elapsed time in update: %f"):
            # Main training code
            trainer.dis_update(images_a.clone(), images_b.clone(), mask_a, mask_b, texture_a, texture_b, config)
            trainer.gen_update(images_a.clone(), images_b.clone(), mask_a, mask_b, texture_a, texture_b, config)
            torch.cuda.synchronize()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b,
                                                    test_display_mask_a, test_display_mask_b, 
                                                    test_display_texture_a, test_display_texture_b,
                                                    config, False )
                train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b,
                                                     train_display_mask_a, train_display_mask_b, 
                                                     train_display_texture_a, train_display_texture_b,
                                                     config, False)
            write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
            write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
            # HTML
            write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')
            del test_image_outputs, train_image_outputs

        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                image_outputs = trainer.sample(train_display_images_a, train_display_images_b,
                                               train_display_mask_a, train_display_mask_b,
                                               train_display_texture_a, train_display_texture_b, 
                                               config, True)
            write_2images(image_outputs, display_size, image_directory, 'train_current')
            del image_outputs

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')

