# -*- coding: utf-8 -*-
from networks import AdaINGen, MsImageDis, VAEGen
from utils import weights_init, get_model_list, vgg_preprocess, load_resnet50, get_scheduler , load_state_dict
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import resnet
import numpy as np

class IPMNet_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(IPMNet_Trainer, self).__init__()
        lr = hyperparameters['lr']
        vgg_weight_file = hyperparameters['vgg_weight_file']
        # Initiate the networks
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = self.gen_a # AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init(hyperparameters['init']))
        self.dis_b.apply(weights_init(hyperparameters['init']))

        # Load VGGFace model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_resnet50(vgg_weight_file)
            self.vgg.eval()
            self.vgg.fc.reset_parameters()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def gen_update(self, x_a, x_b, mask_a, mask_b, texture_a, texture_b, hyperparameters):
        self.gen_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, s_a_prime, x_a_gray_facial = self.gen_a.encode(x_a, mask_a, texture_a)
        c_b, s_b_prime, x_b_gray_facial = self.gen_b.encode(x_b, mask_b, texture_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, s_a_prime, x_a_gray_facial)
        x_b_recon = self.gen_b.decode(c_b, s_b_prime, x_b_gray_facial)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a, x_b_gray_facial) 
        x_ab = self.gen_b.decode(c_a, s_b, x_a_gray_facial) 
        # encode again
        c_a_recon, s_b_recon, x_a_recon_gray_facial = self.gen_b.encode(x_ab, mask_a, texture_a)
        c_b_recon, s_a_recon, x_b_recon_gray_facial = self.gen_a.encode(x_ba, mask_b, texture_b)
        # decode again (if needed)
        x_aba = self.gen_a.decode(c_a_recon, s_a_prime, x_a_recon_gray_facial) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(c_b_recon, s_b_prime, x_b_recon_gray_facial) if hyperparameters['recon_x_cyc_w'] > 0 else None
        # background
        x_a_back = x_a * mask_a.repeat(1, 3, 1, 1)
        x_b_back = x_b * mask_b.repeat(1, 3, 1, 1)
        x_ab_back = x_ab * mask_a.repeat(1, 3, 1, 1)
        x_ba_back = x_ba * mask_b.repeat(1, 3, 1, 1)
        # foreground
        x_a_fore = x_a * (1 - mask_a).repeat(1, 3, 1, 1)
        x_b_fore = x_b * (1 - mask_b).repeat(1, 3, 1, 1)
        x_a_recon_fore = x_a_recon * (1 - mask_a).repeat(1, 3, 1, 1)
        x_b_recon_fore = x_b_recon * (1 - mask_b).repeat(1, 3, 1, 1)

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        # backgrouned loss
        self.loss_back_x_a = self.recon_criterion(x_ab_back, x_a_back) if hyperparameters['back_w'] > 0 else 0
        self.loss_back_x_b = self.recon_criterion(x_ba_back, x_b_back) if hyperparameters['back_w'] > 0 else 0
        # foreground loss
        self.loss_fore_x_a = self.recon_criterion(x_a_recon_fore, x_a_fore) if hyperparameters['fore_w'] > 0 else 0
        self.loss_fore_x_b = self.recon_criterion(x_b_recon_fore, x_b_fore) if hyperparameters['fore_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b +\
                              hyperparameters['back_w'] * self.loss_back_x_a +\
                              hyperparameters['back_w'] * self.loss_back_x_b +\
                              hyperparameters['fore_w'] * self.loss_fore_x_a +\
                              hyperparameters['fore_w'] * self.loss_fore_x_b 
                              
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean(torch.abs(img_fea - target_fea))

    def sample(self, x_a, x_b, mask_a, mask_b, texture_a, texture_b, hyperparameters, train=True):
        self.eval()
        s_a1 = Variable(self.s_a)
        s_b1 = Variable(self.s_b)
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        x_a_recon, x_b_recon, x_a_facial_mask, x_b_facial_mask, x_ba, x_ab, x_aba, x_bab = [], [], [], [], [], [], [], []
        x_ab1, x_ab2, x_ba1, x_ba2 = [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a, x_a_gray_facial = self.gen_a.encode(x_a[i].unsqueeze(0), mask_a[i].unsqueeze(0), texture_a[i].unsqueeze(0))
            c_b, s_b, x_b_gray_facial = self.gen_b.encode(x_b[i].unsqueeze(0), mask_b[i].unsqueeze(0), texture_b[i].unsqueeze(0))
            if train:
                if i == 0:
                    print(s_a.squeeze())
                    print(s_b.squeeze())
            x_a_recon.append(self.gen_a.decode(c_a, s_a, x_a_gray_facial))
            x_b_recon.append(self.gen_b.decode(c_b, s_b, x_b_gray_facial))
            x_a_facial_mask.append(x_a_gray_facial)
            x_b_facial_mask.append(x_b_gray_facial)
            x_ba.append(self.gen_a.decode(c_b, s_a, x_b_gray_facial)) 
            x_ab.append(self.gen_b.decode(c_a, s_b, x_a_gray_facial))
            # randn style
            x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0), x_b_gray_facial)) 
            x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0), x_a_gray_facial))
            x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0), x_b_gray_facial)) 
            x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0), x_a_gray_facial))
            # encode again
            c_a_recon, _, x_a_recon_gray_facial = self.gen_a.encode(x_ab[i], mask_a[i].unsqueeze(0), texture_a[i].unsqueeze(0))
            c_b_recon, _, x_b_recon_gray_facial = self.gen_b.encode(x_ba[i], mask_b[i].unsqueeze(0), texture_b[i].unsqueeze(0))
            # decode again (if needed)
            x_aba_recon = self.gen_a.decode(c_a_recon, s_a, x_a_recon_gray_facial) if hyperparameters['recon_x_cyc_w'] > 0 else None
            x_bab_recon = self.gen_b.decode(c_b_recon, s_b, x_b_recon_gray_facial) if hyperparameters['recon_x_cyc_w'] > 0 else None
            x_aba.append(x_aba_recon)
            x_bab.append(x_bab_recon)

        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_a_facial_mask, x_b_facial_mask = torch.cat(x_a_facial_mask), torch.cat(x_b_facial_mask)
        x_ab, x_ba = torch.cat(x_ab), torch.cat(x_ba)
        x_aba, x_bab = torch.cat(x_aba), torch.cat(x_bab)
        x_ab1, x_ab2, x_ba1, x_ba2 = torch.cat(x_ab1), torch.cat(x_ab2), torch.cat(x_ba1), torch.cat(x_ba2)
        self.train()
        return x_a, x_b, x_a_recon, x_a_facial_mask, x_ab, x_aba, \
               x_b, x_a, x_b_recon, x_b_facial_mask, x_ba, x_bab

    def dis_update(self, x_a, x_b, mask_a, mask_b, texture_a, texture_b, hyperparameters):
        self.dis_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, _, x_a_gray_facial = self.gen_a.encode(x_a, mask_a, texture_a)
        c_b, _, x_b_gray_facial = self.gen_b.encode(x_b, mask_b, texture_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a, x_b_gray_facial)
        x_ab = self.gen_b.decode(c_a, s_b, x_a_gray_facial)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
