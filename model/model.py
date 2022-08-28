import torch
import torch.nn as nn
from .generators import UNetGenerator
from .discriminators import Discriminator
from .losses import CategoryLoss, BinaryLoss
import os
from torch.optim.lr_scheduler import StepLR
from utils.init_net import init_net
import torchvision.utils as vutils
from .TRM import Transformer
# import time


class S2SModel:
    def __init__(self, input_nc=3, ngf=64, ndf=64,
                 Lconst_penalty=15, Lcategory_penalty=1, L1_penalty=100,
                 schedule=10, lr=0.001, gpu_ids=None, save_dir='.', is_training=True,
                 image_size=256, tgt_vocab_size=408):

        if is_training:
            self.use_dropout = True
        else:
            self.use_dropout = False

        # loss function
        self.Lconst_penalty = Lconst_penalty
        self.Lcategory_penalty = Lcategory_penalty
        self.L1_penalty = L1_penalty

        self.schedule = schedule

        self.save_dir = save_dir
        self.gpu_ids = gpu_ids

        self.input_nc = input_nc
        self.ngf = ngf
        self.ndf = ndf
        self.lr = lr
        self.is_training = is_training
        self.image_size = image_size
        self.tgt_vocab_size = tgt_vocab_size

    def setup(self):
        self.netG = UNetGenerator(
            input_nc=self.input_nc,
            output_nc=self.input_nc,
            ngf=self.ngf,
            is_training=self.use_dropout
        )
        self.netD = Discriminator(
            input_nc=2 * self.input_nc,
            ndf=self.ndf,
            image_size=self.image_size
        )
        self.TRM = Transformer(tgt_vocab_size=self.tgt_vocab_size)

        init_net(self.netG, gpu_ids=self.gpu_ids)
        init_net(self.netD, gpu_ids=self.gpu_ids)

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_TRM = torch.optim.Adam(self.TRM.parameters(), lr=self.lr)

        # self.category_loss = CategoryLoss(self.embedding_num)
        self.real_binary_loss = BinaryLoss(True)
        self.fake_binary_loss = BinaryLoss(False)
        self.l1_loss = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()
        self.trm_loss = nn.CrossEntropyLoss()

        if self.gpu_ids:
            self.real_binary_loss.cuda()
            self.fake_binary_loss.cuda()
            self.l1_loss.cuda()
            self.mse.cuda()
            self.sigmoid.cuda()
            self.trm_loss.cuda()
            self.TRM.to(self.gpu_ids[0])
        if self.is_training:
            self.netD.train()
            self.netG.train()
            self.TRM.train()
        else:
            self.netD.eval()
            self.netG.eval()
            self.TRM.eval()

    def set_input(self, labels, real_A, real_B, stan_A):
        # dec_inputs, target_batch
        dec_inputs = labels[:, :-1]
        dec_inputs = torch.where(torch.gt(dec_inputs, 2.9) & torch.lt(dec_inputs, 3.1), torch.zeros_like(dec_inputs), dec_inputs)
        target_batch = labels[:, 1:]

        if self.gpu_ids:
            # seal -> A, traditional ch -> B
            self.real_A = real_A.to(self.gpu_ids[0])
            self.real_B = real_B.to(self.gpu_ids[0])
            self.stan_A = stan_A.to(self.gpu_ids[0])
            self.dec_inputs = dec_inputs.to(self.gpu_ids[0])
            self.target_batch = target_batch.to(self.gpu_ids[0])
        else:
            self.real_A = real_A
            self.real_B = real_B
            self.stan_A = stan_A
            self.dec_inputs = dec_inputs
            self.target_batch = target_batch

    def forward(self):
        # generate fake img
        # seal -> A, traditional ch -> B
        # generate with real seal
        self.real_A_encoder_list, self.fake_AA, self.fake_AB = self.netG(self.real_A)
        # generate with real traditional ch
        self.real_B_encoder_list, self.fake_BA, self.fake_BB = self.netG(self.real_B)

        # # CycleGAN part
        # self.cycl_BA_encoder_list, _, self.cycle_fake_AB = self.netG(self.fake_BA)
        # self.cycl_AB_encoder_list, _, _ = self.netG(self.fake_AB)

        # generate style img
        self.style_A_encoder_list, style_AA, style_AB = self.netG(self.stan_A)

        # Transformer part
        # enc_inputs, dec_inputs, enc_outputs
        gan_encoder_input = self.real_A_encoder_list[5].view(16, 512, 16)
        all_one = torch.ones_like(self.dec_inputs)
        self.logits, _, _ = self.TRM(all_one, self.dec_inputs, gan_encoder_input)

    def backward_D(self, no_target_source=False):

        # discriminator part
        real_AB = torch.cat([self.stan_A, self.real_B], 1)
        fake_AB = torch.cat([self.stan_A, self.fake_AB], 1)
        fake_BB = torch.cat([self.stan_A, self.fake_BB], 1)
        # cycl_AB = torch.cat([self.stan_A, self.cycle_fake_AB], 1)

        real_BA = torch.cat([self.real_B, self.real_A], 1)
        fake_BA = torch.cat([self.real_B, self.fake_BA], 1)
        fake_AA = torch.cat([self.real_B, self.fake_AA], 1)

        real_AB_D_logits = self.netD(real_AB)
        fake_AB_D_logits = self.netD(fake_AB.detach())
        fake_BB_D_logits = self.netD(fake_BB.detach())
        # cycl_AB_D_logits = self.netD(cycl_AB.detach())

        real_BA_D_logits = self.netD(real_BA)
        fake_BA_D_logits = self.netD(fake_BA.detach())
        fake_AA_D_logits = self.netD(fake_AA.detach())

        d_loss_real = self.real_binary_loss(real_AB_D_logits) + self.real_binary_loss(real_BA_D_logits)
        # d_loss_fake = self.fake_binary_loss(fake_AB_D_logits) + self.fake_binary_loss(fake_AA_D_logits) + \
        #               self.fake_binary_loss(fake_BA_D_logits) + self.fake_binary_loss(fake_BB_D_logits) + \
        #               self.fake_binary_loss(cycl_AB_D_logits)
        d_loss_fake = self.fake_binary_loss(fake_AB_D_logits) + self.fake_binary_loss(fake_AA_D_logits) + \
                      self.fake_binary_loss(fake_BA_D_logits) + self.fake_binary_loss(fake_BB_D_logits)

        self.d_loss = d_loss_real + d_loss_fake
        self.d_loss.backward()

    def backward_G(self, no_target_source=False):
        fake_AB = torch.cat([self.stan_A, self.fake_AB], 1)
        fake_BB = torch.cat([self.stan_A, self.fake_BB], 1)
        # cycl_AB = torch.cat([self.stan_A, self.cycle_fake_AB], 1)
        fake_BA = torch.cat([self.real_B, self.fake_BA], 1)
        fake_AA = torch.cat([self.real_B, self.fake_AA], 1)

        fake_AB_D_logits = self.netD(fake_AB)
        fake_BA_D_logits = self.netD(fake_BA)
        fake_AA_D_logits = self.netD(fake_AA)
        fake_BB_D_logits = self.netD(fake_BB)
        # cycl_AB_D_logits = self.netD(cycl_AB)

        # encoding constant loss
        # this loss assume that generated imaged and real image should reside in the same space and close to each other
        if len(self.real_A_encoder_list) != 8:
            raise RuntimeError("len of generators encoder_results is not 8")
        # the fake_AB_encoder_list(CycleGAN part just compare with 7 layer, without 5 layer)
        # const_loss = (self.mse(self.real_A_encoder_list[5], self.real_B_encoder_list[5]) +
        #               self.mse(self.real_A_encoder_list[7], self.real_B_encoder_list[7]) +
        #               self.mse(self.real_A_encoder_list[5], self.style_A_encoder_list[5]) +
        #               self.mse(self.real_A_encoder_list[7], self.style_A_encoder_list[7]) +
        #               self.mse(self.real_B_encoder_list[5], self.style_A_encoder_list[5]) +
        #               self.mse(self.real_B_encoder_list[7], self.style_A_encoder_list[7]) +
        #               self.mse(self.cycl_AB_encoder_list[7], self.real_B_encoder_list[7]) +
        #               self.mse(self.cycl_BA_encoder_list[7], self.style_A_encoder_list[7])
        #               ) * self.Lconst_penalty
        const_loss = (self.mse(self.real_A_encoder_list[5], self.real_B_encoder_list[5]) +
                      self.mse(self.real_A_encoder_list[7], self.real_B_encoder_list[7]) +
                      self.mse(self.real_A_encoder_list[5], self.style_A_encoder_list[5]) +
                      self.mse(self.real_A_encoder_list[7], self.style_A_encoder_list[7]) +
                      self.mse(self.real_B_encoder_list[5], self.style_A_encoder_list[5]) +
                      self.mse(self.real_B_encoder_list[7], self.style_A_encoder_list[7])
                      ) * self.Lconst_penalty
        # L1 loss between real and generated images
        # l1_loss = (self.l1_loss(self.fake_AB, self.real_B) + self.l1_loss(self.fake_BB, self.real_B) +
        #            self.l1_loss(self.fake_BA, self.stan_A) + self.l1_loss(self.fake_AA, self.stan_A) +
        #            self.l1_loss(self.cycle_fake_AB, self.real_B)) * self.L1_penalty
        l1_loss = (self.l1_loss(self.fake_AB, self.real_B) + self.l1_loss(self.fake_BB, self.real_B) +
                   self.l1_loss(self.fake_BA, self.stan_A) + self.l1_loss(self.fake_AA, self.stan_A)) * self.L1_penalty

        # cheat_loss = self.real_binary_loss(fake_AA_D_logits) + self.real_binary_loss(fake_AB_D_logits) + \
        #              self.real_binary_loss(fake_BA_D_logits) + self.real_binary_loss(fake_BB_D_logits) + \
        #              self.real_binary_loss(cycl_AB_D_logits)
        cheat_loss = self.real_binary_loss(fake_AA_D_logits) + self.real_binary_loss(fake_AB_D_logits) + \
                     self.real_binary_loss(fake_BA_D_logits) + self.real_binary_loss(fake_BB_D_logits)

        self.g_loss = cheat_loss + l1_loss + const_loss
        self.g_loss.backward()
        return const_loss, l1_loss, cheat_loss

    def update_lr(self):
        # There should be only one param_group.
        for p in self.optimizer_D.param_groups:
            current_lr = p['lr']
            update_lr = current_lr / 2.0
            # minimum learning rate guarantee
            update_lr = max(update_lr, 0.0002)
            p['lr'] = update_lr
            print("Decay net_D learning rate from %.5f to %.5f." % (current_lr, update_lr))

        for p in self.optimizer_G.param_groups:
            current_lr = p['lr']
            update_lr = current_lr / 2.0
            # minimum learning rate guarantee
            update_lr = max(update_lr, 0.0002)
            p['lr'] = update_lr
            print("Decay net_G learning rate from %.5f to %.5f." % (current_lr, update_lr))

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # udpate G's weights

        # update TRM
        self.set_requires_grad(self.TRM, True)
        self.optimizer_TRM.zero_grad()
        self.trm_pre_loss = self.trm_loss(self.logits, self.target_batch.contiguous().view(-1))
        self.trm_pre_loss.backward()
        self.optimizer_TRM.step()

        # magic move to Optimize G again
        # according to https://github.com/carpedm20/DCGAN-tensorflow
        # collect all the losses along the way
        self.forward()  # compute fake images: G(A)
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        const_loss, l1_loss, cheat_loss = self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # udpate G's weights
        return const_loss, l1_loss, cheat_loss

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def print_networks(self, verbose=False):
        """Print the total number of parameters in the network and (if verbose) network architecture
        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in ['G', 'D']:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def save_networks(self, epoch):
        """Save all the networks to the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in ['G', 'D']:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if self.gpu_ids and torch.cuda.is_available():
                    # torch.save(net.cpu().state_dict(), save_path)
                    torch.save(net.state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch):
        """Load all the networks from the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in ['G', 'D']:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)

                if self.gpu_ids and torch.cuda.is_available():
                    net.load_state_dict(torch.load(load_path))
                else:
                    net.load_state_dict(torch.load(load_path,map_location=torch.device('cpu')))
                # net.eval()
        print('load model %d' % epoch)

    # save the picture
    def sample(self, batch, basename):
        chk_mkdir(basename)
        cnt = 0
        with torch.no_grad():
            self.set_input(batch[0], batch[2], batch[1], batch[3])
            self.forward()
            tensor_to_plot = torch.cat([self.fake_AB, self.real_B], 3)
            for label, image_tensor in zip(batch[0], tensor_to_plot):
                style_name = label.split('_')[1]
                # label_dir = os.path.join(basename)
                chk_mkdir(style_name)
                vutils.save_image(image_tensor, os.path.join(basename, style_name + '_' + str(cnt) + '.png'))
                cnt += 1
            # img = vutils.make_grid(tensor_to_plot)
            # vutils.save_image(tensor_to_plot, basename + "_construct.png")
            '''
            We don't need generate_img currently.
            self.set_input(torch.randn(1, self.embedding_num).repeat(batch[0].shape[0], 1), batch[2], batch[1])
            self.forward()
            tensor_to_plot = torch.cat([self.fake_B, self.real_A], 3)
            vutils.save_image(tensor_to_plot, basename + "_generate.png")
            '''


def chk_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)