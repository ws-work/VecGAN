import os
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

from core.networks import Generator, Discriminator
from config import model_config as config 
from utils import get_model_list

def init_fun(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.01)

class Trainer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gen = Generator()
        self.dis = Discriminator()

        beta1 = config.beta1
        beta2 = config.beta2

        self.dis_opt = torch.optim.Adam(self.dis.parameters(),
                                        lr=config.lr_dis, betas=(beta1, beta2))

        self.gen_opt = torch.optim.Adam(self.gen.parameters(),
                                        lr=config.lr_gen_others, betas=(beta1, beta2))

        self.apply(init_fun)

    def forward(self):
        pass 

    def calc_gan_loss(self, x, y, i, j, j_trg):
        batch = x.size(0)

        # non-translation path
        e = self.gen.encode(x)
        x_n = self.gen.decode(e)

        # self-translation path
        s = self.gen.extract(x, i)
        e_slf = self.gen.translate(e, s, i)
        x_slf = self.gen.decode(e_slf)

        # cycle-translation path
        ## translate
        s_trg = self.gen.map(torch.randn(batch, self.noise_dim).cuda(), i, j_trg)
        e_trg = self.gen.translate(e, s_trg, i)
        x_trg = self.gen.decode(e_trg)
        ## cycle-back
        e_trg_rec = self.gen.encode(x_trg)
        s_trg_rec = self.gen.extract(x_trg, i) 
        e_cyc = self.gen.translate(e_trg_rec, s, i)
        x_cyc = self.gen.decode(e_cyc)

        loss_gen_adv = self.dis.calc_gen_loss_real(x, s, y, i, j) + \
                       self.dis.calc_gen_loss_fake_trg(x_trg, s_trg.detach(), y, i, j_trg) + \
                       self.dis.calc_gen_loss_fake_cyc(x_cyc, s.detach(), y, i, j) 

        loss_gen_sty = F.l1_loss(s_trg_rec, s_trg)


        loss_gen_rec = F.l1_loss(x_n, x) + \
                       F.l1_loss(x_slf, x) + \
                       F.l1_loss(x_cyc, x)

        log_gen_ortho = self.gen.calc_gen_loss_real()

        loss_gen_total = self.hyperparameters['adv_w'] * loss_gen_adv + \
                         self.hyperparameters['sty_w'] * loss_gen_sty + \
                         self.hyperparameters['rec_w'] * loss_gen_rec + \
                         self.hyperparameters['ort_w'] * log_gen_ortho

        loss_gen_total.backward()

        return (loss_gen_adv, loss_gen_sty, loss_gen_rec, log_gen_ortho, 
        x_trg.detach(), x_cyc.detach(), s.detach(), s_trg.detach())
    
    def calc_dis_loss(self, x, x_trg, x_cyc, s, s_trg, y, i, j, j_trg):

        loss_dis_adv = self.dis.calc_dis_loss_real(x, s, y, i, j) + \
                       self.dis.calc_dis_loss_fake_trg(x_trg, s_trg, y, i, j_trg) + \
                       self.dis.calc_dis_loss_fake_cyc(x_cyc, s, y, i, j) 
        loss_dis_adv.backward()

        return loss_dis_adv

    def update(self, x, y, i, j, j_trg):
        # gen 
        for p in self.dis.parameters():
            p.requires_grad = False
        for p in self.gen.parameters():
            p.requires_grad = True

        self.gen_opt.zero_grad()
        breakpoint()
        self.loss_gen_adv, self.loss_gen_sty, self.loss_gen_rec, self.log_gen_ortho,\
        x_trg, x_cyc, s, s_trg = self.calc_gan_loss(x, y, i, j, j_trg)

        self.loss_gen_adv = self.loss_gen_adv.mean()
        self.loss_gen_sty = self.loss_gen_sty.mean()
        self.loss_gen_rec = self.loss_gen_rec.mean()
        
        nn.utils.clip_grad_norm_(self.gen.parameters(), 100)
        self.gen_opt.step()

        # dis
        for p in self.dis.parameters():
            p.requires_grad = True
        for p in self.gen.parameters():
            p.requires_grad = False

        self.dis_opt.zero_grad()

        self.loss_dis_adv = self.calc_dis_loss(x, x_trg, x_cyc, s, s_trg, y, i, j, j_trg)
        self.loss_dis_adv = self.loss_dis_adv.mean()

        nn.utils.clip_grad_norm_(self.dis.parameters(), 100)
        self.dis_opt.step()

        return self.loss_gen_adv.item(), \
               self.loss_gen_sty.item(), \
               self.loss_gen_rec.item(), \
               self.loss_dis_adv.item()


    def sample(self, x, x_trg, j, j_trg, i):
        gen = self.gen

        out = [x]
        with torch.no_grad():

            e = gen.encode(x)

            z = torch.randn(1, config.noise_dim).cuda().repeat(x.size(0), 1)
            s_trg = gen.map(z, i, j_trg)
            x_trg_ = gen.decode(gen.translate(e, s_trg, i))
            out += [x_trg_]

            z = torch.randn(1, config.noise_dim).cuda().repeat(x.size(0), 1)
            s_trg = gen.map(z, i, j_trg)
            x_trg_ = gen.decode(gen.translate(e, s_trg, i))
            out += [x_trg_]

            s_trg = gen.extract(x_trg, i)
            x_trg_ = gen.decode(gen.translate(e, s_trg, i))
            out += [x_trg, x_trg_]

            x_trg_ = gen.decode(gen.translate(e, s_trg.flip([0]), i))
            out += [x_trg.flip([0]), x_trg_]

        return out

    def resume(self, checkpoint_dir):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen.load_state_dict(state_dict['gen'])
        self.gen_test.load_state_dict(state_dict['gen_test'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis.load_state_dict(state_dict['dis'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])

        for state in self.dis_opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        
        for state in self.gen_opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        print('Resume from iteration %d' % iterations)
        return iterations
    

    def save(self, snapshot_dir, iterations):
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'gen': self.gen.state_dict(), 'gen_test': self.gen_test.state_dict()}, gen_name)
        torch.save({'dis': self.dis.state_dict()}, dis_name)
        torch.save({'dis': self.dis_opt.state_dict(), 
                    'gen': self.gen_opt.state_dict()}, opt_name)
