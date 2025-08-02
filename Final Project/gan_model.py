import itertools

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from dataset import ImagetoImageDataset
from model import Generator, Discriminator

class AgingGAN(pl.LightningModule):
    """
    CycleGAN model for face aging (domain A -> domain B).
    """
    def __init__(self, hparams):
        super(AgingGAN, self).__init__()
        self.save_hyperparameters(hparams)
        self.automatic_optimization = False
        # Generators and Discriminators
        self.genA2B = Generator(ngf=self.hparams['ngf'], n_residual_blocks=self.hparams['n_blocks'])
        self.genB2A = Generator(ngf=self.hparams['ngf'], n_residual_blocks=self.hparams['n_blocks'])
        self.disGA = Discriminator(ndf=self.hparams['ndf'])
        self.disGB = Discriminator(ndf=self.hparams['ndf'])
        # placeholders for logging
        self.generated_A = None
        self.generated_B = None
        self.real_A = None
        self.real_B = None

    def forward(self, x):
        # Forward using generator A2B by default
        return self.genA2B(x)

    def training_step(self, batch, batch_idx):
        # fetch both optimizers
        opt_g, opt_d = self.optimizers()

        real_A, real_B = batch  # Domain A (young), Domain B (old)

        # ----- GENERATOR STEP -----
        opt_g.zero_grad()

        # Identity loss
        same_B = self.genA2B(real_B)
        loss_id_B = F.l1_loss(same_B, real_B) * self.hparams['identity_weight']
        same_A = self.genB2A(real_A)
        loss_id_A = F.l1_loss(same_A, real_A) * self.hparams['identity_weight']

        # GAN loss
        fake_B = self.genA2B(real_A)
        pred_fake_B = self.disGB(fake_B)
        loss_gan_A2B = F.mse_loss(pred_fake_B, torch.ones_like(pred_fake_B)) * self.hparams['adv_weight']

        fake_A = self.genB2A(real_B)
        pred_fake_A = self.disGA(fake_A)
        loss_gan_B2A = F.mse_loss(pred_fake_A, torch.ones_like(pred_fake_A)) * self.hparams['adv_weight']

        # Cycle-consistency
        recov_A = self.genB2A(fake_B)
        loss_cycle_ABA = F.l1_loss(recov_A, real_A) * self.hparams['cycle_weight']
        recov_B = self.genA2B(fake_A)
        loss_cycle_BAB = F.l1_loss(recov_B, real_B) * self.hparams['cycle_weight']

        # Total generator loss
        g_loss = (loss_id_A + loss_id_B +
                  loss_gan_A2B + loss_gan_B2A +
                  loss_cycle_ABA + loss_cycle_BAB)

        # backward + step
        self.manual_backward(g_loss)
        opt_g.step()

        # cache for D step & logging
        self.generated_A = fake_A
        self.generated_B = fake_B
        self.real_A = real_A
        self.real_B = real_B
        self.log('Loss/Generator', g_loss, prog_bar=True)

        # optional: log images every N batches
        if batch_idx % 500 == 0:
            self.genA2B.eval()
            self.genB2A.eval()
            img_fake_A = self.genB2A(real_B)
            img_fake_B = self.genA2B(real_A)
            self.logger.experiment.add_image('Real/A',
                                            make_grid(self.real_A, normalize=True),
                                            self.current_epoch)
            self.logger.experiment.add_image('Real/B',
                                            make_grid(self.real_B, normalize=True),
                                            self.current_epoch)
            self.logger.experiment.add_image('Fake/A',
                                            make_grid(img_fake_A, normalize=True),
                                            self.current_epoch)
            self.logger.experiment.add_image('Fake/B',
                                            make_grid(img_fake_B, normalize=True),
                                            self.current_epoch)
            self.genA2B.train()
            self.genB2A.train()

        # ----- DISCRIMINATOR STEP -----
        opt_d.zero_grad()

        # Discriminator A
        pred_real_A = self.disGA(real_A)
        loss_D_real_A = F.mse_loss(pred_real_A, torch.ones_like(pred_real_A))
        pred_fake_A = self.disGA(self.generated_A.detach())
        loss_D_fake_A = F.mse_loss(pred_fake_A, torch.zeros_like(pred_fake_A))
        loss_D_A = 0.5 * (loss_D_real_A + loss_D_fake_A)

        # Discriminator B
        pred_real_B = self.disGB(real_B)
        loss_D_real_B = F.mse_loss(pred_real_B, torch.ones_like(pred_real_B))
        pred_fake_B = self.disGB(self.generated_B.detach())
        loss_D_fake_B = F.mse_loss(pred_fake_B, torch.zeros_like(pred_fake_B))
        loss_D_B = 0.5 * (loss_D_real_B + loss_D_fake_B)

        d_loss = loss_D_A + loss_D_B

        # backward + step
        self.manual_backward(d_loss)
        opt_d.step()
        self.log('Loss/Discriminator', d_loss, prog_bar=True)

        return {'loss': g_loss, 'd_loss': d_loss}


    def configure_optimizers(self):
        g_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),
                                   lr=self.hparams['lr'], betas=(0.5, 0.999),
                                   weight_decay=self.hparams['weight_decay'])
        d_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters()),
                                   lr=self.hparams['lr'], betas=(0.5, 0.999),
                                   weight_decay=self.hparams['weight_decay'])
        return [g_optim, d_optim], []

    def train_dataloader(self):
        # Data augmentation and transformations
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.hparams['img_size'] + 50, self.hparams['img_size'] + 50)),
            transforms.RandomCrop(self.hparams['img_size']),
            transforms.RandomRotation(degrees=(0, int(self.hparams['augment_rotation']))),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        dataset = ImagetoImageDataset(self.hparams['domainA_dir'], self.hparams['domainB_dir'], transform=train_transform)
        loader = DataLoader(dataset, batch_size=self.hparams['batch_size'],
                            shuffle=True, num_workers=self.hparams['num_workers'])
        return loader
