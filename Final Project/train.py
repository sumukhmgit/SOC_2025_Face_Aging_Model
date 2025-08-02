import os
from argparse import ArgumentParser

import pytorch_lightning as pl

from gan_model import AgingGAN

def main():
    parser = ArgumentParser(description="Train the Face Aging GAN model")
    # Data directories
    parser.add_argument('--domainA_dir', type=str, default=os.path.join('data', 'processed', 'trainA'),
                        help='Directory for domain A (young) images')
    parser.add_argument('--domainB_dir', type=str, default=os.path.join('data', 'processed', 'trainB'),
                        help='Directory for domain B (old) images')
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--img_size', type=int, default=256, help='Size of input images (already resized)')
    parser.add_argument('--ngf', type=int, default=64, help='Base channel count for generator')
    parser.add_argument('--ndf', type=int, default=64, help='Base channel count for discriminator')
    parser.add_argument('--n_blocks', type=int, default=9, help='Number of residual blocks in generator')
    parser.add_argument('--adv_weight', type=float, default=1.0, help='Adversarial loss weight')
    parser.add_argument('--cycle_weight', type=float, default=10.0, help='Cycle consistency loss weight')
    parser.add_argument('--identity_weight', type=float, default=5.0, help='Identity loss weight')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for optimizers')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 regularization)')
    parser.add_argument('--augment_rotation', type=float, default=10.0, help='Max degrees for random rotation augmentation')
    parser.add_argument('--max_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use (0 for CPU)')
    args = parser.parse_args()

    # Prepare hyperparameters dict for LightningModule
    hparams = {
        'domainA_dir': args.domainA_dir,
        'domainB_dir': args.domainB_dir,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'img_size': args.img_size,
        'ngf': args.ngf,
        'ndf': args.ndf,
        'n_blocks': args.n_blocks,
        'adv_weight': args.adv_weight,
        'cycle_weight': args.cycle_weight,
        'identity_weight': args.identity_weight,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'augment_rotation': args.augment_rotation
    }

    model = AgingGAN(hparams)
    trainer = pl.Trainer(max_epochs=args.max_epochs)
    trainer.fit(model)

if __name__ == '__main__':
    main()
