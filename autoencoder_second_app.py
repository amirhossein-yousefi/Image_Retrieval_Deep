from torch import nn
import hyperparams
import torch.nn.functional as F
import torch
class ConvAutoEncBase(nn.Module):
    def training_step(self, batch):
        images, _ = batch
        out = self(images)  # Generate predictions
        loss = F.mse_loss(out, images)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, _ = batch
        out = self(images)  # Generate predictions
        loss = F.mse_loss(out, images)  # Calculate loss
        return {'val_loss': loss.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss']))


class Autoencoder(ConvAutoEncBase):
    def __init__(self, hparams=hyperparams):
        super().__init__()
        self.hparams = hparams

        self.encoder = nn.Sequential(
            # input (nc) x 32 x 32
            nn.Conv2d(hparams.nc, hparams.nfe, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfe),
            nn.LeakyReLU(True),
            # input (nfe) x 16 x 16
            nn.Conv2d(hparams.nfe, hparams.nfe * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfe * 2),
            nn.LeakyReLU(True),
            # input (nfe*2) x 8 x 8
            nn.Conv2d(hparams.nfe * 2, hparams.nfe * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfe * 4),
            nn.LeakyReLU(True),
            # input (nfe*4) x 4 x 4
            nn.Conv2d(hparams.nfe * 4, hparams.nfe * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfe * 8),
            nn.LeakyReLU(True),
            # input (nfe*8) x 2 x 2
            nn.Conv2d(hparams.nfe * 8, hparams.nfe * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfe * 16),
            nn.Tanh(),
            # input (nfe*16) x 1 x 1
        )

        self.decoder = nn.Sequential(
            # input (nz) x 1 x 1
            nn.ConvTranspose2d(hparams.nz, hparams.nfd * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hparams.nfd * 16),
            nn.ReLU(True),
            # input (nfd*16) x 4 x 4
            nn.ConvTranspose2d(hparams.nfd * 16, hparams.nfd * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfd * 8),
            nn.ReLU(True),
            # input (nfd*8) x 8 x 8
            nn.ConvTranspose2d(hparams.nfd * 8, hparams.nfd * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfd * 4),
            nn.ReLU(True),
            # input (nfd*4) x 16 x 16
            nn.ConvTranspose2d(hparams.nfd * 4, hparams.nfd * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfd * 2, hparams.nfd * 2),
            nn.ReLU(True),
            # input (nfd*2) x 32 x 32
            nn.ConvTranspose2d(hparams.nfd * 2, hparams.nfd, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hparams.nfd),
            nn.ReLU(True),
            # input (nfd) x 64 x 64
            nn.ConvTranspose2d(hparams.nfd, hparams.nc, 3, 1, 1, bias=False),
            nn.ReLU()
            # output (nc) x 128 x 128
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    from torchsummary import summary
    import torch

    model = Autoencoder()
    model.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    summary(model, input_size=(3, 32, 32))
