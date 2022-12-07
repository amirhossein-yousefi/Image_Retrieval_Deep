import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoEncBase(nn.Module):
    def training_step(self, batch):
        images, _ = batch
        out = self(images)  # Generate predictions
        loss = F.binary_cross_entropy(out, images)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, _ = batch
        out = self(images)  # Generate predictions
        loss = F.binary_cross_entropy(out, images)  # Calculate loss
        return {'val_loss': loss.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss']))


class ConvEncoder(nn.Module):
    """
    A simple Convolutional Encoder Model
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, (3, 3), padding=(1, 1))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d((2, 2))

        self.conv4 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d((2, 2))

    def forward(self, x):
        # Downscale the image with conv maxpool etc.
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        return x


class ConvDecoder(nn.Module):
    """
    A simple Convolutional Decoder Model
    """

    def __init__(self):
        super().__init__()
        self.deconv2 = nn.ConvTranspose2d(128, 64, (2, 2), stride=(2, 2))
        self.relu2 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(64, 64, (2, 2), stride=(2, 2))
        self.relu3 = nn.ReLU()
        self.deconv4 = nn.ConvTranspose2d(64, 32, (2, 2), stride=(2, 2))
        self.relu4 = nn.ReLU()

        self.deconv5 = nn.ConvTranspose2d(32, 32, 1)
        self.relu5 = nn.ReLU()
        self.deconv6 = nn.ConvTranspose2d(32, 16, (2, 2), stride=(2, 2))
        self.relu6 = nn.ReLU()

        self.deconv7 = nn.ConvTranspose2d(16, 3, 1)
        self.relu7 = nn.ReLU()

    def forward(self, x):
        # Upscale the image with convtranspose etc.
        x = self.deconv2(x)
        x = self.relu2(x)

        x = self.deconv3(x)
        x = self.relu3(x)

        x = self.deconv4(x)
        x = self.relu4(x)
        x = self.deconv5(x)
        x = self.relu5(x)
        x = self.deconv6(x)
        x = self.relu6(x)

        x = self.deconv7(x)
        x = self.relu7(x)
        return x


class AutoEncCnnModel(ConvAutoEncBase):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, xb):
        return self.dec(self.enc(xb))
