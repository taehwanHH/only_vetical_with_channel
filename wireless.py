import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        self.in1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=0)
        self.in2 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=0)
        self.in3 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=0)
        self.in4 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=0)

        self.out1 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1)
        self.out2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1)
        self.out3 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1)
        self.out4 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1)

        self.prelu = nn.PReLU()

        self.pool = nn.AdaptiveAvgPool2d((8, 8))

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(2048, self.latent_dim)

        self.essen = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=1)

    def forward(self, x):

        height = x.shape[-1]

        if height == 32:
            encoder_level = 1
            x = self.prelu(self.in1(x))
            x = self.prelu(self.out1(x))
            x = self.prelu(self.out2(x))
            x = self.prelu(self.out3(x))
            x = self.prelu(self.out4(x))

            x = self.prelu(self.essen(x))
            # print(x.shape)

        if height == 26:  # Selection Ratio : 0.765625
            encoder_level = 2
            x = self.prelu(self.in2(x))
            x = self.prelu(self.out2(x))
            x = self.prelu(self.out3(x))
            x = self.prelu(self.out4(x))

            x = self.prelu(self.essen(x))
            # print(x.shape)

        if height == 20:  # Selectio Ratio : 0.5625
            encoder_level = 3
            x = self.prelu(self.in3(x))
            x = self.prelu(self.out3(x))
            x = self.prelu(self.out4(x))
            x = self.prelu(self.essen(x))
            # print(x.shape)

        if height == 16:  # Selection Ratio : 0.390625
            encoder_level = 4
            x = self.prelu(self.in4(x))
            x = self.prelu(self.out4(x))
            x = self.prelu(self.essen(x))
            # print(x.shape)

        x = self.pool(x)

        x = self.flatten(x)
        encoded = self.linear(x)

        return encoded, encoder_level


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim

        self.linear = nn.Linear(self.latent_dim, 2048)
        self.prelu = nn.PReLU()
        self.unflatten = nn.Unflatten(1, (32, 8, 8))

        self.essen = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=1, output_padding=1)

        self.in4 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=1)
        self.in3 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=1)
        self.in2 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=1)
        self.in1 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=0)

        self.out4 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1, padding=4)
        self.out3 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1, padding=3)
        self.out2 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1, padding=1)
        self.out1 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, encoder_level):

        x = self.essen(self.unflatten(self.prelu(self.linear(x))))

        if encoder_level == 1:
            x = self.prelu(self.in4(x))
            x = self.prelu(self.in3(x))
            x = self.prelu(self.in2(x))
            x = self.prelu(self.in1(x))
            x = self.out1(x)


        elif encoder_level == 2:
            x = self.prelu(self.in4(x))
            x = self.prelu(self.in3(x))
            x = self.prelu(self.in2(x))
            x = self.out2(x)


        elif encoder_level == 3:
            x = self.prelu(self.in4(x))
            x = self.prelu(self.in3(x))
            x = self.out3(x)


        elif encoder_level == 4:
            x = self.prelu(self.in4(x))
            x = self.out4(x)

        decoded = self.sigmoid(x)

        return decoded

class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def Power_norm(self, z, P=1 / np.sqrt(2)):

        batch_size, z_dim = z.shape
        z_power = torch.sqrt(torch.sum(z ** 2, 1))
        z_M = z_power.repeat(z_dim, 1)

        return np.sqrt(P * z_dim) * z / z_M.t()
    def Power_norm_complex(self, z, P=1 / np.sqrt(2)):

        batch_size, z_dim = z.shape
        z_com = torch.complex(z[:, 0:z_dim:2], z[:, 1:z_dim:2])
        z_com_conj = torch.complex(z[:, 0:z_dim:2], -z[:, 1:z_dim:2])
        z_power = torch.sum(z_com * z_com_conj, 1).real
        z_M = z_power.repeat(z_dim // 2, 1)
        z_nlz = np.sqrt(P * z_dim) * z_com / torch.sqrt(z_M.t())
        z_out = torch.zeros(batch_size, z_dim).to(device)
        z_out[:, 0:z_dim:2] = z_nlz.real
        z_out[:, 1:z_dim:2] = z_nlz.imag

        return z_out

    def AWGN_channel(self, x, snr, P=1):
        batch_size, length = x.shape
        gamma = 10 ** (snr / 10.0)
        noise = np.sqrt(P / gamma) * torch.randn(batch_size, length).cuda()
        y = x + noise
        return y

    def Fading_channel(self, x, snr, P=1):

        gamma = 10 ** (snr / 10.0)
        [batch_size, feature_length] = x.shape
        K = feature_length // 2

        h_I = torch.randn(batch_size, K).to(device)
        h_R = torch.randn(batch_size, K).to(device)
        h_com = torch.complex(h_I, h_R)
        x_com = torch.complex(x[:, 0:feature_length:2], x[:, 1:feature_length:2])
        y_com = h_com * x_com

        n_I = np.sqrt(P / gamma) * torch.randn(batch_size, K).to(device)
        n_R = np.sqrt(P / gamma) * torch.randn(batch_size, K).to(device)
        noise = torch.complex(n_I, n_R)

        y_add = y_com + noise
        y = y_add / h_com

        y_out = torch.zeros(batch_size, feature_length).to(device)
        y_out[:, 0:feature_length:2] = y.real
        y_out[:, 1:feature_length:2] = y.imag

        return y_out

    def forward(self, x, SNRdB, channel):

        encoded, encoder_level = self.encoder(x)

        if channel == 'AWGN':
            normalized_x = self.Power_norm(encoded)
            channel_output = self.AWGN_channel(normalized_x, SNRdB)
        elif channel == 'Rayleigh':
            normalized_complex_x = self.Power_norm_complex(encoded)
            channel_output = self.Fading_channel(normalized_complex_x, SNRdB)

        decoded = self.decoder(channel_output, encoder_level)

        return decoded