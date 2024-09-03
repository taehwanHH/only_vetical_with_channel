import torch
import torch.nn as nn

from DDPG_module.MLP import MultiLayerPerceptron as MLP
from param import Hyper_Param
from robotic_env import RoboticEnv

DEVICE = Hyper_Param['DEVICE']
SensEnc_latent_dim = Hyper_Param['SensEnc_latent_dim']
CUEnc_latent_dim = Hyper_Param['CUEnc_latent_dim']

class SensorEncoder(nn.Module, RoboticEnv):
    def __init__(self, output_dim=SensEnc_latent_dim):
        super(SensorEncoder, self).__init__()
        RoboticEnv.__init__(self)

        self.output_dim = output_dim

        self.mlp = MLP(self.num_sensor_output, self.output_dim,
                       num_neurons = Hyper_Param['SensEnc_neurons'],
                       hidden_act = 'ReLU',
                       out_act ='Tanh' ## including bottleneck layer
                       )
        self.out = nn.Linear(self.output_dim, self.output_dim)
    def forward(self, x):
        x = self.mlp(x)
        encoded = torch.sigmoid(self.out(x))
        return encoded


class SensorDecoder(nn.Module, RoboticEnv):
    def __init__(self, input_dim=SensEnc_latent_dim):
        super(SensorDecoder, self).__init__()
        RoboticEnv.__init__(self)

        self.input_dim = input_dim*self.num_robot

        self.mlp = MLP(self.input_dim, self.state_dim,
                       num_neurons = Hyper_Param['SensDec_neurons'],
                       hidden_act = 'ReLU',
                       out_act = 'Sigmoid')

    def forward(self,x):
        decoded = self.mlp(x)
        return decoded


class CUEncoder(nn.Module, RoboticEnv):
    def __init__(self, output_dim=CUEnc_latent_dim):
        super(CUEncoder, self).__init__()
        RoboticEnv.__init__(self)

        self.output_dim = output_dim

        self.mlp = MLP(self.action_dim, self.output_dim,
                       num_neurons=Hyper_Param['CUEnc_neurons'],
                       hidden_act='ReLU',
                       out_act='Tanh'  ## including bottleneck layer
                       )
        self.out = nn.Linear(self.output_dim, self.output_dim)

    def forward(self, x):
        x = self.mlp(x)
        encoded = torch.sigmoid(self.out(x))
        return encoded


class CUDecoder(nn.Module, RoboticEnv):
    def __init__(self, input_dim=CUEnc_latent_dim):
        super(CUDecoder, self).__init__()
        RoboticEnv.__init__(self)

        self.input_dim = input_dim

        self.mlp = MLP(self.input_dim, self.action_dim,
                       num_neurons = Hyper_Param['CUDec_neurons'],
                       hidden_act = 'ReLU',
                       out_act = 'Sigmoid')

    def forward(self,x):
        decoded = self.mlp(x)
        return decoded


class NormalizeTX:
    def __init__(self, _iscomplex):
        self._iscomplex = _iscomplex
    def apply(self, x):
        dim = x.shape[1]//2 if self._iscomplex else x.shape[1]
        norm = dim**0.5 / torch.sqrt(torch.sum(x**2, dim=1))
        return x*norm.view(-1,1)


class Channel:
    def __init__(self, _iscomplex):
        self._iscomplex = _iscomplex



    def ideal_channel(self, x):
        return x

    def awgn(self, x, seed,snr):
        torch.manual_seed(seed)
        std = (10 ** (-snr / 10.) / 2) ** 0.5 if self._iscomplex else (10 ** (
                    -snr / 10.)) ** 0.5  # for complex xs.
        # noise = torch.randn_like(x).to(DEVICE) * std
        # y = x + noise
        # print(noise)
        y = x + torch.randn_like(x).to(DEVICE) * std
        return y


    def fading(self, x,seed, snr):
        torch.manual_seed(seed)
        if self._iscomplex:
            x_shape = x.shape
            _dim = x_shape[1] // 2
            _std = (10 ** (-snr / 10.) / 2) ** 0.5
            _mul = torch.abs(torch.randn(x_shape[0], 2) / (2 ** 0.5)).to(DEVICE)# should divide 2**0.5 here.
            x_ = x.clone()
            x_[:, :_dim] *= _mul[:, 0].view(-1, 1)
            x_[:, _dim:] *= _mul[:, 1].view(-1, 1)
            x = x_
        else:
            _std = (10 ** (-snr / 10.)) ** 0.5
            x = x * torch.abs(torch.randn(x.shape[0], 1)).to(x)
        y = x + torch.randn_like(x) * _std
        # print(_std)
        return y
