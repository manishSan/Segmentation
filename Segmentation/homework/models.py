import torch
import torch.nn.functional as F
import torchvision as tv
import math

class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Your code here

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        Hint: Don't be too fancy, this is a one-liner
        """
        return F.cross_entropy(input, target)


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_shape = 64 * 64 * 3
        self.out_shape = 6
        self.linear = torch.nn.Linear(self.in_shape, self.out_shape)

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        reshaped_x = x.view(-1, self.in_shape)
        logit = self.linear(reshaped_x)
        return logit


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_shape = 64 * 64 * 3
        out_shape = 6
        n_hidden1 = 1000
        n_hidden2 = 100
        # linear1 = torch.nn.Linear(self.in_shape, self.n_hidden1)
        # # relu1 = torch.nn.ReLU()
        # linear2 = torch.nn.Linear(self.n_hidden1, self.n_hidden2)
        # # relu2 = torch.nn.ReLU()
        # # linear3 = torch.nn.Linear(self.n_hidden2, self.out_shape)
        # nonLinear = torch.nn.ReLU()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(self.in_shape, n_hidden2),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden2, out_shape),
        )

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        reshaped_x = x.view(-1, self.in_shape)
        logit = self.network(reshaped_x)
        return logit


class CNNClassifier(torch.nn.Module):
    
    class CNNBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1) -> None:
            super().__init__()
            self.layer = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=1, padding=1, stride=stride),
                torch.nn.ReLU()
            )
        
        def forward(self, x):
            return self.layer(x)
        
    class CNNResidualBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, stride = 1, down_sample = None):
            super().__init__()
            inner_layers = []
            conv1 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU()
            )
            inner_layers.append(conv1)

            conv2 = torch.nn.Sequential(
                torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(out_channels),
            )
            inner_layers.append(conv2)

            self.layers = torch.nn.Sequential(*inner_layers)
            self.down_sample = down_sample
            self.relu = torch.nn.ReLU()
            self.out_channel = out_channels

        def forward(self, x):
            residual = x
            out = self.layers(x)
            if self.down_sample:
                residual = self.down_sample(x)
            out += residual
            out = self.relu(out)
            return out
        
    def __init__(self, hidden_layers_count=[3, 1, 1], hidden_layer_channel_size=[64, 128, 256], hidden_layer_strides=[1, 2, 2], n_input_channel=3, n_output=6):
        super().__init__()

        c = 32

        hidden_layers = []
        # Create first layer with kernal_size = 7
        layer0 = torch.nn.Sequential(
            torch.nn.Conv2d(n_input_channel, c, kernel_size=7, padding=3, stride=2),
            torch.nn.BatchNorm2d(c),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        hidden_layers.append(layer0)

        for index, block_count in enumerate(hidden_layers_count):
            out_size = hidden_layer_channel_size[index]
            layer = self._make_layer(c, out_size, block_count, stride=hidden_layer_strides[index])
            hidden_layers.append(layer)
            c = out_size

        self.net = torch.nn.Sequential(*hidden_layers)        
        self.avg_pool = torch.nn.AvgPool2d(7, stride=1)
        self.classifier = torch.nn.Linear(c, n_output)

        # Define proportion or neurons to dropout
        self.dropout = torch.nn.Dropout(0.50)

        # for h in hidden_layers:
        #     L.append(self.CNNBlock(c, h))
        #     c = h

        # self.net = torch.nn.Sequential(*L)
        # self.classifier = torch.nn.Linear(c, n_output)

    def _make_layer(self, in_channel, out_channel, block_count, stride):
        # down sample if we are reducing the number of channels
        down_sample = None
        if (stride != 1 or in_channel != out_channel):
            down_sample = torch.nn.Sequential (
                torch.nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(out_channel)
            )
        
        layers = []
        layers.append(
            CNNClassifier.CNNResidualBlock(in_channel, out_channel, stride, down_sample)
        )
        for i in range(1, block_count):
            layers.append(
                CNNClassifier.CNNResidualBlock(out_channel, out_channel)
            )

        return torch.nn.Sequential(*layers)
        

    def forward(self, x):
        x = self.net(x)
        x = self.dropout(x)

        # x = self.avg_pool(x)
        x = x.mean(dim=[2, 3])
        # x = x.view(x.size(0), -1)
        x = self.classifier(x)                        
        return x

class FCN(torch.nn.Module):
    def __init__(self, n_input_channel=3, n_output=5, retain_dim=True):
        super().__init__()

        self.retain_dim = retain_dim
        c = 32
        hidden_layers_count=[2, 3, 1, 1] 
        hidden_layer_channel_size=[64, 128, 256, 512]
        hidden_layer_strides=[1, 2, 2, 1]

        self.hidden_layers = []
        # Create first layer with kernal_size = 7
        self.layer0 = torch.nn.Sequential(
            torch.nn.Conv2d(n_input_channel, c, kernel_size=7, padding=3, stride=2),
            torch.nn.BatchNorm2d(c),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # self.hidden_layers.append(layer0)

        # for index, block_count in enumerate(hidden_layers_count):
        #     out_size = hidden_layer_channel_size[index]
        #     layer = self._make_layer(c, out_size, block_count, stride=hidden_layer_strides[index])
        #     self.hidden_layers.append(layer)
        #     c = out_size

        self.conv1 = self._make_layer(c, hidden_layer_channel_size[0], 
                                      hidden_layers_count[0], 
                                      stride=hidden_layer_strides[0])
        c = hidden_layer_channel_size[0]

        self.conv2 = self._make_layer(c, hidden_layer_channel_size[1], 
                                      hidden_layers_count[1], 
                                      stride=hidden_layer_strides[1])
        c = hidden_layer_channel_size[1]

        self.conv3 = self._make_layer(c, hidden_layer_channel_size[2], 
                                      hidden_layers_count[2], 
                                      stride=hidden_layer_strides[2])
        c = hidden_layer_channel_size[2]

        # self.conv4 = self._make_layer(c, hidden_layer_channel_size[3], 
        #                               hidden_layers_count[3], 
        #                               stride=hidden_layer_strides[3])
        # c = hidden_layer_channel_size[3]

        # Define proportion or neurons to dropout
        self.dropout = torch.nn.Dropout(0.4)

        # fully connected
        self.fc = torch.nn.Conv2d(c, n_output, kernel_size=1)

        # upConvolutions 
        self.deConv1 = torch.nn.ConvTranspose2d(n_output, n_output, kernel_size=4, padding=1, stride=2)
        self.deConv1.weight.data.copy_(self._bilinear_kernel(n_output, n_output, 4))
        c = hidden_layer_channel_size[1]
        c_input = c + n_output
        self.upConv1 = self._make_layer(c_input, n_output, 1, stride=1)
        
        self.deConv2 = torch.nn.ConvTranspose2d(n_output, n_output, kernel_size=6, padding=2, stride=2)
        self.deConv2.weight.data.copy_(self._bilinear_kernel(n_output, n_output, 6))
        c = hidden_layer_channel_size[0]
        c_input = c + n_output
        self.upConv2 = self._make_layer(c_input, n_output, 1, stride=1)

        self.deConv3 = torch.nn.ConvTranspose2d(n_output, n_output, kernel_size=4, padding=0, stride=4)
        self.deConv3.weight.data.copy_(self._bilinear_kernel(n_output, n_output, 4))
        # c = hidden_layer_channel_size[0]
        # c_input = c + n_output
        # self.upConv3 = self._make_layer(c_input, n_output, 1, stride=1)

        # self.upConv4 = torch.nn.ConvTranspose2d(n_output, n_output, kernel_size=4, padding=0, stride=4)
        # self.upConv4.weight.data.copy_(self._bilinear_kernel(n_output, n_output, 4))

    def _make_layer(self, in_channel, out_channel, block_count, stride):
        # down sample if we are reducing the number of channels
        down_sample = None
        if (stride != 1 or in_channel != out_channel):
            down_sample = torch.nn.Sequential (
                torch.nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(out_channel)
            )
        
        layers = []
        layers.append(
            CNNClassifier.CNNResidualBlock(in_channel, out_channel, stride, down_sample)
        )
        for i in range(1, block_count):
            layers.append(
                CNNClassifier.CNNResidualBlock(out_channel, out_channel)
            )

        return torch.nn.Sequential(*layers)
    
    def _bilinear_kernel(self, in_channels, out_channels, kernel_size):
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = (torch.arange(kernel_size).reshape(-1, 1),
            torch.arange(kernel_size).reshape(1, -1))
        filt = (1 - torch.abs(og[0] - center) / factor) * \
            (1 - torch.abs(og[1] - center) / factor)
        weight = torch.zeros((in_channels, out_channels,
                            kernel_size, kernel_size))
        weight[range(in_channels), range(out_channels), :, :] = filt
        return weight
    
    def crop(self, enc_ftrs, x):
            _, _, H, W = x.shape
            enc_ftrs   = tv.transforms.CenterCrop([H, W])(enc_ftrs)
            return enc_ftrs
    
    def forward(self, x):
        # add convolutions
        out_size = (x.size()[-2], x.size()[-1])

        x = self.layer0(x)
        cnv1 = self.dropout(self.conv1(x))
        x = cnv1
        cnv2 = self.dropout(self.conv2(x))
        x = cnv2
        cnv3 = self.dropout(self.conv3(x))
        x = cnv3
    
        # add dropout
        x = self.dropout(x)

        # # add 1x1 conv
        x = self.fc(x)

        # first up convolution
        x = self.deConv1(x)
        features = self.crop(cnv2, x)
        z = torch.cat([x, features], dim=1)
        # print('size z ', z.size())
        x = self.upConv1(self.dropout(z))
        
        # second up convolution
        x = self.deConv2(x)
        features = self.crop(cnv1, x)
        z = torch.cat([x, features], dim=1)
        x = self.upConv2(self.dropout(z))

        # second up convolution
        x = self.deConv3(x)
        # z = torch.cat([x, cnv1], dim=1)
        # x = self.upConv3(self.dropout(z))

        if self.retain_dim:
            x = F.interpolate(x, out_size)

        return x

class FCN_online(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, in_channel, out_channel) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(in_channel, out_channel, 3),
                torch.nn.ReLU(),
                torch.nn.Conv2d(out_channel, out_channel, 3),
                torch.nn.ReLU()
            )
        
        def forward(self, x):
            return self.net(x)
    
    class Encoder(torch.nn.Module):
        def __init__(self, channels=(3,64,128,256)) -> None:
            super().__init__()
            self.encoder_blocks = torch.nn.ModuleList(
                    [FCN.Block(channels[i], channels[i+1]) for i in range(len(channels)-1)]
                )
            self.pool = torch.nn.MaxPool2d(2)

        def forward(self, x):
            filters = []
            for encoder in self.encoder_blocks:
                # print('size of x before input: ', x.size())
                x = encoder(x)
                filters.append(x)
                x = self.pool(x)
            
            return filters

    class Decoder(torch.nn.Module):
        def __init__(self, channels=(256, 128, 64)) -> None:
            super().__init__()
            self.channels = channels
            self.upConv = torch.nn.ModuleList(
                    [torch.nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2) for i in range(len(channels)-1)]
                )
            self.dec_block = torch.nn.ModuleList(
                    [FCN.Block(channels[i], channels[i+1]) for i in range(len(channels)-1)]
                )

        def forward(self, x, encoder_features):
            for i in range(len(self.channels)-1):
                x = self.upConv[i](x)
                enc_feature = self.crop(encoder_features[i], x)
                x = torch.cat([x, enc_feature], dim=1)
                x = self.dec_block[i](x)
            
            return x

        def crop(self, enc_ftrs, x):
            _, _, H, W = x.shape
            enc_ftrs   = tv.transforms.CenterCrop([H, W])(enc_ftrs)
            return enc_ftrs

    def __init__(self, enc_chs=(3,64,128,256), dec_chs=(256, 128, 64), num_class=5, retain_dim=True):
        super().__init__()
        self.encoder     = FCN.Encoder(enc_chs)
        self.decoder     = FCN.Decoder(dec_chs)
        self.head        = torch.nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim

    def forward(self, x):
        # print('input size', x.size())
        out_size = (x.size()[-2], x.size()[-1])
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, out_size)
        return out


class FCN_modular(torch.nn.Module):

    class FCNBlock(torch.nn.Module):
        def __init__(self, in_channel, out_channel, kernel=3, stride=1, batchNorm=True):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(in_channel, out_channel, kernel, stride, padding=math.ceil(kernel/2)),
                torch.nn.BatchNorm2d(out_channel),
                torch.nn.ReLU(),
                torch.nn.Conv2d(out_channel, out_channel, kernel, stride, padding=math.ceil(kernel/2)),
                torch.nn.BatchNorm2d(out_channel),
                torch.nn.ReLU()
            )
            
        def forward(self, x):
            return self.net(x)
        
    class FCNEncoder(torch.nn.Module):
        def __init__(self, in_channel, hidden_layer_channel_size=[64, 128], dropout_rate=[0.1, 0.2, 0.5]):
            super().__init__()
            c = in_channel
            self.layers = torch.nn.ModuleList()
            for channel, d_rate in zip(hidden_layer_channel_size, dropout_rate):
                self.layers.append(
                    torch.nn.Sequential(
                        FCN.FCNBlock(c, channel),
                        torch.nn.MaxPool2d(2),
                        torch.nn.Dropout(d_rate)
                    )
                )
                c = channel
        
        def forward(self, x):
            results = []
            for l in self.layers:
                # print(x.size())
                # print(l)
                x = l(x)
                # print(x.size())
                results.append(
                    x
                )
            return results, x   

    class FCNDecoder(torch.nn.Module):
        def __init__(self, in_channel, hidden_layer_channel_size=[128, 64], dropout_rate=[0.1, 0.2, 0.5]):
            super().__init__()
            self.deConvs = torch.nn.ModuleList()
            self.upConvs = torch.nn.ModuleList()
            self.dropouts = torch.nn.ModuleList()
            c = in_channel
            for channel, d_rate in zip(hidden_layer_channel_size, dropout_rate):
                de_conv = torch.nn.ConvTranspose2d(in_channels=c, out_channels=channel, kernel_size=3, stride=2)
                # de_conv.weight.copy_(self._bilinear_kernel(c, channel, 3))
                self.deConvs.append(de_conv)
                
                self.upConvs.append(
                    FCN.FCNBlock(channel * 2, channel),
                )
                c = channel
                
                self.dropouts.append(
                    torch.nn.Dropout(d_rate)
                )

        def forward(self, x, encoder_features):
            for de, up, drop, features in zip(self.deConvs, self.upConvs, self.dropouts, encoder_features[::-1]):
                # print('x.size', x.size())
                x = de(x)
                # print('x.size', x.size(), 'features size', features.size())
                f = self.crop(features, x)
                # print('f.size',f.size())
                z = torch.cat([x, f], dim=1)
                # print('z.size',z.size())
                x = up(z)
                # print('x.size',x.size())
                x = drop(x)
                # print('x.size',x.size())

            return x

        def crop(self, enc_ftrs, x):
            # x.size torch.Size([512, 256, 39, 47]) features size torch.Size([512, 256, 15, 19])
            _, _, H, W = x.shape
            enc_ftrs   = tv.transforms.CenterCrop([H, W])(enc_ftrs)
            return enc_ftrs

        def _bilinear_kernel(self, in_channels, out_channels, kernel_size):
            factor = (kernel_size + 1) // 2
            if kernel_size % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            og = (torch.arange(kernel_size).reshape(-1, 1),
                torch.arange(kernel_size).reshape(1, -1))
            filt = (1 - torch.abs(og[0] - center) / factor) * \
                (1 - torch.abs(og[1] - center) / factor)
            weight = torch.zeros((in_channels, out_channels,
                                kernel_size, kernel_size))
            weight[range(in_channels), range(out_channels), :, :] = filt
            return weight
        
    def __init__(self, n_input_channel=3, n_output=5, retain_dim=True):
        super().__init__()

        self.retain_dim = retain_dim
        hidden_layer_channel_size=[64, 128]

        self.encoder = FCN.FCNEncoder(n_input_channel)
        middle_channel = hidden_layer_channel_size[-1] * 2
        self.middle = FCN.FCNBlock(hidden_layer_channel_size[-1], middle_channel)        
        self.decoder = FCN.FCNDecoder(middle_channel)
        self.fc = torch.nn.Conv2d(hidden_layer_channel_size[0], n_output, kernel_size=1)

    
    def forward(self, x):
        # add convolutions
        out_size = (x.size()[-2], x.size()[-1])

        features, x = self.encoder(x)
        # print(self.middle)
        # print(x.size())
        x = self.middle(x)
        # print(x.size())
        x = self.decoder(x, features)
        x = self.fc(x)

        if self.retain_dim:
            x = F.interpolate(x, out_size)

        return x

model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
    'cnn': CNNClassifier,
    'fcn': FCN
}

def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
