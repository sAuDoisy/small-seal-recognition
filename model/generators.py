import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


class UNetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc=3, output_nc=3, ngf=64, is_training=True):
        """
        Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UNetGenerator, self).__init__()
        # construct unet structure

        self.need_dropout = is_training
        self.encoder_input_list = [input_nc, ngf, ngf * 2, ngf * 4, ngf * 8, ngf * 8, ngf * 8, ngf * 8, ngf * 8]
        self.decoder_input_list = [ngf * 8, ngf * 8, ngf * 8, ngf * 8, ngf * 8, ngf * 4, ngf * 2, ngf, output_nc]
        self.encoder = Encoder(self.encoder_input_list)
        self.decoder_concat_layers = [i for i in range(2, 9)]
        self.decoder_for_seal = Decoder(self.decoder_input_list, self.decoder_concat_layers, need_dropout=self.need_dropout)
        self.decoder_for_tra_ch = Decoder(self.decoder_input_list, self.decoder_concat_layers, need_dropout=self.need_dropout)

    def forward(self, img):
        """Standard forward"""
        encoder_results = self.encoder(img)
        fake_seal = self.decoder_for_seal(encoder_results[7], encoder_results)
        fake_tra_ch = self.decoder_for_tra_ch(encoder_results[7], encoder_results)
        # seal -> A, traditional ch -> B
        return encoder_results, fake_seal, fake_tra_ch



class Encoder(nn.Module):
    def __init__(self, channel_list):
        super(Encoder, self).__init__()
        # channel_list = [input_channel(input), ngf, ngf * 2, ngf * 4, ngf * 8, ngf * 8, ngf * 8, ngf * 8, ngf * 8]
        # channel_list = [3, 64, 128, 256, 512, 512, 512, 512, 512]
        self.channel_list = channel_list
        self.encoder_layers_dict = dict()
        self.encoder = self.build_encoder(self.channel_list)

    def forward(self, input_feature):
        encoder_each_layer_result = []
        for i in range(1, 9):
            each_start, each_end = self.encoder_layers_dict[f'e{i}']
            input_feature = self.encoder[each_start: each_end](input_feature)
            encoder_each_layer_result.append(input_feature)
        return encoder_each_layer_result

    def build_encoder(self, channel_list):
        # define 'e0'
        encoder_model = [nn.Conv2d(channel_list[0], channel_list[1], kernel_size=4, stride=2, padding=1)]
        self.encoder_layers_dict['e1'] = (0, 1)

        def encoder_layer(input_filters, output_filters, layer, start):
            act = nn.LeakyReLU(0.2, True)
            # zi2zi conv -> only use bias when it is InstanceNorm2d, don't need bias before BN
            conv = nn.Conv2d(input_filters, output_filters, kernel_size=4, stride=2, padding=1, bias=False)
            norm = nn.BatchNorm2d(output_filters)
            enc_l = [act, conv, norm]
            self.encoder_layers_dict["e%d" % layer] = (start, start + 3)
            # [start: tail] -> ei (i -> NO.layer)
            return start + 3, nn.Sequential(*enc_l)

        tail = 1
        for i in range(1, len(channel_list) - 1):
            tail, turn = encoder_layer(channel_list[i], channel_list[i + 1], i + 1, tail)
            encoder_model += turn
        return nn.Sequential(*encoder_model)


class Decoder(nn.Module):
    def __init__(self, channel_list, concat_layers, need_dropout=False):
        super(Decoder, self).__init__()
        # channel_list = [ngf * 8(input), ngf * 8, ngf * 8, ngf * 8, ngf * 8, ngf * 4, ngf * 2, ngf, output_channel] ->
        # channel_list = [512, 512, 512, 512, 512, 256, 128, 64, 3]
        self.channel_list = channel_list
        self.need_dropout = need_dropout
        self.decoder_layers_dict = dict()
        self.decoder = self.build_decoder(self.channel_list, need_dropout=self.need_dropout)
        # Unet concat from Layer2 to Layer8
        self.concat_layers = concat_layers

    def forward(self, input_feature, encoder_features_list):
        for i in range(1, 9):
            if i in self.concat_layers:
                input_feature = torch.cat([encoder_features_list[8-i], input_feature], 1)
            each_start, each_end = self.decoder_layers_dict[f'd{i}']
            input_feature = self.decoder[each_start: each_end](input_feature)
        return input_feature

    def build_decoder(self, channel_list, dropout_layer=None, need_dropout=False):
        decoder_model = []

        def decoder_layer(input_filters, output_filters, layer, start):
            act = nn.ReLU()
            if layer > 1:
                conv = nn.ConvTranspose2d(input_filters * 2, output_filters, kernel_size=4, stride=2, padding=1, bias=False)
            else:
                conv = nn.ConvTranspose2d(input_filters, output_filters, kernel_size=4, stride=2, padding=1, bias=False)
            norm = nn.BatchNorm2d(output_filters)
            dec = [act, conv, norm]
            cnt = 3
            if need_dropout and 5 <= layer <= 7:
                dec += [nn.Dropout(0.5)]
                cnt += 1
            self.decoder_layers_dict["d%d" % layer] = (start, start + cnt)
            return start + cnt, dec
        pos = 0
        for i in range(len(channel_list) - 2):
            pos, turn = decoder_layer(channel_list[i], channel_list[i + 1], i + 1, pos)
            decoder_model += turn
        # decoder layer 8 (d8)
        d8_act = nn.ReLU()
        d8_conv = nn.ConvTranspose2d(channel_list[-2] * 2, channel_list[-1], kernel_size=4, stride=2, padding=1)
        d8_tanh = nn.Tanh()
        decoder_model += [d8_act, d8_conv, d8_tanh]
        self.decoder_layers_dict['d8'] = (pos, pos + 3)
        return nn.Sequential(*decoder_model)
