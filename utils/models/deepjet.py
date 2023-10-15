import torch.nn as nn
import numpy as np
import torch


class InputConv(nn.Module):
    def __init__(self, in_chn, out_chn, dropout_rate=0.1, **kwargs):
        super(InputConv, self).__init__(**kwargs)

        self.lin = torch.nn.Conv1d(in_chn, out_chn, kernel_size=1)
        self.bn = torch.nn.BatchNorm1d(out_chn, eps=0.001, momentum=0.6)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, norm=True):
        if norm:
            x = self.dropout(self.bn(self.act(self.lin(x))))
        else:
            x = self.act(self.lin(x))
        return x


class LinLayer(nn.Module):
    def __init__(self, in_chn, out_chn, dropout_rate=0.1, **kwargs):
        super(LinLayer, self).__init__(**kwargs)

        self.lin = torch.nn.Linear(in_chn, out_chn)
        self.bn = torch.nn.BatchNorm1d(out_chn, eps=0.001, momentum=0.6)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(self.bn(self.act(self.lin(x))))
        return x


class InputProcess(nn.Module):
    def __init__(self, **kwargs):
        super(InputProcess, self).__init__(**kwargs)

        self.cpf_bn = torch.nn.BatchNorm1d(16, eps=0.001, momentum=0.6)
        self.cpf_conv1 = InputConv(16, 64)
        self.cpf_conv2 = InputConv(64, 32)
        self.cpf_conv3 = InputConv(32, 32)
        self.cpf_conv4 = InputConv(32, 8)

        self.npf_bn = torch.nn.BatchNorm1d(6, eps=0.001, momentum=0.6)
        self.npf_conv1 = InputConv(6, 32)
        self.npf_conv2 = InputConv(32, 16)
        self.npf_conv3 = InputConv(16, 4)

        self.vtx_bn = torch.nn.BatchNorm1d(12, eps=0.001, momentum=0.6)
        self.vtx_conv1 = InputConv(12, 64)
        self.vtx_conv2 = InputConv(64, 32)
        self.vtx_conv3 = InputConv(32, 32)
        self.vtx_conv4 = InputConv(32, 8)

    def forward(self, cpf, npf, vtx):
        cpf = self.cpf_bn(torch.transpose(cpf, 1, 2))
        cpf = self.cpf_conv1(cpf)
        cpf = self.cpf_conv2(cpf)
        cpf = self.cpf_conv3(cpf)
        cpf = self.cpf_conv4(cpf, norm=False)
        cpf = torch.transpose(cpf, 1, 2)

        npf = self.npf_bn(torch.transpose(npf, 1, 2))
        npf = self.npf_conv1(npf)
        npf = self.npf_conv2(npf)
        npf = self.npf_conv3(npf, norm=False)
        npf = torch.transpose(npf, 1, 2)

        vtx = self.vtx_bn(torch.transpose(vtx, 1, 2))
        vtx = self.vtx_conv1(vtx)
        vtx = self.vtx_conv2(vtx)
        vtx = self.vtx_conv3(vtx)
        vtx = self.vtx_conv4(vtx, norm=False)
        vtx = torch.transpose(vtx, 1, 2)

        return cpf, npf, vtx


class DenseClassifier(nn.Module):
    def __init__(self, **kwargs):
        super(DenseClassifier, self).__init__(**kwargs)

        self.LinLayer1 = LinLayer(265, 200)
        self.LinLayer2 = LinLayer(200, 100)
        self.LinLayer3 = LinLayer(100, 100)
        self.LinLayer4 = LinLayer(100, 100)
        self.LinLayer5 = LinLayer(100, 100)
        self.LinLayer6 = LinLayer(100, 100)
        self.LinLayer7 = LinLayer(100, 100)
        self.LinLayer8 = LinLayer(100, 100)

    def forward(self, x):
        x = self.LinLayer1(x)
        x = self.LinLayer2(x)
        x = self.LinLayer3(x)
        x = self.LinLayer4(x)
        x = self.LinLayer5(x)
        x = self.LinLayer6(x)
        x = self.LinLayer7(x)
        x = self.LinLayer8(x)

        return x


class DeepJet(nn.Module):
    def __init__(self, feature_edges, num_classes=6, for_inference = False, **kwargs):
        super(DeepJet, self).__init__(**kwargs)

        self.feature_edges = np.array(feature_edges)
        self.InputProcess = InputProcess()
        self.DenseClassifier = DenseClassifier()

        self.global_bn = torch.nn.BatchNorm1d(15, eps=0.001, momentum=0.6)
        self.cpf_lstm = torch.nn.LSTM(input_size=8, hidden_size=150, num_layers=1, batch_first=True)
        self.npf_lstm = torch.nn.LSTM(input_size=4, hidden_size=50, num_layers=1, batch_first=True)
        self.vtx_lstm = torch.nn.LSTM(input_size=8, hidden_size=50, num_layers=1, batch_first=True)

        self.cpf_bn = torch.nn.BatchNorm1d(150, eps=0.001, momentum=0.6)
        self.npf_bn = torch.nn.BatchNorm1d(50, eps=0.001, momentum=0.6)
        self.vtx_bn = torch.nn.BatchNorm1d(50, eps=0.001, momentum=0.6)

        self.cpf_dropout = nn.Dropout(0.1)
        self.npf_dropout = nn.Dropout(0.1)
        self.vtx_dropout = nn.Dropout(0.1)

        self.Linear = nn.Linear(100, num_classes)
        self.for_inference = for_inference

    def forward(self, x):

        global_vars, cpf, npf, vtx, cpf_4v, npf_4v, vtx_4v = x[0],x[1],x[2],x[3],x[4],x[5],x[6]
        global_vars = self.global_bn(global_vars)[..., 0]

        cpf, npf, vtx = self.InputProcess(cpf, npf, vtx)
        cpf = torch.squeeze(self.cpf_lstm(torch.flip(cpf, dims=[1]))[0][:, -1])
        cpf = self.cpf_dropout(self.cpf_bn(cpf))

        npf = torch.squeeze(self.npf_lstm(torch.flip(npf, dims=[1]))[0][:, -1])
        npf = self.npf_dropout(self.npf_bn(npf))

        vtx = torch.squeeze(self.vtx_lstm(torch.flip(vtx, dims=[1]))[0][:, -1])
        vtx = self.vtx_dropout(self.vtx_bn(vtx))

        fts = torch.cat((global_vars, cpf, npf, vtx), dim=1)
        fts = self.DenseClassifier(fts)

        output = self.Linear(fts)

        if self.for_inference:
            output = torch.softmax(output, dim=1)

        return output
