import copy

import torch
from torch import nn
import torch.nn.functional as F


class EMGFlowNet(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 12, (1, 63), padding=(0, 31))
        self.bn1_1 = torch.nn.BatchNorm2d(12)
        self.depthConv1 = torch.nn.Conv2d(12, 12 * 2, (12, 1), groups=12, bias=False)

        self.lstm0 = torch.nn.LSTM(300, 300, 1, batch_first=True, dropout=0.3)
        self.lstm1 = torch.nn.LSTM(300, 300, 1, batch_first=True, dropout=0.3)
        self.lstm2 = torch.nn.LSTM(300, 300, 1, batch_first=True, dropout=0.3)
        self.lstm3 = torch.nn.LSTM(300, 300, 1, batch_first=True, dropout=0.3)
        self.lstm4 = torch.nn.LSTM(300, 300, 1, batch_first=True, dropout=0.3)
        self.lstm5 = torch.nn.LSTM(300, 300, 1, batch_first=True, dropout=0.3)
        self.lstm6 = torch.nn.LSTM(300, 300, 1, batch_first=True, dropout=0.3)
        self.lstm7 = torch.nn.LSTM(300, 300, 1, batch_first=True, dropout=0.3)
        self.lstm8 = torch.nn.LSTM(300, 300, 1, batch_first=True, dropout=0.3)
        self.lstm9 = torch.nn.LSTM(300, 300, 1, batch_first=True, dropout=0.3)
        self.lstm10 = torch.nn.LSTM(300, 300, 1, batch_first=True, dropout=0.3)
        self.lstm11 = torch.nn.LSTM(300, 300, 1, batch_first=True, dropout=0.3)

        self.bn1_1 = torch.nn.BatchNorm2d(12)
        self.bn1_2 = torch.nn.BatchNorm2d(24)
        self.AvgPool1 = torch.nn.AvgPool2d(1, 4)

        self.conv2_1 = torch.nn.Conv2d(24, 24, (1, 15), groups=24, padding=(0, 7))
        self.conv2_2 = torch.nn.Conv2d(24, 24, 1)
        self.bn2 = torch.nn.BatchNorm2d(24)
        self.AvgPool2 = torch.nn.AvgPool2d(1, 8)

        self.l1 = torch.nn.Linear(240, 4)

        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc1_avg = torch.nn.Conv1d(24, 24, 1, bias=False)
        self.fc2_avg = torch.nn.Conv1d(24, 24, 1, bias=False)

        self.avg_pool_1 = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool_1 = torch.nn.AdaptiveAvgPool1d(1)
        self.fc1_avg_1 = torch.nn.Conv1d(12, 12, 1, bias=False)
        self.fc2_avg_1 = torch.nn.Conv1d(12, 12, 1, bias=False)

        self.param_avg = torch.nn.Parameter(torch.as_tensor(1.), True)
        self.param_max = torch.nn.Parameter(torch.as_tensor(1.), True)

    def forward(self, x):
        bs, tp_bs, channel, length, _ = x.shape
        # ori_x = x.transpose(2, 3)
        output_feature = []
        ori_x = x.reshape(bs, tp_bs, 1, channel, length).type(torch.float)
        # output_feature.append(copy.deepcopy(ori_x))

        self.lstm0.flatten_parameters()
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        self.lstm3.flatten_parameters()
        self.lstm4.flatten_parameters()
        self.lstm5.flatten_parameters()
        self.lstm6.flatten_parameters()
        self.lstm7.flatten_parameters()
        self.lstm8.flatten_parameters()
        self.lstm9.flatten_parameters()
        self.lstm10.flatten_parameters()
        self.lstm11.flatten_parameters()

        ori_x_tmp, (hn, cn) = self.lstm0(ori_x[:, :, 0, 0, :])
        catted_ori_x = ori_x_tmp[:, -1, :].unsqueeze(1)
        ori_x_tmp, (hn, cn) = self.lstm1(ori_x[:, :, 0, 1, :])
        catted_ori_x = torch.cat([catted_ori_x, ori_x_tmp[:, -1, :].unsqueeze(1)], dim=1)
        ori_x_tmp, (hn, cn) = self.lstm2(ori_x[:, :, 0, 2, :])
        catted_ori_x = torch.cat([catted_ori_x, ori_x_tmp[:, -1, :].unsqueeze(1)], dim=1)
        ori_x_tmp, (hn, cn) = self.lstm3(ori_x[:, :, 0, 3, :])
        catted_ori_x = torch.cat([catted_ori_x, ori_x_tmp[:, -1, :].unsqueeze(1)], dim=1)
        ori_x_tmp, (hn, cn) = self.lstm4(ori_x[:, :, 0, 4, :])
        catted_ori_x = torch.cat([catted_ori_x, ori_x_tmp[:, -1, :].unsqueeze(1)], dim=1)
        ori_x_tmp, (hn, cn) = self.lstm5(ori_x[:, :, 0, 5, :])
        catted_ori_x = torch.cat([catted_ori_x, ori_x_tmp[:, -1, :].unsqueeze(1)], dim=1)
        ori_x_tmp, (hn, cn) = self.lstm0(ori_x[:, :, 0, 6, :])
        catted_ori_x = torch.cat([catted_ori_x, ori_x_tmp[:, -1, :].unsqueeze(1)], dim=1)
        ori_x_tmp, (hn, cn) = self.lstm1(ori_x[:, :, 0, 7, :])
        catted_ori_x = torch.cat([catted_ori_x, ori_x_tmp[:, -1, :].unsqueeze(1)], dim=1)
        ori_x_tmp, (hn, cn) = self.lstm2(ori_x[:, :, 0, 8, :])
        catted_ori_x = torch.cat([catted_ori_x, ori_x_tmp[:, -1, :].unsqueeze(1)], dim=1)
        ori_x_tmp, (hn, cn) = self.lstm3(ori_x[:, :, 0, 9, :])
        catted_ori_x = torch.cat([catted_ori_x, ori_x_tmp[:, -1, :].unsqueeze(1)], dim=1)
        ori_x_tmp, (hn, cn) = self.lstm4(ori_x[:, :, 0, 10, :])
        catted_ori_x = torch.cat([catted_ori_x, ori_x_tmp[:, -1, :].unsqueeze(1)], dim=1)
        ori_x_tmp, (hn, cn) = self.lstm5(ori_x[:, :, 0, 11, :])
        catted_ori_x = torch.cat([catted_ori_x, ori_x_tmp[:, -1, :].unsqueeze(1)], dim=1)
        output_feature.append(catted_ori_x.clone().detach())

        # for i in range(0, 12):
        #     ori_x_tmp, (hn, cn) = self.lstm[i](ori_x[:, :, 0, i, :])
        #     # ori_x_tmp = self.conv1(ori_x_tmp[:, -1, :].unsqueeze(1))
        #
        #     if catted_ori_x is not None:
        #         catted_ori_x = torch.cat([catted_ori_x, ori_x_tmp[:, -1, :].unsqueeze(1)], dim=1)
        #     else:
        #         catted_ori_x = ori_x_tmp[:, -1, :].unsqueeze(1)

        # # first attention
        # avg_out = self.fc2_avg_1(F.relu(self.fc1_avg_1(self.avg_pool_1(catted_ori_x))))
        # max_out = self.fc2_avg_1(F.relu(self.fc1_avg_1(self.max_pool_1(catted_ori_x))))
        # out = avg_out + max_out
        # catted_ori_x = catted_ori_x * out
        # output_feature.append(copy.deepcopy(catted_ori_x))
        ori_x = self.conv1(catted_ori_x.unsqueeze(1))
        ori_x = self.bn1_1(ori_x)
        ori_x = F.elu(ori_x)
        output_feature.append(ori_x.clone().detach())
        #
        # for i in range(bs):
        #     ori_x_tmp = self.conv1(ori_x[i])
        #     ori_x_tmp = self.bn1_1(ori_x_tmp)
        #     if catted_ori_x is not None:
        #         catted_ori_x = torch.cat([catted_ori_x, ori_x_tmp.reshape((1, 10, 8, 12, 300))], dim=0)
        #     else:
        #         catted_ori_x = ori_x_tmp.reshape((1, 10, 8, 12, 300))
        # ori_x = self.depthConv1(ori_x)

        # catted_ori_x = catted_ori_x.reshape(bs, tp_bs, -1)

        # ori_x = self.lstm(ori_x)
        ori_x = self.depthConv1(ori_x)
        ori_x = self.bn1_2(ori_x)
        ori_x = F.elu(ori_x)
        ori_x = self.AvgPool1(ori_x)
        output_feature.append(ori_x.clone().detach())
        ori_x = self.conv2_2(self.conv2_1(ori_x))
        ori_x = self.bn2(ori_x)
        ori_x = F.elu(ori_x)
        output_feature.append(ori_x.clone().detach())

        # channel attention
        avg_out = self.fc2_avg(F.relu(self.fc1_avg(self.avg_pool(ori_x.squeeze(2)))))
        max_out = self.fc2_avg(F.relu(self.fc1_avg(self.max_pool(ori_x.squeeze(2)))))
        out = self.param_avg * avg_out + self.param_max * max_out
        ori_x = ori_x * out.unsqueeze(3)
        output_feature.append(out.clone().detach())
        ori_x = F.elu(ori_x)
        ori_x = self.AvgPool2(ori_x)

        ori_x = ori_x.reshape(bs, -1)

        # output_feature.append(copy.deepcopy(ori_x))
        ori_x = F.softmax(self.l1(ori_x), dim=1)

        return ori_x, output_feature

    pass
