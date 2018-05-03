import torch
import torch.nn as nn
import math

def conv_output(shape, Kernel, Padding=(0, 0, 0), Stride=(1, 1, 1)):
    """
    Z : depth
    Y : height
    X : width
    P : padding
    K : kernel
    """
    Z, Y, X = shape
    
    Z_out = ((Z + 2 * Padding[0] - (Kernel[0] - 1) - 1) / Stride[0]) + 1
    Y_out = ((Y + 2 * Padding[1] - (Kernel[1] - 1) - 1) / Stride[1]) + 1
    X_out = ((X + 2 * Padding[2] - (Kernel[2] - 1) - 1) / Stride[2]) + 1
    
    return (Z_out, Y_out, X_out)

def pool_output(shape, Kernel, Padding=(0, 0, 0), Stride=(1, 1, 1)):
    """
    Z : depth
    Y : height
    X : width
    P : padding
    K : kernel
    """
    Z, Y, X = shape

    Z_out = math.floor(((Z + 2 * Padding[0] - (Kernel[0] - 1) - 1) / Stride[0]) + 1)
    Y_out = math.floor(((Y + 2 * Padding[1] - (Kernel[1] - 1) - 1) / Stride[1]) + 1)
    X_out = math.floor(((X + 2 * Padding[2] - (Kernel[2] - 1) - 1) / Stride[2]) + 1)
        
    return (Z_out, Y_out, X_out)

class ConvColumn(nn.Module):

    def __init__(self, num_classes, kernel_size):
        super(ConvColumn, self).__init__()

        self.conv_layer1 = self._make_conv_layer(3, 64, (1, 2, 2), (1, 2, 2), kernel_size=kernel_size)
        self.conv_layer2 = self._make_conv_layer(64, 128, (2, 2, 2), (2, 2, 2), kernel_size=kernel_size)
        self.conv_layer3 = self._make_conv_layer(
            128, 256, (2, 2, 2), (2, 2, 2), kernel_size=kernel_size)
        self.conv_layer4 = self._make_conv_layer(
            256, 256, (2, 2, 2), (2, 2, 2), kernel_size=kernel_size)

        c = self._compute_linear_size(kernel_size)
        linear_size = 256
        for i in c:
            if i != 0:
                linear_size *= i
                
        self.fc5 = nn.Linear(linear_size, 512)
        self.fc5_act = nn.ELU()
        self.fc6 = nn.Linear(512, num_classes)

    def _compute_linear_size(self, kernel_size):
        shape = (18, 84, 84)
        Conv_K = kernel_size
        Padding = (1, 1, 1)
        Stride = (1, 2, 2)
        Pool_K = (1, 2, 2)

        shape = conv_output(shape, Kernel=Conv_K, Padding=Padding)
        shape = pool_output(shape, Kernel=Pool_K, Stride=Stride)

        Pool_K = (2, 2, 2)
        Stride = (2, 2, 2)

        shape = conv_output(shape, Kernel=Conv_K, Padding=Padding)
        shape = pool_output(shape, Kernel=Pool_K, Stride=Stride)

        shape = conv_output(shape, Kernel=Conv_K, Padding=Padding)
        shape = pool_output(shape, Kernel=Pool_K, Stride=Stride)

        shape = conv_output(shape, Kernel=Conv_K, Padding=Padding)
        shape = pool_output(shape, Kernel=Pool_K, Stride=Stride)

        return shape

    def _make_conv_layer(self, in_c, out_c, pool_size, stride, kernel_size=(3, 3, 3)):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ELU(),
            nn.MaxPool3d(pool_size, stride=stride, padding=0)
        )
        return conv_layer

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)

        x = x.view(x.size(0), -1)

        x = self.fc5(x)
        x = self.fc5_act(x)

        x = self.fc6(x)
        return x

class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self, num_classes):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(4608, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = self.pool1(h)
        
        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(h.size(0), -1)
        h = self.relu(self.fc6(h))
        
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)

        logits = self.fc8(h)

        return logits

"""
References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." 
Proceedings of the IEEE international conference on computer vision. 2015.
"""

class LRCN(nn.Module):
    def __init__(self, num_classes, kernel_size):
        super(LRCN, self).__init__()

        self.conv_layer1 = self._make_conv_layer(3, 64, (1, 2, 2), (1, 2, 2), kernel_size=kernel_size)
        self.conv_layer2 = self._make_conv_layer(64, 128, (1, 2, 2), (1, 2, 2), kernel_size=kernel_size)
        self.conv_layer3 = self._make_conv_layer(
            128, 256, (1, 2, 2), (1, 2, 2), kernel_size=kernel_size)
        self.conv_layer4 = self._make_conv_layer(
            256, 256, (1, 2, 2), (1, 2, 2), kernel_size=kernel_size)

        c = self._compute_linear_size(kernel_size)
        linear_size = 256
        for i in c:
            if i != 0:
                linear_size *= i
        linear_size = int(linear_size / 18)
                        
        self.fc5 = nn.Linear(linear_size, 512)
        self.fc5_act = nn.ELU()
        
        self.lstm = nn.LSTM(512, 256, num_layers=1, batch_first=True)
        self.fc6 = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(dim=2)

    def _compute_linear_size(self, kernel_size):
        shape = (18, 84, 84)
        Conv_K = kernel_size
        Padding = (0, 1, 1)
        Stride = (1, 2, 2)
        Pool_K = (1, 2, 2)

        shape = conv_output(shape, Kernel=Conv_K, Padding=Padding)
        shape = pool_output(shape, Kernel=Pool_K, Stride=Stride)

        Pool_K = (1, 2, 2)
        Stride = (1, 2, 2)

        shape = conv_output(shape, Kernel=Conv_K, Padding=Padding)
        shape = pool_output(shape, Kernel=Pool_K, Stride=Stride)

        shape = conv_output(shape, Kernel=Conv_K, Padding=Padding)
        shape = pool_output(shape, Kernel=Pool_K, Stride=Stride)

        shape = conv_output(shape, Kernel=Conv_K, Padding=Padding)
        shape = pool_output(shape, Kernel=Pool_K, Stride=Stride)

        return shape

    def _make_conv_layer(self, in_c, out_c, pool_size, stride, kernel_size=(3, 3, 3)):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=kernel_size, stride=1, padding=(0, 1, 1)),
            nn.BatchNorm3d(out_c),
            nn.ELU(),
            nn.MaxPool3d(pool_size, stride=stride, padding=0)
        )
        return conv_layer

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)

        x = x.permute(0, 2, 1, 3, 4)
        x = x.contiguous().view(x.size(0), x.size(1), -1)
        
        x = self.fc5(x)
        x = self.fc5_act(x)

        lstm_out, (hidden, context) = self.lstm(x)
        print(hidden.size())
        x = torch.squeeze(hidden)
        print(x.size())
        x = self.fc6(x)
        print(x.size())

        return x
