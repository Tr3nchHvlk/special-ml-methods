import torch
import numpy as np

class CCAN(torch.nn.Module):
    """
    Constructive Cascade ANN
    """

    def __init__(self, hidden_size, input_dim, output_dim):
        super(CCAN, self).__init__()
        self.hidden_size = hidden_size
        self.scale = 0
        self.inter_i = []
        self.inter_o = []
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.io = torch.nn.Linear(input_dim, output_dim)
        self.freeze_former_weights = False

    def upfactor(self):
        if (self.scale == 0):
            self.inter_i.append(
                torch.nn.Sequential(
                    torch.nn.Linear(self.input_dim, self.hidden_size),
                    torch.nn.Sigmoid()
                )
            )
        else:
            self.inter_i.append(
                torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_size + self.input_dim, self.hidden_size),
                    torch.nn.Sigmoid()
                )
            )

        self.inter_o.append(
            torch.nn.Linear(self.hidden_size, self.output_dim)
        )

        self.scale += 1

    def forward(self, x):
        y = self.io(x)

        if (self.scale > 0):
            _y = x

            if (self.freeze_former_weights):
                # first hidden layer
                if (self.scale > 1):
                    with torch.no_grad():
                        _y = self.inter_i[0](x)
                else:
                    _y = self.inter_i[0](x)

                y += self.inter_o[0](_y)

                # intermediate hidden layers
                for d in range(1, self.scale-1):
                    with torch.no_grad():
                        _x = torch.cat((x, _y), dim=1)
                        _y = self.inter_i[d](_x)
                    y += self.inter_o[d](_y)

            else:
                # first hidden layer
                _y = self.inter_i[0](x)
                y += self.inter_o[0](_y)

                # intermediate hidden layers
                for d in range(1, self.scale-1):
                    # with torch.no_grad():
                    _x = torch.cat((x, _y), dim=1)
                    _y = self.inter_i[d](_x)
                    y += self.inter_o[d](_y)

            # final hidden layer 
            # (only final hidden layer weights will be trained, weights from previous trained layers will be frozen)
            if (self.scale > 1):
                _x = torch.cat((x, _y), dim=1)
                _y = self.inter_i[self.scale-1](_x)
                y += self.inter_o[self.scale-1](_y)

        y = torch.nn.functional.sigmoid(y)
        y = torch.nn.functional.softmax(y, dim=1)
        
        return y
    
class SCCN(torch.nn.Module):
    """
    Stacked Constructive Cascade ANN
    """

    def __init__(self, hidden_size, input_dim, output_dim):
        super(SCCN, self).__init__()
        self.hidden_size = hidden_size
        self.scale = 0
        self.inter_i = []
        self.inter_o = []
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.io = torch.nn.Linear(input_dim, output_dim)
        self.freeze_former_weights = False

    def upfactor(self):
        self.inter_i.append(
            torch.nn.Sequential(
                torch.nn.Linear(self.hidden_size * self.scale + self.input_dim, self.hidden_size),
                torch.nn.Sigmoid()
            )
        )

        self.inter_o.append(
            torch.nn.Linear(self.hidden_size, self.output_dim)
        )

        self.scale += 1

    def forward(self, x):
        y = self.io(x)

        _ys = []

        if (self.freeze_former_weights):
            for d in range(self.scale-1):
                _y = []
                with torch.no_grad():
                    _x = torch.cat((torch.cat(_ys, dim=1), x), dim=1) if len(_ys) > 0 else x
                    _y = self.inter_i[d](_x)
                _ys.append(_y)
                y += self.inter_o[d](_y)
            
            if (self.scale > 0):
                _x = torch.cat((torch.cat(_ys, dim=1), x), dim=1) if len(_ys) > 0 else x
                _y = self.inter_i[self.scale-1](_x)
                # _ys.append(_y)
                y += self.inter_o[self.scale-1](_y)
        else:
            for d in range(self.scale):
                # _y = []
                _x = torch.cat((torch.cat(_ys, dim=1), x), dim=1) if len(_ys) > 0 else x
                _y = self.inter_i[d](_x)
                _ys.append(_y)
                y += self.inter_o[d](_y)

        y = torch.nn.functional.sigmoid(y)
        y = torch.nn.functional.softmax(y, dim=1)

        return y