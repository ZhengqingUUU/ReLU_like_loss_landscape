"""model.py"""

import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


class two_layer_net(nn.Module):

    def __init__(self, input_size ,hidden_size, output_size, beta_1, beta_2, random_seed = None, alpha = 1, balancedness= True, activation = "relu") -> None:
        """_summary_

        Args:
            input_size (_type_): _description_
            hidden_size (_type_): _description_
            output_size (_type_): _description_
            beta_1 (_type_): standard deviation of the output layer weight initialization
            beta_2 (_type_): standard deviation of the input layer weight initialization
            alpha (int, optional): _description_. Defaults to 1.
            activation (str, optional): _description_. Defaults to "relu".

        Raises:
            NotImplementedError: _description_
        """
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size, bias = False)
        self.width = hidden_size
        if activation.lower() == "relu":
            self.activation = nn.ReLU(True)
        else:
            raise NotImplementedError
        self.output_layer = nn.Linear(hidden_size, output_size, bias = False)
        self.alpha = alpha
        self.weight_init(beta_1, beta_2, balancedness,random_seed = random_seed)
    
    def forward(self, x_1 ):
        s_2 = self.input_layer(x_1)
        x_2 = self.activation(s_2)
        o = self.output_layer(x_2)
        return (1/self.alpha)*o
    
    def weight_init(self, beta_1, beta_2, balancedness, random_seed = None):
        print()
        if random_seed != None:
            torch.manual_seed(random_seed)
        for name, layer in dict(self.named_modules()).items():
            if name == 'input_layer':
                layer.weight.data.normal_(0) ## default std to be 1
                layer.weight.data = layer.weight.data*beta_2
                input_weight_norm = torch.linalg.norm(layer.weight.data,dim=1) # just register the norm of the input weight in case we need it!
            elif name == 'output_layer':
                layer.weight.data.normal_(0)## default std to be 1
                if not balancedness:
                    layer.weight.data = layer.weight.data*beta_1
                else:
                    print("balancedness is imposed!")
                    output_weight_sign = torch.sgn(layer.weight.data)
                    output_weight_norm = input_weight_norm.reshape(output_weight_sign.shape)
                    layer.weight.data = output_weight_sign*output_weight_norm