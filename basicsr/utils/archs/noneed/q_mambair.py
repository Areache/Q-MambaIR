# import torch
# import time
# import math
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.multiprocessing as mp
# from torch.nn import Module, Parameter
# from torch.autograd import Function
# from .quant_modules import QuantConv2d, QuantLinear, QuantAct
# from basicsr.utils.registry import MODEL_REGISTRY

# # def q_mambair(model):
# #     net = Q_MambaIR(model)
# #     return net

# @MODEL_REGISTRY.register()
# class Q_MambaIR(Module):
#     def __init__(self, model):
#         super().__init__()
        
#         import pdb;pdb.set_trace()
#         self.in_proj = QuantLinear()
#         self.conv2d = QuantConv2d()
#         self.in_proj.set_param(
#             model.net_g.module.layers[0].residual_group.blocks[0].self_attention.in_proj
#         )
#         self.conv2d.set_param(
#             model.net_g.module.layers[0].residual_group.blocks[0].self_attention.conv2d
#         )
#         self.act = QuantAct()
#         self.out_proj = QuantLinear()
#         self.out_proj.set_param(
#             model.net_g.module.layers[0].residual_group.blocks[0].self_attention.out_proj
#         )

#     def forward(self, x):
        
#         x = self.quant_output(x)
