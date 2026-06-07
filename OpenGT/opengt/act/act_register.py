import torch.nn as nn
from torch_geometric.graphgym.register import register_act

# Register GELU activation (used by GPS and other GT models)
register_act('gelu', nn.GELU)
