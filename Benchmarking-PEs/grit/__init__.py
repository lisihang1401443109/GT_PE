from .act import * # noqa
from .config import * # noqa
from .encoder import * # noqa
from .head import * # noqa
from .layer import * # noqa
from .loader import * # noqa
from .loss import * # noqa
from .network import * # noqa
from .optimizer import * # noqa
from .pooling import * # noqa
from .stage import * # noqa
from .train import * # noqa
from .transform import * # noqa
import torch_geometric.graphgym.register as register
if 'default' not in register.head_dict:
    if 'inductive_node' in register.head_dict:
        register.head_dict['default'] = register.head_dict['inductive_node']

# Register GraphGym modules
from torch_geometric.graphgym.register import (register_loader,
                                                register_head,
                                                register_node_encoder,
                                                register_edge_encoder,
                                                register_layer,
                                                register_pooling,
                                                register_loss,
                                                register_optimizer,
                                                register_scheduler,
                                                register_stage)

from grit.loader.master_loader import load_dataset_master
register_loader('master_loader', load_dataset_master)


from grit.encoder.gnn_encoder import GNNNodeEncoder
register_node_encoder('GNNNodeEncoder', GNNNodeEncoder)