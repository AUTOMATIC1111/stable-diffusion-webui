from .basic_train import *
from .callback import *
from .core import *
from .basic_data import *
from .data_block import *
from .layers import *
from .metrics import *
from .torch_core import *
from .train import *
from .datasets import *
from .version import *
from . import callbacks

"""
from . import core,torch_core,basic_data,basic_train,callback,data_block,layers,metrics,train,datasets,callbacks

__all__  = [o for o in dir(core) if not o.startswith('_')]
__all__ += [o for o in dir(torch_core) if not o.startswith('_')]
__all__ += [*basic_train.__all__, *callback.__all__, 'core', 'torch_core', 'callbacks',
           *basic_data.__all__, *data_block.__all__, *layers.__all__, *metrics.__all__,
           *train.__all__, *datasets.__all__, '__version__']
"""

try: from .gen_doc.nbdoc import doc
except: pass  # Optional if jupyter is present
    #__all__.append('doc')

__all__ = [o for o in dir(sys.modules[__name__]) if not o.startswith('_')] + ['__version__']

