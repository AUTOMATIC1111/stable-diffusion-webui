import ast
import os.path

import torch

from modules import devices, shared

lazy_load= False #when this is enabled, HNs will be loaded when required.



class DynamicDict(dict): # Brief dict that dynamically unloads Hypernetworks if required.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current = None
        self.hash = None
        self.dict = {**kwargs}

    def prepare(self, key, value):
        if lazy_load and self.current is not None and (key != self.current): # or filename is identical, but somehow hash is changed?
            self.current.to('cpu')
        self.current = value
        if self.current is not None:
            self.current.to(devices.device)

    def __getitem__(self, item):
        value = self.dict[item]
        self.prepare(item, value)
        return value

    def __setitem__(self, key, value):
        if key in self.dict:
            return
        self.dict[key] = value

    def __contains__(self, item):
        return item in self.dict


available_opts= DynamicDict()  # string -> HN itself.



# Behavior definition.
# [[], [], []] -> sequential processing
# [{"A" : 0.8, "B" : 0.1}] -> parallel processing. with weighted sum in this case, A = 8/9 effect, B = 1/9 effect.
# [("A", 0.2), ("B", 0.4)] -> tuple is used to specify strength.
# [{"A", "B", "C"}] -> parallel, but having same effects (set)
# ["A", "B", []] -> sequential processing
# [{"A":0.6}, "B", "C"] -> sequential, dict with single value will be considered as strength modification.
# [["A"], {"B"}, "C"] -> singletons are equal to items without covers, nested singleton will not be parsed, because its inefficient.
# {{'Aa' : 0.2, 'Ab' : 0.8} : 0.8, 'B' : 0.1} (X) -> {"{'Aa' : 0.2, 'Ab' : 0.8}" : 0.8, 'B' : 0.1} (O), When you want complex setups in parallel, you need to cover them with "". You can use backslash too.


# Testing parsing function.

def test_parsing(string = None):

    def test(arg):
        print(arg)
        try:
            obj = str(Forward.parse(arg))
            print(obj)
        except Exception as e:
            print(e)
    if string:
        test(string)
    else:
        for strings in ["[[], [], []]", "[{\"A\" : 0.8, \"B\" : 0.1}]", '[("A", 0.2), ("B", 0.4)]', '[{"A", "B", "C"}]', '[{"A":0.6}, "B", "C"]', '[["A"], {"B"}, "C"]', '{"{\'Aa\' : 0.2, \'Ab\' : 0.8}" : 0.8, \'B\' : 0.1}']:
            test(strings)


class Forward:
    def __init__(self, **kwargs):
        self.name = "defaultForward" if 'name' not in kwargs else kwargs['name']
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, context_k, context_v = None, layer = None):
        raise NotImplementedError

    @staticmethod
    def parse(arg, name=None):
        arg = Forward.unpack(arg)
        arg = Forward.eval(arg)
        if Forward.isSingleTon(arg):
            return SingularForward(*Forward.parseSingleTon(arg))
        elif Forward.isParallel(arg):
            return ParallelForward(Forward.parseParallel(arg), name=name)
        elif Forward.isSequential(arg):
            return SequentialForward(Forward.parseSequential(arg), name=name)
        raise ValueError(f"Cannot parse {arg} into sequences!")

    @staticmethod
    def unpack(arg):  # stop using ({({{((a))}})}) please
        if len(arg) == 1 and type(arg) in (set, list, tuple):
            return Forward.unpack(list(arg)[0])
        if len(arg) == 1 and type(arg) is dict:
            key = list(arg.keys())[0]
            if arg[key] == 1:
                return Forward.unpack(key)
        return arg

    @staticmethod
    def eval(arg): # from "{something}", parse as etc form.
        if arg is None:
            raise ValueError("None cannot be evaluated!")
        try:
            newarg = ast.literal_eval(arg)
            if type(arg) is str and arg.startswith(("{", "[", "(")) and newarg is not None:
                if not newarg:
                    raise RuntimeError(f"Cannot eval false object {arg}!")
                return newarg
        except ValueError:
            return arg
        return arg

    @staticmethod
    def isSingleTon(arg): # Very strict. This applies strength to HN, which cannot happen in combined networks. Only weighting is allowed in complex process.
        if type(arg) is str and not arg.startswith(('[','(','{')): #Strict. only accept str
            return True
        elif type(arg) is dict: #Strict. only accept {str : int/float} - Strength modification can only happen for str.
            return len(arg) == 1 and all(type(value) in (int, float) for value in arg.values()) and all(type(k) is str for k in arg)
        elif type(arg) in (list, set):
            return len(arg) == 1 and all(type(x) is str for x in arg)
        elif type(arg) is tuple:
            return len(arg) == 2 and type(arg[0]) is str and type(arg[1]) in (int, float)
        return False

    @staticmethod
    def parseSingleTon(sequence): #accepts sequence, returns str, float pair. This is Strict.
        if type(sequence) in (list, dict, set):
            assert len(sequence) == 1, f"SingularForward only accepts singletons, but given {sequence}!"
            key = list(sequence)[0]
            if type(sequence) is dict:
                assert type(key) is str, f"Strength modification only accepts single Hypernetwork, but given {key}!"
                return key, sequence[key]
            else:
                key = list(key)[0]
                return key, 1
        elif type(sequence) is tuple:
            assert len(sequence) == 2, f"Tuple with non-couple {sequence} encountered in SingularForward!"
            assert type(sequence[0]) is str, f"Strength modification only accepts single Hypernetwork, but given {sequence[0]}!"
            assert type(sequence[1]) in (int, float), f"Strength tuple only accepts Numbers, but given {sequence[1]}!"
            return sequence[0], sequence[1]
        else:
            assert type(sequence) is str, f"Strength modification only accepts single Hypernetwork, but given {sequence}!"
            return sequence, 1


    @staticmethod
    def isParallel(arg): # Parallel, or Sequential processing is not strict, it can have {"String covered sequence or just HN String" : weight, ...
        if type(arg) in (dict, set) and len(arg) > 1:
            if type(arg) is set:
                return all(type(key) is str for key in arg), f"All keys should be Hypernetwork Name/Sequence for Set but given :{arg}"
            else:
                arg : dict
                return all(type(key) is str for key in arg.keys()), f"All keys should be Hypernetwork Name/Sequence for Set but given :{arg}"
        else:
            return False

    @staticmethod
    def parseParallel(sequence): # accepts sequence, returns {"Name or sequence" : weight...}
        assert len(sequence) > 1, f"Length of sequence {sequence} was not enough for parallel!"
        if type(sequence) is set: # only allows hashable types. otherwise it should be supplied as string cover
            assert all(type(key) in (str, tuple) for key in sequence), f"All keys should be Hypernetwork Name/Sequence for Set but given :{sequence}"
            return {key: 1/len(sequence) for key in sequence}
        elif type(sequence) is dict:
            assert all(type(key) in (str, tuple) for key in sequence.keys()), f"All keys should be Hypernetwork Name/Sequence for Dict but given :{sequence}"
            assert all(type(value) in (int, float) for value in sequence.values()), f"All values should be int/float for Dict but given :{sequence}"
            return sequence
        else:
            raise ValueError(f"Cannot parse parallel sequence {sequence}!")

    @staticmethod
    def isSequential(arg):
        if type(arg) is list and len(arg)>0:
            return True
        return False

    @staticmethod
    def parseSequential(sequence):  # accepts sequence, only checks if its list, then returns sequence.
        if type(sequence) is list and len(sequence)>0:
            return sequence
        else:
            raise ValueError(f"Cannot parse non-list sequence {sequence}!")

from modules.hypernetworks.hypernetwork import Hypernetwork


def find_non_hash_key(target):
    closest = [x for x in shared.hypernetworks if x.rsplit('(', 1)[0] == target or x == target]
    if closest:
        return shared.hypernetworks[closest[0]]
    raise KeyError(f"{target} is not found in Hypernetworks!")
class SingularForward(Forward):

    def __init__(self, processor, strength):
        self.name = processor
        self.processor = processor
        self.strength = strength
        super(SingularForward, self).__init__()
        # parse. We expect parsing Singletons or (k,v) pair here, which is HN Name and Strength.
        available_opts[self.processor] = Hypernetwork()
        available_opts[self.processor].load(find_non_hash_key(self.processor))
        # assert self.processor in available_opts, f"Hypernetwork named {processor} is not ready!"
        assert 0 <= self.strength <=1 , "Strength must be between 0 and 1!"

    def forward(self, context_k, context_v = None, layer=None):
        if self.processor in available_opts:
            context_layers = available_opts[self.processor].layers.get(context_k.shape[2], None)
            if context_layers is None:
                return context_k, context_k
            if context_v is None:
                context_v = context_k
            if layer is not None and hasattr(layer, 'hyper_k') and hasattr(layer, 'hyper_v'):
                layer.hyper_v = context_layers[0], layer.hyper_k = context_layers[1]
            return context_layers[0].forward_strength(context_k, self.strength) , context_layers[1].forward_strength(context_v, self.strength) #define forward_strength, which invokes HNModule with specified strength.
        # Note : we share same HN if it is called multiple time, which means you might not be able to train it via this structure.
        raise KeyError(f"Key {self.processor} is not found in cached Hypernetworks!")

    def __str__(self):
        return "SingularForward>" + str(self.processor)


class ParallelForward(Forward):

    def __init__(self, sequence, name=None):
        self.name = "ParallelForwardHypernet" if name is None else name
        self.callers= {}
        self.weights= {}
        super(ParallelForward, self).__init__()
        # parse
        for keys in sequence:
            self.callers[keys] = Forward.parse(keys)
            self.weights[keys] = sequence[keys]

    def forward(self, context, context_v = None, layer = None):
        ctx_k, ctx_v = torch.zeros_like(context, device = context.device), torch.zeros_like(context, device = context.device)
        for key in self.callers:
            k, v = self.callers[key].forward(context, context_v, layer=layer)
            ctx_k += k * self.weights[key]
            ctx_v += v * self.weights[key]
        return ctx_k, ctx_v

    def __str__(self):
        return "ParallelForward>" +str({str(k): str(v) for (k,v) in self.callers.items()})


class SequentialForward(Forward):
    def __init__(self, sequence, name=None):
        self.name = "SequentialForwardHypernet" if name is None else name
        self.callers = []
        super(SequentialForward, self).__init__()
        for keys in sequence:
            self.callers.append(Forward.parse(keys))

    def forward(self, context, context_v = None, layer=None):
        if context_v is None:
            context_v = context
        for keys in self.callers:
            context, context_v = keys(context, context_v, layer=layer)
        return context, context_v

    def __str__(self):
        return "SequentialForward>" + str([str(x) for x in self.callers])


class EmptyForward(Forward):
    def __init__(self):
        super().__init__()
        self.name = None

    def forward(self, context, context_v=None, layer=None):
        if context_v is None:
            context_v = context
        return context, context_v

    def __str__(self):
        return "EmptyForward"


def load(filename):
    with open(filename, 'r') as file:
        return Forward.parse(file.read(), name=os.path.basename(filename))

