


lazy_load : bool = False #when this is enabled, HNs will be loaded when required.
available_opts : dict[str,HyperNetwork]= {}

# Behavior definition.
# [[], [], []] -> sequential processing
# [{A : 0.8, B : 0.1}] -> parallel processing. with weighted sum in this case, A = 8/9 effect, B = 1/9 effect.
# [(A, 0.2), (B, 0.4)] -> tuple is used to specify strength.
# [{A, B, C}] -> parallel, but having same effects (set)
# [A, B, []] -> sequential processing
# [{A:0.6}, B, C] -> sequential, dict with single value will be considered as strength modification.
# [[A], {B}, C] -> singletons are equal to items without covers

class HyperNetworkProcessing:
    def __init__(self, sequence:[str|list|dict|set]):
        # parse from sequence.
        # calculate weighted sum if parallel.
        # apply weighted forward if weight is specified.

class SingularForward:
    key = None
    processor = None
    strength:int|float = 1
    def __init__(self, sequence:[str|list|dict|set|tuple]):
        if type(sequence) in (list, dict, set):
            assert len(sequence) == 1, "SingularForward only accepts singletons!"
            key = list(sequence)[0]
            if type(sequence) is dict:
                self.strength = sequence[key]
        elif type(sequence) is tuple:
            assert len(sequence) == 2, "Tuple with non-couple encountered in SingularForward!"
            key = sequence[0]
            self.strength = sequence[1]
        else:
            key = sequence
        self.processor = key
        assert 0 <= self.strength <=1 , "Strength must be between 0 and 1!"

    def forward(self, context):
        if self.processor in available_opts:
            return available_opts[self.processor].forward_strength(context, self.strength) #define forward_strength, which invokes HNModule with specified strength.
        # Note : we share same HN if it is called multiple time, which means you might not be able to train it via this structure.
        raise KeyError(f"Key {self.processor} is not found in cached Hypernetworks!")

