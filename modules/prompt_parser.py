from collections import namedtuple
import torch

import modules.shared as shared
from lark import Lark, Transformer, Visitor, v_args

grammar = r"""
start: prompt
prompt: (emphasized | scheduled | plain)*
emphasized: "(" prompt ")" -> emph_more
          | "(" prompt ":" NUMBER ")" -> emph_valued
          | "[" prompt "]" -> emph_less
scheduled: "[" (prompt ":")? prompt ":" NUMBER "]"
plain: /([^\\\[\]():]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
"""
parser = Lark(grammar, parser='lalr')

# a prompt like this: "fantasy landscape with a [mountain:lake:0.25] and [an oak:a christmas tree:0.75][ in foreground::0.6][ in background:0.25] [shoddy:masterful:0.5]"
# will be represented with prompt_schedule like this (assuming steps=100):
# [25, 'fantasy landscape with a mountain and an oak in foreground shoddy']
# [50, 'fantasy landscape with a lake and an oak in foreground in background shoddy']
# [60, 'fantasy landscape with a lake and an oak in foreground in background masterful']
# [75, 'fantasy landscape with a lake and an oak in background masterful']
# [100, 'fantasy landscape with a lake and a christmas tree in background masterful']

def collect_steps(steps, tree):
    l = [steps]
    class CollectSteps(Visitor):
        def scheduled(self, tree):
            swap_position = float(tree.children[-1])
            if swap_position < 1:
                swap_position *= steps
            tree.children[-1] = min(steps, int(swap_position))
            l.append(tree.children[-1])
    CollectSteps().visit(tree)
    return sorted(set(l))
def at_step(step, tree):
    class AtStep(Transformer):
        @v_args(tree=True)
        def prompt(self, tree):
            children = []
            for child in tree.children:
                if child.data != 'scheduled':
                    children.append(child)
                    continue
                p1 = child.children[0].children if len(child.children) == 3 else []
                p2, t = child.children[-2].children, child.children[-1]
                children.extend(p1 if step <= t else p2)
            tree.children = children
            return tree
    return AtStep().transform(tree)

def get_learned_conditioning_prompt_schedules(prompts, steps):
    res = []
    for prompt in prompts:
        tree = parser.parse(prompt)
        key_steps = collect_steps(steps, tree)
        res.append([(k, at_step(k, tree)) for k in key_steps])
    return res


ScheduledPromptConditioning = namedtuple("ScheduledPromptConditioning", ["end_at_step", "cond"])
ScheduledPromptBatch = namedtuple("ScheduledPromptBatch", ["shape", "schedules"])


def get_learned_conditioning(prompts, steps):

    res = []

    prompt_schedules = get_learned_conditioning_prompt_schedules(prompts, steps)
    cache = {}

    for prompt, prompt_schedule in zip(prompts, prompt_schedules):

        cached = cache.get(prompt, None)
        if cached is not None:
            res.append(cached)
            continue

        texts = [x[1] for x in prompt_schedule]
        conds = shared.sd_model.get_learned_conditioning(texts)

        cond_schedule = []
        for i, (end_at_step, text) in enumerate(prompt_schedule):
            cond_schedule.append(ScheduledPromptConditioning(end_at_step, conds[i]))

        cache[prompt] = cond_schedule
        res.append(cond_schedule)
    
    return ScheduledPromptBatch((len(prompts),) + res[0][0].cond.shape, res)


def reconstruct_cond_batch(c: ScheduledPromptBatch, current_step):
    res = torch.zeros(c.shape, device=shared.device, dtype=next(shared.sd_model.parameters()).dtype)
    for i, cond_schedule in enumerate(c.schedules):
        target_index = 0
        for curret_index, (end_at, cond) in enumerate(cond_schedule):
            if current_step <= end_at:
                target_index = curret_index
                break
        res[i] = cond_schedule[target_index].cond

    return res

def parse_prompt_attention(tree):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its assoicated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text

    Example:

        'a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).)'

    produces:

    [
        ['a ', 1.0],
        ['house', 1.5730000000000004],
        [' ', 1.1],
        ['on', 1.0],
        [' a ', 1.1],
        ['hill', 0.55],
        [', sun, ', 1.1],
        ['sky', 1.4641000000000006],
        ['.', 1.1]
    ]
    """
    class T(Transformer):
        def start(self, args):
            return args[0]
        def prompt(self, args):
            return sum(args, start=[])
        def emph_more(self, args):
            return [(t, w*1.1) for t, w in args[0]]
        def emph_less(self, args):
            return [(t, w/1.1) for t, w in args[0]]
        def emph_valued(self, args):
            return [(t, w*float(args[1])) for t, w in args[0]]
        def plain(self, args):
            return [(args[0].value, 1.)]
    res = T().transform(tree)
    return res if res else [("", 1.)]
