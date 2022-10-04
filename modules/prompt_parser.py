import re
from collections import namedtuple
from typing import List

import lark

# a prompt like this: "fantasy landscape with a [mountain:lake:0.25] and [an oak:a christmas tree:0.75][ in foreground::0.6][ in background:0.25] [shoddy:masterful:0.5]"
# will be represented with prompt_schedule like this (assuming steps=100):
# [25, 'fantasy landscape with a mountain and an oak in foreground shoddy']
# [50, 'fantasy landscape with a lake and an oak in foreground in background shoddy']
# [60, 'fantasy landscape with a lake and an oak in foreground in background masterful']
# [75, 'fantasy landscape with a lake and an oak in background masterful']
# [100, 'fantasy landscape with a lake and a christmas tree in background masterful']

schedule_parser = lark.Lark(r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | plain | WHITESPACE)*
!emphasized: "(" prompt ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt ":"] prompt ":" [WHITESPACE] NUMBER "]"
WHITESPACE: /\s+/
plain: /([^\\\[\]():]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
""")

def parse_prompt_schedules(prompts, steps):
    """
    >>> g = lambda p: parse_prompt_schedules([p], 10)[0]
    >>> g("test")
    [[10, 'test']]
    >>> g("a [b:3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [b: 3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [[[b]]:2]")
    [[2, 'a '], [10, 'a [[b]]']]
    >>> g("[(a:2):3]")
    [[3, ''], [10, '(a:2)']]
    >>> g("a [b : c : 1] d")
    [[1, 'a b  d'], [10, 'a  c  d']]
    >>> g("a[b:[c:d:2]:1]e")
    [[1, 'abe'], [2, 'ace'], [10, 'ade']]
    >>> g("a [unbalanced")
    [[10, 'a [unbalanced']]
    >>> g("a [b:.5] c")
    [[5, 'a  c'], [10, 'a b c']]
    >>> g("a [{b|d{:.5] c")  # not handling this right now
    [[5, 'a  c'], [10, 'a {b|d{ c']]
    >>> g("((a][:b:c [d:3]")
    [[3, '((a][:b:c '], [10, '((a][:b:c d']]
    """

    def collect_steps(steps, tree):
        l = [steps]
        class CollectSteps(lark.Visitor):
            def scheduled(self, tree):
                tree.children[-1] = float(tree.children[-1])
                if tree.children[-1] < 1:
                    tree.children[-1] *= steps
                tree.children[-1] = min(steps, int(tree.children[-1]))
                l.append(tree.children[-1])
        CollectSteps().visit(tree)
        return sorted(set(l))

    def at_step(step, tree):
        class AtStep(lark.Transformer):
            def scheduled(self, args):
                before, after, _, when = args
                yield before or () if step <= when else after
            def start(self, args):
                def flatten(x):
                    if type(x) == str:
                        yield x
                    else:
                        for gen in x:
                            yield from flatten(gen)
                return ''.join(flatten(args))
            def plain(self, args):
                yield args[0].value
            def __default__(self, data, children, meta):
                for child in children:
                    yield from child
        return AtStep().transform(tree)

    def get_schedule(prompt):
        try:
            tree = schedule_parser.parse(prompt)
        except lark.exceptions.LarkError as e:
            if 0:
                import traceback
                traceback.print_exc()
            return [[steps, prompt]]
        return [[t, at_step(t, tree)] for t in collect_steps(steps, tree)]

    promptdict = {prompt: get_schedule(prompt) for prompt in set(prompts)}
    return [promptdict[prompt] for prompt in prompts]


re_compound_tokenizer = re.compile(r'''
\\\\|\\(|\\)|(?P<e>\(|\)|\bPLUS\b|\bAND\b|\bNOT\b)|
:\s*(?P<w>[+-]?(?:\d+\.\d*|\.\d+|\d+))
|\w+|.
''', re.VERBOSE)

def parse_compound(prompt):
    """
    Parse a prompt that may include AND, NOT and :weight arguments
    into a list of lists of weighted subprompts that will be evaluated
    for it.

    Used to implement composition, as described in https://arxiv.org/abs/2206.01714
    Compositional Visual Generation with Composable Diffusion Models by Liu et al.

    >>> p = parse_compound
    >>> p('prompt (no compound:.5)')
    [[[('prompt (no compound:.5)', 1.0)], 1.0]]
    >>> p('a AND b')
    [[[('a', 1.0)], 0.5], [[('b', 1.0)], 0.5]]
    >>> p('a:3 AND (b:2) :+1.4')
    [[[('a', 1.0)], 1.5], [[('(b:2)', 1.0)], 0.7]]
    >>> p('pre (a AND b  ) aft')
    [[[('pre a aft', 1.0)], 0.5], [[('pre b aft', 1.0)], 0.5]]
    >>> p('a (NOT b) c')
    [[[('a b c', 1.0)], -0.5]]
    >>> p('a ( NOT b) c')
    [[[('a  c', 1.0)], 1.0], [[('a b c', 1.0)], -0.5]]
    >>> p('a:3:2 PLUS b:3')
    [[[('a:3', 0.4), ('b', 0.6)], 1.0]]
    >>> p('a:3 PLUS b:4 AND c:4 PLUS d:6 :10')
    [[[('a', 0.75), ('b', 0.25)], 2.0], [[('c', 0.4), ('d', 0.6)], 5.0]]
    >>> p('a (b AND c) e (f AND g)')
    [[[('a b e f', 1.0)], 0.5], [[('a c e g', 1.0)], 0.5]]
    >>> p('a (b AND c) (NOT d)')
    [[[('a b d', 1.0)], -0.5], [[('a c ', 1.0)], 1.0]]
    >>> p('a (b AND c) ( NOT d)')
    [[[('a b ', 1.0)], 1.0], [[('a c d', 1.0)], -0.5]]
    >>> p('a (b AND c) (d NOT e NOT f)')
    [[[('a b d', 1.0)], 1.0], [[('a c e', 1.0)], -0.25], [[('a b f', 1.0)], -0.25]]
    >>> p('(a:4 AND b:3)') # degenerate cases
    [[[('a', 1.0)], 2.0], [[('b', 1.0)], 1.5]]
    >>> p('(a:4 AND b:3)(NOT a)')
    [[[('aa', 1.0)], -2.0], [[('b', 1.0)], 3.0]]
    >>> p('(a:4 AND b:3)(NOT a:6)')
    [[[('aa', 1.0)], -2.0], [[('b', 1.0)], 3.0]]
    >>> p('a AND (b AND c)') # ignore recursion
    [[[('a', 1.0)], 0.5], [[('(b AND c)', 1.0)], 0.5]]
    """

    toks = []
    last_normal = False
    for m in re_compound_tokenizer.finditer(prompt):
        t = m.group()
        v = None
        if m['w']:
            v = float(m['w'])
        normal = not m['e'] and not m['w']
        if normal and last_normal:
            # coalesce runs of non-syntactic tokens
            toks[-1][0] += t
        else:
            toks.append([t, v])
        last_normal = normal

    # compute spans for matching parens
    openpars = []
    pars = []
    for n, (t, v) in enumerate(toks):
        if t == '(':
            openpars.append(n)
        elif t == ')':
            if openpars:
                pars.append((openpars.pop(), n))

    def findpars(i):
        try:
            return min((b-a, a+1, b) for a, b in pars if a < i < b)[1:]
        except ValueError:
            return (0, len(toks))

    def handle_plus(ts):
        out = []
        start = 0
        for n, (t, v) in enumerate(ts):
            if t in ('PLUS'):
                out.append([ts[start:n], None])
                start = n + 1
        out.append([ts[start:], None])
        return out

    def handle_andnot(ts):
        out = []
        start = 0
        is_not = False
        for n, (t, v) in enumerate(ts):
            if t in ('NOT', 'AND'):
                if n:
                    out.append([handle_plus(ts[start:n]), is_not])
                start = n + 1
                is_not = t == 'NOT'
        out.append([handle_plus(ts[start:]), is_not])
        return out

    def set_weights(subs):
        # attempt to pop a weight from the end of the token stream for each element
        for andc in subs:
            peek = 1.0
            if len(subs) > 1:
                # weights bind tightest to the ends of AND/NOT clauses
                try:
                    while not andc[0][-1][0][-1][0].strip():
                        andc[0][-1][0].pop()
                    peek = float(andc[0][-1][0][-1][1])  # lol
                    andc[0][-1][0].pop()
                except (TypeError, IndexError):
                    pass
            andc[1] = peek * (-1 if andc[1] else 1)
            for plusc in andc[0]:
                peek = 1.0
                try:
                    while not plusc[0][-1][0].strip():
                        plusc[0].pop()
                    peek = float(plusc[0][-1][1])
                    plusc[0].pop()
                except (TypeError, IndexError):
                    pass
                plusc[0] = ''.join(t for t, v in plusc[0]).strip()
                plusc[1] = peek

    compound_chunks = []

    def fuse_deeper(ts):
        depth = 0
        acc = ''
        for t, v in ts:
            if t == '(':
                depth += 1
            elif t == ')':
                acc += t
                depth -= 1
                if depth == 0:
                    yield acc, None
                    acc = ''
                    continue
            if depth:
                acc += t
            else:
                yield (t, v)

    n = 0
    while n < len(toks):
        t, v = toks[n]

        if t in ('AND', 'NOT', 'PLUS'):
            a, b = findpars(n)
            subs = handle_andnot(list(fuse_deeper(toks[a:b])))
            set_weights(subs)

            # print("?", t, [a, b, before, after], subs)
            compound_chunks.append((a, b, subs))
            n = b
        else:
            n += 1

    last_end = 0
    acc = [[[['', 1.0]], 1.0]]

    def accumulate(subs):
        if isinstance(subs, str):
            for andc in acc:
                for plusc in andc[0]:
                    plusc[0] += subs
            return
        for i, (andc, andw) in enumerate(subs):
            if i >= len(acc):
                # duplicate the first element if appending longer arrays
                acc.append([[list(acc[0][0][0])], 1.0])
            for j, (plusc, plusw) in enumerate(andc):
                if j >= len(acc[i][0]):
                    acc[i][0].append(list(acc[i][0][0]))
        for i, (andc, andw) in enumerate(subs):
            for j, (plusc, plusw) in enumerate(andc):
                acc[i][0][j][0] += plusc
                if acc[i][0][j][1] == 1.0:
                    acc[i][0][j][1] = plusw  # last non-default weight overrides
            if acc[i][1] == 1.0:
                acc[i][1] = andw
            elif andw == -1.0 and acc[i][1] > 0:
                acc[i][1] *= -1

    for a, b, subs in compound_chunks:
        if a > last_end:
            accumulate(''.join(t for t, v in toks[last_end:a-1]))
        accumulate(subs)
        last_end = b + 1
    accumulate(''.join(t for t, v in toks[last_end:]))


    # PLUS normalization: divide by sum(abs(x))
    for andc in acc:
        plussum = sum(abs(w) for _, w in andc[0]) or 1
        andc[0] = [(phrase, weight / plussum)
                    for phrase, weight in andc[0]]

    # AND/NOT normalization: divide by number of terms with same sign
    # from the paper: NOT should be weaker to avoid erasing common style terms
    negation_factor = 2
    poscount = sum(1 for _, w in acc if w >= 0)
    negcount = sum(1 for _, w in acc if w < 0) * negation_factor
    for andc in acc:
        if andc[1] >= 0:
            andc[1] /= poscount
        else:
            andc[1] /= negcount

    return acc

def parse_schedules_and_compounds(prompts, steps, should_parse_compounds=False):
    schedules = parse_prompt_schedules(prompts, steps)

    cache = {}
    ret = []
    for prompt_schedule in schedules:
        parsed = []
        for end_at_step, text in prompt_schedule:
            if text not in cache:
                if should_parse_compounds:
                    compound = parse_compound(text)
                else:
                    compound = [([(text, 1.0)], 1.0)]
                cache[text] = compound
            parsed.append((end_at_step, cache[text]))
        ret.append(parsed)
    return ret

def dump_parsed_schedule(prompt, steps, schedule):
    print(f"Prompt equation for {prompt!r} over {steps} steps:")
    for end_at_step, parsed in schedule:
        print(f"to {end_at_step:-3}:", end=' ')
        spacer = ' ' * 8
        for i, (promptweights, weight) in enumerate(parsed):
            if len(promptweights) == 1:
                print(f'{spacer if i > 0 else ""}{weight:1.2f} x {promptweights[0][0]}')
            else:
                print(f'{spacer}{weight:1.2f} x ({promptweights[0][1]:.2f} x {promptweights[0][0]}')
                for n, (prompt, weight) in enumerate(promptweights[1:], 2):
                    print(f'{spacer}      + {weight:.2f} x {prompt}{")" if n == len(promptweights) else ""}')


ScheduledPromptConditioning = namedtuple("ScheduledPromptConditioning", ["end_at_step", "cond"])
Schedules = List[List[ScheduledPromptConditioning]]
CompoundPrompt = namedtuple('CompoundPrompt', ["clauses", "weights"])

def get_conditioning_for_weighted_subprompts(model, subprompts):
    if len(subprompts) <= 1:
        c = model.get_learned_conditioning([subprompts[0][0]])
    else:
        c = None
        for subtext, subweight in subprompts:
            if c is None:
                c = model.get_learned_conditioning([subtext])
                c *= subweight
            else:
                c.add_(model.get_learned_conditioning([subtext]), alpha=subweight)
    return c[0]

def get_conditioning_for_compound(model, parsed):
    subprompt_count = sum(len(x[0]) for x in parsed)

    conds = [get_conditioning_for_weighted_subprompts(model, subprompts) for subprompts, _ in parsed]

    nextparam = next(model.parameters())
    weights = torch.Tensor([w for _, w in parsed]).type(nextparam.dtype).to(nextparam.device)

    return CompoundPrompt(torch.cat(conds), weights)

def get_learned_conditioning(model, prompts, steps, should_parse_compounds=True) -> Schedules:
    res = []
    cache = {}

    subprompt_schedules = parse_schedules_and_compounds(prompts, steps, should_parse_compounds)

    for prompt, prompt_schedule in zip(prompts, subprompt_schedules):
        if prompt not in cache:
            should_print = sum(sum(len(subprompts) for subprompts, _ in parsed) for _, parsed in prompt_schedule) > 1
            if should_print:
                if len(prompts) > 1 and max(len(parsed) for _, parsed in prompt_schedule) > 1:
                    print("BUG: prompt composition batches produce incorrect results")
                dump_parsed_schedule(prompt, steps, prompt_schedule)
            cond_schedule = []
            for end_at_step, parsed in prompt_schedule:
                cond = get_conditioning_for_compound(model, parsed)
                cond_schedule.append(ScheduledPromptConditioning(end_at_step, cond))
            cache[prompt] = cond_schedule
        res.append(cache[prompt])

    return res

def reconstruct_cond_batch(c: Schedules, current_step):
    res = None
    weights = None

    for i, cond_schedule in enumerate(c):
        target_index = 0
        for current_index, (end_at, cond) in enumerate(cond_schedule):
            if current_step <= end_at:
                target_index = current_index
                break
        cond = cond_schedule[target_index].cond
        if res is None:
            shape = (len(c),) + cond.clauses.shape
            res = torch.zeros(shape, device=cond.weights.device, dtype=cond.weights.dtype)
            weights = torch.zeros((len(c),) + cond.weights.shape, device=res.device, dtype=res.dtype)
        res[i] = cond.clauses
        weights[i] = cond.weights

    return res, weights


re_attention = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)


def parse_prompt_attention(text):
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

    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith('\\'):
            res.append([text[1:], 1.0])
        elif text == '(':
            round_brackets.append(len(res))
        elif text == '[':
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ')' and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == ']' and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)

    if 0:  # compound fuzztest
        import random

        choices = 'a b c d ( ) AND NOT PLUS :2 :3 :4 1'.split() + [' ', ' ']
        for _ in range(10000):
            c = random.randint(0, 100)
            p = ' '.join(random.choice(choices) for _ in range(c))
            o = parse_compound(p)
            dump_parsed_schedule(p, 1, [(1, o)])
    if 0:  # schedule fuzztest
        import random

        choices = 'a b ( ) c d [ ] : : :2 :3 :4 1 .1 .3 .8 [a:3]'.split() + [' ', ' ']
        for _ in range(10000):
            c = random.randint(0, 100)
            p = ''.join(random.choice(choices) for _ in range(c))
            o = parse_prompt_schedules([p], 100)
            print(o)
else:
    import torch  # doctest faster
