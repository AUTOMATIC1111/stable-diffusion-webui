import itertools

OPEN = '{'
CLOSE = '}'
SEPARATE = '|'
MARK = '@'
REAL_MARK = ':'

def combine(left, right):
    return map(lambda p: (p[0][0] + p[1][0], p[0][1] * p[1][1]), itertools.product(left, right))


def get_weighted_prompt(prompt_weight):
    (prompt, full_weight) = prompt_weight
    results = [('', full_weight)]
    alts = []
    start = 0
    mark = -1
    open_count = 0
    first_open = 0
    nested = False

    for i, c in enumerate(prompt):
        add_alt = False
        do_combine = False
        if c == OPEN:
            open_count += 1
            if open_count == 1:
                first_open = i
                results = list(combine(results, [(prompt[start:i], 1)]))
                start = i + 1
            else:
                nested = True

        if c == MARK and open_count == 1:
            mark = i

        if c == SEPARATE and open_count == 1:
            add_alt = True

        if c == CLOSE:
            open_count -= 1
            if open_count == 0:
                add_alt = True
                do_combine = True
        if i == len(prompt) - 1 and open_count > 0:
            add_alt = True
            do_combine = True

        if add_alt:
            end = i
            weight = 1
            if mark != -1:
                weight_str = prompt[mark + 1:i]
                try:
                    weight = float(weight_str)
                    end = mark
                except ValueError:
                    print("warning, not a number:", weight_str)



            alt = (prompt[start:end], weight)
            alts += get_weighted_prompt(alt) if nested else [alt]
            nested = False
            mark = -1
            start = i + 1

        if do_combine:
            if len(alts) <= 1:
                alts = [(prompt[first_open:i + 1], 1)]

            results = list(combine(results, alts))
            alts = []

    # rest of the prompt
    results = list(combine(results, [(prompt[start:], 1)]))
    weight_sum = sum(map(lambda r: r[1], results))
    results = list(map(lambda p: (p[0], p[1] / weight_sum * full_weight), results))

    return results


def switch_syntax(prompt):
    p = list(prompt)
    stack = []
    for i, c in enumerate(p):
        if c == '{' or c == '[' or c == '(':
            stack.append(c)

        if c == '}' or c == ']' or c == ')':
            stack.pop()

        if c == REAL_MARK and stack[-1] == '{':
            p[i] = MARK

    return "".join(p)

# def test(p, w=1):
#     print('')
#     print(p)
#     result = get_weighted_prompt((p, w))
#     print(result)
#     print(sum(map(lambda x: x[1], result)))
#
#
# test("fantasy landscape")
# test("fantasy {landscape|city}, dark")
# test("fantasy {landscape|city}, {fire|ice} ")
# test("fantasy {landscape|city}, {fire|ice}, {dark|light} ")
# test("fantasy landscape, {{fire|lava}|ice}")
# test("fantasy landscape, {{fire@4|lava@1}|ice@2}")
# test("fantasy landscape, {{fire@error|lava@1}|ice@2}")
# test("fantasy landscape, {{fire|lava}|ice@2")
# test("fantasy landscape, {fire|lava} {cool} {ice,water}")
# test("fantasy landscape, {fire|lava} {cool} {ice,water")
# test("{lava|ice|water@5}")
# test("{fire@4|lava@1}", 5)
# test("{{fire@4|lava@1}|ice@2|water@5}")
# test("{fire|lava@3.5}")
