import itertools

OPEN = '{'
CLOSE = '}'
SEPARATE = '|'
MARK = '@'


def combine(left, right):
    return map(lambda p: (p[0][0] + p[1][0], p[0][1] * p[1][1]), itertools.product(left, right))


def get_weighted_prompt(prompt_weight):
    (prompt, weight) = prompt_weight
    results = [('', weight)]
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
                if weight_str.isdecimal():
                    weight = int(weight_str)
                    end = mark

            alt = (prompt[start:end], weight)
            alts += get_weighted_prompt(alt) if nested else [alt]
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
    results = list(map(lambda p: (p[0], p[1] / weight_sum), results))

    return results


# def test(p):
#     print('')
#     print(p)
#     print(get_weighted_prompt((p, 1)))

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
