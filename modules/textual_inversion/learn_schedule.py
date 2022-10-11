
class LearnSchedule:
    def __init__(self, learn_rate, max_steps, cur_step=0):
        pairs = learn_rate.split(',')
        self.rates = []
        self.it = 0
        self.maxit = 0
        for i, pair in enumerate(pairs):
            tmp = pair.split(':')
            if len(tmp) == 2:
                step = int(tmp[1])
                if step > cur_step:
                    self.rates.append((float(tmp[0]), min(step, max_steps)))
                    self.maxit += 1
                    if step > max_steps:
                        return
                elif step == -1:
                    self.rates.append((float(tmp[0]), max_steps))
                    self.maxit += 1
                    return
            else:
                self.rates.append((float(tmp[0]), max_steps))
                self.maxit += 1
                return

    def __iter__(self):
        return self

    def __next__(self):
        if self.it < self.maxit:
            self.it += 1
            return self.rates[self.it - 1]
        else:
            raise StopIteration
