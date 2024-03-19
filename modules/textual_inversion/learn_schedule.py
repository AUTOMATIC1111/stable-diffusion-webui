import tqdm


class LearnScheduleIterator:
    def __init__(self, learn_rate, max_steps, cur_step=0):
        """
        specify learn_rate as "0.001:100, 0.00001:1000, 1e-5:10000" to have lr of 0.001 until step 100, 0.00001 until 1000, and 1e-5 until 10000
        """

        pairs = learn_rate.split(',')
        self.rates = []
        self.it = 0
        self.maxit = 0
        try:
            for pair in pairs:
                if not pair.strip():
                    continue
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
            assert self.rates
        except (ValueError, AssertionError) as e:
            raise Exception('Invalid learning rate schedule. It should be a number or, for example, like "0.001:100, 0.00001:1000, 1e-5:10000" to have lr of 0.001 until step 100, 0.00001 until 1000, and 1e-5 until 10000.') from e


    def __iter__(self):
        return self

    def __next__(self):
        if self.it < self.maxit:
            self.it += 1
            return self.rates[self.it - 1]
        else:
            raise StopIteration


class LearnRateScheduler:
    def __init__(self, learn_rate, max_steps, cur_step=0, verbose=True):
        self.schedules = LearnScheduleIterator(learn_rate, max_steps, cur_step)
        (self.learn_rate,  self.end_step) = next(self.schedules)
        self.verbose = verbose

        if self.verbose:
            print(f'Training at rate of {self.learn_rate} until step {self.end_step}')

        self.finished = False

    def step(self, step_number):
        if step_number < self.end_step:
            return False

        try:
            (self.learn_rate, self.end_step) = next(self.schedules)
        except StopIteration:
            self.finished = True
            return False
        return True

    def apply(self, optimizer, step_number):
        if not self.step(step_number):
            return

        if self.verbose:
            tqdm.tqdm.write(f'Training at rate of {self.learn_rate} until step {self.end_step}')

        for pg in optimizer.param_groups:
            pg['lr'] = self.learn_rate

