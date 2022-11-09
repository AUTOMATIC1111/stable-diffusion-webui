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
            for i, pair in enumerate(pairs):
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
        except (ValueError, AssertionError):
            raise Exception('Invalid learning rate schedule. It should be a number or, for example, like "0.001:100, 0.00001:1000, 1e-5:10000" to have lr of 0.001 until step 100, 0.00001 until 1000, and 1e-5 until 10000.')


    def __iter__(self):
        return self

    def __next__(self):
        if self.it < self.maxit:
            self.it += 1
            return self.rates[self.it - 1]
        else:
            raise StopIteration


class LearnRateScheduler:
    def __init__(self, learn_rate, max_steps, cur_step=0, last_lr=None, verbose=True):
        if len(learn_rate) >= 2 and learn_rate[0] == "=":
            self.schedules = None
            splits = learn_rate.split("/")
            try:
                self.initial_learn_rate = float(splits[0][1:])
            except (ValueError, AssertionError):
                raise Exception('Invalid auto-learning rate. Auto-learning should be of the form "=0.0001" or "=2e-5", where the number is the initial learning rate to use. Optionally you can also add "/" followed by a target change rate, such as "=1e-5/0.08, for an 8% image change per cycle. An additional section separated by a slash can be added for the curve half life, such as "=8e-5/0.08/30000". Make sure to enable saving images to the log directory at a reasonable interval, as rates only change when images are generated, and only slowly.')
            try:
                self.max_image_differential = float(splits[1])
            except (ValueError, AssertionError):
                raise Exception('Invalid auto-learning rate. Image change rate is not a parseable number. Should be for example "0.08" for 8%, or whatnot.')
            except (KeyError, IndexError):
                self.max_image_differential = 0.08	# How much do we want the average pixel change to be per cycle (%) at step 1?
            try:
                self.differential_halflife = float(splits[2])
            except (ValueError, AssertionError):
                raise Exception('Invalid auto-learning rate. Half life is not a parseable number.  Should be a number of cycles, such as "30000"')
            except (KeyError, IndexError):
                self.differential_halflife = 30000	# After how many cycles should the change rate be halved?
            
            tqdm.tqdm.write(f'Auto-learning selected.')
            
            self.LR_SCALEDOWN_MINUS      = 0.15  # How much to maintain the old learning rate (vs. the new one when one arrives)
            self.LR_SCALEDOWN_PLUS       = 0.65 # How much to maintain the old learning rate (vs. the new one when one arrives)
            self.LR_SCALEDOWN_MINUS_EXP  = 2.0	# How much to maintain the old learning rate (vs. the new one when one arrives)
            self.LR_SCALEDOWN_PLUS_EXP   = 1.0	# How much to maintain the old learning rate (vs. the new one when one arrives)
            self.MAX_GROWTH              = 1.35 # Cap on how much the LR can grow in each preview cycle.
            
            self.max_steps = max_steps
            self.target_image_differential = self.max_image_differential * (0.5 ** (cur_step / self.differential_halflife))
            if last_lr:
                self.learn_rate = last_lr
            else:
                if cur_step != 0:
                    print("Could not retrieve the previous learning rate; will guestimate where to start.")
                self.learn_rate = 0.1 * self.initial_learn_rate * (self.target_image_differential / self.max_image_differential) ** (1 + (cur_step / self.differential_halflife))	# Be very pessimistic, as we lack an optimizer, so training tends to explode.
            
        else:
            self.schedules = LearnScheduleIterator(learn_rate, max_steps, cur_step)
            (self.learn_rate,  self.end_step) = next(self.schedules)
            print(f'Learning schedule selected.')
        
        
        self.verbose = verbose

        if self.verbose:
            if self.schedules:
                print(f'Training at a rate of {self.learn_rate} until step {self.end_step}')
            else:
                print(f'Training at a rate of {self.learn_rate}')

        self.finished = False


    def apply(self, optimizer, step_number):
        if self.schedules:
            if step_number < self.end_step:
                return

            try:
                (self.learn_rate, self.end_step) = next(self.schedules)
            except Exception:
                self.finished = True
                return
        else:
            self.target_image_differential = self.max_image_differential * (0.5 ** (step_number / self.differential_halflife))

        if self.verbose:
            if self.schedules:
                tqdm.tqdm.write(f'Training at a rate of {self.learn_rate} until step {self.end_step}')

        for pg in optimizer.param_groups:
            pg['lr'] = self.learn_rate


    def apply_image_differential(self, actual_image_differential):
        if self.schedules is None:
            old_learn_rate = self.learn_rate
            ratio = self.target_image_differential / actual_image_differential
            if self.target_image_differential > actual_image_differential:
                ratio **= self.LR_SCALEDOWN_PLUS_EXP
                limit = self.MAX_GROWTH / self.LR_SCALEDOWN_PLUS
                if ratio > limit:
                    ratio = limit
                self.learn_rate = self.learn_rate * self.LR_SCALEDOWN_PLUS + self.learn_rate * ratio * (1 - self.LR_SCALEDOWN_PLUS)
            else: 
                ratio **= self.LR_SCALEDOWN_MINUS_EXP
                self.learn_rate = self.learn_rate * self.LR_SCALEDOWN_MINUS + self.learn_rate * ratio * (1 - self.LR_SCALEDOWN_MINUS)
            if self.verbose:
                tqdm.tqdm.write(f'Image differential={actual_image_differential}, target={self.target_image_differential}, lr={old_learn_rate}->{self.learn_rate}')

