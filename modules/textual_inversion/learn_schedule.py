import tqdm
import math

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
    def __init__(self, learn_rate, target_image_differential, change_rate_weight, differential_halflife, differential_saturation_halflife, saturation_halflife, max_steps, cur_step=0, last_lr=None, verbose=True):
        print("Learning rate parameters:", [learn_rate, target_image_differential, change_rate_weight, differential_halflife, differential_saturation_halflife, saturation_halflife, max_steps])
        if change_rate_weight == "":
            if target_image_differential != "":
                if saturation_halflife != "":
                    change_rate_weight = 0.5
                else:
                    change_rate_weight = 1.0
            else:
                change_rate_weight = 0.0
        if target_image_differential == "":
            target_image_differential = 0.08
        if differential_halflife == "":
            differential_halflife = 1e16
        if saturation_halflife == "":
            saturation_halflife = 1e16
        try:
            self.target_image_differential = float(target_image_differential)
        except (ValueError, AssertionError):
            raise Exception(f'Invalid image change rate "{target_image_differential}". It should be a number, like "0.08" (aka 8%).')
        if self.target_image_differential <= 0 or self.target_image_differential > 1:
            raise Exception(f'Invalid image change rate "{self.target_image_differential}"; must be between 0 and 1')
        try:
            self.change_rate_weight = float(change_rate_weight)
        except (ValueError, AssertionError):
            raise Exception(f'Invalid image change weighting "{change_rate_weight}". It should be a number in the range of 0-1.')
        if self.change_rate_weight < 0 or self.change_rate_weight > 1:
            raise Exception(f'Invalid image change weighting "{change_rate_weight}"; must be between 0 and 1')
        try:
            self.differential_halflife = float(differential_halflife)
        except (ValueError, AssertionError):
            raise Exception(f'Invalid image change rate half life "{differential_halflife}". It should be a number, like "30000".')
        if self.differential_halflife <= 0:
            raise Exception(f'Invalid image change rate half life "{differential_halflife}"; must be greater than 0.')
        if differential_saturation_halflife == "":
            self.differential_saturation_halflife = None
        else:
            try:
                self.differential_saturation_halflife = float(differential_saturation_halflife)
            except (ValueError, AssertionError):
                raise Exception(f'Invalid image change halflife (steps) "{differential_saturation_halflife}". It should be a number between 0-1, like "0.04".')
            if self.differential_saturation_halflife <= 0:
                raise Exception(f'Invalid image change halflife (steps) "{differential_saturation_halflife}"; must be greater than 0, like "0.04".')
        try:
            self.saturation_halflife = float(saturation_halflife)
        except (ValueError, AssertionError):
            raise Exception(f'Invalid learning rate halflife (saturation) "{saturation_halflife}". It should be a number greater than 0, like "0.04".')
        if self.saturation_halflife <= 0:
            raise Exception(f'Invalid learning rate halflife (saturation) "{saturation_halflife}"; must be greater than 0, like "0.04".')

        self.LR_SCALEDOWN_MINUS      = 0.15  # How much to maintain the old learning rate (vs. the new one when one arrives)
        self.LR_SCALEDOWN_PLUS       = 0.65 # How much to maintain the old learning rate (vs. the new one when one arrives)
        self.LR_SCALEDOWN_MINUS_EXP  = 2.0	# How much to maintain the old learning rate (vs. the new one when one arrives)
        self.LR_SCALEDOWN_PLUS_EXP   = 1.0	# How much to maintain the old learning rate (vs. the new one when one arrives)
        self.MAX_GROWTH              = 1.35 # Cap on how much the LR can grow in each preview cycle.
        self.SATURATION_SKIP         = 4    # Skip the first X images before starting to average saturation as a baseline
        self.SATURATION_LAST         = 10   # Last image for which to build up the saturation average
        self.SATURATION_SCALEDOWN    = 0.95 # For resetting the saturation mean
            
        self.mean_saturation = 0.3
        self.mean_saturation_count = 0
        self.saturation_scalar = 1.0
        self.step_number = cur_step
        self.orig_target_image_differential = self.target_image_differential
        self.target_image_differential = self.orig_target_image_differential * (0.5 ** (cur_step / self.differential_halflife))

#        else:
        try:
            self.schedules = LearnScheduleIterator(learn_rate, max_steps, cur_step)
            (self.learn_rate,  self.end_step) = next(self.schedules)
        except:
            self.schedules = None
            self.learn_rate = None
            self.end_step = max_steps
        
        if self.learn_rate is None or self.learn_rate == 0 or self.learn_rate == "":
            self.learn_rate = last_lr
        
        if self.learn_rate is None:
           self.learn_rate = 1e-6
           
        self.change_learn_rate = self.learn_rate

        self.initial_learn_rate = self.learn_rate
        print(f'Learning schedule selected.')
        
        if cur_step != 0 and self.change_rate_weight > 0 and not last_lr:
            print("Could not retrieve the previous learning rate; will guestimate where to start.")
            self.learn_rate = 0.1 * self.initial_learn_rate * (self.target_image_differential / self.target_image_differential) ** (1 + (cur_step / self.differential_halflife))	# Be very pessimistic, as we lack an optimizer, so training tends to explode.
        
        self.verbose = verbose

        print("After adjustment of params:", [self.learn_rate, self.target_image_differential, self.change_rate_weight, self.differential_halflife, self.differential_saturation_halflife, self.saturation_halflife, self.end_step])
 
        if self.verbose:
            print(f'Training at a rate of {self.learn_rate}')

        self.max_steps = max_steps
        if cur_step >= self.max_steps:
            self.finished = True
        else:
            self.finished = False


    def apply(self, optimizer, step_number):
        self.step_number = step_number
        if step_number >= self.max_steps:
            self.finished = True
            return

        self.target_image_differential = self.orig_target_image_differential * (0.5 ** (step_number / self.differential_halflife))

        if step_number < self.end_step:
            return

        try:
            (self.initial_learn_rate, self.end_step) = next(self.schedules)
        except Exception:
            self.finished = True
            return
        
        if self.verbose:
            tqdm.tqdm.write(f'step={step_number}, target_image_differential={self.target_image_differential}, end_step={self.end_step}, initial_learn_rate={self.initial_learn_rate}')

        self.recalc_learn_rate(optimizer)


    def recalc_learn_rate(self, optimizer):
        nonchange_weight = 1.0 - self.change_rate_weight
        nonchange_learn_rate = self.initial_learn_rate  * self.saturation_scalar
        self.learn_rate = math.e ** (math.log(nonchange_learn_rate) * nonchange_weight + math.log(self.change_learn_rate) * self.change_rate_weight)	# Logarithmic average
#        if self.verbose:
#            tqdm.tqdm.write(f'Image differential={image_differential}, target={self.target_image_differential}, lr={old_learn_rate}->{self.learn_rate}')

        if self.verbose:
            tqdm.tqdm.write(f'Training at a rate of {self.learn_rate}: nonchange rate={nonchange_learn_rate} * {nonchange_weight}, change rate={self.change_learn_rate} * {self.change_rate_weight}')

        for pg in optimizer.param_groups:
            pg['lr'] = self.learn_rate


    def apply_image_stats(self, optimizer, image_differential, saturation):
        saturation_rise = 0
        if self.mean_saturation_count > self.SATURATION_LAST:
            saturation_rise = saturation - self.mean_saturation
            if saturation_rise < 0:
                self.mean_saturation = self.mean_saturation * self.SATURATION_SCALEDOWN + saturation * (1 - self.SATURATION_SCALEDOWN)
        else:
            if self.mean_saturation_count >= self.SATURATION_SKIP:
                self.mean_saturation *= (self.mean_saturation_count - self.SATURATION_SKIP)
                self.mean_saturation_count += 1
                self.mean_saturation = (self.mean_saturation + saturation) / (self.mean_saturation_count - self.SATURATION_SKIP)
            else:
                self.mean_saturation_count += 1

        if saturation_rise > 0:
            if self.differential_saturation_halflife:
                differential_saturation_divisor = (saturation_rise / self.differential_saturation_halflife)
            else:
                differential_saturation_divisor = 1.0
            self.saturation_scalar = 0.5 ** (saturation_rise / self.saturation_halflife)
        else:
            differential_saturation_divisor = 1.0
            self.saturation_scalar = 1.0

        change_ratio = None
        if image_differential is not None:
            if image_differential == 0:
                change_ratio = 1e6
            else:
                change_ratio = (self.target_image_differential / (1 + differential_saturation_divisor)) / image_differential
            if change_ratio > 1:
                change_ratio **= self.LR_SCALEDOWN_PLUS_EXP
                limit = self.MAX_GROWTH / self.LR_SCALEDOWN_PLUS
                if change_ratio > limit:
                    change_ratio = limit
                self.change_learn_rate = self.learn_rate * self.LR_SCALEDOWN_PLUS + self.learn_rate * change_ratio * (1 - self.LR_SCALEDOWN_PLUS)
            else: 
                change_ratio **= self.LR_SCALEDOWN_MINUS_EXP
                self.change_learn_rate = self.learn_rate * self.LR_SCALEDOWN_MINUS + self.learn_rate * change_ratio * (1 - self.LR_SCALEDOWN_MINUS)
        else:
            self.change_learn_rate = self.learn_rate

        if self.verbose:
            tqdm.tqdm.write(f'Saturation stats: saturation_rise={saturation_rise} (cur={saturation}, avg={self.mean_saturation}), saturation_scalar={self.saturation_scalar}')
            tqdm.tqdm.write(f'Change stats: change_ratio={change_ratio}, target={self.target_image_differential}, saturation_divisor={differential_saturation_divisor}, image_differential={image_differential}')
            tqdm.tqdm.write(f'Change LR: change_rate={self.change_learn_rate}, learn_rate={self.learn_rate}')

        self.recalc_learn_rate(optimizer)

