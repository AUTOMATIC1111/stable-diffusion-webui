import time

# to microseconds * to millis * to seconds
NUM_NANO_SECONDS_IN_SECONDS = 1000 \
                              * 1000 \
                              * 1000


class Timer:
    def __init__(self):
        self.start = time.perf_counter_ns()
        self.records = {}
        self.total = 0

    def elapsed(self):
        end = time.perf_counter_ns()
        res = end - self.start
        self.start = end
        return res

    def record(self, category, extra_time_ns=0):
        e = self.elapsed()
        if category not in self.records:
            self.records[category] = 0

        self.records[category] += e + extra_time_ns
        self.total += e + extra_time_ns

    def summary(self):

        res = f"{(self.total / NUM_NANO_SECONDS_IN_SECONDS):.3f}s"

        additions = [x for x in self.records.items() if x[1] >= 0.1]
        if not additions:
            return res

        res += " ("
        res += ", ".join(
            [f"{category}: {(time_taken / NUM_NANO_SECONDS_IN_SECONDS):.3f}s" for category, time_taken in additions])
        res += ")"

        return res

    def reset(self):
        self.__init__()

    def decorator(self, func, category, extra_time_ns=0):
        def wrapper(*args, **kwargs):
            result = func(args, kwargs)
            self.record(category, extra_time_ns)
            return result

        return wrapper
