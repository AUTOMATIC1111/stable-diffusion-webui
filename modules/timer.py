import time


class Timer:
    def __init__(self):
        self.start = time.time()
        self.records = {}
        self.total = 0

    def elapsed(self):
        end = time.time()
        res = end - self.start
        self.start = end
        return res

    def record(self, category, extra_time=0):
        e = self.elapsed()
        if category not in self.records:
            self.records[category] = 0

        self.records[category] += e + extra_time
        self.total += e + extra_time

    def summary(self, min_time=0.05):
        res = f"{self.total:.2f}"
        additions = [x for x in self.records.items() if x[1] >= min_time]
        if not additions:
            return res
        res += " { " + " ".join([f"{category}={time_taken:.2f}" for category, time_taken in additions]) + " }"
        return res

    def reset(self):
        self.__init__()

startup = Timer()
