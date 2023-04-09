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

    def summary(self):
        res = f"{self.total:.1f}s"

        additions = [x for x in self.records.items() if x[1] >= 0.1]
        if not additions:
            return res

        res += " ("
        res += ", ".join([f"{category}: {time_taken:.1f}s" for category, time_taken in additions])
        res += ")"

        return res

    def reset(self):
        self.__init__()
