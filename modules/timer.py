import time
import argparse


class TimerSubcategory:
    def __init__(self, timer, category):
        self.timer = timer
        self.category = category
        self.start = None
        self.original_base_category = timer.base_category

    def __enter__(self):
        self.start = time.time()
        self.timer.base_category = self.original_base_category + self.category + "/"
        self.timer.subcategory_level += 1

        if self.timer.print_log:
            print(f"{'  ' * self.timer.subcategory_level}{self.category}:")

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_for_subcategroy = time.time() - self.start
        self.timer.base_category = self.original_base_category
        self.timer.add_time_to_record(self.original_base_category + self.category, elapsed_for_subcategroy)
        self.timer.subcategory_level -= 1
        self.timer.record(self.category, disable_log=True)


class Timer:
    def __init__(self, print_log=False):
        self.start = time.time()
        self.records = {}
        self.total = 0
        self.base_category = ''
        self.print_log = print_log
        self.subcategory_level = 0

    def elapsed(self):
        end = time.time()
        res = end - self.start
        self.start = end
        return res

    def add_time_to_record(self, category, amount):
        if category not in self.records:
            self.records[category] = 0

        self.records[category] += amount

    def record(self, category, extra_time=0, disable_log=False):
        e = self.elapsed()

        self.add_time_to_record(self.base_category + category, e + extra_time)

        self.total += e + extra_time

        if self.print_log and not disable_log:
            print(f"{'  ' * self.subcategory_level}{category}: done in {e + extra_time:.3f}s")

    def subcategory(self, name):
        self.elapsed()

        subcat = TimerSubcategory(self, name)
        return subcat

    def summary(self):
        res = f"{self.total:.1f}s"

        additions = [(category, time_taken) for category, time_taken in self.records.items() if time_taken >= 0.1 and '/' not in category]
        if not additions:
            return res

        res += " ("
        res += ", ".join([f"{category}: {time_taken:.1f}s" for category, time_taken in additions])
        res += ")"

        return res

    def dump(self):
        return {'total': self.total, 'records': self.records}

    def reset(self):
        self.__init__()


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--log-startup", action='store_true', help="print a detailed log of what's happening at startup")
args = parser.parse_known_args()[0]

startup_timer = Timer(print_log=args.log_startup)

startup_record = None
