from os import getpid
from collections import defaultdict
from modules.dml.pdh import HQuery, HCounter, expand_wildcard_path


class MemoryProvider:
    hQuery: HQuery
    hCounters: defaultdict[str, list[HCounter]]

    def __init__(self):
        self.hQuery = HQuery()
        self.hCounters = defaultdict(list)

    def get_memory(self, device_id: int) -> dict[str, int]:
        if len(self.hCounters) == 0:
            pid = getpid()
            paths_dedicated = expand_wildcard_path(f"\\GPU Process Memory(pid_{pid}_*_phys_{device_id})\\Dedicated Usage")
            paths_committed = expand_wildcard_path(f"\\GPU Process Memory(pid_{pid}_*_phys_{device_id})\\Total Committed")
            for path in paths_dedicated:
                self.hCounters["dedicated_usage"].append(self.hQuery.add_counter(path))
            for path in paths_committed:
                self.hCounters["total_committed"].append(self.hQuery.add_counter(path))
        self.hQuery.collect_data()
        result = defaultdict(int)
        for key in self.hCounters:
            for hCounter in self.hCounters[key]:
                result[key] += hCounter.get_formatted_value(int)
        return dict(result)

    def __del__(self):
        self.hQuery.close()
