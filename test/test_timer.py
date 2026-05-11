import importlib

import pytest


class FakeClock:
    def __init__(self, current):
        self.current = current

    def __call__(self):
        return self.current

    def advance(self, seconds):
        self.current += seconds


def import_timer(monkeypatch, start=0):
    clock = FakeClock(start)
    monkeypatch.setattr("time.time", clock)
    import modules.timer as timer

    return importlib.reload(timer), clock


def test_timer_records_elapsed_time_and_summary(monkeypatch):
    timer, clock = import_timer(monkeypatch, start=1.0)

    subject = timer.Timer()
    clock.advance(0.2)
    subject.record("load", extra_time=0.05)
    clock.advance(0.05)
    subject.record("small")

    assert subject.dump() == {
        "total": pytest.approx(0.3),
        "records": {"load": pytest.approx(0.25), "small": pytest.approx(0.05)},
    }
    assert subject.summary() == "0.3s (load: 0.2s)"

    subject.reset()

    assert subject.dump() == {"total": 0, "records": {}}
    assert subject.summary() == "0.0s"


def test_timer_subcategory_records_nested_time_and_logs(monkeypatch, capsys):
    timer, clock = import_timer(monkeypatch, start=5.0)

    subject = timer.Timer(print_log=True)
    clock.advance(0.1)
    with subject.subcategory("model"):
        assert subject.base_category == "model/"
        assert subject.subcategory_level == 1
        clock.advance(0.2)
        subject.record("load")
        clock.advance(0.7)

    assert subject.base_category == ""
    assert subject.subcategory_level == 0
    assert subject.records == {
        "model/load": pytest.approx(0.2),
        "model": pytest.approx(1.6),
    }
    assert subject.total == pytest.approx(0.9)
    assert capsys.readouterr().out.splitlines() == [
        "  model:",
        "  load: done in 0.200s",
    ]


def test_timer_subcategory_without_logging(monkeypatch, capsys):
    timer, clock = import_timer(monkeypatch, start=7.0)

    subject = timer.Timer()
    clock.advance(0.1)
    with subject.subcategory("quiet"):
        clock.advance(0.4)
        pass

    assert subject.records == {"quiet": pytest.approx(0.8)}
    assert capsys.readouterr().out == ""
