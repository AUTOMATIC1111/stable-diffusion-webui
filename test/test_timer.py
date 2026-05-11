import importlib

import pytest


def import_timer(monkeypatch, times):
    monkeypatch.setattr("time.time", lambda: next(times))

    import modules.timer as timer

    return importlib.reload(timer)


def test_timer_records_elapsed_time_and_summary(monkeypatch):
    timer = import_timer(monkeypatch, iter([1.0, 1.0, 1.0, 1.2, 1.25, 1.3]))

    subject = timer.Timer()
    subject.record("load", extra_time=0.05)
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
    timer = import_timer(monkeypatch, iter([5.0, 5.0, 5.0, 5.1, 5.2, 5.5, 5.7, 5.8]))

    subject = timer.Timer(print_log=True)
    with subject.subcategory("model"):
        assert subject.base_category == "model/"
        assert subject.subcategory_level == 1
        subject.record("load")

    assert subject.base_category == ""
    assert subject.subcategory_level == 0
    assert subject.records == {
        "model/load": pytest.approx(0.2),
        "model": pytest.approx(0.9),
    }
    assert subject.total == pytest.approx(0.7)
    assert capsys.readouterr().out.splitlines() == [
        "  model:",
        "  load: done in 0.200s",
    ]


def test_timer_subcategory_without_logging(monkeypatch, capsys):
    timer = import_timer(monkeypatch, iter([7.0, 7.0, 7.1, 7.2, 7.3, 7.4]))

    subject = timer.Timer()
    with subject.subcategory("quiet"):
        pass

    assert subject.records == {"quiet": pytest.approx(0.4)}
    assert capsys.readouterr().out == ""
