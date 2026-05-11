import importlib
import types


def import_infotext_versions(monkeypatch, *, auto_backcompat=True):
    shared = types.SimpleNamespace(opts=types.SimpleNamespace(auto_backcompat=auto_backcompat))
    import modules

    monkeypatch.setattr(modules, "shared", shared, raising=False)
    monkeypatch.setitem(__import__("sys").modules, "modules.shared", shared)

    import modules.infotext_versions as infotext_versions

    return importlib.reload(infotext_versions)


def test_parse_version_accepts_supported_shapes(monkeypatch):
    infotext_versions = import_infotext_versions(monkeypatch)

    assert infotext_versions.parse_version(None) is None
    assert str(infotext_versions.parse_version("1.8.0")) == "1.8.0"
    assert str(infotext_versions.parse_version("v1.7.0-225-gabc123")) == "1.7.0.post225"
    assert infotext_versions.parse_version("not a version") is None


def test_backcompat_returns_early_when_disabled_or_version_missing(monkeypatch):
    infotext_versions = import_infotext_versions(monkeypatch, auto_backcompat=False)
    disabled = {"Version": "1.5.0", "Prompt": "[old]", "Sampler": "DDIM", "Refiner": "yes"}

    infotext_versions.backcompat(disabled)

    assert disabled == {"Version": "1.5.0", "Prompt": "[old]", "Sampler": "DDIM", "Refiner": "yes"}

    infotext_versions = import_infotext_versions(monkeypatch)
    missing_version = {}

    infotext_versions.backcompat(missing_version)

    assert missing_version == {}


def test_backcompat_sets_flags_for_older_versions(monkeypatch):
    infotext_versions = import_infotext_versions(monkeypatch)
    data = {"Version": "1.5.0", "Prompt": "a [b:c:1]", "Sampler": "PLMS", "Refiner": "yes"}

    infotext_versions.backcompat(data)

    assert data["Old prompt editing timelines"] is True
    assert data["Pad conds v0"] is True
    assert data["Downcast alphas_cumprod"] is True
    assert data["Refiner switch by sampling steps"] is True


def test_backcompat_only_sets_matching_flags(monkeypatch):
    infotext_versions = import_infotext_versions(monkeypatch)
    data = {"Version": "1.7.0-225", "Prompt": "plain", "Sampler": "Euler", "Refiner": ""}

    infotext_versions.backcompat(data)

    assert data == {"Version": "1.7.0-225", "Prompt": "plain", "Sampler": "Euler", "Refiner": ""}
