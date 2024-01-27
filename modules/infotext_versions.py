from modules import shared
from packaging import version
import re


v160 = version.parse("1.6.0")
v170_tsnr = version.parse("v1.7.0-225")


def parse_version(text):
    if text is None:
        return None

    m = re.match(r'([^-]+-[^-]+)-.*', text)
    if m:
        text = m.group(1)

    try:
        return version.parse(text)
    except Exception:
        return None


def backcompat(d):
    """Checks infotext Version field, and enables backwards compatibility options according to it."""

    if not shared.opts.auto_backcompat:
        return

    ver = parse_version(d.get("Version"))
    if ver is None:
        return

    if ver < v160 and '[' in d.get('Prompt', ''):
        d["Old prompt editing timelines"] = True

    if ver < v160 and d.get('Sampler', '') in ('DDIM', 'PLMS'):
        d["Pad conds v0"] = True

    if ver < v170_tsnr:
        d["Downcast alphas_cumprod"] = True

