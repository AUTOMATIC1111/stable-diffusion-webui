from __future__ import annotations

import time


def format_performance_html(start_time: float, mem_stats: dict[str, int | None]) -> str:
    elapsed_m, elapsed_s = divmod(time.perf_counter() - start_time, 60)
    elapsed_text = f"{elapsed_s:.2f}s"
    if elapsed_m > 0:
        elapsed_text = f"{elapsed_m}m {elapsed_text}"
    return (
        f"<div class='performance'>"
        f"<p class='time'>Time taken: <wbr>{elapsed_text}</p>"
        f"{format_vram_html(mem_stats)}"
        f"</div>"
    )


def format_vram_html(mem_stats: dict[str, int | None]) -> str:
    mem_stats_mb = {k: -(v // -(1024 * 1024)) for k, v in mem_stats.items()}
    vram_bits = []

    # CUDA uses active_peak/reserved_peak, MPS uses max_active/max_active_cached
    active_peak = mem_stats_mb.get("active_peak") or mem_stats_mb.get("max_active")
    reserved_peak = mem_stats_mb.get("reserved_peak") or mem_stats_mb.get("max_active_cached")
    if active_peak and reserved_peak:
        vram_bits.append(f"Torch active/reserved: {active_peak}/{reserved_peak} MiB")

    sys_peak = mem_stats_mb.get("system_peak")
    sys_total = mem_stats_mb.get("total")
    if sys_peak and sys_total:
        sys_pct = round(sys_peak / max(sys_total, 1) * 100, 2)
        vram_bits.append(f"System: {sys_peak}/{sys_total} MiB ({sys_pct}%)")

    vram_html = f"<p class='vram'>{'<wbr>'.join(vram_bits)}</p>" if vram_bits else ""
    return vram_html
