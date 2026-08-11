from modules import mps_utils


GIB = 1024**3


def test_sdp_is_used_for_normal_sd_attention_on_16gb_mac():
    assert mps_utils.should_use_sdp(2, 8, 4096, 4096, 2, total_memory=16 * GIB, available_memory=12 * GIB)


def test_sdp_is_avoided_for_high_resolution_self_attention():
    assert not mps_utils.should_use_sdp(1, 8, 9216, 9216, 2, total_memory=16 * GIB, available_memory=12 * GIB)


def test_sdp_budget_scales_down_on_8gb_mac():
    assert not mps_utils.should_use_sdp(2, 8, 4096, 4096, 2, total_memory=8 * GIB, available_memory=6 * GIB)


def test_cross_attention_remains_on_fast_path_at_high_resolution():
    assert mps_utils.should_use_sdp(2, 8, 9216, 77, 2, total_memory=8 * GIB, available_memory=4 * GIB)


def test_query_chunk_is_reduced_for_large_self_attention():
    chunk_size = mps_utils.attention_query_chunk_size(
        1024,
        16,
        9216,
        2,
        total_memory=16 * GIB,
        available_memory=12 * GIB,
    )
    assert 1 <= chunk_size < 1024
    assert chunk_size % 64 == 0


def test_query_chunk_is_unchanged_for_cross_attention():
    chunk_size = mps_utils.attention_query_chunk_size(
        1024,
        16,
        77,
        2,
        total_memory=8 * GIB,
        available_memory=4 * GIB,
    )
    assert chunk_size == 1024
