import numpy as np
from rulervla.training.main_data import update_indices


def test_shuffled_epoch_covers_every_frame_and_resume_has_no_gap():
    count = 37
    order = np.concatenate([update_indices(count,42,step) for step in range(4)])
    np.testing.assert_array_equal(np.sort(order[:count]), np.arange(count))
    np.testing.assert_array_equal(np.sort(order[count:count*2]), np.arange(count))
    np.testing.assert_array_equal(update_indices(count,42,3), order[96:128])
    assert not np.array_equal(update_indices(count,42,0),update_indices(count,43,0))
