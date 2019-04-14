import numpy as np
import pandas as pd

import pandas_implants.pace


def test_pace_array():
    series = pd.Series(["4:00", "4:20"], dtype="pace")
    assert np.allclose(series.values, [240, 260])