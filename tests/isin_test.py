import vaex
import numpy as np


def test_isin():
    x = np.array([1.01, 2.02, 3.03])
    y = np.array([1, 3, 5])
    s = np.array(['dog', 'cat', 'mouse'])
    sm = np.array(['dog', 'cat', None])
    w = np.array([2, '1.1', None])
    m = np.ma.MaskedArray(data=[np.nan, 1, 1], mask=[True, True, False])
    n = np.array([-5, np.nan, 1])
    df = vaex.from_arrays(x=x, y=y, s=s, sm=sm, w=w, m=m, n=n)

    assert df.x.isin([1, 2.02, 5, 6]).tolist() == [False, True, False]
    assert df.y.isin([5, -1, 0]).tolist() == [False, False, True]
    assert df.s.isin(['elephant', 'dog']).tolist() == [True, False, False]
    assert df.sm.isin(['cat', 'dog']).tolist() == [True, True, False]
    assert df.w.isin([2, None]) == [True, False, True]
    assert df.m.isin([1, 2, 3]) == [False, False, True]
    assert df.n.isin([2, np.nan]) == [False, True, False]
