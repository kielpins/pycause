import numpy as np
import datastats as dst

nsamp = 1000
rand = list()
for it in range(4):
    rand.append(np.random.rand(nsamp))
prec = np.finfo(rand[0].dtype).eps

r0 = rand[0]
raw_locorr = np.column_stack([np.hstack([r0,r0]),np.hstack([r0,-r0])])
locorr = dst.DataStats(raw_data=raw_locorr)
raw_hicorr = np.column_stack([np.hstack([r0,r0]),np.hstack([r0,r0])])
hicorr = dst.DataStats(raw_data=raw_hicorr)
from_corr = dst.DataStats()
corr_mat = np.array([[1,0.25,0.5],[0.25,1,0.5],[0.5,0.5,1]])

def test_set_get_corr_mat():
    from_corr.set_corr_mat(corr_mat)
    assert np.array_equal(from_corr.get_corr_mat(),corr_mat)
    
def test_calc_corr_mat():
    corr = locorr.get_corr_mat()
    nomcorr = np.identity(2)
    diff = np.amax(corr-nomcorr)
    assert diff is not np.NaN
    assert diff<2*prec

def test_get_nsamp():
    assert locorr.get_nsamp() == 2*nsamp

def test_set_get_nsamp():
    from_corr.set_nsamp(nsamp)
    assert from_corr.get_nsamp() == nsamp

def test_get_part_corr():
    pcorr = dst.DataStats()
    corr_mat = np.array([[1,0.25,0.5],[0.25,1,0.5],[0.5,0.5,1]])
    pcorr.nsamp = 1000
    pcorr.set_corr_mat(corr_mat)
    pcorr_xy = pcorr.get_part_corr((0,1),{2})[0]
    assert abs(pcorr_xy) < 2*prec
    