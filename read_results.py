import numpy as np
from lytools import *
from pprint import pprint
T = Tools()
fdir = '/home/yangli/UCONN_Projects/Primary_Forests/h04v03'

for f in T.listdir(fdir):
    fpath = join(fdir, f)
    array = np.load(fpath)
    pprint(array[0])
    exit(0)
    print(fpath)
    print(array.shape)
    # exit(0)

pass