import sys
sys.path.append("..")

from utils import *
from HPC_func import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

T = Tools_Extend()
# this_root = '/home/ygo26002/Project_data/Primary_Forests_JiWon' # for HPC
this_root = '/Users/liyang/Projects_data/Primary_Forests_JiWon' # for MacMini
data_root = join(this_root,'data')