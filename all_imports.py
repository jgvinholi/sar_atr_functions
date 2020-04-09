%tensorflow_version 2.x

from matplotlib.image import imread
from matplotlib import rcParams
import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy
import os, glob
import pickle
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import fclusterdata
from numba import jit, njit
import warnings
from operator import itemgetter 
from joblib import Parallel, delayed
import multiprocessing