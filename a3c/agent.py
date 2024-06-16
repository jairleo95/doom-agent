import pylab
import cv2
import numpy as np
from a3c.networks import A3CModel
import os
from tensorflow.keras.models import load_model
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class A3CAgent:
