# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 17:52:56 2014

@author: mildbyte
"""

import numpy as np
        
def safe_log(x):
    return np.log(np.clip(x, a_min=1e-6, a_max=np.max(x)))
