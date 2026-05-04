"""
Datasets module

Created on Thu Sep 25 2025

Author: Zizhong Yan
"""

import pandas as pd
import pkg_resources 

def helpman08():
    """
    Load the data and return a dataset class instance.
    """
    path = pkg_resources.resource_filename('networkfm', 'database/')
    data = pd.read_csv(path+'/helpman08.csv')  
    return data


def trade1986():
    """
    Load the data and return a dataset class instance.
    """
    path = pkg_resources.resource_filename('networkfm', 'database/')
    data = pd.read_csv(path+'/trade1986.csv')  
    return data
