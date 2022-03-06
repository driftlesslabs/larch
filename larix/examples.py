import pandas as pd
import numpy as np
from .dataset import Dataset, DataArray
import os

def example_file(filename):
    warehouse_file = os.path.normpath( os.path.join( os.path.dirname(__file__), 'data_warehouse', filename) )
    if os.path.exists(warehouse_file):
        return warehouse_file
    raise FileNotFoundError(f"there is no example data file '{warehouse_file}' in data_warehouse")

def MTC(**kwargs):
    ca = pd.read_csv(example_file('MTCwork.csv.gz'), index_col=('casenum', 'altnum'), dtype={'chose':np.float32})
    dataset = Dataset.construct.from_idca(
        ca.rename_axis(index=('caseid', 'altid')),
        altnames=['DA', 'SR2', 'SR3+', 'Transit', 'Bike', 'Walk'],
        fill_missing=0,
    )
    dataset['avail'] = DataArray(dataset['_avail_'].values, dims=['caseid', 'altid'], coords=dataset.coords)
    dataset['chose'] = dataset['chose'].fillna(0.0)
    return dataset