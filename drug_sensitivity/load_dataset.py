"""
Helper method for loading the specified file. 
Return (R,M) where M is the mask, and R is the dataset (missing values set to 0).
"""

import numpy

''' Load the datasets and return the mask and values (missing = NaN) '''
def load_data(file_name):
    dataset = numpy.genfromtxt(file_name,dtype=float,usemask=True,missing_values=numpy.nan)
    R, M = dataset.data, ~dataset.mask
    R[numpy.isnan(R)] = 0.
    return (R,M)
    
''' Same as load_data but remove entirely missing rows and columns, and return indices or rows and cols with observations '''
def load_data_without_empty(file_name):
    dataset = numpy.genfromtxt(file_name,dtype=float,usemask=True,missing_values=numpy.nan)
    R, M = dataset.data, ~dataset.mask
    R[numpy.isnan(R)] = 0.
    
    I,J = R.shape
    rows_with_observations = numpy.array([i for i in range(0,I) if sum(M[i,:]) > 0])
    columns_with_observations = numpy.array([j for j in range(0,J) if sum(M[:,j]) > 0])
    R = R[rows_with_observations[:,None],columns_with_observations]
    M = M[rows_with_observations[:,None],columns_with_observations]
    return (R,M,rows_with_observations,columns_with_observations)
    
''' Same as load_data but only use the specified row and column indices '''
def load_data_filter(file_name,rows=None,columns=None):
    assert rows is not None or columns is not None, "Either rows or columns needs to be specified!"    
    
    dataset = numpy.genfromtxt(file_name,dtype=float,usemask=True,missing_values=numpy.nan)
    R, M = dataset.data, ~dataset.mask
    R[numpy.isnan(R)] = 0.
    
    if columns is not None and rows is not None:
        R = R[rows[:,None],columns]
        M = M[rows[:,None],columns]
    elif columns is None:    
        I,J = R.shape
        R = R[rows[:,None],range(0,J)]
        M = M[rows[:,None],range(0,J)]
    elif rows is None:    
        I,J = R.shape
        R = R[:,columns]
        M = M[:,columns]
        
    return (R,M)