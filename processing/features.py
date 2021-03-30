# Importing necessary libraries
import numpy as np

def calculate_max_peaks(data):
  '''maximum of the peaks
    Funcion that finds the maximum of the peaks
    
    Parameters
    ----------
    data : 1-d array 
        array containing set of peaks
    
    Returns
    -------
    max(data) : int or float
        maximum value of the peak
  '''
  if (len(data) == 0):
    return 0
  else:
    return np.max(data)
  
def calculate_mean_eda(data):
  '''mean of the eda data
    Funcion that finds the mean of the eda data
	
    Parameters
    ----------
    data : 1-d array 
        array containing eda data
    
    Returns
    -------
    mean(data) : int or float
        mean value of the eda data
  '''
  return np.mean(data)
  
def calculate_number_of_peaks(data):
  '''number of the peaks
    Funcion that finds the number of the peaks
    
    Parameters
    ----------
    data : 1-d array 
        array containing set of peaks
    
    Returns
    -------
    max(data) : int
        number of the peak
  '''
  return len(data)
