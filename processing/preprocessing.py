# Importing necessary libraries
import numpy as np
import scipy.signal as sps


def resample_data(edadata, prevSR, newSR):
  '''calculates rolling mean
    Function to calculate moving average over the passed data
	
    Parameters
    ----------
    edadata : 1-d array
        array containing the eda data
    prevSR : int or float 
        the previous sample rate of the data
    newSR : int or float
        the new sample rate of the data
		
    Returns
    -------
    data : 1-d array
        array containing the resampled data
  '''
  number_of_samples = int(round(len(edadata) * float(newSR) / prevSR))
  data = sps.resample(edadata, number_of_samples)
  
  return data

	
def normalization(edadata):
  '''min max normalization
    Function to calculate normalized eda data
	
    Parameters
    ----------
    edadata : 1-d array
        array containing the eda data
		
    Returns
    -------
    n_edadata : 1-d array
        normalized eda data
  '''
  edadata = edadata-(np.min(edadata))
  edadata /= (np.max(edadata) - np.min(edadata))
  n_edadata = edadata
  return n_edadata

def rolling_mean(data, windowsize, sample_rate):
  '''calculates rolling mean
    Function to calculate moving average over the passed data
	
    Parameters
    ----------
    data : 1-d array
        array containing the eda data
    windowsize : int or float 
        the moving average window size in seconds 
    sample_rate : int or float
        the sample rate of the data set
		
    Returns
    -------
    rol_mean : 1-d array
        array containing computed rolling mean
  '''
  avg_hr = (np.mean(data))
  data_arr = np.array(data)
	
  t_windowsize = int(windowsize*sample_rate)
  t_shape = data_arr.shape[:-1] + (data_arr.shape[-1] - t_windowsize + 1, t_windowsize)
  t_strides = data_arr.strides + (data_arr.strides[-1],)
  sep_win = np.lib.stride_tricks.as_strided(data_arr, shape=t_shape, strides=t_strides)
  rol_mean = np.mean(sep_win, axis=1)
	
  missing_vals = np.array([avg_hr for i in range(0, int(abs(len(data_arr) - len(rol_mean))/2))])
  rol_mean = np.insert(rol_mean, 0, missing_vals)
  rol_mean = np.append(rol_mean, missing_vals)

  #only to catch length errors that sometimes unexplicably occur. 
  ##Generally not executed, excluded from testing and coverage
  if len(rol_mean) != len(data): # pragma: no cover
    lendiff = len(rol_mean) - len(data)
    if lendiff < 0:
      rol_mean = np.append(rol_mean, 0)
    else:
      rol_mean = rol_mean[:-1]
	  
  return rol_mean
