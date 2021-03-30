# Importing necessary libraries
import numpy as np
import time
import scipy.signal
import matplotlib.pyplot as plt
from scipy import stats

# Importing necessary functions
from features import *
from filtering import *
from preprocessing import *
from windowing import *

'''
'''
def statistical_feature_extraction(preprocessed_eda, sample_rate, windowsize=0.75,  use_scipy=True, measures={},
                        working_data={}):
	'''processes passed edadata.
	
	Processes the passed eda data. Returns measures{} dict containing results.
	Parameters
	----------
	preprocessed_eda : 1d array or list 
		array or list containing normalized eda data to be analysed
	sample_rate : int or float
		the sample rate with which the eda data is sampled
	windowsize : int or float
		the window size in seconds to use in the calculation of the moving average.
		Calculated as windowsize * sample_rate
		default : 0.75
	measures : dict
		dictionary object used by heartpy to store computed measures. Will be created
		if not passed to function.
	working_data : dict
		dictionary object that contains all heartpy's working data (temp) objects.
		will be created if not passed to function
	Returns
	-------
	working_data : dict
		dictionary object used to store temporary values.
	
	measures : dict
		dictionary object used by heartpy to store computed measures.
	'''
	t1 = time.time()
	
	
	# Extracting phasic and tonic components of from normalized eda
	[phasic_eda, p, tonic_eda, l, d, e, obj] = cvxEDA(preprocessed_eda, 1./sample_rate)
	
	# Removing line noise
	filtered_phasic_eda = phasic_eda # comment out the next line if the line noise in negligble in your data
	filtered_phasic_eda = butter_lowpassfilter(phasic_eda, 5./sample_rate, sample_rate, order=4)
	
	# Update working_data
	working_data['filtered_phasic_eda'] = filtered_phasic_eda
	working_data['phasic_eda'] = phasic_eda
	working_data['tonic_eda'] = tonic_eda
	
	peaklist = []
	indexlist = []
	
	if (use_scipy):
		indexlist, _ = scipy.signal.find_peaks(filtered_phasic_eda)
		for i in indexlist:
			peaklist.append(preprocessed_eda[i])
	else:
		# Calculate the onSet and offSet based on Phasic GSR signal
		onSet_offSet = calculate_onSetOffSet(filtered_phasic_eda, sample_rate)
		# Calculate the peaks using onSet and offSet of Phasic GSR signal
		if (len(onSet_offSet) != 0):
			peaklist, indexlist = calculate_thepeaks(preprocessed_eda, onSet_offSet)
	
	working_data['peaklist'] = peaklist
	working_data['indexlist'] = indexlist
	# Calculate the number of peaks
	measures['number_of_peaks'] = calculate_number_of_peaks(peaklist)
	# Calculate the std mean of EDA
	measures['mean_eda'] = calculate_mean_eda(preprocessed_eda)
	# Calculate the maximum value of peaks of EDA
	measures['max_of_peaks'] = calculate_max_peaks(peaklist)
	
	return working_data, measures

	

'''
process EDA signal with windowing of size segment_width*sample_rate
'''
def segmentwise(edadata, sample_rate, segment_width=120, segment_overlap=0,
                        segment_min_size=5):
	'''processes passed edadata.
	Processes the passed eda data. Returns measures{} dict containing results.
	
	Parameters
	----------
	edadata : 1d array or list 
		array or list containing eda data to be analysed
	sample_rate : int or float
		the sample rate with which the eda data is sampled
	segment_width : int or float
		width of segments in seconds
		default : 120
	segment_overlap: float
		overlap fraction of adjacent segments.
		Needs to be 0 <= segment_overlap < 1.
		default : 0 (no overlap)
	segment_min_size : int
		often a tail end of the data remains after segmenting into segments.
		default : 20
	
	Returns
	-------
	edadata_segmentwise : 2d array or list 
		array or list containing segmentwised eda data to be analysed
	orking_data : dict
		dictionary object used to store temporary values.
	s_measures : dict
		dictionary object used by heartpy to store computed measures.
	'''
	slice_indices = make_windows(edadata, sample_rate, segment_width, segment_overlap, segment_min_size)
	
	s_measures = {}
	s_working_data = {}
	
	edadata_segmentwise = []
	for i, ii in slice_indices:
		edadata_segmentwise.append(edadata[i:ii])
		s_measures = append_dict(s_measures, 'segment_indices', (i, ii))
		s_working_data = append_dict(s_working_data, 'segment_indices', (i, ii))
	return s_working_data, s_measures, edadata_segmentwise
