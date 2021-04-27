import os
import numpy as np
import pylab
from sklearn.decomposition import FastICA
import scipy.stats
import h5py
import image_data_analysis_full


def truncate_non_neg(x):
    """Function that truncates arrays od real numbers into arrays of non negatives.
    Args:
    x(numpy.array): input array
    Returns:
    y(numpy.array): array with positive or zero numbers
    """
    #write your function here and erase the current return
    return 0


def get_power_spectrum_whitening_filter(average_PS,noise_variance):
    """Function that estimates the whitening and denoising power spectrum filter
    Args:
    average_PS(numpy.array): average power spectrum of the observation
    noise_variance(double): variance of the gaussian white noite.
    Returns:
    w(numpy.array): whitening denoising filter
    """
    #write your function here and erase the current return
    return 0



def make_whitening_filters_figure(whitening_filters):
    pylab.figure()
    for i,whiteningFilter in enumerate(whitening_filters):
        pylab.subplot(1,len(whitening_filters),i+1)
        vmax = np.max(np.abs(whiteningFilter))
        vmin = -vmax
        pylab.imshow(whiteningFilter,cmap = 'gray',vmax = vmax, vmin = vmin)
        pylab.axis("off")



def get_ICA_input_data(dataset_file_name, sample_size, number_of_samples):
    """ Function that samples the input directory for later to be used by FastICA
    Args:
    inputFileName(str):: Absolute pathway to the image database hdf5 file
    sample_size (tuple(int,int)): size of the samples that are extrated from the images
    nSamples(int): number of samples that should be taken from the database
    Returns:
    X(numpy.array)nSamples, sample_size
    """
    #write your function here and erase the current return
    return 0



def pre_process(X):
    """Function that preprocess the data to be fed to the ICA algorithm
    Args:
    X(numpy array): input to be preprocessed
    Returns:
    X(numpy.array)
    """
    #write your function here and erase the current return
    return 0

    
def get_IC(X):
    """Function that estimates the independent components of the data
    Args:
    X(numpy.array):preprocessed data
    Returns:
    W(numpy.array) the matrix of the independent sources of the data
    """
    #write your function here and erase the current return
    return 0

    
   
def make_idependent_components_figure(W, sample_size): 
    W = W.reshape([-1,]+sample_size)
    pylab.figure()
    for i in range(W.shape[0]):
        pylab.subplot(sample_size[0],sample_size[1],i+1)
        pylab.imshow(W[i],cmap = 'gray')
        pylab.axis("off")


   
   
def estimate_sources(W,X):
    """Function that estimates the independent sources of the data
    Args:
    W(numpy.array):The matrix of the independent components
    X(numpy.array):preprocessed data
    Returns:
    S(numpy.array) the matrix of the sources of X
    """
    #write your function here and erase the current return
    return 0


