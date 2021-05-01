import os
import numpy as np
import pylab
from sklearn.decomposition import FastICA
import scipy.stats
import h5py
import random 

import io_image_data
import image_data_analysis

#import image_data_analysis_full

def relu(x) : 
	return max(0., x)

#vectorisation 
relu = np.vectorize(relu)

 
def truncate_non_neg(x):
    """Function that truncates arrays od real numbers into arrays of non negatives.
    Args:
    x(numpy.array): input array
    Returns:
    y(numpy.array): array with positive or zero numbers
    """
    
    y = relu(x)
    return y


def get_power_spectrum_whitening_filter(average_PS,noise_variance):
    """Function that estimates the whitening and denoising power spectrum filter
    Args:
    average_PS(numpy.array): average power spectrum of the observation
    noise_variance(double): variance of the gaussian white noite.
    Returns:
    w(numpy.array): whitening denoising filter
    """    

    M = average_PS.size

    #reshape the average PS 
    average_PS = np.fft.ifftshift(average_PS) 
    
    #compute filter
    w = np.fft.ifft2(  truncate_non_neg( (average_PS - noise_variance**2*M*np.ones((average_PS.shape[0], average_PS.shape[0]) ) ) / average_PS ) / np.sqrt(average_PS) )
    
    #reshape the average PS 
    average_PS = np.fft.fftshift(average_PS) 


    return   np.fft.fftshift( w.real )  


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

    #load data 
    data = io_image_data.readH5(dataset_file_name, 'images')
    #recover image size
    database_size , l , L = data.shape

    #samples
    samples = []

    for i in range(number_of_samples) : 

    	if i == database_size : 
    		break 

    	#choose position
    	top_left_corner =  image_data_analysis.get_sample_top_left_corner(0,l - sample_size[0] ,0 , L - sample_size[1] )
    	#get sample
    	sample = image_data_analysis.get_sample_image(data[i], sample_size, top_left_corner)    
    	samples.append(sample)

    return np.array(samples)



def get_line_index(dSize):
    """
    Cette fonction permet d'obtenir l'ordre de parcours des pixels d'une image carrée selon un parcours ligne par ligne
    :param dSize: largeur ou longueur de l'image en pixel (peu importe, la fonction ne fonctionne qu'avec des images carrées)
    :return: une liste de taille 2*dSize*dSize qui correspond aux coordonnées de chaque pixel ordonnée selon le parcours ligne par ligne
    """
    return [a.flatten() for a in np.indices((dSize, dSize))]

  

def line_transform_img(img):
    """
    Cette fonction prend une image carrée en entrée, et retourne l'image applatie (1 dimension) selon le parcours ligne par ligne
    :param img: une image (donc un numpy array 2 dimensions)
    :return: un numpy array 1 dimension
    """
    assert img.shape[0] == img.shape[1], 'veuillez donner une image carrée en entrée'
    idx = get_line_index(img.shape[0])
    return img[idx[0], idx[1]]


def pre_process(X):
    """Function that preprocess the data to be fed to the ICA algorithm
    Args:
    X(numpy array): input to be preprocessed
    Returns:
    X(numpy.array)
    """

    X_new = []
    for sample in X : 
    	x = line_transform_img(sample)
    	x = x - x.mean()    	
    	X_new.append( x )

    return np.array(X_new)

    
def get_IC(X):
    """Function that estimates the independent components of the data
    Args:
    X(numpy.array):preprocessed data
    Returns:
    W(numpy.array) the matrix of the independent sources of the data
    """

    
    W = FastICA( algorithm='parallel' , max_iter=300, n_components=10).fit(X)     
    return W.components_

    
   
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


