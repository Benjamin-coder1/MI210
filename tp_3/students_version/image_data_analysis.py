import numpy as np
import scipy.fftpack
import os
import pylab
import h5py
import io_image_data
import random 

def get_sample_top_left_corner(iMin,iMax,jMin,jMax ):
	""" Function that genereates randomly a position between i,j intervals [iMin,iMax], [jMin,jMax]
	Args:
		iMin (int): the i minimum coordinate (i is the column-position of an array)
		iMax (int): the i maximum coordinate (i is the column-position of an array)
		jMin (int): the j minimum coordinate (j is the row-position of an array)
		jMax (int): the j maximum coordinate (j is the row-position of an array)
	Returns:
		[i,j] (tuple(int,int)): random integers such iMin<=i<iMax,jMin<=j<jMax,
	""" 

	i = random.randint(iMin, iMax - 1)
	j = random.randint(jMin, jMax - 1)

	return (i,j)

def get_sample_image(image, sample_size, top_left_corner):
	""" Function that extracts a sample of an image with a given size and a given position
	Args:
		image (numpy.array) : input image to be sampled
		sample_size (tuple(int,int)): size of the sample
		top_left_corner (tuple(int,int)): positon of the top left corner of the sample within the image
	Returns:
		sample (numpy.array): image sample
	""" 

	i,j = top_left_corner
	return image[ i:i+sample_size[0], j:j+sample_size[1]]



def get_sample_PS(sample):
	""" Function that calculates the power spectrum of a image sample
	Args:
		sample (numpy.array): image sample
	Returns:
		sample_PS (numpy.array): power spectrum of the sample. The axis are shifted such the low frequencies are in the center of the array (see scipy.ffpack.fftshift)
	""" 

	#calculation fourier transformation + centrage 
	Sample = np.fft.fftshift( np.fft.fft2(sample) ) 
	#calculation of power spectrum
	PS = np.power(abs(Sample), 2)

	return PS

def get_average_PS(input_file_name, sample_size, number_of_samples):
	""" Function that estimates the average power spectrum of a image database
	Args:
		input_file_name (str) : Absolute pathway to the image database stored in the hdf5
		sample_size (tuple(int,int)): size of the samples that are extrated from the images
		number_of_samples (int): number of image samples to consider in calculating the average
	Returns:
		average_PS (numpy.array): average power spectrum of the database samples. The axis are shifted such the low frequencies are in the center of the array (see scipy.ffpack.fftshift)
	"""

	#load data 
	data = io_image_data.readH5(input_file_name, 'images')
	#recover image size
	database_size , l , L = data.shape

	PS_mean = np.zeros( (sample_size[0],sample_size[1]) )

	for i in range(number_of_samples) : 

		if i == database_size : 
			break 

		#choose position
		top_left_corner =  get_sample_top_left_corner(0,l - sample_size[0] ,0 , L - sample_size[1] )
		#get sample
		sample = get_sample_image(data[i], sample_size, top_left_corner)    
		#compute PS
		PS_mean += get_sample_PS( sample )

		 

	return  PS_mean/number_of_samples


 

def get_radial_freq(PS_size):
	""" Function that returns the Discrete Fourier Transform radial frequencies
	Args:
		PS_size (tuple(int,int)): the size of the window to calculate the frequencies
	Returns:
		radial_freq (numpy.array): radial frequencies in crescent order
	"""
	fx = np.fft.fftshift(np.fft.fftfreq(PS_size[0], 1./PS_size[0]));
	fy = np.fft.fftshift(np.fft.fftfreq(PS_size[1], 1./PS_size[1]));
	[X,Y] = np.meshgrid(fx,fy);
	R = np.sqrt(X**2+Y**2);
	radial_freq = np.unique(R);
	radial_freq.sort()
	return radial_freq[radial_freq!=0]


def compute_mean_circle(X, radius) : 
	"""
	Cette fonction calcul la valeur moyenne surfacique du PS sur un cercle 
	"""

	l,L = ( X.shape[0] , X.shape[1] )
	somme = 0
	for i in range(X.shape[0]) : 
		for j in range(X.shape[1]) : 
			if np.sqrt((i - l/2)**2 + (j - L/2)**2) <= radius : 
				somme += X[i][j]

	return somme / ( np.pi*radius**2)

#vectorization for performing 
compute_mean_circle = np.vectorize(compute_mean_circle)
compute_mean_circle.excluded.add(0)


def get_radial_PS(average_PS):
	""" Function that estimates the average power radial spectrum of a image database
	Args:
		average_PS (numpy.array) : average power spectrum of the database samples.
	Returns:
		average_PS_radial (numpy.array): average radial power spectrum of the database samples.
	""" 

	radial_freq = get_radial_freq(average_PS.shape) 
	return radial_freq, compute_mean_circle(average_PS, radial_freq)


def decoupe_image(image, grid_size ) : 
	""" 
	Allow to cut a picturewith grid size 
	"""

	l,L = image.shape
	l_n, L_n = int(l/grid_size[0]) , int(L/grid_size[1])
	images = []

	for i in range(grid_size[0]) : 
		for j in range(grid_size[1]) : 
			images.append( image[i*l_n:(i+1)*l_n, j*L_n:(j+1)*L_n ]    )


	return images





def get_average_PS_local(input_file_name, sample_size, grid_size, number_of_samples):
	""" Function that estimates the local average power spectrum of a image database
	Args:
		input_file_name (str) : Absolute pathway to the image database
		sample_size (tuple(int,int)): size of the samples that are extrated from the images
		grid_size (tuple(int,int)): size of the grid that define the borders of each local region
		number_of_samples (int): number of image samples to consider in calculating the average
	Returns:
		average_PS_local (numpy.array): average power spectrum of the database samples. The axis are shifted such the low frequencies are in the center of the array (see scipy.ffpack.fftshift)
	""" 
	
	#load data 
	data = io_image_data.readH5(input_file_name, 'images')
	#recover image size
	database_size , l , L = data.shape
	l, L = int(l/grid_size[0]), int(L/grid_size[1])

	PS_mean = np.zeros( (grid_size[0]*grid_size[1], sample_size[0],sample_size[1]) )


	for i in range(number_of_samples) : 

		if i == database_size : 
			break 

		#l'image 
		images = decoupe_image( data[i], grid_size ) 

		for k in range(len(images)) : 
			#choose position
			top_left_corner =  get_sample_top_left_corner(0,l - sample_size[0] , 0 , L - sample_size[1] )
			#get sample
			sample = get_sample_image(images[k], sample_size, top_left_corner)    
			#compute PS
			PS_mean[k] += get_sample_PS( sample )


	for k in range(len(images)) :
		PS_mean[k] /= number_of_samples 

 
	return  PS_mean



def make_average_PS_figure(average_PS):
	""" Function that makes and save the figure with the power spectrum
	Args:
		average_PS (numpy.array): the average power spectrum in an array of shape [sampleShape[0],sampleShape[1]]
	""" 
	pylab.figure()
	pylab.imshow(np.log(average_PS),cmap = "gray")
	pylab.contour(np.log(average_PS))
	pylab.axis("off")
  #  pylab.savefig(figure_file_name)

def make_average_PS_radial_figure(radial_freq,average_PS_radial):
	""" Function that makes and save the figure with the power spectrum
	Args:
		average_PS (numpy.array) : the average power spectrum
		average_PS_radial (numpy.array): the average radial power spectrum
	""" 
	pylab.figure()
	pylab.loglog(radial_freq,average_PS_radial,'.')
	pylab.xlabel("Frequecy")
	pylab.ylabel("Radial Power Spectrum")
	
	


def make_average_PS_local_figure(average_PS_local,grid_size):
	""" Function that makes and save the figure with the local power spectrum
	Args:
		average_PS_local (numpy.array): the average power spectrum in an array of shape [grid_size[0],grid_size[1],sampleShape[0],sampleShape[1]
		grid_size (tuple): size of the grid
	""" 
	pylab.figure()
	for i in range(grid_size[0]):
		for j in range(grid_size[1]):
			pylab.subplot(grid_size[0],grid_size[1],i*grid_size[1]+j+1)
			pylab.imshow(np.log(average_PS_local[grid_size[0]*i + j]),cmap = "gray")
			pylab.contour(np.log(average_PS_local[grid_size[0]*i + j]))
			pylab.axis("off")


def make_average_PS_local_figure(average_PS_local,grid_size):
	""" Function that makes and save the figure with the local power spectrum
	Args:
		average_PS_local (numpy.array): the average power spectrum in an array of shape [grid_size[0],grid_size[1],sampleShape[0],sampleShape[1]
		grid_size (tuple): size of the grid
	""" 
	pylab.figure()
	for i in range(grid_size[0]):
		for j in range(grid_size[1]):
			pylab.subplot(grid_size[0],grid_size[1],i*grid_size[1]+j+1)
			pylab.imshow(np.log(average_PS_local[grid_size[0]*i + j]),cmap = "gray")
			pylab.contour(np.log(average_PS_local[grid_size[0]*i + j]))
			pylab.axis("off")


def dispay_picture(N1, N2) : 
	"""
	Function that display the N first pictures of the dataset 
	"""

	pylab.figure()
	#load data 
	data = io_image_data.readH5('domestic_robot.hdf5', 'images')

	for i in range(N1):
		for j in range(N2):
			pylab.subplot(N1,N2,i*N2+j+1)
			pylab.imshow(data[i])
			pylab.axis("off")


		








