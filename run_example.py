# Z. L. Szpak, W. Chojnacki, and A. van den Hengel. 
# Guaranteed ellipse fitting with a confidence region and an uncertainty
# measure for centre, axes, and orientation. 
# J. Math. Imaging Vision, 2015. 
# http://dx.doi.org/10.1007/s10851-014-0536-x

import numpy as np
import colorama
colorama.init()
from colorama import Fore, Style
from ellipses import direct, fge, cov, utils
from ellipses.mpl_support import *
# noisy data points sampled from an ellipse
data_points = np.loadtxt("sample_data/sample_ellipse.csv", delimiter=',')
data_points = data_points[10:40:4]

#add noise
covList = []
for iPts in range(len(data_points)):
    standardDevX = 1 + (10-1)*np.random.rand()
    standardDevY = 1 + (10-1)*np.random.rand()
    covList.append( np.diag([standardDevX**2, standardDevY**2]) )
    perturbationX = standardDevX * np.random.randn();
    perturbationY = standardDevY * np.random.randn();
    data_points[iPts] =\
        data_points[iPts] + [perturbationX, perturbationY];   

# Example with ALL data points
# An example of fitting to all the data points
fprintf('**************************************************************\n')
fprintf('* Example with ALL data points assuming homogeneous Gaussian *\n')
fprintf('* noise with the noise level automatically estimated from    *\n')
fprintf('* data points                                                *\n')
fprintf('**************************************************************\n')

fprintf('Algebraic ellipse parameters of direct ellipse fit: \n')
print(f"{Fore.GREEN} Trying {Style.RESET_ALL}")
theta_dir  = direct.compute_directellipse_estimates(data_points)
print(theta_dir)

fprintf('Algebraic ellipse parameters of our method: \n')
print(f"{Fore.GREEN} Trying {Style.RESET_ALL}")
theta_fastguaranteed = fge.fast_guaranteed_ellipse_estimate(data_points, covList)

fprintf('Geometric ellipse parameters \n')
fprintf('(majAxis, minAxis, xCenter,yCenter, orientation (radians)): \n')
geometricEllipseParameters = utils.fromAlgebraicToGeometricParameters(theta_fastguaranteed)
print(geometricEllipseParameters)
print(theta_fastguaranteed)
fprintf('Covariance matrix of geometric parameters: \n')
geoCov =  cov.compute_covariance_of_geometric_parameters(
                               theta_fastguaranteed, data_points)

fprintf('Standard deviation of geometric parameters: \n')                          
stds = np.sqrt(np.diag(geoCov)) 
print(stds) 
 
S, thetaCovarianceMatrixNormalisedSpace = cov.compute_covariance_of_sampson_estimate(
                      theta_fastguaranteed, data_points)

# plot the data points
x = data_points;
n = len(x);
import matplotlib.pyplot as plt

# determine data range
plt.plot(x[:,0],x[:,1],'k.')
minX,minY = np.min(x,axis=0) - 20
maxX,maxY = np.max(x,axis=0) + 20

# the particular way we plot confidence regions in matlab
# requires that we work with square axes
minX = -100;
maxX = 600;
minY = -100;
maxY = 600;

# plot the direct ellipse fit
a,b,c,d,e,f = theta_dir

xres=700;
yres=700;
xv = np.linspace(minX, maxX, xres);
yv = np.linspace(minY, maxY, yres);
x,y = np.meshgrid(xv,yv)
fh = lambda x,y : a*x**2 + b*x*y + c*y**2 + d*x + e*y + f
z=fh(x,y)
plt.contour(xv,yv,z,[0],colors='r')


#plot the guaranteed ellipse fit
a,b,c,d,e,f = theta_fastguaranteed
fh = lambda x,y : a*x**2 + b*x*y + c*y**2 + d*x + e*y + f
z=fh(x,y)
plt.contour(xv,yv,z,[0],colors='g',ls=":")

                                             

# scale and translate data points so that they lie inside a unit box
# in this instance we are actually just after the transformation matrix T
dummy, T = utils.normalize_data_isotropically(data_points);
theta_guaranteed_covweightedNormalised = utils.normalize_coordinate_system(T,theta_fastguaranteed)

# now we can transform our grid points to normalised space
pts = np.vstack([xv, yv, np.ones(xres)])
pts_normalised_space = T@pts
xv_normalised = pts_normalised_space[0]
yv_normalised = pts_normalised_space[1]
disp=print

#Z = cov.compute_confidence_band(xv_normalised,yv_normalised,
#                          theta_guaranteed_covweightedNormalised,
#                          thetaCovarianceMatrixNormalisedSpace,11.07)
#
#plt.figure()
#plt.imshow(Z)
plt.savefig('example.pdf')
#CL fprintf("l228\n")
#CL 
#CL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#CL 

