import numpy as np
from numpy import abs, diag, kron, eye
from numpy.linalg import norm, inv, pinv, svd
from .. import direct
from .. import utils
from .common import *
def NoiseLevelLoop(parameters,dataPts, covList):
    t = parameters
    t = t / norm(t)
    numberOfPoints = len(dataPts)
    M = zeros((6,6))
    aml = 0
    for i in range(numberOfPoints):
        m = dataPts[i]
    

        # transformed data point
        ux_i = np.array([m[0]**2, m[0]*m[1], m[1]**2, m[0], m[1], 1]).reshape(6,1)
        # derivative of transformed data point
        dux_i =np.array([[2*m[0], m[1], 0, 1, 0, 0],
                         [0, m[0], 2*m[1], 0, 1, 0]]).T

        # outer product
        A = ux_i @ ux_i.T

        # covariance matrix of the ith data pont
        covX_i = covList[i] 
        B = dux_i @ covX_i @ dux_i.T
        tAt = t.T @ A @ t
        tBt = t.T @ B @ t
        aml = aml + abs(tAt/tBt)
        M = M + A /tBt
    return M, aml, t

def estimateNoiseLevel(algebraicEllipseParameters,dataPts, covList):
    M,aml,t = NoiseLevelLoop(algebraicEllipseParameters,dataPts, covList)
    
    numberOfPoints = len(dataPts)
    # estimate of the noise level based on average residual
    sigma_squared = aml / (numberOfPoints-5)
    return sigma_squared


def compute_confidence_band(xv,yv,tNormalisedSpace,covNormalisedSpace,criticalValue):
    """% Author: Zygmunt L. Szpak (zygmunt.szpak@gmail.com)
     Date: September 2014
    
     Description:  This procedure samples point over a regular grid and tests 
                   each point to see if it could plausibly have originated
                   from the ellipse parameterised by 'tNormalisedSpace'
                   with covariance matrix 'covNormalisedSpace'.
                   The test cut-off value is specified by 'criticalValue'. 
                   Points that pass the test form the confidence region for
                   the ellipse.
                   
    
                  
    
     Parameters :  xv                        -  vector specifing x-coordinates
                                                that will be used to form
                                                a grid of 2D coordinates.
    
                   yv                        -  vector specifing y-coordinates
                                                that will be used to form a
                                                grid of 2D coordinates.
    
                   criticalValue             -  a threshold on the Chi-Squared
                                                value associated with
                                                a chosen level of confidence
                                                and p-value with 5 degrees
                                                of freedom. Refer to
                                                http://en.wikipedia.org/wiki/
                                                Chi-squared_distribution
                                                for more details.
                                                
    
    
    
    
     Return     : a grid (represented as a matrix) with a value of 0
                  if a point could have originated from the ellipse
                  and a value of 1 otherwise. 
               
    
    
     Example of Usage:
    
    
    
     Credit:
    
    
     Last Modified:
     19/03/2014
     """

    Xinterp,Yinterp = np.meshgrid(xv,yv)
    Z = np.ones_like(Xinterp)#len(xv),len(yv))
    for i in range(len(xv)):
      for j in range(len(yv)):       
        x = Xinterp[i,j]
        y = Yinterp[i,j]        

        tNormalisedSpace = tNormalisedSpace /  norm(tNormalisedSpace);
       
        ux = np.array([x**2, x*y, y**2, x, y, 1]).reshape(6,1)
        zSquared = tNormalisedSpace.T @(ux @ ux.T) @ tNormalisedSpace
        var = ux.T @ covNormalisedSpace @ ux
        
        if (zSquared/var < criticalValue):
            Z[i,j] = 0
        disp=print
    return Z



def compute_covariance_of_sampson_estimate(algebraicEllipseParameters, dataPts, covList=None):
    """ function  [covarianceMatrix, covarianceMatrixNormalisedSpace]  = ...
             compute_covariance_of_sampson_estimate(...
                       algebraicEllipseParameters, dataPts, covList)
    
    
     Author: Zygmunt L. Szpak (zygmunt.szpak@gmail.com)
     Date: February 2013
    
     Description: This procedure takes as input an estimated of ellipse 
                  parameters [a b c d e f] associated with the equation
    
                            a x^2 + b x y + c y^2 + d x + e y + f = 0
    
                  where it is assumed that the parameters were estimated
                  using the approximate maximum likelihood (Sampson distance).
                  
                   The function also takes as input the data points on which
                   the ellipse parameters were estimated, and optionally
                   a list of covariance matrices for the data points. If the
                   list of data point covariance matrices is not specified,
                   then it is assumed that the data points were corrupted
                   by isotropic homogeneous Gaussian noise, in which case
                   the noise level is estimated from the data points. 
    
                   This function return a covariance matrix for the estimated
                   algebraic parameters in the original data space. It also
                   returns the covariance matrix in a normalised coordinate
                   system because the covariance matrix in the original 
                   coordinate system usually consists of tiny numbers.
                   Both covariance matrices encode the same information,
                   but the covariance matrix in the normalised coordinate
                   system produces values that lie in a more sensible range
                   and is numerically more useful for subsequent analysis. 
                   
    
                  
    
     Parameters :  algebraicEllipseParameters -  an estimate of the algebraic
                                                 ellipse parameters based on
                                                 the Sampson cost function in 
                                                 the original (unnormalised) 
                                                 data space.

                   dataPts                   -  an nPoints x 2 matrix of 
                                                coordinates
    
                   covList                   - a list of N 2x2 covariance 
                                               matrices representing the
                                               uncertainty of the coordinates
                                               of each data point.
                                               if this parameter is not 
                                               specified then  default 
                                               isotropic  (diagonal)and 
                                               homogeneous (same noise level 
                                               for each data point) covariance
                                               matrices are assumed, and the
                                               noise level is estimated from
                                               the data points
    
    
    
     Return     : covarianceMatrix summarising the uncertainty of the 
                  algebraic parameter estimate associated with the Sampson
                  cost function (aka approximate maximum likelihood cost)
               
    
    
     Example of Usage:
    
    
    
     Credit:
    
    
     Last Modified:
     19/03/2014
    """

    nPts = len(dataPts)

    # Check to see if the user passed in their own list of covariance matrices
    if covList is None:
        # Generate a list of diagonal covariance matrices
        covList = covList = [np.eye(2) for i in range(nPts)]
        sigma_squared = estimateNoiseLevel(algebraicEllipseParameters,
                                                  dataPts, covList)

        # ensure that the isotropic covariance matrices are scaled with
        # an estimate of the noise level
        for iPts in range(nPts):
          covList[iPts] = covList[iPts] * sigma_squared

    # the algebraicEllipseParameters were computed in a hartley normalised
    # coordinate system so in order to correctly characterise the uncertainty
    # of the estimate, we need to know the transformation matrix T that maps
    # between the original coordinate system and the hartley normalised
    # coordinate system.
    #
    # scale and translate data points so that they lie inside a unit box


    dataPts, T = utils.normalize_data_isotropically(dataPts)
    dataPts=dataPts.T

    # transfer initialParameters to normalized coordinate system
    # the formula appears in the paper Z.Szpak, W. Chojnacki and A. van den
    # Hengel, "A comparison of ellipse fitting methods and implications for
    # multiple view geometry", Digital Image Computing Techniques and
    # Applications, Dec 2012, pp 1--8
    algebraicEllipseParameters = (algebraicEllipseParameters / 
                                           norm(algebraicEllipseParameters))
    algebraicEllipseParametersNormalisedSpace = utils.normalize_coordinate_system(T, algebraicEllipseParameters)
    # Becase the data points are now in a new normalised coordinate system,
    # the data covariance matrices also need to be tranformed into the 
    # new normalised coordinate system. The transformation of the covariance
    # matrices into the new coordinate system can be achieved by embedding the
    # covariance matrices in a 3x3 matrix (by padding the 2x2 covariance
    # matrices by zeros) and by  multiply the covariance matrices by the 
    # matrix T from the left and T' from the right. 
    normalised_CovList = utils.normalize_cov_list(covList,T)

    M,aml,t = NoiseLevelLoop(algebraicEllipseParametersNormalisedSpace,dataPts, normalised_CovList)
    
    Pt = eye(6) - np.outer(t,t)/norm(t)**2
    # compute rank-5 constrained pseudo-inverse of M
    U, D, V = svd(M)

    for i in range(5):
        D[i] = 1/D[i]

    D[5] = 0
    pinvM = V.T @ diag(D) @ U.T
    
    covarianceMatrixNormalisedSpace = Pt @ pinvM @ Pt
    # transform covariance matrix from normalised coordinate system
    # back to the original coordinate system                         
    E,P34,D3 = utils.epd()
    F = inv(E) @ P34 @ pinv(D3) @ kron(T,T).T @ D3 @ P34 @ E    
    t = F @ algebraicEllipseParametersNormalisedSpace
    P = eye(6) - np.outer(t,t)/norm(t)**2;
    covarianceMatrix = norm(t)**(-2) * P @ F @ covarianceMatrixNormalisedSpace @ F.T @ P
    return covarianceMatrix, covarianceMatrixNormalisedSpace


def compute_covariance_of_geometric_parameters(algebraicEllipseParameters, dataPts, covList=None):
    """ function geometricCovarianceMatrix = ...
               compute_covariance_of_geometric_parameters(...
                                 algebraicEllipseParameters, dataPts, covList)
    
    
     Author: Zygmunt L. Szpak (zygmunt.szpak@gmail.com)
     Date: February 2013
    
                  This procedure takes as input an estimated of ellipse 
                  parameters [a b c d e f] associated with the equation
    
                            a x^2 + b x y + c y^2 + d x + e y + f = 0
    
                  where it is assumed that the parameters were estimated
                  using the approximate maximum likelihood (Sampson distance).
                  
                  The function also takes as input the data points on which
                  the ellipse parameters were estimated, and optionally
                  a list of covariance matrices for the data points. If the
                  list of data point covariance matrices is not specified,
                  then it is assumed that the data points were corrupted
                  by isotropic homogeneous Gaussian noise, in which case
                  the noise level is estimated from the data points. 
    
                  This function return a covariance matrix for geometric
                  ellipse parameters in the original data space.
    
                  Note than when the ellipse as actually a circle the 
                  covariance of the orientation becomes undefined.
                   
    
                  
    
     Parameters :  algebraicEllipseParameters -  an estimate of the algebraic
                                                 ellipse parameters based on
                                                 the Sampson cost function in 
                                                 the original (unnormalised) 
                                                 data space.

                   dataPts                   -  an nPoints x 2 matrix of 
                                                coordinates
    
                   covList                   - a list of N 2x2 covariance 
                                               matrices representing the
                                               uncertainty of the coordinates
                                               of each data point.
                                               if this parameter is not 
                                               specified then  default 
                                               isotropic  (diagonal)and 
                                               homogeneous (same noise level 
                                               for each data point) covariance
                                               matrices are assumed, and the
                                               noise level is estimated from
                                               the data points
    
    
    
     Return     : covarianceMatrix summarising the uncertainty of  
                  geometric ellipse parameters associated with the Sampson
                  cost function (aka approximate maximum likelihood cost)
    
                  the diagonal of the covariance matrix containts the
                  uncertainty associated with the major axis, minor axis,
                  x-centre, y-centre and orientation (in radians)
                  respectively.
               
    
    
     Example of Usage:
    
    
    
     Credit:
    
    
     Last Modified:
    """
    covarianceMatrix, thetaCovarianceMatrixNormalisedSpace = \
              compute_covariance_of_sampson_estimate(algebraicEllipseParameters,dataPts, covList)
    # the algebraicEllipseParameters were computed in a hartley normalised
    # coordinate system so in order to correctly characterise the uncertainty
    # of the estimate, we need to know the transformation matrix T that maps
    # between the original coordinate system and the hartley normalised
    # coordinate system.
    #
    # scale and translate data points so that they lie inside a unit box
    scaledData, T = utils.normalize_data_isotropically(dataPts);

    # extract isotropic scaling factor from matrix T which we will require
    # in the transformation of the geometric parameter covariance matrix
    # from a normalised coordinate system to the original data space
    s = T[0,0]**-1

    # transfer initialParameters to normalized coordinate system
    # the formula appears in the paper Z.Szpak, W. Chojnacki and A. van den
    # Hengel, "A comparison of ellipse fitting methods and implications for
    # multiple view geometry", Digital Image Computing Techniques and
    # Applications, Dec 2012, pp 1--8
    algebraicEllipseParameters = (algebraicEllipseParameters /
                                          norm(algebraicEllipseParameters))

    algebraicEllipseParametersNormalisedSpace = utils.normalize_coordinate_system(T, algebraicEllipseParameters)
                         
  
    a,b,c,d,e,f = algebraicEllipseParametersNormalisedSpace
    
    # various computations needed to build the jacobian matrix
    # of the transfromation from theta (algebraic parameters) to
    # eta (geometric parameters)
    delta = b**2 - 4*a*c
    lambdaPlus = 0.5*(a + c - (b**2 + (a - c)**2)**0.5)
    lambdaMinus = 0.5*(a + c + (b**2 + (a - c)**2)**0.5)

    psi = b*d*e - a*e**2 - b**2*f + c*(4*a*f - d**2);
    Vplus = (psi/(lambdaPlus*delta))**0.5;
    Vminus = (psi/(lambdaMinus*delta))**0.5;

    dXcenter = derivativeXcenter(a,b,c,d,e,delta)
    dYcenter = derivativeYcenter(a,b,c,d,e,delta)

    dTau = derivativeTau(a,b,c)

    dVplus = derivativeVplus(a,b,c,d,e,f,psi,lambdaPlus,delta)
    dVminus = derivativeVminus(a,b,c,d,e,f,psi,lambdaMinus,delta)

    #A = max(Vplus,Vminus)
    #B = min(Vplus,Vminus)
    if Vplus>Vminus:
        A,B = Vplus, Vminus
        dA,dB = dVplus, dVminus
    else:
        B,A = Vplus, Vminus
        dB,dA = dVplus, dVminus
    # jacobian matrix of the transformation from theta to eta (geometric)
    # parameters
    #etaDtheta = [dA';dB';dXcenter';dYcenter';dTau'];
    etaDtheta = np.hstack([dA,dB,dXcenter,dYcenter,dTau]).T

    # propogate uncertainty from the algebraic parameter space (theta)
    # to the geometric parameter space (eta) in a normalised coordinate
    # system for maximum numerical accuracy
    etaCovarianceMatrixNormalisedSpace =(
                  etaDtheta @ thetaCovarianceMatrixNormalisedSpace @ etaDtheta.T)
    # apply denormalisation step to determine geometric parameter covariance
    # matrix in the original data space
    denormalisationMatrix = diag([s,s,s,s,1]);
    etaCovarianceMatrix =(
        denormalisationMatrix @ etaCovarianceMatrixNormalisedSpace @
                                                denormalisationMatrix.T)


    geometricCovarianceMatrix = etaCovarianceMatrix;
    return geometricCovarianceMatrix
