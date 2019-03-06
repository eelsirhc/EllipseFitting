import numpy as np
def normalize_data_isotropically(data):
    """ Author: Zygmunt L. Szpak (zygmunt.szpak@gmail.com)
        Date: February 2013
        
        Description: This procedure takes as input a matrix of two-dimensional 
                     coordinates and normalizes the coordinates so that they
                     lie inside a unit box. 
        
        Parameters : dataPts               - an nPoints x 2 matrix of 
                                             coordinates
        
        Return     : normalizedPts         - an nPoints x 2 matrix of 
                                             coordinates which are constrained
                                             to lie inside a unit box. 
        
                   : T                     - a 3x3 affine transformation matrix 
                                             T that was used to transform the 
                                             (homogenous coordinates) of the data 
                                             points so that they lie inside a 
                                             unit box.
                             
        Citation: W. Chojnacki and M. Brookes, "On the Consistency of the
                  Normalized Eight-Point Algorithm", J Math Imaging Vis (2007)
                  28: 19-27

    """

    nPoints = data.shape[0]
    # homogenous representation of data points resulting in a 3 x nPoints
    # matrix, where the first row contains all the x-coordinates, the second
    # row contains all the y-coordinates and the last row contains the
    # homogenous coordinate 1.

    points = np.hstack([data,np.ones((nPoints,1))]).T
    
    meanX = np.mean(data[:,0])
    meanY = np.mean(data[:,1])

    #isotropic scaling factor
    s = np.sqrt(  (1/(2*nPoints))*sum((points[0] - meanX)**2 +
                                      (points[1]-meanY)**2))

    T = np.array([[1/s, 0, -meanX/s],
                  [0, 1/s, -meanY/s],
                  [0, 0, 1]])

    normalizedPts = np.dot(T, points)
    #Remove homogeneous coordinate
    
    normalizedPts = normalizedPts[:2]

    return normalizedPts, T

def epd():
    E = np.diag([1,2**-1,1,2**-1,2**-1,1])
    P34 = np.eye(6)
    P34[2:4] = P34[3:1:-1]
    #P34 = np.linalg.kron(np.diag([0,1,0]), [0 1;1 0]) + kron(diag([1,0,1]), [1 0; 0 1]);
    #print("P34=",P34)
    # 9 x 6 duplication matrix
    D3 = np.array(
         [[1, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0],
          [0, 1, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0],  
          [0, 0, 0, 0, 1, 0],
          [0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0, 1]])

    return E, P34, D3


def normalize_cov_list(covList,T):
    normalised_CovList = []
    for cov in covList:
        covX_i = np.zeros((3, 3))
        covX_i[:2,:2] = cov
        covX_i = T @ covX_i @ T.T
        # the upper-left 2x2 matrix now represents the covariance of the 
        # coordinates of the data point in the normalised coordinate system
        normalised_CovList.append( covX_i[:2, :2] )
    return normalised_CovList


def normalize_coordinate_system(T, ellipsePar):

    E,P34,D3 = epd()
    inv,kron, pinv, norm = np.linalg.inv, np.kron, np.linalg.pinv, np.linalg.norm
    ellipseParNormalized = (inv(E) @ P34 @ pinv(D3) @
                          inv(kron(T,T)).T@D3@P34@E@ellipsePar)


    ellipseParNormalized = (ellipseParNormalized / 
                                                 norm(ellipseParNormalized))

    return ellipseParNormalized


def fromAlgebraicToGeometricParameters(algebraicEllipseParameters):
    """
    Author: Zygmunt L. Szpak (zygmunt.szpak@gmail.com)
    Date: February 2013
    
    Description: This procedure takes as input a vector representing the 
                 algebraic parameters of an ellipse equation and produces 
                 equivalent geometric parameters so that the ellipse can be
                 expressed in parametric form.
                 
    
    Parameters : algebraicEllipseParameters  - a length-6 vector containing
                                               an the ellipse parameters 
                                               theta = [a b c d e f]
                                               associated with the ellipse
                                               equation,
    
                               a*x^2+ b * x y + c * y^2 + d * x + e*y + f = 0
              
                             with the additional result that b^2 - 4 a c < 0.
    
    
    
    Return     : geometricEllipseParameters - a length-5 vector of 
                                              geometrically meaningful 
                                              ellipse parameters,
    
                                 majAxis: half-length of major axis
                                 minAxis: half-length of minor axis
                                 xCenter: x-coordinate of ellipse centroid
                                 yCenter: y-coordinate of ellipse centroid
                                 tilt   : rotation of ellipse axis in radians 
                                          as measured counter-clockwise from
                                          the postive x-axis 
                                             
                 
    
    Example of Usage:
    
    
    
    
    Last Modified:
    
    """


    a,b,c,d,e,f = algebraicEllipseParameters


    delta = b**2 - 4*a*c
    lambdaPlus = 0.5*(a + c - (b**2 + (a - c)**2)**0.5)
    lambdaMinus = 0.5*(a + c + (b**2 + (a - c)**2)**0.5)

    psi = b*d*e - a*e**2 - b**2*f + c*(4*a*f - d**2)
    Vplus = (psi/(lambdaPlus*delta))**0.5
    Vminus = (psi/(lambdaMinus*delta))**0.5

    # major semi-axis
#    axisA = max(Vplus,Vminus)
    # minor semi-axis
#    axisB = min(Vplus,Vminus)

    # determine x-coordinate of ellipse centroid
    xCenter = (2*c*d - b*e)/(delta)
    yCenter = (2*a*e - b*d)/(delta)

    # angle between x-axis and major axis
    tau = 0
    acot = lambda x: np.pi/2 - np.arctan(x)
    pi = np.pi
    # determine tilt of ellipse in radians
    cat = 0
    if (Vplus >= Vminus):
        axisA, axisB = Vplus, Vminus
        if(b == 0 and a < c):
            cat=1
            tau = 0
        elif (b == 0 and a >= c):
            cat=2
            tau = 0.5*pi
        elif (b < 0 and a < c):
            cat=2
            tau = 0.5*acot((a - c)/b)
        elif (b < 0 and a == c):
            cat=4
            tau = pi*3/4
        elif (b < 0 and a > c):
            cat=5
            tau = 0.5*acot((a - c)/b) #+ pi/2
        elif (b > 0 and a < c):
            cat=6
            tau = 0.5*acot((a - c)/b) + pi/2
        elif (b > 0 and a == c):
            cat=7
            tau = pi*(1/4)
        elif (b > 0 and a > c):
            cat=8
            tau = 0.5*acot((a - c)/b) + pi/2
    elif (Vplus < Vminus):
        axisA, axisB = Vminus, Vplus
        if(b == 0 and a < c):
            cat=10
            tau = pi/2
        elif (b == 0 and a >= c):
            cat=11
            tau = 0
        elif (b < 0 and a < c):
            cat=12
            tau = 0.5*acot((a - c)/b) + pi/2
        elif (b < 0 and a == c):
            cat=13
            tau = pi*(3/4)
        elif (b < 0 and a > c):
            cat=14
            tau = 0.5*acot((a - c)/b) + pi/2
        elif (b > 0 and a < c):
            cat=15
            tau = 0.5*acot((a - c)/b) #+ pi/2
        elif (b > 0 and a == c):
            cat=16
            tau = pi*3/4
        elif (b > 0and a > c):
            cat=17
            tau = 0.5*acot((a - c)/b);

    # notation in paper
    geometricEllipseParameters = np.array([axisA, axisB, xCenter, yCenter, tau]).reshape(5, 1)

    return geometricEllipseParameters  # ,cat


