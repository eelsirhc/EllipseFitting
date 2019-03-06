import numpy as np
from .. import utils


def direct_ellipse_fit(data):
    """This code is an implementation of the following paper

        R. Halif and J. Flusser
        Numerically stable direct least squares fitting of ellipses
        Proc. 6th International Conference in Central Europe on Computer Graphics
        and Visualization. WSCG '98
        Czech Republic,125--132, feb, 1998.
        """

    x = data[0]
    y = data[1]

    D1 = np.stack([x**2, x*y, y**2]).T # quadratic part of the design matrix
    D2 = np.stack([x, y, np.ones(x.size)]).T # linear part of the design matrix
    S1 = np.dot(D1.T, D1) # quadratic part of the scatter matrix
    S2 = np.dot(D1.T, D2) # combined part of the scatter matrix
    S3 = np.dot(D2.T, D2) # linear part of the scatter matrix
    T = - np.dot(np.linalg.inv(S3), S2.T) # for getting a2 from a1
    M = S1 + np.dot(S2, T) # reduce scatter matrix
    M = np.stack([M[2] / 2, - M[1], M[0] / 2]) # premultiply by inv(C1)
    evalue, evec = np.linalg.eig(M)       # solve eigensystem
    cond = 4 * evec[0] * evec[2] - evec[1]**2 # evaluate a'Ca
    fw = np.where(cond > 0)[0][0]
    al = np.real(evec[:, fw])  # eigenvector for min. pos. eigenvalue
    if np.max(np.imag(evec[:, fw])) > 1e-6:
        raise ValueError("Complex eigenvector")

    a = np.hstack([al, np.dot(T, al)])              # ellipse coefficients
    a = a/np.linalg.norm(a)
    return a

def compute_directellipse_estimates(data):
    """
    This function is a wrapper for the numerically stable direct ellipse
    fit due to 
    
    R. Halif and J. Flusser
    "Numerically stable direct least squares fitting of ellipses"
    Proc. 6th International Conference in Central Europe on Computer 
    Graphics and Visualization. WSCG '98 Czech Republic,125--132, feb, 1998
    
    which is a modificaiton of
    
    A. W. Fitzgibbon, M. Pilu, R. B. Fisher
    "Direct Least Squares Fitting of Ellipses"
    IEEE Trans. PAMI, Vol. 21, pages 476-480 (1999)

    Parameters:
        data - Nx2 matrix where N is the number of data points
    Returns:
        A length 6 tuple (a,b,c,d,e,f) representing the parameters of the equation
            ax^2 + bxy + cy^2 + dx + ey + f = 0
    
        with the constraint that b^2 - 4ac < 0 (an ellipse)

    """

    nPoints = data.shape[0]

    # Scale and translate data to lie inside a unit box

    normalized_points, T = utils.normalize_data_isotropically(data)
    normalized_points = np.vstack([normalized_points, np.ones(nPoints)])

    theta = direct_ellipse_fit(normalized_points)
    theta = theta / np.linalg.norm(theta)

    a, b, c, d, e, f = theta
    C = np.array([[a, b/2, d/2], [b/2, c, e/2], [d/2, e/2, f]])
    #denormalize

    C = np.dot(T.T, np.dot(C, T))
    
    theta = (C[0, 0], C[1, 0]*2, C[1, 1], C[2, 0]*2, C[2, 1]*2, C[2, 2])
    
    theta = theta / np.linalg.norm(theta)
    return theta