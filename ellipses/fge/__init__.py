import numpy as np
from .. import direct
from .. import utils
import dotmap
from numpy.linalg import norm, inv, pinv, det
from numpy import kron, sign

def fastLevenbergMarquardtStep(struct, rho):
    """%   Function: fastLevenbergMarquardtStep
    %
    %   This function is used in the main loop of guaranteedEllipseFit in the
    %   process of minimizing an approximate maximum likelihood cost function 
    %   of an ellipse fit to data.  It computes an update for the parameters
    %   representing the ellipse, using the method of Levenberg-Marquardt for
    %   non-linear optimisation. 
    %   See: http://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
    %
    %   However, unlike the traditional LevenbergMarquardt step, we do not
    %   add a multiple of the identity matrix to the approximate Hessian,
    %   but instead a different positive semi-definite matrix. Our choice
    %   particular choice of the different matrix corresponds to the 
    %   gradient descent direction in the theta coordinate system, 
    %   transformed to the eta coordinate system. We found empirically
    %   that taking steps according to the theta coordinate system
    %   instead of the eta coordinate system lead to faster convergence.
    %
    %   Parameters:
    %
    %      struct     - a data structure containing various parameters
    %                   needed for the optimisation process.  
    %
    %   Returns: 
    %
    %     the same data structure 'struct', except that relevant fields have
    %     been updated
    %
    %   See Also: 
    %
    %    fastGuaranteedEllipseFit
    %
    %  Zygmunt L. Szpak (c) 2014
    %  Last modified 18/3/2014 
    """
 
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # extract variables from data structure                              %
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    jacobian_matrix = struct.jacobian_matrix
    r = struct.r
    lam = struct.lam
    delta = struct.delta[struct.k]
    damping_multiplier = struct.damping_multiplier
    damping_divisor = struct.damping_divisor
    current_cost = struct.cost[struct.k]
    data_points = struct.data_points
    covList = struct.covList
    numberOfPoints = struct.numberOfPoints
    H = struct.H;    
    jlp = struct.jacob_latentParameters
    eta = struct.eta[struct.k]
    
    # convert latent variables into length-6 vector (called t) representing
    # the equation of an ellipse
    t = np.array([1, 2*eta[0], eta[1]**2 +  np.abs(eta[1])**rho, eta[2],eta[3], eta[4]]).reshape(6,1).T
    # we impose unit norm constraint on theta
    t = t /norm(t)
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #    % compute two potential updates for theta based on different         %
    #    % weightings of the identity matrix.                                 %
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
    jacob = jacobian_matrix.T@r           
    DMP = (jlp.T@jlp) * lam
    update_a = - inv(H+DMP)@jacob     
    # In a similar fashion, the second potential search direction 
    # is computed
    
    DMP = (jlp.T@jlp)*lam/damping_divisor
    update_b = - inv(H+DMP)@jacob

    # the potential new parameters are then 
    eta_potential_a = eta + update_a
    eta_potential_b = eta + update_b
    #print("-----eta a b ----")
    #print(eta_potential_a)
    #print(eta_potential_b)
    # we need to convert from eta to theta and impose unit norm constraint
    t_potential_a =  np.array([1, 2*eta_potential_a[0], eta_potential_a[0]**2 +
                               np.abs(eta_potential_a[1])**rho, eta_potential_a[2],
                                  eta_potential_a[3], eta_potential_a[4]]).reshape(6,1)
    t_potential_a = t_potential_a/norm(t_potential_a);
    
    t_potential_b =   np.array([1, 2*eta_potential_b[0], eta_potential_b[0]**2 +
                       np.abs(eta_potential_b[1])**rho, eta_potential_b[2],
                           eta_potential_b[3], eta_potential_b[4]]).reshape(6,1)
    t_potential_b = t_potential_b/norm(t_potential_b)
     
    
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % compute new residuals and costs based on these updates             %
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
    # % residuals computed on data points
    cost_a = 0;
    cost_b = 0;
    for i in range(numberOfPoints):
        m = data_points[i]
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

        #print("shapes")
        #print(t_potential_a.shape, t_potential_b.shape, A.shape, B.shape)
        t_aBt_a = t_potential_a.T @ B @ t_potential_a
        t_aAt_a = t_potential_a.T @ A @ t_potential_a

        t_bBt_b = t_potential_b.T @ B @ t_potential_b
        t_bAt_b = t_potential_b.T @ A @ t_potential_b

         # AML cost for i'th data point
        cost_a = cost_a +  np.abs(t_aAt_a@inv(t_aBt_a))
        cost_b = cost_b +  np.abs(t_bAt_b@inv(t_bBt_b))   
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% determine appropriate damping and if possible select an update     %
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #print("cost")
    #print(cost_a, cost_b, current_cost)
    if (cost_a >= current_cost and cost_b >= current_cost):
        # neither update reduced the cost
        struct.eta_updated = False
        # no change in the cost
        struct.cost[struct.k+1] = current_cost
        # no change in parameters
        struct.eta[struct.k+1] = eta
        struct.t[struct.k+1] = t
        # no changes in step direction
        struct.delta[struct.k+1] = delta
        # next iteration add more Identity matrix
        struct.lam = lam * damping_multiplier
    elif (cost_b < current_cost):
       # update 'b' reduced the cost function
       struct.eta_updated = True;
       # store the new cost
       struct.cost[struct.k+1] = cost_b
       # choose update 'b'
       struct.eta[struct.k+1] = eta_potential_b
       struct.t[struct.k+1] = t_potential_b[:,0]
       # store the step direction
       struct.delta[struct.k+1] = update_b.T
       # next iteration add less Identity matrix
       struct.lam = lam / damping_divisor
    else:
       # update 'a' reduced the cost function
       struct.eta_updated = True;
       # store the new cost
       struct.cost[struct.k+1] = cost_a
       # choose update 'a'
       struct.eta[struct.k+1] = eta_potential_a
       struct.t[struct.k+1] = t_potential_a[:,0]
       # store the step direction
       struct.delta[struct.k+1] = update_a.T
       # keep the same damping for the next iteration
       struct.lam = lam;
    return struct

def fastGuaranteedEllipseFit(latentParameters,dataPts,covList):
    """Function: fastGuaranteedEllipseFit
    
       This function implements the ellipse fitting algorithm described in
       Z.Szpak, W. Chojnacki and A. van den Hengel
       "Guaranteed Ellipse Fitting with an Uncertainty Measure for Centre, 
        Axes, and Orientation"
    
    
       Parameters:
    
          latentParameters    - an initial seed for latent parameters
                                [p q r s t] which through a transformation
                                are related to parameters  [a b c d e f] 
                                associated with the conic equation  
                                
                                 a x^2 + b x y + c y^2 + d x + e y + f = 0
    
          dataPts             - a 2xN matrix where N is the number of data
                                points
    
          covList             - a list of N 2x2 covariance matrices 
                                representing the uncertainty of the 
                                coordinates of each data point.
    
    
       Returns: 
    
         a length-6 vector [a b c d e f] representing the parameters of the
         equation
        
         a x^2 + b x y + c y^2 + d x + e y + f = 0
    
         with the additional result that b^2 - 4 a c < 0.
    
       See Also: 
    
        compute_guaranteedellipse_estimates
        levenbergMarquardtStep
        lineSearchStep
    
      Zygmunt L. Szpak (c) 2014
      Last modified 18/3/2014
    """
 
    eta = latentParameters;
    # convert latent variables into length-6 vector (called t) representing
    # the equation of an ellipse
    t = np.array([1, 2*eta[0], eta[0]**2 +  abs(eta[1])**2, eta[2],eta[3], eta[4]])
    t = t / np.linalg.norm(t)

    # various variable initialisations
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # primary loop variable
    keep_going = True
    # in some case a LevenbergMarquardtStep does not decrease the cost
    # function and so the parameters (eta) are not updated
    struct=dotmap.DotMap()
    struct.eta_updated = False
    # damping parameter in LevenbergMarquadtStep
    struct.lam = 0.01
    # loop counter (matlab arrays start at index 1, not index 0)
    struct.k = 0
    # used to modify the tradeoff between gradient descent and hessian based
    # descent in LevenbergMarquadtStep
    struct.damping_multiplier = 15
    # used to modify the tradeoff between gradient descent and hessian based
    # descent in LevenbergMarquadtStep
    struct.damping_divisor = 1.2
    # number of data points
    struct.numberOfPoints =  len(dataPts)
    # data points that we are going to fit an ellipse to
    struct.data_points = dataPts
    # a list of 2x2 covariance matrices representing the uncertainty
    # in the coordinates of the data points
    struct.covList = covList

    # various parameters that determine stopping criteria
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # maximum loop iterations 
    maxIter = 200
    # step-size tolerance
    struct.tolDelta = 1e-7
    # cost tolerance
    struct.tolCost = 1e-7
    # parameter tolerance
    struct.tolEta = 1e-7
    # gradient tolerance
    struct.tolGrad = 1e-7
    # barrier tolerance (prevent ellipse from converging on parabola)
    struct.tolBar = 15.5
    # minimum allowable magnitude of conic determinant (prevent ellipse from 
    # convering on degenerate parabola (eg. two parallel lines) 
    struct.tolDet = 1e-5

    Fprim = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]])
    F = np.zeros((6,6))
    F[:3,:3] = Fprim
    I = np.eye(6)
 
    # various initial memory allocations
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # allocate space for cost of each iteration
    struct.cost = np.zeros((maxIter))
    # allocate space for the latent parameters of each iteration
    struct.eta = np.zeros((maxIter,5))
    # and for the parameters representing the ellipse equation
    struct.t = np.zeros((maxIter,6))
    # allocate space for the parameter direction of each iteration
    struct.delta = np.zeros((maxIter,5))
    # make parameter vector a unit norm vector for numerical stability
    # store the parameters associated with the first iteration
    struct.t[struct.k] = t
    struct.eta[struct.k] = eta
    # start with some random search direction (here we choose all 1's)
    # we can initialise with anything we want, so long as the norm of the
    # vector is not smaller than tolDeta. The initial search direction 
    # is not used in any way in the algorithm. 
    struct.delta[struct.k] = np.ones(5)
    # main estimation loop
    while (keep_going and struct.k < maxIter):
        #print(struct.k)
        # allocate space for residuals 
        struct.r = np.zeros(struct.numberOfPoints)
        # allocate space for the jacobian matrix based on AML component
        struct.jacobian_matrix = np.zeros((struct.numberOfPoints,5))
        # grab the current latent parameter estimates
        eta = struct.eta[struct.k];
        # convert latent variables into length-6 vector (called t) representing
        # the equation of an ellipse
        t = np.array([1, 2*eta[0], eta[0]**2 +  abs(eta[1])**2,
                     eta[2],eta[3], eta[4]]).T
  
        # jacobian matrix of the transformation from eta to theta parameters
        jacob_latentParameters =np.array(
            [[0       ,              0                  ,        0, 0, 0],
             [2       ,              0                  ,        0, 0, 0],
             [2*eta[0], 2*abs(eta[1])**(2-1)*np.sign(eta[1]),        0, 0, 0],
             [0       ,              0                  ,        1, 0, 0],
             [0       ,              0                  ,        0, 1, 0],
             [0       ,              0                  ,        0, 0, 1]])
        
        # we impose the additional constraint that theta will be unit norm
        # so we need to modify the jacobian matrix accordingly
        t=t.reshape(len(t),1)
        #print('a=',t@t.T)
        #print('b=',(np.linalg.norm(t,2)**2))
        Pt = np.eye(6) - ((t@t.T)/(np.linalg.norm(t,2)**2));
        jacob_latentParameters = (1/np.linalg.norm(t,2))*Pt@jacob_latentParameters
        # unit norm constraint
        eps = 3e-16
        t = t / np.linalg.norm(t); 
        #print("Pt=",Pt)
        #print("jac=",jacob_latentParameters)
        #print("t",t)
        # residuals computed on data points
        for i in range(struct.numberOfPoints):
            m = dataPts[i]
      #      print("m=",m)
            # transformed data point
          #  ux_i = [m(1)^2 m(1)*m(2) m(2)^2 m(1) m(2) 1]';
          #dux_i =[2*m(1) m(2) 0 1 0 0; 0 m(1) 2*m(2) 0 1 0]';
          
            ux_i = np.array([m[0]**2, m[0]*m[1], m[1]**2, m[0], m[1], 1]).reshape(1,6).T
            # derivative of transformed data point
            dux_i =np.array([[2*m[0], m[1], 0, 1, 0, 0],[0, m[0], 2*m[1], 0, 1, 0]]).reshape(2,6).T
           
            # outer product
            A = ux_i @ ux_i.T
  
            # covariance matrix of the ith data pont
            covX_i = covList[i] 
  
            B = dux_i @ covX_i @ dux_i.T
      #      print("t=",t)
      #      print("dux_i=",dux_i)
      #      print("covX_i=",covX_i)
      #      print("B=",B)
            tBt = t.T @ B @ t
            tAt = t.T @ A @ t
    
            # AML cost for i'th data point
            struct.r[i] = np.sqrt(np.abs(tAt@np.linalg.inv(tBt)))
       #     print("A=",A)
        #    print("tBt=",tBt)
            
            
            # derivative AML component
            M = (A * np.linalg.inv(tBt))
        #    print("sB=",B.shape)
        #    print("stAt=",tAt.shape)
        #    print("sinv=",np.linalg.inv(tBt.T@tBt).shape)
            Xbits = B * ((tAt @ np.linalg.inv(tBt.T@tBt)))[0]
            X = M - Xbits;
               
            # gradient for AML cost function (row vector)
            #grad = ((X*t) / sqrt((abs(tAt/tBt)+eps)))';
            grad = ((X@t) @ np.linalg.inv( np.sqrt((np.abs(tAt@np.linalg.inv(tBt))+eps)))).T
            # build up jacobian matrix
            struct.jacobian_matrix[i] = grad @ jacob_latentParameters
  
  
          # approximate Hessian matrix
        struct.H =  struct.jacobian_matrix.T@ struct.jacobian_matrix
        # sum of squares cost for the current iteration
        struct.cost[struct.k] = (struct.r.T@struct.r)
  
        struct.jacob_latentParameters =  jacob_latentParameters
  
        # use LevenbergMarquadt step to update parameters
        struct = fastLevenbergMarquardtStep(struct,2);
        #print(struct.jacobian_matrix[i])
        # Preparations for various stopping criteria tests
   
        # convert latent variables into length-6 vector (called t) representing
        # the equation of an ellipse
        eta = struct.eta[struct.k+1]
        t = np.array([1, 2*eta[0], eta[0]**2 +  np.abs(eta[1])**2, eta[2],eta[3], eta[4]]).reshape(6,1)
        t = t / norm(t);  
  
        # First criterion checks to see if discriminant approaches zero by using
        # a barrier 
        tIt = t.T @ I @ t
        tFt = t.T @ F @ t;
        barrier = (tIt*inv(tFt))
  
        # Second criterion checks to see if the determinant of conic approaches
        # zero
        t=t[:,0]
        M = np.array([[t[0] , t[1]/2 ,t[3]/2],
                      [t[1]/2 , t[2] ,t[4]/2],
                      [t[3]/2 ,t[4]/2 , t[5]]])
        #print(M.shape,t.shape)
        DeterminantConic = det(M);
  
        # Check for various stopping criteria to end the main loop
        if (min(norm(struct.eta[struct.k+1]-struct.eta[struct.k]),
                norm(struct.eta[struct.k+1]+struct.eta[struct.k])) <
                                         struct.tolEta and struct.eta_updated):
            keep_going = False
        elif (np.abs(struct.cost[struct.k] - struct.cost[struct.k+1])
                                       < struct.tolCost and struct.eta_updated):
            keep_going = False
        elif (norm(struct.delta[struct.k+1]) < 
                                        struct.tolDelta and struct.eta_updated):
            keep_going = False
        elif norm(grad) < struct.tolGrad:
            keep_going = false;
        elif (np.log(barrier) > struct.tolBar or
                                       np.abs(DeterminantConic) < struct.tolDet):
            keep_going = False
        struct.k = struct.k + 1
    iterations = struct.k
    theta = struct.t[struct.k]
    theta = theta / norm(theta)
    return theta, iterations


def fast_guaranteed_ellipse_estimate(dataPts, covList=None, return_iterations=False):
    """function estimatedParameters = fast_guaranteed_ellipse_estimate(...
                                            dataPts, covList)
     
     Author: Zygmunt L. Szpak (zygmunt.szpak@gmail.com)
     Date: March 2014
     
     Description: This procedure takes as input a matrix of two-dimensional 
                  coordinates and estimates a best fit ellipse using the
                  sampson distance between a data point and the ellipse
                  equations as the error measure. The Sampson distance is
                  an excellent approximation to the orthogonal distance for
                  small noise levels. The Sampson distance is often also
                  referred to as the approximate maximum likelihood (AML).
                  The user can specify a list of covariance matrices for the
                  data points. If the user does not specify a list of
                  covariance matrices then isotropic homogeneous Gaussian
                  noise is assumed. 
     
     Parameters : initialParameters    - initial parameters use to seed the
                                         iterative estimation process
     
                  dataPts                - an nPoints x 2 matrix of 
                                          coordinates
     
                  covList               - a list of N 2x2 covariance matrices 
                                          representing the uncertainty of the 
                                          coordinates of each data point.
                                          if this parameter is not specified 
                                          then  default isotropic  (diagonal)
                                          and homogeneous (same noise level 
                                          for each data point) covariance
                                          matrices are assumed.
     
     
     Return     : a length-6 vector containing an estimate of the ellipse
                  parameters theta = [a b c d e f] associated with the ellipse
                  equation 
     
                       a*x^2+ b * x y + c * y^2 + d * x + e*y + f = 0
               
                  with the additional result that b^2 - 4 a c < 0.
     
     Example of Usage:
     
     
     
     
     Last Modified:
     18/03/2014
     """


    nPts = len(dataPts)

    # Check to see if the user passed in their own list of covariance matrices
    if covList is None:
        # Generate a list of diagonal covariance matrices   
        #print("not implemented")
        covList = [np.eye(2) for i in range(nPts)]
  
    # estimate an initial ellipse using the direct ellipse fit method
    initialEllipseParameters   = direct.compute_directellipse_estimates(dataPts)
    #print("INITAI=",initialEllipseParameters)
    # scale and translate data points so that they lie inside a unit box
    normalizedPoints, T = utils.normalize_data_isotropically(dataPts)


    # transfer initialParameters to normalized coordinate system
    # the formula appears in the paper Z.Szpak, W. Chojnacki and A. van den
    # Hengel, "A comparison of ellipse fitting methods and implications for
    # multiple view geometry", Digital Image Computing Techniques and
    # Applications, Dec 2012, pp 1--8
    initialEllipseParameters = (initialEllipseParameters /
                                np.linalg.norm(initialEllipseParameters))

    E = np.diag([1,2**-1,1,2**-1,2**-1,1])
    #print(E)
    # permutation matrix for interchanging 3rd and 4th
    # entries of a length-6 vector
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

   # print("D3=",D3)  
   # print("E/P34=",np.linalg.inv(E) @ P34)
   # print("initial=",initialEllipseParameters)
   # print("pinv(d3)=",np.linalg.pinv(D3))
    initialEllipseParametersNormalizedSpace = (np.linalg.inv(E) @ P34 @ np.linalg.pinv(D3) @
                          np.linalg.inv(np.kron(T,T)).T@D3@P34@E@initialEllipseParameters)
   # disp=print
   # print("-a----\n")
   # disp(np.linalg.inv(E)@ P34@np.linalg.pinv(D3))
   # print("-b----\n")
   # disp(np.linalg.inv(np.kron(T,T)).T)
   # print("-c----\n")
   # disp(D3@P34)
   # print("-d----\n")
   # disp(E@initialEllipseParameters)
   # print("-e----\n")
   # print("initialEllipseParametersNormalizedSpace=", initialEllipseParametersNormalizedSpace)

    initialEllipseParametersNormalizedSpace =( initialEllipseParametersNormalizedSpace /
                                np.linalg.norm(initialEllipseParametersNormalizedSpace))
# Becase the data points are now in a new normalised coordinate system,
# the data covariance matrices also need to be tranformed into the 
# new normalised coordinate system. The transformation of the covariance
# matrices into the new coordinate system can be achieved by embedding the
# covariance matrices in a 3x3 matrix (by padding the 2x2 covariance
# matrices by zeros) and by  multiply the covariance matrices by the 
# matrix T from the left and T' from the right. 
    normalised_CovList = utils.normalize_cov_list(covList,T)

    # To guarantee an ellipse we utilise a special parameterisation which
    # by definition excludes the possiblity of a hyperbola. In theory 
    # a parabola could be estimated, but this is very unlikely because
    # an equality constraint is difficult to satisfy when there is noisy data.
    # As an extra guard to ensure that a parabolic fit is not possible we
    # terminate our algorithm when the discriminant of the conic equation
    # approaches zero. 
    #
    #
    # convert our original parameterisation to one that excludes hyperbolas
    # NB, it is assumed that the initialParameters that were passed into the
    # function do not represent a hyperbola or parabola.
    #matlab p = para(2)/(2*para(1));
    #matlab q = (para(3)/para(1) - (para(2)/(2*para(1)))^2)^(1/2);
    #matlab r = para(4) / para(1);
    #matlab s = para(5) / para(1);
    #matlab t = para(6) / para(1);
    para = initialEllipseParametersNormalizedSpace;
    p = para[1]/(2*para[0]);
    q = (para[2]/para[0] - (para[1]/(2*para[0]))**2)**(1/2);
    r = para[3] / para[0]
    s = para[4] / para[0]
    t = para[5] / para[0]

    latentParameters  = np.array([p,q,r,s,t])
    #print(latentParameters)
    ellipseParametersFinal, iterations =\
           fastGuaranteedEllipseFit(latentParameters,normalizedPoints.T,normalised_CovList)
     

    ellipseParametersFinal = ellipseParametersFinal / norm(ellipseParametersFinal)
                                     
    # convert final ellipse parameters back to the original coordinate system
#    estimatedParameters =  E \ P34*pinv(D3)*...
#                     np.kron(T,T)'*D3*P34*E*ellipseParametersFinal;


    estimatedParameters =  (inv(E) @ P34 @ pinv(D3) @
                     np.kron(T,T).T@D3@P34@E@ellipseParametersFinal)

    estimatedParameters = estimatedParameters / norm(estimatedParameters);
    estimatedParameters = estimatedParameters / sign(estimatedParameters[-1]);
    #print(estimatedParameters, iterations)
    if return_iterations:
        return estimatedParameters, iterations
    return estimatedParameters
    
#CL end

 





