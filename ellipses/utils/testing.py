import numpy as np
def gen_ellipse(xcenter, ycenter, width, height, angle, theta=None):
    """Make an ellipse.

    Arguments
    ---------
        xcenter : horizontal position
        ycenter : vertical position
        width : size in the horizontal
        height : size in the vertical
        theta : angular positions to calculate

    Returns:
        x,y : ellipse points
        xlim,ylim : min, max limits in both axes
    """

    if theta is None:
        theta = np.arange(0, 360, 1)
    theta_ = np.deg2rad(theta)

    x = 0.5 * width * np.cos(theta_)
    y = 0.5 * height * np.sin(theta_)

    rtheta = np.deg2rad(angle)
    R = np.array([
        [np.cos(rtheta), -np.sin(rtheta)],
        [np.sin(rtheta),  np.cos(rtheta)],
    ])

    x, y = np.dot(R, np.array([x, y]))

    x += xcenter
    y += ycenter

    tlim = np.deg2rad([0, 90, 180, 270]) + np.pi/2-rtheta
    xlim = 0.5 * width * np.cos(tlim)
    ylim = 0.5 * height * np.sin(tlim)

    xlim, ylim = np.dot(R, np.array([xlim, ylim]))
    xlim = np.array([xlim.min(), xlim.max()])
    ylim = np.array([ylim.min(), ylim.max()])
    return x, y, xlim, ylim



def make_ellipse(plim=360,  # Limit on position
                 slow=10, shigh=100,  # Limit on size
                 alim=360, sigma=1,  # Limit on rotation angle
                 angle=None, width=None, height=None,  # Fixed values of angle, width, height
                 xcenter=None, ycenter=None,  # Fixed position
                 drop=False, nmax=1e5,  # Drop data, number of points to keep
                 theta_range=360,  theta_offset=0):

    # generate random locations, size, angle
    if xcenter is None:
        xcenter = np.random.randint(0, plim)

    if ycenter is None:
        ycenter = np.random.randint(0, plim)

    if width is None and height is None and angle is None:
        width = np.random.randint(slow, shigh)
        height = np.random.randint(slow, shigh)
        angle = np.random.rand()*alim
    else:
        width = width or np.random.randint(slow, shigh)
        height = height or np.random.randint(slow, shigh)
        angle = angle or np.random.rand()*alim

    if width < height:
        # swap the width and height to make the longer one width,
        # then rotation is relative to the major axis.
        width, height = height, width

    # find a random number of points
    approx_radius = np.clip(0.5*(width+height), 5, 1e5)  # guess a radius
    npoints = np.random.randint(4, min(approx_radius*6, nmax))
    # generate the random points
    theta = np.sort(np.random.rand(npoints))*theta_range + theta_offset
    theta = theta % 360

    # block out part of the angles
    thetau = np.array(theta)
    if drop:
        npoints_block = np.random.randint(2, 20)
        theta_block = np.sort(np.random.rand(npoints))*2*np.pi
        theta = np.hstack([theta[np.where((theta > b[0]) & (theta < b[1]))]
                           for b in zip(theta_block[::2], theta_block[1::2])])

    if len(theta) == 0:
        # backup just in case theta is not defined
        theta = np.linspace(0, 360, 10)

    x, y, xlim, ylim = gen_ellipse(xcenter,
                                   ycenter,
                                   width,
                                   height,
                                   angle,
                                   theta=theta)

    xn = np.round(x).astype(int) + np.random.randn(len(x))*sigma
    yn = np.round(y).astype(int) + np.random.randn(len(y))*sigma
    # extract the unique values
    xi, yi = np.array(list(set(zip(xn, yn)))).T

    return xi, yi, dict(xcenter=xcenter,
                        ycenter=ycenter,
                        width=width,
                        height=height,
                        angle=angle,
                        theta=theta, thetau=thetau,
                        xi=xi, yi=yi,
                        x=x, y=y, xlim=xlim, ylim=ylim)
