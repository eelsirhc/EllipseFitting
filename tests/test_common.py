from .context import (testing,
                     direct_ellipse_fit,
                     fromAlgebraicToGeometricParameters,
                     fast_guaranteed_ellipse_estimate)
import numpy as np

def test_generate_ellipse():
    x, y, p = testing.make_ellipse()

def test_fit_direct_simple(rtol_percent=20):
    kwargs = dict(sigma=0., xcenter=100, ycenter=200, width=20, height=10, angle=0)
    x, y, p = testing.make_ellipse(**kwargs)
    #plot_ellipse(p)
    data = np.array([x,y])
    res_dir = direct_ellipse_fit(data)
    #theta_fastguaranteed = fast_guaranteed_ellipse_estimate(data.T)
    smaj, smin, xcen, ycen, tilt = fromAlgebraicToGeometricParameters(res_dir)
    tilt*=360/(2*np.pi)
    rtol = 0.01 * rtol_percent
    assert np.isclose(smaj,0.5*kwargs["width"],rtol=rtol)
    assert np.isclose(smin,0.5*kwargs["height"],rtol=rtol)
    r = 0.5 * (smaj + smin)
    assert np.isclose(xcen, kwargs["xcenter"],atol=rtol*r)
    assert np.isclose(ycen, kwargs["ycenter"],atol=rtol*r)
    delta_angle = (tilt - kwargs["angle"])
    delta_angle = delta_angle % 180
    assert np.isclose(delta_angle, 0, atol=20)


def test_fit_fge_simple(rtol_percent=20):
    kwargs = dict(sigma=0., xcenter=100, ycenter=200, width=20, height=10, angle=0)
    x, y, p = testing.make_ellipse(**kwargs)
    #plot_ellipse(p)
    data = np.array([x,y])
    res_dir = direct_ellipse_fit(data)
    res_fge = fast_guaranteed_ellipse_estimate(data.T)
    smaj, smin, xcen, ycen, tilt = fromAlgebraicToGeometricParameters(res_fge)
    tilt *= 360/(2*np.pi)
    rtol = 0.01 * rtol_percent
    assert np.isclose(smaj,0.5*kwargs["width"],rtol=rtol)
    assert np.isclose(smin,0.5*kwargs["height"],rtol=rtol)
    r = 0.5 * (smaj + smin)
    assert np.isclose(xcen, kwargs["xcenter"],atol=rtol*r)
    assert np.isclose(ycen, kwargs["ycenter"],atol=rtol*r)
    delta_angle = (tilt - kwargs["angle"])
    delta_angle = delta_angle % 180
    print(delta_angle, tilt, kwargs["angle"])
    assert np.isclose(delta_angle, 0, atol=20, rtol=1)


def test_long_fit(test_num = 10, rtol_percent=10):
    match_direct = []
    match_fge = []
    rtol = 0.01 * rtol_percent
    target = np.array([rtol, rtol, rtol, rtol, 20])
    for i in range(test_num):
        x, y, p = testing.make_ellipse(alim=180)
        gen = np.array([p["width"]*0.5,
                        p["height"]*0.5,
                        p["xcenter"],
                        p["ycenter"],
                        p["angle"]])
        data = np.array([x, y])
        res_dir = direct_ellipse_fit(data)
        try:
            direct = np.array(fromAlgebraicToGeometricParameters(res_dir))[:,0]
            direct[-1] *= 360/(2*np.pi)
            diff_dir = [(gen[0]-direct[0])/min(gen[0], direct[0]),
                        (gen[1]-direct[1])/min(gen[1], direct[1]),
                        (gen[2]-direct[2])/min(gen[0], direct[0]),
                        (gen[3]-direct[3])/min(gen[1], direct[1]),
                        (gen[4]-direct[4])]
            match_direct.append(
                np.mean(np.abs(diff_dir) < target)
                )
        except Exception as e:
            match_direct.append(0)

        try:
            res_fge = fast_guaranteed_ellipse_estimate(data.T)
            fge = np.array(fromAlgebraicToGeometricParameters(res_fge))[:,0]
            fge[-1] *= 360/(2*np.pi)
            diff_fge = [(gen[0]-fge[0])/min(gen[0], fge[0]),
                        (gen[1]-fge[1])/min(gen[1], fge[1]),
                        (gen[2]-fge[2])/min(gen[0], fge[0]),
                        (gen[3]-fge[3])/min(gen[1], fge[1]),
                        (gen[4]-fge[4])]
            match_fge.append(
                np.mean(np.abs(diff_fge) < target)
                )
        except Exception as e:
            match_fge.append(0)
    assert np.sum(match_direct) > 0.75 * test_num
    assert np.sum(match_fge) > 0.75 * test_num