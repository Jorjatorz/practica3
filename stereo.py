import numpy as np
import misc
from numpy import linalg as la
from scipy import ndimage
from matplotlib import pyplot as ppl

import maxflow
from maxflow.fastmin import aexpansion_grid_step


def plothline_TODO(line, axes=None):
    """Plot a line given its homogeneous coordinates.
    
    Parameters
    ----------
    line : array_like
        Homogeneous coordinates of the line.
    axes : AxesSubplot
        Axes where the line should be plotted. If not given,
        line will be plotted in the active axis.
    """

    if axes == None:
        axes = ppl.gca()

    [x0, x1, y0, y1] = axes.axis()
    #     (x0, y0) ._____________________. (x1, y0)
    #              |                     |
    #              |                     |
    #              |                     |
    #              |                     |
    #              |                     |
    #              |                     |
    #     (x0, y1) .---------------------. (x1, y1)

    # TODO: Compute the intersection of the line with the image
    # borders.

    # Obtenemos los parametros de la recta - La formula de la recta es ax + yb + c = 0 en coordenadas homogenas [a,b,c]. De ahi se puede sacar la pendiente y la interseccion con y
    m = -line[0] / line[1]
    b = -line[2] / line[1]

    # Calculamos intersecciones con los bordes
    p0 = x0 * m + b
    p1 = x1 * m + b

    # TODO: Plot the line with axes.plot.
    axes.plot([0, x1], [p0, p1])

    axes.axis([x0, x1, y0, y1])


def plot_epipolar_lines_TODO(image1, image2, F):
    """Ask for points in one image and draw the epipolar lines for those points.
    
    Parameters
    ----------
    image1 : array_like
        First image.
    image2 : array_like
        Second image.
    F : array_like
        3x3 fundamental matrix from image1 to image2.
    """
    # Prepare the two images.
    fig = ppl.gcf()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image1)
    ax1.axis('image')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(image2)
    ax2.axis('image')
    ppl.draw()

    ax1.set_xlabel("Choose points in left image (or right click to end)")
    point = ppl.ginput(1, timeout=-1, show_clicks=False, mouse_pop=2, mouse_stop=3)
    while len(point) != 0:
        # point has the coordinates of the selected point in the first image.
        point = np.hstack([np.array(point[0]), 1])
        ax1.plot(point[0], point[1], '.r')

        # TODO: Determine the epipolar line.
        l = np.matmul(F, point.reshape(3, 1))

        # TODO: Plot the epipolar line with plothline (the parameter 'axes' should be ax2).
        plothline_TODO(l, axes=ax2)

        ppl.draw()
        # Ask for a new point.
        point = ppl.ginput(1, timeout=-1, show_clicks=False, mouse_pop=2, mouse_stop=3)

    ax1.set_xlabel('')
    ppl.draw()


def plot_correspondences_TODO(image1, image2, S, H1, H2):
    """
    Ask for points in the first image and plot their correspondences in
    the second image.
    
    Parameters
    ----------
    image1, image2 : array_like
        The images (before rectification)
    S : array_like
        The matrix of disparities.
    H1, H2 : array_like
        The homographies which rectify both images.
    """
    # Prepare the two images.
    fig = ppl.gcf()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image1)
    ax1.axis('image')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(image2)
    ax2.axis('image')
    ppl.draw()

    ax1.set_xlabel("Choose points in left image (or right click to end)")
    point = ppl.ginput(1, timeout=-1, show_clicks=False, mouse_pop=2, mouse_stop=3)
    while len(point) != 0:
        # point has the coordinates of the selected point in the first image.
        point = np.c_[np.array(point), 1].T
        ax1.plot(point[0, :], point[1, :], '.r')

        # TODO: Determine the correspondence of 'point' in the second image.

        # TODO: Plot the correspondence with ax2.plot.
        # ax2.plot(...)

        ppl.draw()
        # Ask for a new point.
        point = ppl.ginput(1, timeout=-1, show_clicks=False, mouse_pop=2, mouse_stop=3)

    ax1.set_xlabel('')
    ppl.draw()


def main():
    # Ejercicio 1
    cameras = np.load("cameras.npz")
    P1 = cameras["left"]
    P2 = cameras["right"]

    def triangulate(point1, point2, P1, P2):
        from numpy.linalg import lstsq
        p = np.array([
            P1[0] - P1[2] * (point1[0] / point1[2]),
            P1[1] - P1[2] * (point1[1] / point1[2]),
            P2[0] - P2[2] * (point2[0] / point2[2]),
            P2[1] - P2[2] * (point2[1] / point2[2])
        ])
        result = lstsq(p[:, :-1], (-1) * p[:, -1])[0]
        return np.append(result, 1)

    def reconstruct(points1, points2, P1, P2):
        return np.array([triangulate(p1, p2, P1, P2) for p1, p2 in zip(points1.T, points2.T)]).T

    # Ejercicio 2
    from scipy.misc import imread
    img1 = imread('./images/minoru_cube3_left.jpg')
    img2 = imread('./images/minoru_cube3_right.jpg')
    points1, points2 = misc.askpoints(img1, img2)
    r = reconstruct(points1, points2, P1, P2)
    misc.plot3D(r[0], r[1], r[2])
    ppl.show()

    # Ejercicio 3
    from matplotlib.pyplot import imshow
    misc.plothom(np.matmul(P1, r))
    imshow(img1)
    ppl.show()

    from matplotlib.pyplot import imshow
    misc.plothom(np.matmul(P2, r))
    imshow(img2)
    ppl.show()

    # Ejercicio 4
    F = projmat2f(P1, P2)

    # Ejercicio 5
    P1_f, P2_f = f2projmat(F)

    # Ejercicio 6,7,8
    plot_epipolar_lines_TODO(img1, img2, F)
    plot_epipolar_lines_TODO(img2, img1, F.T)

    # Ejercicio 10
    print('ejercicio 10')
    img1 = misc.rgb2gray(img1 / 255.0)
    img2 = misc.rgb2gray(img2 / 255.0)
    H1, H2 = misc.projmat2rectify(P1, P2,
                                  img1.shape[:2])
    O1, O2 = misc.rectify_images(img1, img2, H1,
                                 H2)
    from numpy.linalg import inv
    F = np.matmul(np.matmul(inv(H2).T, F), inv(H1))  # resultado del ejercicio 9
    plot_epipolar_lines_TODO(O1, O2, F)
    plot_epipolar_lines_TODO(O2, O1, F.T)

    # Ejercicio 11
    def localssd(im1, im2, K):
        from scipy.ndimage import convolve
        return convolve((im1 - im2) ** 2, np.ones((2 * K + 1, 2 * K + 1)))


def projmat2f(P1, P2):
    A = P1[:, [0, 1, 2]]
    b = P1[:, [3]]
    B = P2[:, [0, 1, 2]]
    d = P2[:, [3]]

    from numpy.linalg import inv
    Ainv = inv(A)
    Binv = inv(B)

    skewM = np.matmul(Binv.transpose(), misc.skew(np.matmul(Binv, d) - np.matmul(Ainv, b)))

    return np.matmul(skewM, inv(A))


def f2projmat(F):
    # Obtenemos los epipolos dado F
    i, e = misc.epipoles(F)
    e = e.reshape(3, 1)

    P1 = np.hstack((np.identity(3), np.array([0, 0, 0]).reshape(3, 1)))

    # Calculamos P2
    v = np.array([0, 1, 0]).reshape(3, 1)
    ex = misc.skew(e)
    P2 = np.hstack((np.matmul(ex, F) + np.matmul(e, v.T), e.reshape(3, 1)))

    return P1, P2


if __name__ == "__main__":
    main()
