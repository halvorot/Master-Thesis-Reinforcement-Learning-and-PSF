
import matlab.engine
import numpy as np


class PSF:
    """
    A class used to interact with the Predictive Safety Filter (PSF) implemented in Matlab

    ...

    Attributes
    ----------
    None

    Methods
    -------
    calc(x, u_L)
        Calculates a "safe" input from state and proposed input

    """

    def __init__(self, N, A, B, Hx, hx, Hu, hu):
        """
        Initialize the Matlab engine and passes discrete linear model to the PSF with horizon N.
            Model: x[t+1]=Ax + Bu s.t. Hx<=hx , Hu<=hu

            Parameters
            ----------
            N: float
            A : np.ndarray
            B : np.ndarray
            Hx : np.ndarray
            hx : np.ndarray
            Hu : np.ndarray
            hu : np.ndarray

            Returns
            -------
            None
        """
        self._eng = matlab.engine.start_matlab()

        _A = matlab.double(A.tolist())
        _B = matlab.double(B.tolist())
        _Hx = matlab.double(Hx.tolist())
        _hx = matlab.double(hx.tolist())
        _Hu = matlab.double(Hu.tolist())
        _hu = matlab.double(hu.tolist())
        _N = matlab.double([N])

        self._m_PSF = self._eng.PSF(_A, _B, _Hx, _hx, _Hu, _hu, _N)

    def calc(self, x, u_L):
        """Performs the argmin ||u-u_L|| for the given model.

        Parameters
        ----------
        x: np.ndarray
            Current state
        u_L : np.ndarray
            Desired input

        Returns
        -------
        u: np.ndarray
            Allowed input

        Examples
        --------
        Given the class system example:
        Ts = 0.1
        A = np.asarray([[1, Ts, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, Ts],
                        [0, 0, 0, 1]])

        B = np.asarray([[Ts * Ts * 0.5, 0],
                        [Ts, 0],
                        [0, Ts * Ts * 0.5],
                        [0, Ts]])
        Hx = np.asarray([[1, 0, 0, 0],
                         [-1, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, -1, 0, 0],
                         [0, 0, 1, 0],
                         [-1, 0, -1, 0],
                         [0, 0, 0, 1],
                         [0, 0, 0, -1]])
        hx = np.asarray([[1],
                         [2],
                         [1],
                         [1],
                         [2],
                         [2],
                         [1],
                         [1]])
        Hu = np.asarray([[1, 0],
                         [-1, 0],
                         [0, 1],
                         [0, -1]])
        hu = np.asarray([[2],
                         [2],
                         [2],
                         [2]])
        psf = PSF(15, A, B, Hx, hx, Hu, hu)

        x = np.asarray([[0],
            [0],
            [0],
            [0]])

        u_L = np.asarray([[0.5],
                          [0.5]])
        print(psf.calc(x, u_L))

        # Input constrain violation
        x = np.asarray([[0],
                        [0],
                        [0],
                        [0]])

        u_L = np.asarray([[0],
                          [3]])
        print(psf.calc(x, u_L))
        # Recursive feasibility violation

        x = np.asarray([[0.5],
                        [1],
                        [0.5],
                        [1]])

        u_L = np.asarray([[0.3],
                          [0.8]])
        print(psf.calc(x, u_L))

        See Also
        --------


        """

        _x = matlab.double(x.tolist())
        _u_L = matlab.double(u_L.tolist())

        m = self._eng.calc(self._m_PSF, _x, _u_L)
        u = np.asarray(m)

        return u


if __name__ == '__main__':
    Ts = 0.1
    A = np.asarray([[1, Ts, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, Ts],
                    [0, 0, 0, 1]])

    B = np.asarray([[Ts * Ts * 0.5, 0],
                    [Ts, 0],
                    [0, Ts * Ts * 0.5],
                    [0, Ts]])

    Hx = np.asarray([[1, 0, 0, 0],
                     [-1, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, 1, 0],
                     [-1, 0, -1, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, -1]])

    hx = np.asarray([[1],
                     [2],
                     [1],
                     [1],
                     [2],
                     [2],
                     [1],
                     [1]])

    Hu = np.asarray([[1, 0],
                     [-1, 0],
                     [0, 1],
                     [0, -1]])

    hu = np.asarray([[2],
                     [2],
                     [2],
                     [2]])

    psf = PSF(15.0, A, B, Hx, hx, Hu, hu)

    # No constrain violation
    x = np.asarray([[0],
                    [0],
                    [0],
                    [0]])

    u_L = np.asarray([[0.5],
                      [0.5]])
    print(psf.calc(x, u_L))

    # Input constrain violation
    x = np.asarray([[0],
                    [0],
                    [0],
                    [0]])

    u_L = np.asarray([[0],
                      [3]])
    print(psf.calc(x, u_L))

    # Recursive feasibility violation
    x = np.asarray([[0.5],
                    [1],
                    [0.5],
                    [1]])

    u_L = np.asarray([[0.3],
                      [0.8]])
    print(psf.calc(x, u_L))
