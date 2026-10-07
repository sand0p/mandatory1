import numpy as np
import sympy as sp
from scipy import sparse

x, y, t = sp.symbols("x,y,t")


class Wave2D:
    """Class for solving the 2D wave equation"""


    def create_mesh(self, N: int, sparse: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Return 2D mesh created using np.meshgrid

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        sparse : bool, optional
            Whether to create a sparse mesh or not. Default is False.
        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh"""
        xi = np.linspace(0, 1, N + 1)
        return np.meshgrid(xi, xi, indexing="ij", sparse=sparse)

    def D2(self, N: int) -> sparse.lil_matrix:
        """Return second order differentiation matrix

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Returns
        -------
        D : scipy sparse LIL matrix
            The second order differentiation matrix
        """
        return sparse.diags([1.0, -2.0, 1.0], [-1, 0, 1], (N + 1, N + 1), format="lil") * N**2

    @property
    def w(self):
        """Return the dispersion coefficient"""
        return self.c * np.pi * np.sqrt(self.mx**2 + self.my**2)

    def ue(self, mx: int, my: int) -> sp.Expr:
        """Return the exact standing wave

        Parameters
        ----------
        mx, my : int
            Parameters for the standing wave
        Returns
        -------
        ue : Sympy expression
            The exact solution as a Sympy expression in x, y and t
        """
        return sp.sin(mx * sp.pi * x) * sp.sin(my * sp.pi * y) * sp.cos(self.w * t)

    def initialize(self, N: int, mx: int, my: int) -> np.ndarray:
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.h = 1 / N
        self.mx, self.my = mx, my
        self.xij, self.yij = self.create_mesh(N, sparse=True)
        self.ue_func = sp.lambdify((x, y, t), self.ue(mx, my), "numpy")
        self.D = self.D2(N).tocsr()
        U = np.zeros((2, N + 1, N + 1))
        U[0] = self.ue_func(self.xij, self.yij, 0)
        self.apply_bcs(U[0])
        U[1] = U[0] + 0.5 * (self.c * self.dt)**2 * (self.D @ U[0] + (self.D @ U[0].T).T)
        self.apply_bcs(U[1])
        return U

    @property
    def dt(self) -> float:
        """Return the time step"""
        return self.cfl * self.h / self.c

    def l2_error(self, u: np.ndarray, t0: float) -> float:
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        error = u - self.ue_func(self.xij, self.yij, t0)
        return np.sqrt(self.h**2 * np.sum(error**2))

    def apply_bcs(self, u: np.ndarray):
        """Apply boundary conditions to the solution mesh function

        Parameters
        ----------
        u : array
            The solution mesh function
        """
        u[[0, -1], :] = 0
        u[:, [0, -1]] = 0

    def __call__(
        self,
        N: int,
        Nt: int,
        cfl: float = 0.5,
        c: float = 1.0,
        mx: int = 3,
        my: int = 3,
        store_data: int = -1,
    ):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.cfl = cfl
        self.c = c
        u, um = self.initialize(N, mx, my)
        if store_data == -1:
            E = [self.l2_error(u, 0)]
        else:
            data = {0: u.copy()}
        for n in range(1, Nt + 1):
            up = 2 * u - um + (c * self.dt)**2 * (self.D @ u + (self.D @ u.T).T)
            self.apply_bcs(up)
            um, u = u, up
            if store_data == -1:
                E.append(self.l2_error(u, n * self.dt))
            elif n % store_data == 0:
                data[n] = u.copy()
        if store_data == -1:
            return self.h, np.array(E)
        return data

    def convergence_rates(
        self, m: int = 4, cfl: float = 0.1, Nt: int = 10, mx: int = 3, my: int = 3
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i]) for i in range(1, m, 1)]
        return np.array(r), np.array(E), np.array(h)


class Wave2D_Neumann(Wave2D):
    def D2(self, N: int) -> sparse.lil_matrix:
        D = super().D2(N)
        D[0, 1] = 2 * N**2
        D[-1, -2] = 2 * N**2
        return D

    def ue(self, mx: int, my: int) -> sp.Expr:
        return sp.cos(mx * sp.pi * x) * sp.cos(my * sp.pi * y) * sp.cos(self.w * t)

    def apply_bcs(self, u: np.ndarray):
        pass


def test_convergence_wave2d():
    sol = Wave2D()
    r, _, _ = sol.convergence_rates(m=5, mx=2, my=3)
    assert abs(r[-1] - 2) < 1e-2, r


def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, _, _ = solN.convergence_rates(mx=3, my=3)
    assert abs(r[-1] - 2) < 0.05


def test_exact_wave2d():
    for solver in (Wave2D(), Wave2D_Neumann()):
        for m in (1, 2, 3):
            _, err = solver(32, 100, cfl=1 / np.sqrt(2), mx=m, my=m)
            assert np.max(err) < 1e-12, np.max(err)

if __name__=='__main__':
    test_convergence_wave2d()
    test_convergence_wave2d_neumann()
    test_exact_wave2d()
    print('test passed!')
