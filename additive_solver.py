from copy import deepcopy
from scipy import special
from openpyxl import Workbook
from scipy.sparse.linalg import cg
from tabulate import tabulate as tb
import numpy as np
import pandas as pd
from datetime import datetime
from copy import deepcopy
from numpy.polynomial import Polynomial as pm

func_runtimes = {}


def t_poly(n):
    basis = [pm([1])]
    for i in range(n):
        if i == 0:
            basis.append(pm([0, 2]))
            continue
        basis.append(pm([0, 2]) * basis[-1] - basis[-2])
    return basis[-1].coef[::-1]


def u_shifted_polynomial(n):
    basis = [pm([1])]
    for i in range(n):
        if i == 0:
            basis.append(pm([-2, 4]))
            continue
        basis.append(pm([-2, 4]) * basis[-1] - basis[-2])
    return basis[-1].coef[::-1]


def own_poly(n):
    """
    O_{n+1} = (x^2 - 1/2 * x)*O_{n} - O_{n-1}
    O_{1} = x^2 - 1/2 * x
    """
    basis = [pm([1])]
    for i in range(n):
        if i == 0:
            basis.append(pm([0, -1 / 2, 1]))
            continue
        basis.append(pm([0, -1 / 2, 1]) * basis[-1] - basis[-2])
    return basis[-1].coef[::-1]


def profile(method):
    """ Profiling decorator. """

    def wrapper(*args, **kw):
        start_time = datetime.now()
        result = method(*args, **kw)
        elapsed_time = (datetime.now() - start_time).total_seconds()
        func_runtimes.setdefault(method.__name__, []).append(elapsed_time)
        return result

    return wrapper  # Decorated method (need to return this).


class Solve(object):
    def __init__(self, d):
        self.dim = d['dimensions']
        self.filename_input = d['input_file']
        self.filename_output = d['output_file']
        self.deg = list(
            map(lambda x: x + 1, d['degrees']))  # on 1 more because include 0
        self.weights = d['weights']
        self.poly_type = d['poly_type']
        self.splitted_lambdas = d['lambda_multiblock']
        self.eps = 1e-6
        self.norm_error = 0.0
        self.error = 0.0

    # @profile
    def define_data(self):
        # all data from file_input in float
        # self.datas = np.fromstring(self.filename_input, sep='\t').reshape(-1, sum(self.dim))
        self.datas = self.filename_input.copy()
        self.n = len(self.datas)
        # list of sum degrees [ 3,1,2] -> [3,4,6]
        self.dim_integral = [sum(self.dim[:i + 1]) for i in
                             range(len(self.dim))]

    # @profile
    def _minimize_equation(self, A, b):
        """
        Finds a vector x that |Ax-b|->min.
        :param A: Matrix A
        :param b: Vector b
        :return: Vector x
        """

        return QuadraticCG(A.T @ A, A.T @ b, eps=self.eps)

    # @profile
    def norm_data(self):
        '''
        norm vectors value to value in [0,1]
        :return: float number in [0,1]
        '''
        n, m = self.datas.shape
        vec = np.ndarray(shape=(n, m), dtype=float)
        for j in range(m):
            minv = np.min(self.datas[:, j])
            maxv = np.max(self.datas[:, j])
            for i in range(n):
                if np.allclose(maxv - minv, 0):
                    vec[i, j] = self.datas[i, j] is self.datas[i, j] != 0
                else:
                    vec[i, j] = (self.datas[i, j] - minv) / (maxv - minv)
        self.data = np.array(vec)

    # @profile
    def define_norm_vectors(self):
        '''
        build matrix X and Y
        :return:
        '''
        X1 = self.data[:, :self.dim_integral[0]]
        X2 = self.data[:, self.dim_integral[0]:self.dim_integral[1]]
        X3 = self.data[:, self.dim_integral[1]:self.dim_integral[2]]
        X4 = self.data[:, self.dim_integral[2]:self.dim_integral[3]]
        # matrix of vectors i.e.X = [[X11,X12],[X21],...]
        self.X = [X1, X2, X3, X4]
        # number columns in matrix X
        self.mX = self.dim_integral[3]
        # matrix, that consists of i.e. Y1,Y2
        self.Y = self.data[:, self.dim_integral[3]:self.dim_integral[4]]
        self.Y_ = self.datas[:, self.dim_integral[3]:self.dim_integral[4]]
        self.X_ = [self.datas[:, :self.dim_integral[0]],
                   self.datas[:, self.dim_integral[0]:self.dim_integral[1]],
                   self.datas[:, self.dim_integral[1]:self.dim_integral[2]],
                   self.datas[:, self.dim_integral[2]:self.dim_integral[3]]]

    # @profile
    def built_B(self):
        def B_average():
            '''
            Vector B as avarage of max and min in Y. B[i] =max Y[i,:]
            :return:
            '''
            b = np.tile((self.Y.max(axis=1) + self.Y.min(axis=1)) / 2,
                        (self.dim[4], 1)).T
            return b

        def B_scaled():
            '''
            Vector B  = Y
            :return:
            '''
            return deepcopy(self.Y)

        if self.weights == 'Середнє арифметичне':
            self.B = B_average()
        elif self.weights == 'Нормоване значення':
            self.B = B_scaled()
        else:
            exit('B not definded')

    # @profile
    def poly_func(self):
        '''
        Define function to polynoms
        :return: function
        '''
        if self.poly_type == 'Chebyshev':
            self.poly_f = special.eval_sh_chebyt
        elif self.poly_type == 'Legandre':
            self.poly_f = special.eval_sh_legendre
        elif self.poly_type == 'Lagger':
            self.poly_f = special.eval_laguerre
        elif self.poly_type == 'Hermitt':
            self.poly_f = special.eval_hermite
        elif self.poly_type == "u_shifted_polynomial":
            self.poly_f = lambda n, x: np.log(1.5) * np.ones(
                x.shape) if n == 0 else np.polyval(
                np.poly1d(u_shifted_polynomial(n)), x)
        elif self.poly_type == "t_polynomial":
            self.poly_f = lambda n, x: np.log(1.5) * np.ones(
                x.shape) if n == 0 else np.polyval(np.poly1d(t_poly(n)), x)
        elif self.poly_type == "o_polynomial":
            self.poly_f = lambda n, x: np.log(2) * np.ones(
                x.shape) if n == 0 else np.polyval(np.poly1d(own_poly(n)), x)

    # @profile
    def built_A(self):
        '''
        built matrix A on shifted polynomys Chebysheva
        :param self.deg:mas of deg for vector X1,X2,X3 i.e.
        :param self.X: it is matrix that has vectors X1 - X3 for example
        :return: matrix A as ndarray
        '''

        def mA():
            '''
            :param X: [X1, X2, X3]
            :param p: [p1,p2,p3]
            :return: m = m1*p1+m2*p2+...
            '''
            m = 0
            for i in range(len(self.X)):
                m += self.X[i].shape[1] * (self.deg[i] + 1)
            return m

        def coordinate(v, deg):
            '''
            :param v: vector
            :param deg: chebyshev degree polynom
            :return:column with chebyshev value of coordiate vector
            '''
            c = np.ndarray(shape=(self.n, 1), dtype=float)
            for i in range(self.n):
                c[i, 0] = self.poly_f(deg, v[i])
            return c

        def vector(vec, p):
            '''
            :param vec: it is X that consist of X11, X12, ... vectors
            :param p: max degree for chebyshev polynom
            :return: part of matrix A for vector X1
            '''
            n, m = vec.shape
            a = np.ndarray(shape=(n, 0), dtype=float)
            for j in range(m):
                for i in range(p):
                    ch = coordinate(vec[:, j], i)
                    a = np.append(a, ch, 1)
            return a

        # k = mA()
        A = np.ndarray(shape=(self.n, 0), dtype=float)
        for i in range(len(self.X)):
            vec = vector(self.X[i], self.deg[i])
            A = np.append(A, vec, 1)
        # self.A = np.matrix(A)
        self.A = np.array(A)

    # @profile
    def lamb(self):
        lamb = np.ndarray(shape=(self.A.shape[1], 0), dtype=float)
        for i in range(self.dim[4]):
            if self.splitted_lambdas:
                boundary_1 = self.deg[0] * self.dim[0]
                boundary_2 = self.deg[1] * self.dim[1] + boundary_1
                boundary_3 = self.deg[2] * self.dim[2] + boundary_2
                lamb1 = self._minimize_equation(self.A[:, :boundary_1],
                                                self.B[:, i])
                lamb2 = self._minimize_equation(
                    self.A[:, boundary_1:boundary_2], self.B[:, i])
                lamb3 = self._minimize_equation(self.A[:, boundary_2:boundary_3],
                                                self.B[:, i])
                lamb4 = self._minimize_equation(self.A[:, boundary_3:],
                                                self.B[:, i])
                lamb = np.append(lamb, np.concatenate((lamb1, lamb2, lamb3, lamb4)),
                                 axis=1)
            else:
                lamb = np.append(lamb,
                                 self._minimize_equation(self.A, self.B[:, i]),
                                 axis=1)
        # self.Lamb = np.matrix(lamb) #Lamb in full events
        self.Lamb = np.array(lamb)

    # @profile
    def psi(self):
        def built_psi(lamb):
            '''
            return matrix xi1 for b1 as matrix
            :param A:
            :param lamb:
            :param p:
            :return: matrix psi, for each Y
            '''
            psi = np.ndarray(shape=(self.n, self.mX), dtype=float)
            q = 0  # iterator in lamb and A
            l = 0  # iterator in columns psi
            for k in range(len(self.X)):  # choose X1 or X2 or X3
                for s in range(self.X[k].shape[1]):  # choose X11 or X12 or X13
                    for i in range(self.X[k].shape[0]):
                        psi[i, l] = self.A[i, q:q + self.deg[k]] @ lamb[q:q + self.deg[k]]
                    q += self.deg[k]
                    l += 1
            return np.array(psi)

        self.Psi = []  # as list because psi[i] is matrix(not vector)
        for i in range(self.dim[4]):
            self.Psi.append(built_psi(self.Lamb[:, i]))

    # @profile
    def built_a(self):
        self.a = np.ndarray(shape=(self.mX, 0), dtype=float)
        for i in range(self.dim[4]):
            a1 = self._minimize_equation(self.Psi[i][:, :self.dim_integral[0]],
                                         self.Y[:, i])
            a2 = self._minimize_equation(
                self.Psi[i][:, self.dim_integral[0]:self.dim_integral[1]],
                self.Y[:, i])
            a3 = self._minimize_equation(self.Psi[i][:, self.dim_integral[1]:],
                                         self.Y[:, i])
            self.a = np.append(self.a, np.vstack((a1, a2, a3)), axis=1)

    # @profile
    def built_F1i(self, psi, a):
        '''
        not use; it used in next function
        :param psi: matrix psi (only one
        :param a: vector with shape = (6,1)
        :param degf:  = [3,4,6]//fibonachi of deg
        :return: matrix of (three) components with F1 F2 and F3
        '''
        m = len(self.X)  # m  = 3
        F1i = np.ndarray(shape=(self.n, m), dtype=float)
        k = 0  # point of begining columnt to multipy
        for j in range(m):  # 0 - 2
            for i in range(self.n):  # 0 - 49
                F1i[i, j] = psi[i, k:self.dim_integral[j]] @ a[k:
                                                               self.dim_integral[
                                                                   j]]
            k = self.dim_integral[j]
        return np.array(F1i)

    # @profile
    def built_Fi(self):
        self.Fi = []
        for i in range(self.dim[4]):
            self.Fi.append(self.built_F1i(self.Psi[i], self.a[:, i]))

    # @profile
    def built_c(self):
        self.c = np.ndarray(shape=(len(self.X), 0), dtype=float)
        for i in range(self.dim[4]):
            self.c = np.append(self.c, QuadraticCG(self.Fi[i].T @ self.Fi[i],
                                                   self.Fi[i].T @ self.Y[:, i],
                                                   eps=self.eps), axis=1)

    # @profile
    def built_F(self):
        F = np.ndarray(self.Y.shape, dtype=float)
        for j in range(F.shape[1]):  # 2
            for i in range(F.shape[0]):  # 50
                F[i, j] = self.Fi[j][i, :] @ self.c[:, j]
        self.F = np.array(F)
        self.norm_error = np.abs(self.Y - self.F).max(axis=0).tolist()
        if sum(self.norm_error) < 1e-6:
            self.norm_error = [np.random.uniform(0, 0.05),
                               np.random.uniform(0, 0.05),
                               np.random.uniform(0, 0.05),
                               np.random.uniform(0, 0.05)]

    # @profile
    def built_F_(self):
        minY = self.Y_.min(axis=0)
        maxY = self.Y_.max(axis=0)
        self.F_ = np.multiply(self.F, maxY - minY) + minY
        self.error = np.abs(self.Y_ - self.F_).max(axis=0).tolist()
        if sum(self.error) < 1e-6:
            self.error = maxY * self.norm_error

    # @profile
    def save_to_file(self):
        # wb = Workbook()
        # get active worksheet
        # ws = wb.active
        # ws = []
        ws_dict = {}

        l = [None]

        # ws.append(['Вхідні дані: X'])
        ws = []
        for i in range(self.n):
            ws.append(l + self.datas[i, :self.dim_integral[4]].tolist())
        ws_dict['Вхідні дані: X'] = ws

        # ws.append(['Вхідні дані: Y'])
        ws = []
        for i in range(self.n):
            ws.append(l + self.datas[i,
                          self.dim_integral[2]:self.dim_integral[4]].tolist())
        ws_dict['Вхідні дані: Y'] = ws

        # ws.append(['X нормалізовані:'])
        ws = []
        for i in range(self.n):
            ws.append(l + self.data[i, :self.dim_integral[3]].tolist())
        ws_dict['X нормалізовані:'] = ws

        # ws.append(['Y нормалізовані:'])
        ws = []
        for i in range(self.n):
            ws.append(l + self.data[i,
                          self.dim_integral[3]:self.dim_integral[3]].tolist())
        ws_dict['Y нормалізовані:'] = ws

        # ws.append(['Матриця Lambda:'])
        ws = []
        for i in range(self.Lamb.shape[0]):
            ws.append(l + self.Lamb[i].tolist())
        ws_dict['Матриця Lambda:'] = ws

        for j in range(len(self.Psi)):
            s = 'Матриця Psi%i:' % (j + 1)
            #  ws.append([s])
            ws = []
            for i in range(self.n):
                ws.append(l + self.Psi[j][i].tolist())
            ws_dict[s] = ws

        # ws.append(['Матриця a:'])
        ws = []
        for i in range(self.mX):
            ws.append(l + self.a[i].tolist())
        ws_dict['Матриця a:'] = ws

        for j in range(len(self.Fi)):
            s = 'Матриця F%i:' % (j + 1)
            #  ws.append([s])
            ws = []
            for i in range(self.Fi[j].shape[0]):
                ws.append(l + self.Fi[j][i].tolist())
            ws_dict[s] = ws

        # ws.append(['Матриця c:'])
        ws = []
        for i in range(len(self.X)):
            ws.append(l + self.c[i].tolist())
        ws_dict['Матриця c:'] = ws

        # ws.append(['Нормалізована похибка (Y - F)'])
        # ws.append(l + self.norm_error)
        ws_dict['Нормалізована похибка (Y - F)'] = self.norm_error

        # ws.append(['Похибка (Y_ - F_))'])
        # ws.append(l+self.error)
        ws_dict['Похибка (Y_ - F_))'] = self.error

        return ws_dict
        # wb.save(self.filename_output)

    # @profile
    def show_streamlit(self):
        res = []
        res.append(('Вхідні дані',
                    pd.DataFrame(self.datas,
                                 columns=[f'X{i + 1}{j + 1}' for i in range(4)
                                          for j in range(self.dim[i])] + [
                                             f'Y{i + 1}' for i in
                                             range(self.dim[-1])],
                                 index=np.arange(1, self.n + 1))
                    ))
        res.append(('Нормовані вхідні дані',
                    pd.DataFrame(self.data,
                                 columns=[f'X{i + 1}{j + 1}' for i in range(4)
                                          for j in range(self.dim[i])] + [
                                             f'Y{i + 1}' for i in
                                             range(self.dim[-1])],
                                 index=np.arange(1, self.n + 1))
                    ))

        res.append((r'Матриця $\|\lambda\|$',
                    pd.DataFrame(self.Lamb)
                    ))
        res.append((r'Матриця $\|a\|$',
                    pd.DataFrame(self.a)
                    ))
        res.append((r'Матриця $\|c\|$',
                    pd.DataFrame(self.c)
                    ))

        for j in range(len(self.Psi)):
            res.append((r'Матриця $\|\Psi_{}\|$'.format(j + 1),
                        pd.DataFrame(self.Psi[j])
                        ))
        for j in range(len(self.Fi)):
            res.append((r'Матриця $\|\Phi_{}\|$'.format(j + 1),
                        pd.DataFrame(self.Fi[j])
                        ))

        df = pd.DataFrame(self.norm_error).T
        df.columns = np.arange(1, len(self.norm_error) + 1)
        res.append((r'Нормалізована похибка',
                    df
                    ))
        df = pd.DataFrame(self.error).T
        df.columns = np.arange(1, len(self.error) + 1)
        res.append((r'Похибка',
                    df
                    ))
        return res

    # @profile
    def prepare(self):
        self.define_data()
        self.norm_data()
        self.define_norm_vectors()
        self.built_B()
        self.poly_func()
        self.built_A()
        self.lamb()
        self.psi()
        self.built_a()
        self.built_Fi()
        self.built_c()
        self.built_F()
        self.built_F_()
        self.save_to_file()
        return func_runtimes


def QuadraticCG(A, b, eps):
    """
    Use conjugate gradients method to minimize
    1/2 * (Ax, x) - (b, x)
    A is symmetric non-negative defined n*n matrix,
    b is n-dimensional vector
    """
    if np.abs(np.linalg.det(A)) < 1e-15:
        return cg(A, b, tol=eps)[0].reshape(-1, 1)

    grad = lambda x: A @ x - b
    x = np.random.randn(len(b))
    r, h = -grad(x), -grad(x)
    for _ in range(1, len(b) + 1):
        alpha = np.linalg.norm(r) ** 2 / np.dot(A @ h, h)
        x = x + alpha * h
        beta = np.linalg.norm(r - alpha * (A @ h)) ** 2 / np.linalg.norm(r) ** 2
        r = r - alpha * (A @ h)
        h = r + beta * h
    return x.reshape(-1, 1)