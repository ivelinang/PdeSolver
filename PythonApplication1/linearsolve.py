import numpy as np

from decomposer import *

class LinearSolver(object):

    def __init__(self):
        pass

    def solve(self, a:np.matrix, b:np.matrix):
        pass

class ForwardSubSolve(LinearSolver):

    '''
    solve Ax=b where A is lower Triangular
    A - NxN lower triangular

    output: 1xn vector that makes Lx = b
    '''

    def __init__(self):
        super().__init__()

    @staticmethod
    def solve(A:np.matrix, b:np.matrix):
        '''
        forward substitution
        lower triangular matrix
        '''
        n = A.shape[0]# no rows

        x = np.zeros((n))#create the x vector
        x[0] = b[0]/A[0,0]

        for i in range(1,n):
            sum=0
            for j in range(i):
                sum = sum + A[i,j]*x[j]
            x[i] = (b[i] - sum)/A[i,i]

        return x


class BackwardSubSolve(LinearSolver):

    '''
    solve Ax=b where A is upper Triangular
    A - NxN upper triangular

    output: 1xn vector that makes Lx = b
    '''

    def __init__(self):
        super().__init__()

    @staticmethod
    def solve(A:np.matrix, b:np.matrix):
        '''
        backward substitution
        upper triangular matrix
        '''
        n = A.shape[0]# no rows

        x = np.zeros((n))#create the x vector
        x[n-1] = b[n-1]/A[n-1,n-1]

        for i in range(n-2, -1, -1):
            sum = 0
            for j in range(i+1,n):
                sum = sum + A[i,j]*x[j]
            x[i] = (b[i] - sum)/A[i,i]

        return x


class LuNoPivSolve(LinearSolver):

    def __init__(self):
        super().__init__()

    @staticmethod
    def solve(A:np.matrix, b:np.matrix):
        '''
        lu decomposition without pivoting
        '''

        n = A.shape[0]# no rows   
        m = A.shape[1]#no cols
        assert(n==m, "Matrix must be square; dim1 == dim2")

        l,u =lu_no_pivoting(A)

        y = ForwardSubSolve.solve(l ,b)
        x = BackwardSubSolve.solve(u,y)

        return x

class LuRowPivSolve(LinearSolver):


    def __init__(self):
       super().__init__()

    @staticmethod
    def solve(A:np.matrix, b:np.matrix):

        n = A.shape[0]# no rows

        l,p,u = lu_row_pivoting(A)

        y = ForwardSubSolve.solve(l,  p@b)
        x = BackwardSubSolve.solve(u,y)

        return x

class CholeskySolve(LinearSolver):


    def __init__(self):
       super().__init__()

    @staticmethod
    def solve(A:np.matrix, b:np.matrix):
        n = A.shape[0]# no rows

        u = cholesky_general(A)
        uT = u.T

        y = ForwardSubSolve.solve(uT, b)
        x = BackwardSubSolve.solve(u, y)

        return x



class Tridag(object):

    '''
    Solve Following system

    |b1 c1 0  ...                  |   |u1  |   |r1  |
    |a2 b2 c2 0  ...               |   |u2  |   |r2  |
    |0  a3 b3 c3 0 ...             | x |    | = |    |
    |... .......................   |   |    |   |    |
    |0               aN-1 bN-1 cN-1|   |uN-1|   |rN-1|
    |0               0    aN   bN  |   |uN  |   |rN  |
    '''

    def __init__(self):
        super().__init__()

    @staticmethod
    def solve(a:np.array, b:np.array, c:np.array, r:np.array , n):
        '''
        a[0...n-1]   a[0] can be anything
        b[0...n-1]
        c[0..n-2]    or c[n-1] can be anything
        r[0...n-1]
        '''

        gam = np.zeros((n));
        u = np.zeros((n));

        if b[0]==0.0:
            raise Exception("rewrite your equations as set of order N-1, with u2 trivially eliminated")

        bet= b[0]
        u[0]= r[0]/bet

        for j in range(1,n): #decomposition and forward substitution
            gam[j] = c[j-1]/bet;
            bet = b[j]-a[j]*gam[j];

            if bet == 0.0:
                raise Exception("bet = 0 error in tridag")

            u[j] = (r[j] - a[j]*u[j-1])/bet

        for j in range(n-2, -1, -1): #back substitution
            u[j] -= gam[j+1]*u[j+1]

        return u


    @staticmethod
    def solve_matrix(A:np.array, r:np.array):
        '''
         Solve Following system
         AxU=R

        |b1 c1 0  ...                  |   |u1  |   |r1  |
        |a2 b2 c2 0  ...               |   |u2  |   |r2  |
        |0  a3 b3 c3 0 ...             | x |    | = |    |
        |... .......................   |   |    |   |    |
        |0               aN-1 bN-1 cN-1|   |uN-1|   |rN-1|
        |0               0    aN   bN  |   |uN  |   |rN  |
        '''

        n = A.shape[0]# no rows
        m = A.shape[1]# no cols

        rn = r.shape[0]
        assert(n==rn, 'rows of A must match rows of r')


        x = np.zeros((n));
        D = np.zeros((n));
        C = np.zeros((n,m));
        B = np.zeros((n,m));

        for i in range(n):
            for j in range(m):
                C[i,j] = 0
                B[i,j] = 0
                if i==j:
                    B[i,j] = 1

        for i in range(n):
            if i>0:
                C[i,i-1] = A[i,i-1]
            temp = C[i,i-1]*B[i-1, i] if i>0 else 0
            C[i,i] = A[i,i] - temp
            if i<n-1:
                B[i,i+1] = A[i,i+1]/C[i,i]

        for i in range(n):
            temp = C[i, i-1]*D[i-1] if i>0 else 0
            D[i] = (r[i] - (temp))/C[i,i]

        for i in range(n-1,-1, -1):
            temp = B[i, i+1]*x[i+1] if i<n-1 else 0
            x[i] = D[i] - temp

        return x

