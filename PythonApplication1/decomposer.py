import numpy as np

def lu_no_pivoting(A):

    n = A.shape[0]# no rows

    l = np.identity(n)
    u = np.zeros((n,n))
   
    B = np.copy(A)

    for i in range(n):
        for j in range(i,n):
            l[j,i] = B[j,i]/B[i,i]
            u[i,j] = B[i,j]

        for j in range(i+1,n):
            for k in range(i+1, n):
                B[j,k] = B[j,k] - l[j,i]*u[i,k];

    l[n-1,n-1]=1
    u[n-1,n-1]=B[n-1,n-1]

    return l,u

def lu_row_pivoting(B):

    A = np.copy(B)
    n = A.shape[0]# no rows

    p = np.identity(n)
    l = np.identity(n)
    u = np.zeros((n,n))

    for i in range(n-1):

        i_max = find_i_max(A,i)

        switch_row(A, i, i_max)

        if i>=1:
            switch_row_L(l, i, i_max);

        switch_row(p, i, i_max)

        for j in range(i,n):
            l[j,i] = A[j,i]/A[i,i]
            u[i,j] = A[i,j]

        for j in range(i+1,n):
            for k in range(i+1, n):
                A[j,k] = A[j,k] - l[j,i]*u[i,k]

    l[n-1, n-1] = 1
    u[n-1, n-1] = A[n-1, n-1]

    return l,p,u






def cholesky_general(A):
    '''
    returns upper trianguler matrix U from matrix A -> s.t U.T x U = A
    '''

    #copy so that you do not change A
    B = np.copy(A)
    n = B.shape[0]# no rows

    u = np.zeros((n,n))

    for i in range(n-1):
        u[i,i] = np.sqrt(B[i,i])

        for j in range(i+1, n):
            u[i,j] = B[i,j]/u[i,i]

        for j in range(i+1, n):
            for k in range(j, n):
                B[j,k] = B[j,k] - u[i,j]*u[i,k]

    u[n-1, n-1] = np.sqrt(B[n-1, n-1])

    return u


def cholesky_banded(A, m:int):
    '''
    definitive positive matrix A with bandwidth m
    returns upper trianguler matrix U from matrix A -> s.t U.T x U = A
    '''

    #copy so that you do not change A
    B = np.copy(A)
    n = B.shape[0]# no rows

    u = np.zeros((n,n))

    for i in range(n-1):
        u[i,i] = np.sqrt(B[i,i])

        for j in range(i+1, min(i+m,n)):
            u[i,j] = B[i,j]/u[i,i]

        for j in range(i+1, min(i+m,n)):
            for k in range(j, min(i+m,n)):
                B[j,k] = B[j,k] - u[i,j]*u[i,k]

    u[n-1, n-1] = np.sqrt(B[n-1, n-1])

    return u

def switch_row(mat, i:int, i_max:int):
    '''
    func to switch fows i and i_max of matrix mat
    '''
    col = mat.shape[1]# no cols

    v = np.zeros(col)

    for j in range(col):
        v[j] = mat[i,j]

    for j in range(col):
        mat[i,j] = mat[i_max, j]

    for j in range(col):
        mat[i_max, j] = v[j]



def switch_row_L(mat, i:int, i_max:int):
    '''
    func to switch rows 1:(i-1) parh of the rows i and and i_max of matrix mat
    '''

    ww = np.zeros(i)

    for j in range(i-1):
        ww[j] = mat[i,j]

    for j in range(i-1):
        mat[i,j] = mat[i_max, j]

    for j in range(i-1):
        mat[i_max, j] = ww[j]


def find_i_max(A, i:int):

    '''
    largest entry
    '''

    row = A.shape[0]# no cols

    i_max = i

    for j in range(i, row):
        if np.abs(A[j,i]) > np.abs(A[i_max, i]):
            i_max = j

    return i_max
