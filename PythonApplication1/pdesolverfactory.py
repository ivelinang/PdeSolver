from PDEsolver import *
from bspdesolver import *
from forwardeuler import *
from backwardeuler import *
from cranknicolson import *

def make_euro_call_fe(s, k, vol, t, r, q):

    left = BSCallLeft(s, k, vol, t, r, q)
    right = BSCallRight(s, k, vol, t, r, q)
    ftau = BSCallTau(s, k, vol, t, r, q)

    xleft = left.get_x_left()
    xright = right.get_x_right()
    taufinal = ftau.get_tau_final()

    fe = ForwardEuler(xleft, xright, taufinal, left, right, ftau)
    solver = BsPdeSolver(s, k, vol, t, r, q, fe)
    return solver

def make_euro_call_be_lu(s, k, vol, t, r, q):

    left = BSCallLeft(s, k, vol, t, r, q)
    right = BSCallRight(s, k, vol, t, r, q)
    ftau = BSCallTau(s, k, vol, t, r, q)

    xleft = left.get_x_left()
    xright = right.get_x_right()
    taufinal = ftau.get_tau_final()

    sol = LuNoPivSolve()
    be = BackwardEuler(xleft, xright, taufinal, left, right, ftau, sol)
    solver = BsPdeSolver(s, k, vol, t, r, q, be)
    return solver

def make_euro_call_cn_lu(s, k, vol, t, r, q):

    left = BSCallLeft(s, k, vol, t, r, q)
    right = BSCallRight(s, k, vol, t, r, q)
    ftau = BSCallTau(s, k, vol, t, r, q)

    xleft = left.get_x_left()
    xright = right.get_x_right()
    taufinal = ftau.get_tau_final()

    sol = LuNoPivSolve()
    be = CrankNicolson(xleft, xright, taufinal, left, right, ftau, sol)
    solver = BsPdeSolver(s, k, vol, t, r, q, be)
    return solver
