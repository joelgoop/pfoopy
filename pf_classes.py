# coding=utf-8
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve
from collections import namedtuple
from copy import deepcopy
import numpy as np

class PowerSystem(object):
    """Class to represent power system for which to solve PF problem"""
    def __init__(self, buses, lines):
        super(PowerSystem, self).__init__()
        self.buses = deepcopy(buses)
        self.lines = lines
        self.n_buses = len(self.buses)
        slack_bus = next(i for i in xrange(self.n_buses) if self.buses[i].slack)
        # Put slack bus first in list
        self.buses[0],self.buses[slack_bus] = self.buses[slack_bus],self.buses[0]
        self.v_contr_buses = [i for i in xrange(self.n_buses) if self.buses[i].v_contr]

        self.init_bus_adm_matr()
        self.init_ps_qs()

    # Initialise P and Q (target) vectors
    def init_ps_qs(self):
        self.p_idx = [i for i in xrange(self.n_buses) if not self.buses[i].slack]
        self.q_idx = [i for i in xrange(self.n_buses) if not (self.buses[i].slack or self.buses[i].v_contr)]
        self.lp, self.lq = len(self.p_idx), len(self.q_idx)

        self.ps = np.zeros(self.lp)
        self.qs = np.zeros(self.lq)
        for i,k in enumerate(self.p_idx):
            self.ps[i] = self.buses[k].pg-self.buses[k].pl
        for i,k in enumerate(self.q_idx):
            self.qs[i] = self.buses[k].qg-self.buses[k].ql
        self.delta_y = np.zeros(self.lp+self.lq)
        self.update_delta_y()

    # Calculate y_bus from line (and transformer) data
    def init_bus_adm_matr(self):
        self.y_bus = np.zeros(shape=(self.n_buses,self.n_buses),dtype=np.complex)
        for pos, line in self.lines:
            self.y_bus[pos] += -1/(line.r+1j*line.x)
            # Add half shunt admittance to y[i,i]
            # for buses at both ends of line
            self.y_bus[pos[0],pos[0]] += 1j*line.b/2
            self.y_bus[pos[1],pos[1]] += 1j*line.b/2
        # Make symmetric
        for i in xrange(self.n_buses):
            for j in xrange(i+1,self.n_buses):
                self.y_bus[j,i] = self.y_bus[i,j]
        # Set y
        for i in xrange(self.n_buses): self.y_bus[i,i] -= sum([self.y_bus[i,j] for j in xrange(self.n_buses) if j != i])

    def update_delta_y(self):
        for i,k in enumerate(self.p_idx):
            self.delta_y[i] = self.ps[i] - self.buses[k].v*sum([self.buses[n].v*(self.y_bus[k,n].real*np.cos(self.buses[k].delta-self.buses[n].delta) + \
                self.y_bus[k,n].imag*np.sin(self.buses[k].delta-self.buses[n].delta)) for n in xrange(self.n_buses)])
        for i,k in enumerate(self.q_idx):
            self.delta_y[self.lp+i] = self.qs[i] - self.buses[k].v*sum([self.buses[n].v*(self.y_bus[k,n].real*np.sin(self.buses[k].delta-self.buses[n].delta) - \
                self.y_bus[k,n].imag*np.cos(self.buses[k].delta-self.buses[n].delta)) for n in xrange(self.n_buses)])

class PowerFlowSolver(object):
    """Power flow solver class."""
    def __init__(self,ps):
        super(PowerFlowSolver, self).__init__()
        self.ps = deepcopy(ps)

    def solve(self):
        raise NotImplementedError("This class should be subclassed, implementing the solve() method.")

class PowerFlowFullNewtonSolver(PowerFlowSolver):
    """Solver class implementing the full Newton-Raphson method"""
    def __init__(self, ps,tol,max_iter):
        super(PowerFlowFullNewtonSolver, self).__init__(ps)
        self.jac = lil_matrix((self.ps.lp+self.ps.lq,self.ps.lp+self.ps.lq))
        self.tol = tol
        self.max_iter = max_iter

    def solve(self):
        self.n_iterations = 0
        delta_x = np.zeros(self.ps.lp+self.ps.lq)
        for newton_i in xrange(self.max_iter):
            self.ps.update_delta_y()
            if all(abs(dy)<self.tol for dy in self.ps.delta_y):
                self.n_iterations = newton_i
                return True
            self.update_jacobian()
            delta_x = spsolve(self.jac,self.ps.delta_y)
            # Update angles and voltages
            for i,k in enumerate(self.ps.p_idx):
                self.ps.buses[k].delta += delta_x[i]
            for i,k in enumerate(self.ps.q_idx):
                self.ps.buses[k].v += delta_x[i+self.ps.lp]
        return False

    def update_jacobian(self):
        # J1
        for i,k in enumerate(self.ps.p_idx):
            for j,n in enumerate(self.ps.p_idx):
                if n==k: self.jac[i,j] = -self.ps.buses[k].v*sum([np.abs(self.ps.y_bus[k,n])*self.ps.buses[n].v*np.sin(self.ps.buses[k].delta-self.ps.buses[n].delta-np.angle(self.ps.y_bus[k,n])) for n in xrange(self.ps.n_buses) if n != k])
                else: self.jac[i,j] = self.ps.buses[k].v * np.abs(self.ps.y_bus[k,n]) * self.ps.buses[n].v * np.sin(self.ps.buses[k].delta-self.ps.buses[n].delta-np.angle(self.ps.y_bus[k,n]))
        # J2
        for i,k in enumerate(self.ps.p_idx):
            for j,n in enumerate(self.ps.q_idx):
                if n==k: self.jac[i,j+self.ps.lp] = self.ps.buses[k].v*np.abs(self.ps.y_bus[k,k])*np.cos(np.angle(self.ps.y_bus[k,k])) + \
                            sum([np.abs(self.ps.y_bus[k,n])*self.ps.buses[n].v*np.cos(self.ps.buses[k].delta-self.ps.buses[n].delta-np.angle(self.ps.y_bus[k,n])) for n in xrange(self.ps.n_buses)])
                else: self.jac[i,j+self.ps.lp] = self.ps.buses[k].v * np.abs(self.ps.y_bus[k,n]) * np.cos(self.ps.buses[k].delta-self.ps.buses[n].delta-np.angle(self.ps.y_bus[k,n]))
        # J3
        for i,k in enumerate(self.ps.q_idx):
            for j,n in enumerate(self.ps.p_idx):
                if n==k: self.jac[i+self.ps.lp,j] = self.ps.buses[k].v*sum([np.abs(self.ps.y_bus[k,n])*self.ps.buses[n].v*np.cos(self.ps.buses[k].delta-self.ps.buses[n].delta-np.angle(self.ps.y_bus[k,n])) for n in xrange(self.ps.n_buses) if n != k])
                else: self.jac[i+self.ps.lp,j] = -self.ps.buses[k].v * np.abs(self.ps.y_bus[k,n]) * self.ps.buses[n].v * np.cos(self.ps.buses[k].delta-self.ps.buses[n].delta-np.angle(self.ps.y_bus[k,n]))
        # J4
        for i,k in enumerate(self.ps.q_idx):
            for j,n in enumerate(self.ps.q_idx):
                if n==k: self.jac[i+self.ps.lp,j+self.ps.lp] = -self.ps.buses[k].v*np.abs(self.ps.y_bus[k,k])*np.sin(np.angle(self.ps.y_bus[k,k])) + \
                            sum([np.abs(self.ps.y_bus[k,n])*self.ps.buses[n].v*np.sin(self.ps.buses[k].delta-self.ps.buses[n].delta-np.angle(self.ps.y_bus[k,n])) for n in xrange(self.ps.n_buses)])
                else: self.jac[i+self.ps.lp,j+self.ps.lp] = self.ps.buses[k].v * np.abs(self.ps.y_bus[k,n]) * np.sin(self.ps.buses[k].delta-self.ps.buses[n].delta-np.angle(self.ps.y_bus[k,n]))

class PowerFlowFastDecoupledSolver(PowerFlowSolver):
    """Solver class implementing the full Newton-Raphson method"""
    def __init__(self, ps,tol,max_iter):
        super(PowerFlowFastDecoupledSolver, self).__init__(ps)
        self.tol = tol
        self.max_iter = max_iter
        self.jac1 = lil_matrix((self.ps.lp,self.ps.lp))
        self.jac4 = lil_matrix((self.ps.lq,self.ps.lq))

    def solve(self):
        self.n_iterations = 0
        delta_x = np.zeros(self.ps.lp+self.ps.lq)
        for newton_i in xrange(self.max_iter):
            self.ps.update_delta_y()
            if all(abs(dy)<self.tol for dy in self.ps.delta_y):
                self.n_iterations = newton_i
                return True
            self.update_jacobians()
            delta_x[0:self.ps.lp] = spsolve(self.jac1,self.ps.delta_y[0:self.ps.lp])
            delta_x[self.ps.lp:] = spsolve(self.jac4,self.ps.delta_y[self.ps.lp:])
            # Update angles and voltages
            for i,k in enumerate(self.ps.p_idx):
                self.ps.buses[k].delta += delta_x[i]
            for i,k in enumerate(self.ps.q_idx):
                self.ps.buses[k].v += delta_x[i+self.ps.lp]
        return False

    def update_jacobians(self):
        # J1
        for i,k in enumerate(self.ps.p_idx):
            for j,n in enumerate(self.ps.p_idx):
                if n==k: self.jac1[i,j] = -self.ps.buses[k].v*sum([np.abs(self.ps.y_bus[k,n])*self.ps.buses[n].v*np.sin(self.ps.buses[k].delta-self.ps.buses[n].delta-np.angle(self.ps.y_bus[k,n])) for n in xrange(self.ps.n_buses) if n != k])
                else: self.jac1[i,j] = self.ps.buses[k].v * np.abs(self.ps.y_bus[k,n]) * self.ps.buses[n].v * np.sin(self.ps.buses[k].delta-self.ps.buses[n].delta-np.angle(self.ps.y_bus[k,n]))
        # J4
        for i,k in enumerate(self.ps.q_idx):
            for j,n in enumerate(self.ps.q_idx):
                if n==k: self.jac4[i,j] = -self.ps.buses[k].v*np.abs(self.ps.y_bus[k,k])*np.sin(np.angle(self.ps.y_bus[k,k])) + \
                            sum([np.abs(self.ps.y_bus[k,n])*self.ps.buses[n].v*np.sin(self.ps.buses[k].delta-self.ps.buses[n].delta-np.angle(self.ps.y_bus[k,n])) for n in xrange(self.ps.n_buses)])
                else: self.jac4[i,j] = self.ps.buses[k].v * np.abs(self.ps.y_bus[k,n]) * np.sin(self.ps.buses[k].delta-self.ps.buses[n].delta-np.angle(self.ps.y_bus[k,n]))

class PowerFlowDCSolver(PowerFlowSolver):
    """Solve a power flow problem with DC method"""
    def __init__(self, ps):
        super(PowerFlowDCSolver, self).__init__(ps)

    def solve(self):
        # Solve -B*d=P
        matr = -np.imag(self.ps.y_bus[1:,1:])
        vec = self.ps.ps
        deltas = solve(matr,vec)
        for i,k in enumerate(self.ps.p_idx):
            self.ps.buses[k].delta = deltas[i]
        return True

# Create a struct-like class with given fields
def Struct(*args, **kwargs):
    def init(self, *iargs, **ikwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        for i in range(len(iargs)):
            setattr(self, args[i], iargs[i])
        for k,v in ikwargs.items():
            setattr(self, k, v)

    name = kwargs.pop("name","DefaultStructName")
    kwargs.update(dict((k, None) for k in args))
    return type(name, (object,), {'__init__': init, '__slots__': kwargs.keys()})

# Immutable, inherits from namedtuple
class Options(namedtuple('Options','tol max_iter decoupled decoupled_dc fast_decoupled')):
    def __new__(cls,tol=1e-14,max_iter=1000,decoupled=False,dc_decoupled=False,fast_decoupled=False):
        return super(Options,cls).__new__(cls,tol,max_iter,decoupled,dc_decoupled,fast_decoupled)

Line = namedtuple('Line','r x b max_mva')

# Mutable, inherits from Struct
class Bus(Struct('v','delta','pg','qg','pl','ql','qgmax','qgmin','slack','v_contr')):
    def __init__(self, v=1., delta=0., pg=0., qg=0., pl=0., ql=0., qgmax=None, qgmin=None, slack=False, v_contr=False):
        super(Bus, self).__init__(v, delta, pg, qg, pl, ql, qgmax, qgmin, slack, v_contr)

def write_matrix_to_file(matrix,f_name):
    f = open(f_name, 'w')
    m,n = matrix.shape
    for i in xrange(m):
        for j in xrange(n):
            f.write(str(matrix[i,j]).replace('.',',')+'; ')
        f.write('\n')
    f.close()
