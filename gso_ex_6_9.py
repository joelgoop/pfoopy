from pf_classes import *

lines = [
    ((1,3), Line(0.009,0.1,1.72,12.)),
    ((1,4), Line(0.0045,0.05,0.88,12.)),
    ((3,4), Line(0.00225,0.025,0.44,12.)),
    ((0,4), Line(0.0015,0.02,0.,6.)), # Transformer
    ((2,3), Line(0.00075,0.01,0.,10.)) # Transformer
]
buses = [
    Bus(slack=True, pl=0., ql=0.),
    Bus(pg=0.,qg=0.,pl=8.,ql=2.8),
    Bus(v=1.05,v_contr=True,pg=5.2,pl=0.8,ql=0.4,qgmax=4.,qgmin=-2.8),
    Bus(pg=0.,qg=0.,pl=0.,ql=0),
    Bus(pg=0.,qg=0.,pl=0,ql=0)
]

tolerance = 1e-14
max_iterations = 1000

ps = PowerSystem(buses,lines)
solver = PowerFlowFullNewtonSolver(ps,tolerance,max_iterations)
dc_solver = PowerFlowDCSolver(ps)

if dc_solver.solve() and solver.solve():
    print 'Success!'
    # print 'Success! Solved in {0} iterations.'.format(solver.n_iterations)
    print [b.v for b in dc_solver.ps.buses]
    print [np.degrees(b.delta) for b in dc_solver.ps.buses]
    print [solver.ps.buses[i].v-dc_solver.ps.buses[i].v for i in xrange(solver.ps.n_buses)]
    print [np.degrees(solver.ps.buses[i].delta-dc_solver.ps.buses[i].delta) for i in xrange(solver.ps.n_buses)]
else:
    print 'Failure.'