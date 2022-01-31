import bnlearn as bn
from pgmpy.factors.discrete import TabularCPD

alarm_clock = TabularCPD(variable='alarm_clock', variable_card=2, values=[[0.01], [0.99]])
traffic = TabularCPD(variable='traffic', variable_card=2, values=[[0.38], [0.62]])

oversleep = TabularCPD(variable='oversleep', variable_card=2,
                  values=[[0.12, 0.81],
                          [0.88, 0.19]],
                  evidence=['alarm_clock'], evidence_card=[2])

on_time = TabularCPD(variable='on_time', variable_card=2,
                   values=[[0.02, 0.42, 0.73, 0.95],
                           [0.98, 0.58, 0.27, 0.05]],
                   evidence=['oversleep', 'traffic'],
                   evidence_card=[2, 2])

edges = [('alarm_clock', 'oversleep'),
         ('oversleep', 'on_time'),
         ('traffic', 'on_time')]

DAG = bn.make_DAG(edges)
DAG = bn.make_DAG(DAG, CPD=[alarm_clock, traffic, on_time, oversleep])

bn.print_CPD(DAG)

bn.inference.fit(DAG, variables=['on_time'], evidence={})
bn.inference.fit(DAG, variables=['oversleep'], evidence={})