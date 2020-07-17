from casadi import *
import numpy as np

## INTERFACE 1

# # Symbolic representation
# x=SX.sym('x')
# y=SX.sym('y')
# z=y-(1-x)**2
# discrete = [True, False]
# options = {"discrete" : discrete}

# f=x**2+100*z**2
# P=dict(x=vertcat(x,y),f=f)

# # Create solver instance
# F=nlpsol('F','bonmin',P,options)

# # Solve the problem
# r=F(x0=[2.5,3.0])
# print(r['x'])


## INTERFACE 2

# opti = Opti();

# x = opti.variable()
# y = opti.variable()
# z = opti.variable()
# discrete = [False, False, True]
# options = {"discrete" : discrete}

# opti.minimize(x**2+100*z**2)
# opti.subject_to(z+(1-x)**2-y==0)

# opti.set_initial(x,2.5)
# opti.set_initial(y,3.0)
# opti.set_initial(z,0.75)

# opti.solver('bonmin',options)
# sol = opti.solve()

# print(sol.value(x),sol.value(y),sol.value(z))
# print(sol.value(opti.f))

## INTERFACE 2, EXPERIMENT 2

opti = Opti();

x = opti.variable()
y = opti.variable()
z = opti.variable()
discrete = [True,True,False]
options = {"discrete" : discrete}


opti.minimize(2*x**4 - x + 2*y**4 - y + 2*z**4 - z)
opti.subject_to([x >= 0.5, x>= 0.75])

# opti.set_initial(x,2.5)

opti.solver('bonmin')
sol = opti.solve()

print(f"x: {sol.value(x)}")
print(sol.value(opti.f))
