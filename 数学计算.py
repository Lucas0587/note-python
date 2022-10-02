'''import sympy
x=sympy.Symbol('x')
y=sympy.Symbol('y')
R=sympy.Symbol('R')
T=sympy.Symbol('T')
Vm=sympy.Symbol('Vm')
p=sympy.Symbol('p')
a=sympy.Symbol('a')
b=sympy.Symbol('b')
f=R*T*Vm/(Vm-b)-a/Vm
print(sympy.diff(f,Vm))
print(sympy.solve([x + 5*y - 2, -3*x + 6*y - 15], [x, y]))'''
'''f=R*T*sympy.exp(-a/(R*T*Vm))/(Vm-b)
print(sympy.solve([sympy.diff(f,Vm),sympy.diff(f,Vm,2)],[Vm,T]))'''


'''from sympy import *
f = symbols('f', cls=Function)
x = symbols('x')
eq = Eq(f(x).diff(x,1)+f(x)+f(x)**2, 0)
print(dsolve(eq, f(x)))
C1 = symbols('C1')
eqr = -C1/(C1 - exp(x))
eqr1 = eqr.subs(x, 0)
print(solveset(eqr1 - 1, C1))
eqr2 = eqr.subs(C1, 1/2)'''


'''import scipy.integrate as integrate
import numpy as np
result = integrate.quad(lambda x: x**2 + np.exp(x) + 1, 0, 1)
print(result)'''

'''from sympy import *
x = symbols('x')
def Deinteg(Func):
    print('∫(', Func, ')dx')
    print('=', integrate(Func, x), '+ C\n')
f1x = 4 * pow(x, 3) + 3 * x * x + 2 * x + 1
f2x = x * exp(x) + exp(x)
f3x = exp(2 * x) + cos(3 * x)
f4x = sin(x) / (1 + sin(x) + cos(x))
Deinteg(f1x)
Deinteg(f2x)
Deinteg(f3x)
Deinteg(f4x)'''

'''from sympy import *
x = symbols('x')
def Deinteg(Func, upperL, lowerL):
    print('∫[', upperL, ',', lowerL, '](', Func, ')dx')
    print('=(', integrate(Func, x), ')|[', upperL, ',', lowerL, ']')
    print('=', integrate(Func, (x, upperL, lowerL)), '\n')
fx = 3* pow(x, 3) + 3 * x * x + 2 * x + 1
gx = sin(x) / (1 + sin(x) + cos(x))

Deinteg(fx, 0, 2)
Deinteg(gx, 0, pi / 2)'''

# 画图
import matplotlib.pyplot as plt
import numpy as np
from sympy import *
x_1 = np.arange(-10, 10, 0.01)
#y_1 = [-0.5/(1 - exp(x)) for x in x_1]
y_1=[(x**3-3*x**2+1) for x in x_1]
#plt.plot(x_1, y_1)
y_2=[(exp(-x**2)) for x in x_1]
plt.plot(x_1,y_2)
#plt.axis([-6,6,-10,10])
plt.grid()
plt.show()