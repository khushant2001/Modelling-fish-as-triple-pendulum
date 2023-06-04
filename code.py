#Khushant khurana 
#Simulating underwater motion using three simple pendulums 
#swinging out of phase. 

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy import signal

t = sp.symbols('t')
theta1 = sp.Function('theta1')(t)
theta2 = sp.Function('theta2')(t)
theta3 = sp.Function('theta3')(t)

#l1, l2, l3, g = sp.symbols('l1 l2 l3 g')
tau1, tau2, tau3 = sp.symbols('tau1 tau2 tau3')

plt.close('all')
#parameters

l1 = 1
l2 = 1
l3 = 1
b1 = .1
b2 = .1
b3 = .1
m1 = 1
m2 = .1
m3 = .1
k = 10
b = 30

#arrays for storing final values
x_final = []
y_final = []
x1 = []
x2 = []
y1 = []
y2 = []

#defining all inertias

J3_3 = (1/12)*m3*(l3**2 + b3**2)
J2_3 = (1/12)*m2*(l2**2 + b2**2)
J1_3 = (1/12)*m1*(l1**2 + b1**2)
theta1_dot = sp.diff(theta1 , t)
theta2_dot = sp.diff(theta2 , t)
theta3_dot = sp.diff(theta3 , t)
J1_1 = J2_1 = J3_1 = 0
J1_2 = J2_2 = J3_2 = 0

#DEFINING GENERALIZED VELOCITIES
q_dot = sp.Matrix([
    [theta1_dot],
    [theta2_dot],
    [theta3_dot]
    ])

q_2dot = sp.diff(q_dot , t)

#DEFINING BASIC MATRIX TRANSFORMATIONS
def skew(a):
    return sp.Matrix([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0]])
def unskew(a):
    return sp.Matrix([
        [a[2,1]],
        [a[0,2]],
        [a[1,0]]])

def rot1(t):
    return sp.Matrix([
        [1, 0, 0],
        [0, sp.cos(t), -sp.sin(t)],
        [0, sp.sin(t), sp.cos(t)]])

def rot2(t):
    return sp.Matrix([
        [sp.cos(t), 0, sp.sin(t)],
        [0, 1, 0],
        [-sp.sin(t), 0, sp.cos(t)]])

def rot3(t):
    return sp.Matrix([
        [sp.cos(t), -sp.sin(t), 0],
        [sp.sin(t), sp.cos(t), 0],
        [0, 0, 1]])

#All position vectors of center of mass and end of linkages in their respective frames. 
rc_1 = sp.Matrix([
    [l1/2],
    [0],
    [0]])
rc_2 = sp.Matrix([
    [l2/2],
    [0],
    [0]
    ])
r_1 = sp.Matrix([
    [l1],
    [0],
    [0]
    ]) 
r_2 = sp.Matrix([
    [l2],
    [0],
    [0]
    ])
rc_3 = sp.Matrix([
    [l3/2],
    [0],
    [0]
    ]) 

#Getting all rotation matrices
R01 = rot3(theta1)
R12 = rot3(theta2)
R23 = rot3(theta3)
R03 = R01 * R12 * R23
R02 = R01*R12

#ROTATIONAL VELOCITIES
omega01_11 = R01.T * sp.diff(R01, t)
omega02_22 = R02.T * sp.diff(R02, t)
omega03_33 = R03.T * sp.diff(R03, t)

#Linear velocities COM in respective frames
vc_1 = sp.simplify(omega01_11 * rc_1)
vc_2 = sp.simplify(omega02_22 * rc_2)
vc_3 = sp.simplify(omega03_33 * rc_3)

#Linear velocities in 0 frame
vC1_0 = R01 * vc_1
vC2_0 = sp.simplify(R02 * vc_2 + sp.simplify(R01 * (omega01_11 * r_1)))

#//CONFUSING!!!!!!!!!!! about the vectors. 
vC3_0 = R03 * vc_3 + sp.simplify(R01 * (omega01_11 * r_1)) + sp.simplify(R02 * (omega02_22 * r_2))

#FINDING JACOBIANS FOR LINEAR VELOCITIES
B1 = (vC1_0.jacobian(q_dot))
B2 = (vC2_0.jacobian(q_dot))
B3 = (vC3_0.jacobian(q_dot))

#FINDING JACOBIANS FOR ROTATIONAL VELOCITIES
B4 = (unskew(omega01_11).jacobian(q_dot))
B5 = (unskew(omega02_22).jacobian(q_dot))
B6 = (unskew(omega03_33).jacobian(q_dot))

#DEFINING THE FULL JACOBIAN MATRIX
B = (sp.Matrix([[B1],[B2],[B3],[B4],[B5],[B6]]))

#DEFINING GRAVITATION FORCE FOR ALL LINKAGES
W_1 = sp.Matrix([
    [0],
    [0],
    [-9.81]
])

W_2 = sp.Matrix([
    [0],
    [0],
    [-9.81]
])

W_3 = sp.Matrix([
    [0],
    [0],
    [-9.81]
])

#Defining the mass matrix - contating mass and inertias
M = sp.zeros(18,18)
M[:3,:3] = m1*sp.eye(3)
M[3:6,3:6] = m2*sp.eye(3)
M[6:9, 6:9] = m3*sp.eye(3)
M[9:12,9:12] = sp.diag(J1_1, J1_2, J1_3)
M[12:15,12:15] = sp.diag(J2_1, J2_2, J2_3)
M[15:18,15:18] = sp.diag(J3_1, J3_2, J3_3)

#Defining the frame rotation
D = sp.zeros(18,18)
D[9:12,9:12] = omega01_11
D[12:15,12:15] = omega02_22
D[15:18, 15:18] = omega03_33

#Defining the external forces and torques
G = sp.zeros(18,1)
G[:3,:] = W_1
G[3:6,:] = W_2
G[6:9,:] = W_3
G[11] = tau1-tau2 + k*(theta2 - theta1)*(l1/2) + b*(theta2_dot - theta1_dot)*l1/2
G[14] = tau2-tau3 + k*(theta3 - theta2)*(l2/2) + b*(theta3_dot - theta1_dot)*l2/2
G[17] = tau3

Bdot = sp.diff(B, t)

Mstar = sp.simplify(B.T * M * B)
Nstar = sp.simplify(B.T * (D*M*B + M*Bdot))
Gstar = sp.simplify(B.T * G)
#eom = (Mstar.inv() * (Gstar - Nstar*q_dot))


#Defining state vector for solving the differential equations
from scipy.integrate import odeint
from numpy import linalg as la
y = sp.Matrix([
    [theta1],
    [theta2],
    [theta3],
    [theta1_dot],
    [theta2_dot],
    [theta3_dot]
])

t = np.linspace(0,10,1000)

#Defining torques that will act on the system
def t1(time):
    return 1*np.sin(.5*time)
def t2(time):
    return 0#-40*signal.square(2 * np.pi * 20 * time)
def t3(time):
    return 2*np.sin(4*time)


#Creating numpy functions to calculate numberical values for the sympy matrices. 

M1 = sp.lambdify([theta1,theta2,theta3,theta1_dot,theta2_dot,theta3_dot], Mstar, "numpy")
N1= sp.lambdify([theta1,theta2,theta3,theta1_dot,theta2_dot,theta3_dot], Nstar, "numpy")
G1 = sp.lambdify([theta1,theta2,theta3,theta1_dot,theta2_dot,theta3_dot, tau1, tau2, tau3], Gstar, "numpy")
a = sp.lambdify([theta1,theta2,theta3,theta1_dot,theta2_dot,theta3_dot], y, "numpy")

#Defining derivatives function. 
def derivatives(x,time):
  m11 = M1(x[0],x[1],x[2],x[3],x[4],x[5])
  n11 = N1(x[0],x[1],x[2],x[3],x[4],x[5])
  g11 = G1(x[0],x[1],x[2],x[3],x[4],x[5],t1(time),t2(time),t3(time))
  x = a(x[0],x[1],x[2],x[3],x[4],x[5])
  tempN = -la.inv(m11)@n11
  tempG = la.inv(m11)@g11
  tempB = np.zeros((6,6))
  tempB[0:3,3:6] = np.eye(3)
  tempB[3:6,3:6] = tempN
  finalG = np.zeros((6,1))
  finalG[3:6,] = tempG
  sol = tempB@x + finalG
  one = sol.item(0)
  two = sol.item(1)
  three = sol.item(2)
  four = sol.item(3)
  five = sol.item(4)
  six = sol.item(5)
  return [one,two,three,four,five,six]

initial = [0,0,0,0,0,0]
X = odeint(derivatives, initial,t)

theta = X[:,0] + X[:,1] + X[:,2]

for n in range(1000):
    x_final.append(l1*np.cos(X[:,0][n]) + l2*np.cos(X[:,1][n]+X[:,0][n]) + l3*np.cos(theta[n]))
    y_final.append(l1*np.sin(X[:,0][n]) + l2*np.sin(X[:,1][n]+X[:,0][n]) + l3*np.sin(theta[n]))
    x1.append(l1*np.cos(X[:,0][n]))
    y1.append(l1*np.sin(X[:,0][n]))
    x2.append(l1*np.cos(X[:,0][n]) + l2*np.cos(X[:,1][n]+X[:,0][n]))
    y2.append(l1*np.sin(X[:,0][n]) + l2*np.sin(X[:,1][n]+X[:,0][n]))


#Plotting all plots
plt.figure()
plt.plot(t,X[:,0])
plt.title("Theta 1 vs time")
plt.xlabel("Time [sec]")
plt.ylabel("Theta1 [rad]")
plt.grid()
plt.xlim(0,t[-1])

plt.figure()
plt.plot(t,X[:,1])
plt.title("Theta 2 vs time")
plt.xlabel("Time [sec]")
plt.ylabel("Theta2 [rad]")
plt.grid()
plt.xlim(0,t[-1])

plt.figure()
plt.plot(t,X[:,2])
plt.title("Theta 3 vs time")
plt.xlabel("Time [sec]")
plt.ylabel("Theta3 [rad]")
plt.grid()
plt.xlim(0,t[-1])

plt.figure()
plt.plot(t,X[:,3])
plt.title("Theta 1_dot vs time")
plt.xlabel("Time [sec]")
plt.ylabel("Theta1_dot [rad/sec]")
plt.grid()
plt.xlim(0,t[-1])

plt.figure()
plt.plot(t,X[:,4])
plt.title("Theta 2_dot vs time")
plt.xlabel("Time [sec]")
plt.ylabel("Theta2_dot [rad/sec]")
plt.grid()
plt.xlim(0,t[-1])

plt.figure()
plt.plot(t,X[:,5])
plt.title("Theta 3_dot vs time")
plt.xlabel("Time [sec]")
plt.ylabel("Theta3_dot [rad/sec]")
plt.grid()
plt.xlim(0,t[-1])


plt.figure()
plt.plot(t,theta)
plt.grid()
plt.title("Final theta of the fish")
plt.xlabel("Time [sec]")
plt.ylabel("Theta_final [rad]")
plt.xlim(0,t[-1])


plt.figure()
plt.plot(t,x_final)
plt.grid()
plt.xlabel("Time [sec]")
plt.ylabel("X position [m]")
plt.title("X position of the fish")
plt.xlim(0,t[-1])

plt.figure()
plt.title("Y position of the fish")
plt.grid()
plt.plot(t,y_final)
plt.xlabel("Time [sec]")
plt.ylabel("Y position [m]")
plt.xlim(0,t[-1])

plt.figure()
plt.plot(x_final,y_final)
plt.grid()
plt.title("Y vs X position")
plt.xlabel("X position [m]")
plt.ylabel("Y position [m]")
plt.xlim(min(x_final),max(x_final))

#The following code runs the simulation. Once it opens to host on the browser, you should be able to 
#see the animation. Once you are done with it. Close the tab and manually stop the program in ide. 
import vpython
from vpython import *

#Initializing ball positions.
ball1 = vpython.sphere(color = color.green, radius = 0.3, make_trail=True, retain=20)
ball2 = vpython.sphere(color = color.blue, radius = 0.3, make_trail=True, retain=20)
ball3 = vpython.sphere(color = color.red, radius = 0.3, make_trail=True, retain=20)

#Initializing rod positions. 
rod1 = cylinder(pos=vector(0,0,0),axis=vector(0,0,0), radius=0.03)
rod2 = cylinder(pos=vector(0,0,0),axis=vector(0,0,0), radius=0.03)
rod3 = cylinder(pos=vector(0,0,0),axis=vector(0,0,0), radius=0.03)

#Initializing the base 
base  = box(pos=vector(0,-4.25,0),axis=vector(1,0,0),
            size=vector(10,0.5,10) )

#Adding shadows for dramatic effect. 
s1 = cylinder(pos=vector(0,-3.99,0),axis=vector(0,-0.1,0), radius=0.8, color=color.gray(luminance=0.7))
s2 = cylinder(pos=vector(0,-3.99,0),axis=vector(0,-0.1,0), radius=0.8, color=color.gray(luminance=0.7))
s3 = cylinder(pos=vector(0,-3.99,0),axis=vector(0,-0.1,0), radius=0.8, color=color.gray(luminance=0.7))

#Starting the simulation. 
print('Start')
i = 0
while (True):
    rate(40)
    i = i + 1
    i = i % len(x1)
    ball1.pos = vector(x1[i], 0, y1[i])
    ball2.pos = vector(x2[i], 0, y2[i])
    ball3.pos = vector(x_final[i], 0, y_final[i])
    rod1.axis = vector(x1[i], 0, y1[i])
    rod2.pos = vector(x1[i], 0, y1[i])
    rod2.axis = vector(x2[i]-x1[i], 0, y2[i]-y1[i])
    rod3.pos = vector(x2[i], 0, y2[i])
    rod3.axis = vector(x_final[i]-x2[i], 0, y_final[i]-y2[i])
    s1.pos = vector(x1[i], -3.99, y1[i])
    s2.pos = vector(x2[i], -3.99, y2[i])
    s3.pos = vector(x_final[i], -3.99, y_final[i])
