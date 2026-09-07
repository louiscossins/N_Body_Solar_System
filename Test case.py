#Two body probelm
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

M_star_kg = (1.989e30)
G = 6.67430e-11
m1 = 1*M_star_kg
m2 = 1*M_star_kg
m_AU =(1.496e11)
Years = (3.154e7)
def bodies_2(t,data):
    global positions
    positions = np.array([data[::4],data[1::4]]) 
    velocities = np.array([data[2::4],data[3::4]]) 
    d = (positions[0,1] - positions[0,0])**2 + (positions[1,1] - positions[1,0])**2
    change_in_r = positions[:,1]-positions[:,0]
    dv_x_dt = np.array([G*m2*(change_in_r[0])/ (d**(3/2)),-G*m1*(change_in_r[0])/ (d**(3/2))])
    dv_y_dt = np.array([G*m2*(change_in_r[1])/(d**(3/2)),-G*m1*(change_in_r[1])/ (d**(3/2))])
    output_data = np.array([velocities[0,0],velocities[1,0],dv_x_dt[0],dv_y_dt[0],velocities[0,1],velocities[1,1],dv_x_dt[1],dv_y_dt[1]])
    return output_data

def convergence(space_dynamics):
    d_12 = (space_dynamics[4,:] - space_dynamics[0,:])**2 + (space_dynamics[5,:] - space_dynamics[1,:])**2
    KE_total = (1/2)*m1*((space_dynamics[2, :])**2 +(space_dynamics[3, :])**2) + (1/2)*m2*((space_dynamics[6, :])**2 + (space_dynamics[7, :])**2)
    PE_total = -G*m1*m2/(2*np.sqrt(d_12)) -G*m1*m2/(2*np.sqrt(d_12))
    E_total = KE_total + PE_total
    print('Simulation completed one cycle')
    return E_total,KE_total,PE_total
#body 1
x1_0 = -0.5*m_AU
y1_0 = 0*m_AU
v_x1_0 = 0
v_y1_0 = -(15e3)
#body 2
x2_0= 0.5*m_AU
y2_0 = 0*m_AU
v_x2_0 = 0
v_y2_0 = (15e3)

N = 10000
Time = 5*(3.154e7)
second = np.linspace(0, Time, N)
Years = second/(3.154e7)

t_length = np.array([0,Time])
#Convergence test 
c = -2
i = True
while i == True:
    tol = 10**c
    dynamics = solve_ivp(fun = bodies_2,t_span = t_length ,y0 = [x1_0,y1_0,v_x1_0,v_y1_0,x2_0,y2_0,v_x2_0,v_y2_0], t_eval = second,rtol = tol,atol = tol)
    space_dynamics = dynamics.y
    E_total = convergence(space_dynamics)[0]
    if abs((E_total-E_total[0])/E_total[0])[-1] <1e-5:
            i = False
            print('It has converged with tolerance rtol and etol = ' + str(tol) )
    elif c == -13.5:
        i = False
        print('System requires byond the computers tolerance capacity to converge')
    else:
        c -= 1
print('Convergence test finished')
#Plotting data
fig, ax = plt.subplots(nrows=1, ncols=2,figsize = (12,5))

ax[0].plot(space_dynamics[0, :]/m_AU,space_dynamics[1, :]/m_AU,color = 'red',label = 'body 1')
ax[0].plot(space_dynamics[4, :]/m_AU,space_dynamics[5, :]/m_AU,color = 'black',label = 'body 2')
ax[0].legend(loc = 'upper left')
ax[0].axis('equal')
ax[0].set_xlabel('x[AU]')
ax[0].set_ylabel('y[AU]')

ax[1].plot(Years,space_dynamics[0, :]/m_AU,color = 'red',label = 'body 1')
ax[1].plot(Years,space_dynamics[4, :]/m_AU,color = 'black',label = 'body 2')
ax[1].set_ylabel('x[AU]')
ax[1].set_xlabel('t[Yr]')
'''Plotting relative change in energy against time to see how fast it diverges from full conservation of energy '''
#Energy
PE_total = convergence(space_dynamics)[2]
KE_total = convergence(space_dynamics)[1]
E_total = convergence(space_dynamics)[0]
plt.figure()
plt.plot(Years,KE_total,label = 'Kinetic energy')
plt.plot(Years,PE_total,label = 'Potential energy')
plt.plot(Years,E_total,label = 'Total energy')
plt.xlabel('T[yrs]')
plt.ylabel('E[J]')
plt.legend()
year_p = Years[space_dynamics[1,:]/m_AU <= 0.00000000001]
hours = 365*24*(year_p)
plt.figure()
plt.plot(Years,abs((E_total-E_total[0])/E_total[0]),label = 'tolerance = '+ str(tol))
plt.xlabel('T[yr]')
plt.ylabel(r'$|\Delta E|$')
#plt.axhline(y = 1e-10,color = 'r',linestyle = '--',label = 'tolerance')
plt.legend()
###########################Should result in a stable body##########################



#%% Bonus
#Two body problem,large and small body
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

M_star_kg = (1.989e30)
G = 6.67430e-11
m1 = 100*M_star_kg
m2 = 0.02*M_star_kg
m_AU =(1.496e11)
Years = (3.154e7)
def bodies_2(t,data):
    global positions
    positions = np.array([data[::4],data[1::4]]) 
    velocities = np.array([data[2::4],data[3::4]]) 
    d = (positions[0,1] - positions[0,0])**2 + (positions[1,1] - positions[1,0])**2
    change_in_r = positions[:,1]-positions[:,0]
    dv_x_dt = np.array([G*m2*(change_in_r[0])/ (d**(3/2)),-G*m1*(change_in_r[0])/ (d**(3/2))])
    dv_y_dt = np.array([G*m2*(change_in_r[1])/(d**(3/2)),-G*m1*(change_in_r[1])/ (d**(3/2))])
    output_data = np.array([velocities[0,0],velocities[1,0],dv_x_dt[0],dv_y_dt[0],velocities[0,1],velocities[1,1],dv_x_dt[1],dv_y_dt[1]])
    return output_data

def convergence(space_dynamics):
    d_12 = (space_dynamics[4,:] - space_dynamics[0,:])**2 + (space_dynamics[5,:] - space_dynamics[1,:])**2
    KE_total = (1/2)*m1*((space_dynamics[2, :])**2 +(space_dynamics[3, :])**2) + (1/2)*m2*((space_dynamics[6, :])**2 + (space_dynamics[7, :])**2)
    PE_total = -G*m1*m2/(2*np.sqrt(d_12)) -G*m1*m2/(2*np.sqrt(d_12))
    E_total = KE_total + PE_total
    print('Simulation completed one cycle')
    return E_total,KE_total,PE_total
#body 1
x1_0 = -0.5*m_AU
y1_0 = 0*m_AU
v_x1_0 = 0
v_y1_0 = 0
#body 2
x2_0= 0.5*m_AU
y2_0 = 0*m_AU
v_x2_0 = 0
#recall that for a stable orbit entripetal force has to equal gravitational therefore v = sqrt(G*M/r)
v_y2_0 = (297e3)

N = 10000
Time = 5*(3.154e7)
second = np.linspace(0, Time, N)
Years = second/(3.154e7)

t_length = np.array([0,Time])
#Convergence test 
c = -2
i = True
while i == True:
    tol = 10**c
    dynamics = solve_ivp(fun = bodies_2,t_span = t_length ,y0 = [x1_0,y1_0,v_x1_0,v_y1_0,x2_0,y2_0,v_x2_0,v_y2_0], t_eval = second,rtol = tol,atol = tol)
    space_dynamics = dynamics.y
    E_total = convergence(space_dynamics)[0]
    if abs((E_total-E_total[0])/E_total[0])[-1] <1e-5:
            i = False
            print('It has converged with tolerance rtol and etol = ' + str(tol) )
    elif c == -13.5:
        i = False
        print('System requires byond the computers tolerance capacity to converge')
    else:
        c -= 1
print('Convergence test finished')
#Plotting data
fig, ax = plt.subplots(nrows=1, ncols=2,figsize = (12,5))

ax[0].plot(space_dynamics[0, :]/m_AU,space_dynamics[1, :]/m_AU,color = 'red',label = 'body 1')
ax[0].plot(space_dynamics[4, :]/m_AU,space_dynamics[5, :]/m_AU,color = 'black',label = 'body 2')
ax[0].legend(loc = 'upper left')
ax[0].axis('equal')
ax[0].set_xlabel('x[AU]')
ax[0].set_ylabel('y[AU]')

ax[1].plot(Years,space_dynamics[0, :]/m_AU,color = 'red',label = 'body 1')
ax[1].plot(Years,space_dynamics[4, :]/m_AU,color = 'black',label = 'body 2')
ax[1].set_ylabel('x[AU]')
ax[1].set_xlabel('t[Yr]')
'''Plotting relative change in energy against time to see how fast it diverges from full conservation of energy '''
#Energy
PE_total = convergence(space_dynamics)[2]
KE_total = convergence(space_dynamics)[1]
E_total = convergence(space_dynamics)[0]
plt.figure()
plt.plot(Years,KE_total,label = 'Kinetic energy')
plt.plot(Years,PE_total,label = 'Potential energy')
plt.plot(Years,E_total,label = 'Total energy')
plt.xlabel('T[yrs]')
plt.ylabel('E[J]')
plt.legend()
year_p = Years[space_dynamics[1,:]/m_AU <= 0.00000000001]
hours = 365*24*(year_p)
plt.figure()
plt.plot(Years,abs((E_total-E_total[0])/E_total[0]),label = 'tolerance = '+ str(tol))
plt.xlabel('T[yr]')
plt.ylabel(r'$|\Delta E|$')
#plt.axhline(y = 1e-10,color = 'r',linestyle = '--',label = 'tolerance')
plt.legend()
###########################Should result in a stable body##########################

