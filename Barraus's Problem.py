#Barraus Problem
'''In this file we are verifying that our code is functional by replicating burraus probelm also known as the prythagorus problem for its initial conditions'''
import numpy as np
from scipy.integrate  import solve_ivp
import matplotlib.pyplot as plt

G = 1
m1 = 3
m2 = 4
m3 = 5
def bodies_3(t,data):
    #Corrdinates , [body1,body2,body3]
    positions = np.array([data[::4],data[1::4]]) 
    velocities = np.array([data[2::4],data[3::4]])
    #Absolute distance,, [body 2 - body 1,body 3 - body 2,body 3 - body 1]
    d =np.array([(positions[0,1] - positions[0,0])**2 + (positions[1,1] - positions[1,0])**2,(positions[0,2] - positions[0,1])**2 + (positions[1,2] - positions[1,1])**2,(positions[0,2] - positions[0,0])**2 + (positions[1,2] - positions[1,0])**2])
    #Relative distance, [body 2 - body 1,body 3 - body 2,body 3 - body 1]
    change_in_r = np.array([positions[:,1]-positions[:,0],positions[:,2]-positions[:,1],positions[:,2]-positions[:,0]])
    #x - axis acceleration ,[body1,body2,body3]
    dv_x_dt = np.array([G*m2*(change_in_r[0,0])/ (d[0]**(3/2)) + G*m3*(change_in_r[2,0])/ (d[2]**(3/2)),G*m1*(-change_in_r[0,0])/ (d[0]**(3/2)) + G*m3*(change_in_r[1,0])/ (d[1]**(3/2)),G*m1*(-change_in_r[2,0])/(d[2]**(3/2)) + G*m2*(-change_in_r[1,0])/(d[1]**(3/2))])
    #y - axis acceleration ,[body1,body2,body3]
    dv_y_dt = np.array([G*m2*(change_in_r[0,1])/ (d[0]**(3/2)) + G*m3*(change_in_r[2,1])/ (d[2]**(3/2)),G*m1*(-change_in_r[0,1])/(d[0]**(3/2)) + G*m3*(change_in_r[1,1])/(d[1]**(3/2)),G*m1*(-change_in_r[2,1])/(d[2]**(3/2)) + G*m2*(-change_in_r[1,1])/(d[1]**(3/2))])
    
    output_data = np.array([velocities[0,0],velocities[1,0],dv_x_dt[0],dv_y_dt[0],velocities[0,1],velocities[1,1],dv_x_dt[1],dv_y_dt[1],velocities[0,2],velocities[1,2],dv_x_dt[2],dv_y_dt[2]])
    return output_data
def convergence(body):
    d_12 = (body[4,:] - body[0,:])**2 + (body[5,:] - body[1,:])**2
    d_23 = (body[8,:] - body[4,:])**2 + (body[9,:] - body[5,:])**2
    d_13 = (body[8,:] - body[0,:])**2 + (body[9,:] - body[1,:])**2
    KE_total = (1/2)*m1*((body[2, :])**2 +(body[3, :])**2) + (1/2)*m2*((body[6, :])**2 + (body[7, :])**2)+ (1/2)*m3*((body[10, :])**2 + (body[11, :])**2)
    PE_total = -G*m1*m2/(2*np.sqrt(d_12)) -G*m1*m2/(2*np.sqrt(d_12)) -G*m1*m3/(2*np.sqrt(d_13)) -G*m1*m3/(2*np.sqrt(d_13)) -G*m2*m3/(2*np.sqrt(d_23)) -G*m2*m3/(2*np.sqrt(d_23))
    E_total = KE_total + PE_total
    print('simulation finished')
    return E_total, KE_total ,PE_total
#body 1
x1_0 = 1
y1_0 = 3
v_x1_0 = 0
v_y1_0 = 0
#body 2
x2_0= -2
y2_0 = -1
v_x2_0 = 0
v_y2_0 = 0
#body 3
x3_0= 1
y3_0 = -1
v_x3_0 = 0
v_y3_0 = 0




N = 10000
Time = 70
Years = np.linspace(0, Time, N)

t_length = np.array([0,Time])
#Convergence test 
True_tol = 4e-9
c = -4
i = True
fig = plt.figure(figsize=(10,8))
while i == True:
    tol = 10**c
    dynamics = solve_ivp(fun = bodies_3,t_span = t_length ,y0 =[x1_0,y1_0,v_x1_0,v_y1_0,x2_0,y2_0,v_x2_0,v_y2_0,x3_0,y3_0,v_x3_0,v_y3_0], t_eval = Years,rtol = tol,atol = tol)
    body = dynamics.y
    E_total = convergence(body)[0]
    if abs((E_total-E_total[0])/E_total[0])[-1] <True_tol:
            i = False
            print('It has converged with tolerance rtol and etol = ' + str(tol) )
    elif c == -13.5:
        i = False
        print('System requires beyond the computers tolerance capacity to converge')
    else:
        c -= 0.2




T1,T2 = 60,70
filter1 = (Years >= T1) & (Years <= T2)

fig = plt.figure(figsize=(16, 10))
ax_tsnaps1 = plt.axes([0.03, 0.55, 0.225/2, 0.35/2])
T1,T2 = 0,10
filter1 = (Years >= T1) & (Years <= T2)
ax_tsnaps1.plot(body[0,:][filter1],body[1,:][filter1],label = 'body one')
ax_tsnaps1.plot(body[4,:][filter1],body[5,:][filter1],label = 'body two')
ax_tsnaps1.plot(body[8, :][filter1],body[9,:][filter1],label = 'body Three')
ax_tsnaps1.set_xlabel('x')
ax_tsnaps1.set_ylabel('y')
ax_tsnaps1.legend(loc = 'upper left')
ax_tsnaps1.axis('equal')
ax_tsnaps1.grid()
ax_tsnaps1.set_title('T: 0 to 10')

ax_tsnaps2 = plt.axes([0.03 +0.262/2, 0.55, 0.225/2, 0.35/2])
T1,T2 = 10,20
filter1 = (Years >= T1) & (Years <= T2)
ax_tsnaps2.plot(body[0,:][filter1],body[1,:][filter1],label = 'body one')
ax_tsnaps2.plot(body[4,:][filter1],body[5,:][filter1],label = 'body two')
ax_tsnaps2.plot(body[8, :][filter1],body[9,:][filter1],label = 'body Three')
ax_tsnaps2.set_xlabel('x')
ax_tsnaps2.axis('equal')
ax_tsnaps2.grid()
ax_tsnaps2.set_title('T: 10 to 20')

ax_tsnaps3 = plt.axes([0.03 +2*0.262/2, 0.55, 0.225/2, 0.35/2])
T1,T2 = 20,30
filter1 = (Years >= T1) & (Years <= T2)
ax_tsnaps3.plot(body[0,:][filter1],body[1,:][filter1],label = 'body one')
ax_tsnaps3.plot(body[4,:][filter1],body[5,:][filter1],label = 'body two')
ax_tsnaps3.plot(body[8, :][filter1],body[9,:][filter1],label = 'body Three')
ax_tsnaps3.set_xlabel('x')
ax_tsnaps3.axis('equal')
ax_tsnaps3.grid()
ax_tsnaps3.set_title('T: 20 to 30')

ax_tsnaps4 = plt.axes([0.03 +3*0.262/2, 0.55, 0.225/2, 0.35/2])
T1,T2 = 30,40
filter1 = (Years >= T1) & (Years <= T2)
ax_tsnaps4.plot(body[0,:][filter1],body[1,:][filter1],label = 'body one')
ax_tsnaps4.plot(body[4,:][filter1],body[5,:][filter1],label = 'body two')
ax_tsnaps4.plot(body[8, :][filter1],body[9,:][filter1],label = 'body Three')
ax_tsnaps4.set_xlabel('x')
ax_tsnaps4.axis('equal')
ax_tsnaps4.grid()
ax_tsnaps4.set_title('T: 30 to 40')

ax_tsnaps5 = plt.axes([0.03 +4*0.262/2, 0.55, 0.225/2, 0.35/2])
T1,T2 = 40,50
filter1 = (Years >= T1) & (Years <= T2)
ax_tsnaps5.plot(body[0,:][filter1],body[1,:][filter1],label = 'body one')
ax_tsnaps5.plot(body[4,:][filter1],body[5,:][filter1],label = 'body two')
ax_tsnaps5.plot(body[8, :][filter1],body[9,:][filter1],label = 'body Three')
ax_tsnaps5.set_xlabel('x')
ax_tsnaps5.axis('equal')
ax_tsnaps5.grid()
ax_tsnaps5.set_title('T: 40 to 50')

ax_tsnaps6 = plt.axes([0.03 +5*0.262/2, 0.55, 0.225/2, 0.35/2])
T1,T2 = 50,60
filter1 = (Years >= T1) & (Years <= T2)
ax_tsnaps6.plot(body[0,:][filter1],body[1,:][filter1],label = 'body one')
ax_tsnaps6.plot(body[4,:][filter1],body[5,:][filter1],label = 'body two')
ax_tsnaps6.plot(body[8, :][filter1],body[9,:][filter1],label = 'body Three')
ax_tsnaps6.set_xlabel('x')
ax_tsnaps6.axis('equal')
ax_tsnaps6.grid()
ax_tsnaps6.set_title('T: 50 to 60')

ax_tsnaps7 = plt.axes([0.032 +6*0.262/2, 0.55, 0.150/2, 0.35/2])
T1,T2 = 60,70
filter1 = (Years >= T1) & (Years <= T2)
ax_tsnaps7.plot(body[0,:][filter1],body[1,:][filter1],label = 'body one')
ax_tsnaps7.plot(body[4,:][filter1],body[5,:][filter1],label = 'body two')
ax_tsnaps7.plot(body[8, :][filter1],body[9,:][filter1],label = 'body Three')
ax_tsnaps7.set_xlabel('x')
ax_tsnaps7.axis('equal')
ax_tsnaps7.grid()
ax_tsnaps7.set_title('T: 60 to 70')
ax_tsnaps7.set_xlim(-5,10)

ax_span = plt.axes([0.035,0.045,0.6275,0.5])

#plots of distance
d_12 = (body[4,:] - body[0,:])**2 + (body[5,:] - body[1,:])**2
d_23 = (body[8,:] - body[4,:])**2 + (body[9,:] - body[5,:])**2
d_13 = (body[8,:] - body[0,:])**2 + (body[9,:] - body[1,:])**2  
ax_span.plot(Years,np.sqrt(d_12),color = 'red',label = r'$d_{12}$')
ax_span.plot(Years,np.sqrt(d_23),color = 'purple',label = r'$d_{23}$')
ax_span.plot(Years,np.sqrt(d_13),color = 'blue',label = r'$d_{13}$')
ax_span.legend(loc = 'upper left')
T_border = np.array([10,20,30,40,50,60])
for f in T_border:
    ax_span.axvline(f, color = 'grey',linestyle ='--')
ax_span.set_ylim(0,12)
ax_span.set_xlim(0,70)
ax_span.set_xlabel('T')
ax_span.set_ylabel('distance between bodies')
ax_span.set_aspect('equal')
ax_span.grid(True)


E_total = convergence(body)[0]
KE_total = convergence(body)[1]
PE_total = convergence(body)[2]

ax_energy_change = plt.axes([0.04 + 0.6615,0.045,0.27,0.4])
ax_energy_change.plot(Years,abs((E_total-E_total[0])/E_total[0]),label = 'Tolarance =' + str(tol))
ax_energy_change.set_xlabel('T')
ax_energy_change.set_ylabel(r'$|\Delta E|$')
ax_energy_change.legend()
ax_energy_change.set_xlim(0,70)
ax_energy_change.grid(True)

#%% Bonus animation

import numpy as np
from scipy.integrate  import solve_ivp
import matplotlib.pyplot as plt

G = 1
m1 = 3
m2 = 4
m3 = 5
def bodies_3(t,data):
    #Corrdinates , [body1,body,body3]
    positions = np.array([data[::4],data[1::4]]) 
    velocities = np.array([data[2::4],data[3::4]])
    #Absolute distance,, [body 2 - body 1,body 3 - body 2,body 3 - body 1]
    d =np.array([(positions[0,1] - positions[0,0])**2 + (positions[1,1] - positions[1,0])**2,(positions[0,2] - positions[0,1])**2 + (positions[1,2] - positions[1,1])**2,(positions[0,2] - positions[0,0])**2 + (positions[1,2] - positions[1,0])**2])
    #Relative distance, [body 2 - body 1,body 3 - body 2,body 3 - body 1]
    change_in_r = np.array([positions[:,1]-positions[:,0],positions[:,2]-positions[:,1],positions[:,2]-positions[:,0]])
    #x - axis acceleration ,[body1,body2,body3]
    x_acceleration = np.array([G*m2*(change_in_r[0,0])/ (d[0]**(3/2)) + G*m3*(change_in_r[2,0])/ (d[2]**(3/2)),G*m1*(-change_in_r[0,0])/ (d[0]**(3/2)) + G*m3*(change_in_r[1,0])/ (d[1]**(3/2)),G*m1*(-change_in_r[2,0])/(d[2]**(3/2)) + G*m2*(-change_in_r[1,0])/(d[1]**(3/2))])
    #y - axis acceleration ,[body1,body2,body3]
    y_acceleration = np.array([G*m2*(change_in_r[0,1])/ (d[0]**(3/2)) + G*m3*(change_in_r[2,1])/ (d[2]**(3/2)),G*m1*(-change_in_r[0,1])/(d[0]**(3/2)) + G*m3*(change_in_r[1,1])/(d[1]**(3/2)),G*m1*(-change_in_r[2,1])/(d[2]**(3/2)) + G*m2*(-change_in_r[1,1])/(d[1]**(3/2))])
    
    output_data = np.array([velocities[0,0],velocities[1,0],x_acceleration[0],y_acceleration[0],velocities[0,1],velocities[1,1],x_acceleration[1],y_acceleration[1],velocities[0,2],velocities[1,2],x_acceleration[2],y_acceleration[2]])
    return output_data

#body 1
x1_0 = 1
y1_0 = 3
v_x1_0 = 0
v_y1_0 = 0
#body 2
x2_0= -2
y2_0 = -1
v_x2_0 = 0
v_y2_0 = 0
#body 3
x3_0= 1
y3_0 = -1
v_x3_0 = 0
v_y3_0 = 0

N = 10000
Time = 70
Years = np.linspace(0, Time, N)

t_length = np.array([0,Time])
#Convergence test 
True_tol = 4e-9
c = -4
i = True
fig = plt.figure(figsize=(10,8))
while i == True:
    tol = 10**c
    sol = solve_ivp(fun = bodies_3,t_span = t_length ,y0 =[x1_0,y1_0,v_x1_0,v_y1_0,x2_0,y2_0,v_x2_0,v_y2_0,x3_0,y3_0,v_x3_0,v_y3_0], t_eval = Years,rtol = tol,atol = tol)
    body = sol.y
    d_12 = (body[4,:] - body[0,:])**2 + (body[5,:] - body[1,:])**2
    d_23 = (body[8,:] - body[4,:])**2 + (body[9,:] - body[5,:])**2
    d_13 = (body[8,:] - body[0,:])**2 + (body[9,:] - body[1,:])**2
    KE_total = (1/2)*m1*((body[2, :])**2 +(body[3, :])**2) + (1/2)*m2*((body[6, :])**2 + (body[7, :])**2)+ (1/2)*m3*((body[10, :])**2 + (body[11, :])**2)
    PE_total = -G*m1*m2/(2*np.sqrt(d_12)) -G*m1*m2/(2*np.sqrt(d_12)) -G*m1*m3/(2*np.sqrt(d_13)) -G*m1*m3/(2*np.sqrt(d_13)) -G*m2*m3/(2*np.sqrt(d_23)) -G*m2*m3/(2*np.sqrt(d_23))
    E_total = KE_total + PE_total
    if abs((E_total-E_total[0])/E_total[0])[-1] <True_tol:
            i = False
            print('It has converged with tolerance rtol and etol = ' + str(tol) )
    elif c == -13.5:
        i = False
        print('System requires beyond the computers tolerance capacity to converge')
    else:
        c -= 0.2
        
plt.figure()
i = 0
while i <= N- 20:
    plt.clf()
    if i <= 300:
        s = 0
    elif i > 300:
        s = i -280
    plt.xlim()
    plt.scatter(body[0,:][i],body[1,:][i],label = 'body 1')
    plt.scatter(body[4,:][i],body[5,:][i],label = 'body 2')
    plt.scatter(body[8,:][i],body[9,:][i],label = 'body 3')
    plt.plot(body[0,:][s:i],body[1,:][s:i],color = 'blue',linestyle = 'dashed')
    plt.plot(body[4,:][s:i],body[5,:][s:i],color = 'orange',linestyle = 'dashed')
    plt.plot(body[8,:][s:i],body[9,:][s:i],color = 'green',linestyle = 'dashed')
    i += 20
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc = 'upper left')
    plt.pause(0.1)

