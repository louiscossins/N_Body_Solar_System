#%% Phobos and diemos

#Key for notation
#------------- means between these code is not original and is a modification of code not written by the user,base code usually above modified code

import numpy as np
from scipy.integrate  import solve_ivp
import matplotlib.pyplot as plt
from itertools import combinations


NASA_scale_mass = 10**24
NASA_dist = 10**9
G = 6.67430e-11
m_AU =(1.496e11)
Year = (3.154e7)
'''In this file we will be building a solar system with the inclusion of phobos and deimos excluding neptune,uranus and pluto as the simulation would take too much time.Focusing on the orbit of phobos and deimos around mars and the orbit of the bary center since we developed the code in previous files'''

def Sim_brain(t,data):
    #Instead of doing nested loops i researched a more computationally more efficient way and found this
    #Corrdinates , [N*bodies]
    positions = np.array([data[::6],data[1::6],data[2::6]])
    velocities = np.array([data[3::6],data[4::6],data[5::6]])
    # convert to State matrix containing positions and velocities fo that we can follow the following code
    #https://numpy.org/doc/stable/reference/generated/numpy.vstack.html
    state_matrix = np.vstack((positions, velocities)).T
    #--------------------------------------------------------------------------------------------------------------------------
    #code below up to return is sourced from https://patrickyoussef.com/blog/nbody/
    # Useful Variables
   #N, D = X.shape # Get the number of bodies, and dimensionality
   #D = D // 2 # Get the number of dimensions, are we 2d or 3d?
   #R = X[:, :D] # Submatrix with all positions
   #V = X[:, D:] # Submatrix with all velocities

   # Build Placeholder Structure
   #Xdot = np.zeros_like(X) # Xdot is the same size as X
  # Xdot[:, :D] = V # Fill in velocities from state 

   # Iterate Over Pairs and Fill Out Acceleration
   # self.pairs gets defined when we start a sim
   # body_i, body_j are the indices of the bodies
   #for body_i, body_j in self.pairs:

       # Get vector from body_i => body_j and its magnitude
       #r1, r2 = R[body_i], R[body_j] # Positions of body_i and body_j
       #r_vec = r2 - r1 # Vector from body_i => body_j
       #r = np.linalg.norm(r_vec) # Distance from body_i => body_j

       # Find Force from body_i => body_j
       #F = self.G * self.masses[body_i] * self.masses[body_j] * r_vec / r**3
       #a1 =  F / self.masses[body_i] # Compute acceleration for body_i
      # a2 = -F / self.masses[body_j] # Compute acceleration for body_j

       # Apply acceleration to body_i and body_j
       #Xdot[body_i, D:] += a1
       #Xdot[body_j, D:] += a2
    #---------------------------------------------------------------------------------------------------------------------------
    D = 3 # taking the modulus to find the dimensions
    Positions_state = state_matrix[:,:D]
    Velocities_state = state_matrix[:,D:]
    
    Output_function = np.zeros_like(data)#Feeds back into the numerical simulation for the next interval time step,needs to indentical to the input so that it functions
    Output_function[::6] = Velocities_state[:, 0]  
    Output_function[1::6] = Velocities_state[:, 1]
    Output_function[2::6] = Velocities_state[:, 2]
    #Finding acceleration:
    for body_i, body_j in Couple:
        r1,r2 = Positions_state[body_i],Positions_state[body_j]
        r_vec = r2 - r1
        r = np.linalg.norm(r_vec)
        
        F = G*M_b[body_i]*M_b[body_j]* r_vec/r**3
        a1 = np.array(F/M_b[body_i],dtype = 'float')
        a2 = np.array(-F / M_b[body_j],dtype = 'float')
        Output_function[3 + body_i * 6:6 + body_i * 6] += a1
        Output_function[3 + body_j * 6:6 + body_j * 6] += a2
    
    return Output_function
#-------------------------------------------------------------------------------------------------------------------------------------------

#2460643.500000000 = A.D. 2024-Nov-29 00:00:00.0000 TDB 
#Sun
M_s = 1988410*NASA_scale_mass
body_s = np.array([-8.922905734358741E+08,-7.154997853244003E+08,2.742300083338033E+07,1.213292310243211E+01,-7.049355666497667E+00,-1.944610352606711E-01])
#Mercury
M_merc = 0.3302*NASA_scale_mass
body_merc = np.array([4.065582633618618E+10,2.439690962257454E+10,-1.731217784872803E+09,-3.471456613000211E+04,4.381366609870928E+04,6.766100659069389E+03])
#Venus
M_ven = 4.8685*NASA_scale_mass
body_ven = np.array([1.075907003909110E+11,-3.702201986827331E+09,-6.273161000821998E+09,8.246393864502393E+02,3.484051240797015E+04,4.315104798935323E+02])
#Earth
M_e = 5.97219*NASA_scale_mass
body_e = np.array([5.698009428496168E+10,1.350324079211623E+11,1.969863120234013E+07,-2.788372956226473E+04,1.157178541777992E+04,1.328595477572136E-01])
#Mars
M_ma = 0.64171*NASA_scale_mass
body_ma = np.array([-1.396428587693312E+10,2.348282601446951E+11,5.283859000303820E+09,-2.326455688279275E+04,7.089862997605068E+02,5.856725364605979E+02])
#Phobos
M_pho = (1.08e-8)*NASA_scale_mass
body_pho = np.array([-1.396699448517580E+10,2.348191838614369E+11,5.284785003186896E+09,-2.146904683613360E+04,8.331506496892999E+01,-3.190933769923456E+02])
#Deimos
M_dei = (1.8e-9)*NASA_scale_mass
body_dei = np.array([-1.394670943635384E+10,2.348154184135526E+11,5.275100603325665E+09,-2.255720431705495E+04,1.836154567323201E+03,3.530532659311515E+02])
#Jupiter
M_J = 1898.18722*NASA_scale_mass
body_j = np.array([1.937852552375109E+11,7.325905039155092E+11,-7.374262366160095E+09,-1.277913422014432E+04,3.963901398244901E+03,2.695090744358801E+02])
#Saturn
M_Sat = 568.34 *NASA_scale_mass
body_sat = np.array([1.410707180996843E+12,-2.916821078346632E+11,-5.109572887813808E+10,1.418791993311057E+03,9.438404940180696E+03,-2.200312908699624E+02])


M_b= np.array([M_s,M_merc,M_ven,M_e,M_ma,M_pho,M_dei,M_J,M_Sat])
n_body = len(M_b)
start_body_coord = np.concatenate((body_s,body_merc,body_ven,body_e,body_ma,body_pho,body_dei,body_j,body_sat))

#------------------------------------------------------------------------------------------------------------------------
#Code below is sourced from https://patrickyoussef.com/blog/nbody/
# Determine Force Pair Indexes
    #self.pairs = list(combinations(range(N), 2))
#------------------------------------------------------------------------------------------------------------------------
Couple = list(combinations(range(n_body), 2))
#----------------------------------------------------------------------------------------------------------------------------
N = 1000000
T_begin = 0 
T_end = 0.5*Year
second = np.linspace(T_begin, T_end, N)
Years = second/Year

T_span = np.array([T_begin,T_end])
T_length = T_end - T_begin
dynamics = solve_ivp(fun = Sim_brain,t_span = T_span ,y0 = start_body_coord, t_eval = second,rtol = 10**-10,atol = 10**-10)
space_dynamics = dynamics.y
x_pos = space_dynamics[::6,:]
y_pos = space_dynamics[1::6,:]
z_pos = space_dynamics[2::6,:]
x_vel = space_dynamics[3::6,:]
y_vel = space_dynamics[4::6,:]
z_vel = space_dynamics[5::6,:]


#Only plot mars,phobos and deimos
#mars
x_mars = x_pos[4,:] - x_pos[4,:]
y_mars = y_pos[4,:] - y_pos[4,:]
z_mars = z_pos[4,:] - z_pos[4,:]
#The equatorial radius of mars is 3396.2 km
R_mars = 3396.2e3*np.ones(100)
#Mars,constructing a planet that isnt a point so we can see what phobos and deimos will look like orbiting it
th = np.linspace(0,2*np.pi,100)
p = np.linspace(0,np.pi,100)
th,p = np.meshgrid(th,p)
x_surf = R_mars*np.sin(p)*np.cos(th)
y_surf = R_mars*np.sin(p)*np.sin(th)
z_surf = R_mars*np.cos(p)

#Phobos
x_mp = x_pos[5,:]-x_pos[4,:]
y_mp = y_pos[5,:]-y_pos[4,:]
z_mp = z_pos[5,:]-z_pos[4,:]
r_mp = np.sqrt(x_mp**2 + y_mp**2 + z_mp**2)
#Deimos
x_md = x_pos[6,:]-x_pos[4,:]
y_md = y_pos[6,:]-y_pos[4,:]
z_md = z_pos[6,:]-z_pos[4,:]
r_md = np.sqrt(x_md**2 + y_md**2 + z_md**2)
#CoM of mars system
x_cm_mpd = (M_ma*x_mars + M_pho*x_mp + M_dei*x_md)/(M_ma + M_pho + M_dei)
y_cm_mpd = (M_ma*y_mars + M_pho*y_mp + M_dei*y_md)/(M_ma + M_pho + M_dei)
z_cm_mpd = (M_ma*z_mars + M_pho*z_mp + M_dei*z_md)/(M_ma + M_pho + M_dei)
r_cm_mpd = np.sqrt(x_cm_mpd**2 + y_cm_mpd**2 + z_cm_mpd**2 )

fig = plt.figure(figsize=(16, 10))
ax_solar_mars = plt.axes([0.03,0.2,0.4,0.5], projection='3d')
ax_solar_energy = plt.axes([0.50,0.2,0.4,0.5])
ax_solar_mars.plot(x_md/(10**3),y_md/(10**3),z_md/(10**3),color = 'purple',label = 'deimos')
ax_solar_mars.plot(x_mp/(10**3),y_mp/(10**3),z_mp/(10**3),color = 'grey',label = 'phobos')
ax_solar_mars.plot(x_cm_mpd/(10**3),y_cm_mpd/(10**3),z_cm_mpd/(10**3),color = 'blue',label = 'CoM of Mars-moons system')
ax_solar_mars.plot_surface(x_surf/(10**3),y_surf/(10**3),z_surf/(10**3),color = 'red',alpha = 0.6)
ax_solar_mars.set_aspect('equal')
ax_solar_mars.set_xlabel('x[km]')
ax_solar_mars.set_ylabel('y[km]')
ax_solar_mars.set_zlabel('z[km]')
ax_solar_mars.legend(loc = 'lower right')
KE = np.zeros(N)
PE = np.zeros(N)
'''Allows us to determine whether energy is conserved which is essential for the simulation to be accurate'''
for i in range(n_body):
    #Sum up KE
    KE += (0.5)*M_b[i]*(x_vel[i]**2 + y_vel[i]**2 + z_vel[i]**2)
    for j in range(n_body):
        if i != j:
            #Sum up potentials using PE = G*m_j*m_i/(2*d_ij)
            d_x = x_pos[j] - x_pos[i]
            d_y = y_pos[j] - y_pos[i]
            d_z = z_pos[j] - z_pos[i]
            d_ij = d_x**2 + d_y**2 + d_z**2
            PE -= G*M_b[i]*M_b[j]/(2*np.sqrt(d_ij))
E_total = KE + PE
E_change = (E_total - E_total[0])/(E_total[0])
ax_solar_energy.plot(Years,np.abs(E_change),label = 'Tolerance:10^-11')
ax_solar_energy.grid(True)
ax_solar_energy.set_xlabel('T[yr]')
ax_solar_energy.set_ylabel(r'$|\Delta E|$')
ax_solar_energy.legend(loc = 'upper left')
ax_solar_energy.set_xlim(Years[0],Years[-1])


'''Calculating eccentricity and period of orbit which we can compare to observational data to check the validity of the model'''
#mars-phobos
e_p = (1 -(np.min(r_mp))/(np.max(r_mp)))/(1 +(np.min(r_mp))/(np.max(r_mp)))
T_p_s = np.sqrt(((4*(np.pi**2))/(G*M_ma))*((np.max(r_mp)/(1 +e_p))**3))
T_p_d = T_p_s/(24*(3600))
#mars-deimos
e_d = (1 -(np.min(r_md))/(np.max(r_md)))/(1 +(np.min(r_md))/(np.max(r_md)))
T_d_s = np.sqrt(((4*(np.pi**2))/(G*M_ma))*((np.max(r_md)/(1 +e_d))**3))
T_d_d = T_d_s/(24*(3600))
