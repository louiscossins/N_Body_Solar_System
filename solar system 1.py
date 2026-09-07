#%% Solar system
#Key for notation
#------------- means between these code is not original and is a modification of code not written by the user,base code usually above modified code
'''Building a n body simulator so that we can build a solar system to varify that it is functional and matchs observations'''
import numpy as np
from scipy.integrate  import solve_ivp
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift, ifft, ifftshift
from itertools import combinations
NASA_scale_mass = 10**24
NASA_dist = 10**9
G = 6.67430e-11
m_AU =(1.496e11)
Year = (3.154e7)

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
#Jupiter
M_J = 1898.18722*NASA_scale_mass
body_j = np.array([1.937852552375109E+11,7.325905039155092E+11,-7.374262366160095E+09,-1.277913422014432E+04,3.963901398244901E+03,2.695090744358801E+02])
#Saturn
M_Sat = 568.34 *NASA_scale_mass
body_sat = np.array([1.410707180996843E+12,-2.916821078346632E+11,-5.109572887813808E+10,1.418791993311057E+03,9.438404940180696E+03,-2.200312908699624E+02])
#Uranus
M_uran = 86.813*NASA_scale_mass
body_uran = np.array([1.676313087168491E+12,2.396794872661520E+12,-1.281525290605831E+10,-5.630482020671903E+03,3.585571980546193E+03,8.619471980214621E+01])
#Neptune
M_nep = 102.409*NASA_scale_mass
body_nep = np.array([4.468860154108258E+12,-1.111925628534143E+11,-1.006997114441983E+11,9.982679348704859E+01,5.465877955204348E+03,-1.152170813608620E+02])
#Pluto
M_pluto =0.01307*NASA_scale_mass
body_pluto =  np.array([2.712463601300879E+12,-4.494472309103653E+12,-3.036717897176201E+11,4.782325727343739E+03,1.600559714286440E+03,-1.544190231239440E+03])

M_b = np.array([M_s,M_merc,M_ven,M_e,M_ma,M_J,M_Sat,M_uran,M_nep,M_pluto])#Stores masses of planets in solar system as we are dealing with a n body system that needs to index a arbitary number of bodies in the system
n_body = len(M_b)#Determines number of planets in system for us to simulate the interactions between
start_body_coord = np.concatenate((body_s,body_merc,body_ven,body_e,body_ma,body_j,body_sat,body_uran,body_nep,body_pluto))

#------------------------------------------------------------------------------------------------------------------------
#Code below is sourced from https://patrickyoussef.com/blog/nbody/
# Determine Force Pair Indexes
    #self.pairs = list(combinations(range(N), 2))
#------------------------------------------------------------------------------------------------------------------------
Couple = list(combinations(range(n_body), 2))
#----------------------------------------------------------------------------------------------------------------------------


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
N = 1000000
T_begin = 0 
T_end = 220*Year
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

fig = plt.figure(figsize=(16, 10))
ax_solar_sun = plt.axes([0.03,0.2,0.4,0.5], projection='3d')
ax_solar_vd = plt.axes([0.50,0.2,0.4,0.5])
#Bodys plot wobbles removed
'''Loop that will plot for us the planets in the solar system so that we dont have to do it manually'''
for k in range(n_body):
    if k != 0 :
        data =ax_solar_sun.plot((x_pos[k,:]-x_pos[0,:])/m_AU,(y_pos[k,:]-y_pos[0,:])/m_AU,(z_pos[k,:]-z_pos[0,:])/m_AU,linewidth = 0.5)
        ax_solar_sun.set_aspect('equal')
        ax_solar_sun.set_xlabel('x[AU]')
        ax_solar_sun.set_ylabel('y[AU]')
        ax_solar_sun.set_zlabel('z[AU]')
ax_solar_sun.legend(labels = ['Mercury','Venus','Earth','Mars','Jupiter','Saturn','Uranus','Neptune','Pluto'],loc = 'lower right', prop={'size': 8})#Source for prop size:https://stackoverflow.com/questions/7125009/how-to-change-legend-fontsize-with-matplotlib-pyplot 
ax_solar_sun.set_aspect('equal')
#From focus    
e_l_f = np.zeros(int(n_body - 1))
T_so_y_3rdf = np.zeros(int(n_body - 1))
T_so_y_3rdr = np.zeros(int(n_body - 1))
a_solar = np.zeros(int(n_body - 1))
Avg_orbital_vel = np.zeros(int(n_body - 1))
#Calculates orbital velocity,eccentricity,time period and average orbital distance which we will compare to the theoretical stable orbit
for l in range(n_body):
    if l != 0:
        rel_x_l = x_pos[l,:] - x_pos[0,:]
        rel_y_l = y_pos[l,:] - y_pos[0,:]
        rel_z_l = z_pos[l,:] - z_pos[0,:]
        r_l = np.sqrt(rel_x_l**2 + rel_y_l**2 + rel_z_l**2)
        #https://en.wikipedia.org/wiki/Eccentricity_(mathematics) e = (1 - r_min/r_max)/(1+ r_min/r_max)
        e = (1 -(np.min(r_l))/(np.max(r_l)))/(1 +(np.min(r_l))/(np.max(r_l)))
        e_l_f[ l - 1] = e
        a_l = (np.max(r_l)/(1 +e))
        a_solar[l - 1] = a_l
        T_so_y_3rdf[l - 1] = (np.sqrt(((4*(np.pi**2))/(G*(M_s + float(M_b[l]))))*((a_l)**3))/(365*(24)*(3600)))
        T_so_y_3rdr[l - 1] = (np.sqrt(((4*(np.pi**2))/(G*(M_s)))*((a_l)**3))/(365*(24)*(3600)))
        
        rel_vx_l = x_vel[l,:] - x_vel[0,:]
        rel_vy_l = y_vel[l,:] - y_vel[0,:]
        rel_vz_l = z_vel[l,:] - z_vel[0,:]
        r_vl = np.sqrt(rel_vx_l**2 + rel_vy_l**2 + rel_vz_l**2) 
        r_vlmean = np.mean(r_vl)
        Avg_orbital_vel[l - 1] = (r_vlmean)
colour = ['Blue','Orange','Green','Red','Purple','Brown','Pink','Gray','Olive']
ax_solar_vd.scatter(np.log10(a_solar/m_AU),np.log10(Avg_orbital_vel/1e3),color = colour)
ax_solar_vd.grid(True)

r = np.linspace(0.09*m_AU,45*m_AU,1000)
v = np.sqrt((G*M_s)/r)
ax_solar_vd.plot(np.log10(r/m_AU),np.log10(v/1e3),linestyle = 'dashed',color = 'red',label = 'logv = 0.5*log(GM) - 0.5log(r)')
ax_solar_vd.set_aspect('equal')
ax_solar_vd.set_xlabel('log[d[AU]]')
ax_solar_vd.set_ylabel('log[v[kms^-1]')
ax_solar_vd.grid(True)
ax_solar_vd.legend()



