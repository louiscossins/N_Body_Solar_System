#Earth to mars closest approach
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

Mass_array = np.array([M_s,M_merc,M_ven,M_e,M_ma,M_J,M_Sat,M_uran,M_nep,M_pluto])
n_body = len(Mass_array)
initial_body = np.concatenate((body_s,body_merc,body_ven,body_e,body_ma,body_j,body_sat,body_uran,body_nep,body_pluto))

####################################################################################################################
#Code below is sourced from https://patrickyoussef.com/blog/nbody/
# Determine Force Pair Indexes
    #self.pairs = list(combinations(range(N), 2))
###################################################################################################################
Couple = list(combinations(range(n_body), 2))

def Gen_body(t,data):
    #Instead of doing nested loops i researched a more computationally more efficient way and found this
    #Corrdinates , [N*bodies]
    positions = np.array([data[::6],data[1::6],data[2::6]])
    velocities = np.array([data[3::6],data[4::6],data[5::6]])
    # convert to State matrix containing positions and velocities fo that we can follow the following code
    #https://numpy.org/doc/stable/reference/generated/numpy.vstack.html
    state_matrix = np.vstack((positions, velocities)).T
    #############################################################################################################
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
    #################################################################################################################
    D = 3 # taking the modulus to find the dimensions
    Positions_state = state_matrix[:,:D]
    Velocities_state = state_matrix[:,D:]
    
    Output_function = np.zeros_like(data)
    Output_function[::6] = Velocities_state[:, 0]  
    Output_function[1::6] = Velocities_state[:, 1]
    Output_function[2::6] = Velocities_state[:, 2]
    #Finding acceleration:
    for body_i, body_j in Couple:
        r1,r2 = Positions_state[body_i],Positions_state[body_j]
        r_vec = r2 - r1
        r = np.linalg.norm(r_vec)
        
        F = G*Mass_array[body_i]*Mass_array[body_j]* r_vec/r**3
        a1 = np.array(F/Mass_array[body_i],dtype = 'float')
        a2 = np.array(-F / Mass_array[body_j],dtype = 'float')
        Output_function[3 + body_i * 6:6 + body_i * 6] += a1
        Output_function[3 + body_j * 6:6 + body_j * 6] += a2
    
    return Output_function

N = 1000000
Time = 250*Year
second = np.linspace(0, Time, N)
Years = second/Year

t_length = [0,Time]
sol = solve_ivp(fun = Gen_body,t_span = t_length ,y0 = initial_body, t_eval = second,rtol = 10**-8,atol = 10**-8)
body = sol.y
x_pos = body[::6,:]
y_pos = body[1::6,:]
z_pos = body[2::6,:]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for k in range(n_body):
    ax.plot((x_pos[k,:])/m_AU,(y_pos[k,:])/m_AU,(z_pos[k,:])/m_AU,linewidth = 0.5)
    ax.set_aspect('equal')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#Bodys plot wobbles removed
for k in range(n_body):
    ax.plot((x_pos[k,:]-x_pos[0,:])/m_AU,(y_pos[k,:]-y_pos[0,:])/m_AU,(z_pos[k,:]-z_pos[0,:])/m_AU,linewidth = 0.5)
    ax.set_aspect('equal')

#Orbital approach
fig = plt.figure(figsize=(16, 10))
ax_solar_me = plt.axes([0.03,0.2,0.4,0.5])
ax_solar_me_f = plt.axes([0.50,0.2,0.4,0.5])
#earth-venus
x_ev = x_pos[3,:]-x_pos[2,:]
y_ev = y_pos[3,:]-y_pos[2,:]
z_ev = z_pos[3,:]-z_pos[2,:]
r_ev = np.sqrt(x_ev**2 + y_ev**2 + z_ev**2)

#mars-earth
x_me = x_pos[4,:]-x_pos[3,:]
y_me = y_pos[4,:]-y_pos[3,:]
z_me = z_pos[4,:]-z_pos[3,:]
r_me = np.sqrt(x_me**2 + y_me**2 + z_me**2)


ax_solar_me.plot(Years,r_me/m_AU,color = 'blue',label = 'Distance from earth to mars over time')
ax_solar_me.set_xlim(Years[0],Years[-1])
ax_solar_me.set_xlabel('T[yr]')
ax_solar_me.set_ylabel('d[AU]')
ax_solar_me.legend(loc = 'upper right')
#We want to know global closest approach and local closest apprach so to find that we will transform this into reciprical shape
r_me_centred = r_me - np.mean(r_me)
f = fftshift(fftfreq(N,Time/N))[int(N/2 - 1):int(N-1)]
r_em_f = np.abs(fftshift(fft(r_me_centred)))
Power = ((r_em_f**2)/(N**2))[int(N/2 - 1):int(N-1)]

ax_solar_me_f.plot(f,Power,color = 'blue',label = 'Power Spectrum of evolution of earth mars distance over time')
ax_solar_me_f.set_xlim(0,2e-7)
ax_solar_me_f.set_ylim(1e13,3e22)
ax_solar_me_f.set_yscale("log")
ax_solar_me_f.set_xlabel('frequency[Hz]')
ax_solar_me_f.set_ylabel('Amplituded^2/N^2')

filt_f = f[Power - 1e19 >= 0 ]
filt_Power = Power[Power - 1e19 >= 0 ]
f_peak_1and2 = np.array([filt_f[0],filt_f[6]])
pow_peak_1and2 = np.array([filt_Power[0],filt_Power[6]])
T_period_1 = 1/(365*(24)*(3600)*(f_peak_1and2))
ax_solar_me_f.scatter(f_peak_1and2[1],pow_peak_1and2[1],color = 'red',label = 'Highest peak period ,T:'+ str(T_period_1[1]) + 'yrs')
ax_solar_me_f.scatter(f_peak_1and2[0],pow_peak_1and2[0],color = 'green',label = 'Lowest frequency peak period , T:'+ str(T_period_1[0]) + 'yrs')
ax_solar_me_f.legend(loc = 'upper right')