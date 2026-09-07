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

#2431436.500000000 = A.D. 1944-Dec-12 00:00:00.0000 TDB
#Sun
M_s = 1988410*NASA_scale_mass
body_s = np.array([9.942195684052139E+08,-6.635976549065267E+08,-1.703488891972436E+07,6.331882937016421E+00,1.199077332002465E+01,-2.491245826368336E-01])
#Mercury
M_merc = 0.3302*NASA_scale_mass
body_merc = np.array([4.586373076234342E+10,1.976097890825299E+10,-2.474724679199295E+09,-2.965150058039233E+04,4.652423676095690E+04,6.522280259781271E+03])
#Venus
M_ven = 4.8685*NASA_scale_mass
body_ven = np.array([1.090609236842270E+11,8.687873466293210E+09,-6.133014097758080E+09,-3.153233283630953E+03,3.474291516505002E+04,6.515932322230125E+02])
#Earth
M_e = 5.97219*NASA_scale_mass
body_e = np.array([2.497267510244667E+10,1.446490224506460E+11,9.948907862827182E+05,-2.987888198109100E+04,4.760727938008329E+03,4.454427974542874E-01])
#Mars
M_ma = 0.64171*NASA_scale_mass
body_ma = np.array([-8.619197115561332E+10,-2.085449162875009E+11,-2.215390094058022E+09,2.327350021700786E+04,-7.284418868025958E+03,-7.275772622089720E+02])
#Jupiter
M_J = 1898.18722*NASA_scale_mass
body_j = np.array([-7.873399254468387E+11,1.904848700680841E+11,1.686338256879952E+10,-3.235136611888204E+03,-1.208127626798907E+04,1.220235931361113E+02])
#Saturn
M_Sat = 568.34 *NASA_scale_mass
body_sat = np.array([-1.738799162414872E+11,1.338699308753722E+12,-1.652659276849341E+10,-1.009378184256943E+04,-1.268109409275259E+03,4.228397694194276E+02])
#Uranus
M_uran = 86.813*NASA_scale_mass
body_uran = np.array([8.937693159790881E+11,2.743723118703001E+12,-1.401821178790808E+09,-6.527072349930102E+03,1.791251278804028E+03,9.137438556628175E+01])
#Neptune
M_nep = 102.409*NASA_scale_mass
body_nep = np.array([-4.507458615468074E+12,-4.123777459259757E+11,1.123320495734106E+11,4.645054607582844E+02,-5.378165049799225E+03,1.006892398983119E+02])
#Pluto
M_pluto =0.01307*NASA_scale_mass
body_pluto =  np.array([-3.574336116242614E+12,4.310141913129216E+12,5.723921885278347E+11,-2.695501085805976E+03,-3.974247503128041E+03,1.222636433946770E+03])

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
Time = 76*Year
Year_intervals = int(np.round(1000000/76,decimals = 0))
second = np.linspace(0, Time, N)
Years = second/Year

t_length = [0,Time]
sol = solve_ivp(fun = Gen_body,t_span = t_length ,y0 = initial_body, t_eval = second,rtol = 10**-10,atol = 10**-10)
body = sol.y
x_pos = body[::6,:]
y_pos = body[1::6,:]
z_pos = body[2::6,:]
x_vel = body[3::6,:]
y_vel = body[4::6,:]
z_vel = body[5::6,:]

#sun with fixed barycentre
fig = plt.figure(figsize=(16, 10))
ax_solar_sun_var1 = plt.axes([0.06,0.2,0.4,0.5])
ax_error = plt.axes([0.56,0.2,0.4,0.5])


moment_x = np.zeros(N)
moment_y = np.zeros(N)
moment_z = np.zeros(N)
for i in range(n_body):
    moment_x += np.array(Mass_array[i]*x_pos[i,:], dtype = 'float')
    moment_y += np.array(Mass_array[i]*y_pos[i,:], dtype = 'float')
    moment_z += np.array(Mass_array[i]*z_pos[i,:], dtype = 'float')
x_barycentre = moment_x/sum(Mass_array)
y_barycentre = moment_y/sum(Mass_array)
z_barycentre = moment_z/sum(Mass_array)
    
rel_x_bs = x_barycentre - x_pos[0]
rel_y_bs = y_barycentre - y_pos[0]
rel_z_bs = z_barycentre - z_pos[0]
#The equatorial radius of sun is 696,340 km
R_sun = (696340e3)*np.ones(100)
theta = np.linspace(0,2*np.pi,100)
x_sun = R_sun*np.cos(theta)
y_sun = R_sun*np.sin(theta)
ax_solar_sun_var1.plot(rel_x_bs/1e9,rel_y_bs/1e9,color = 'black',label = 'Barycener of solarsystem')
ax_solar_sun_var1.fill(x_sun/1e9,y_sun/1e9,color = 'yellow',alpha = 0.6)
ax_solar_sun_var1.scatter(rel_x_bs[::2*13157]/1e9,rel_y_bs[::2*13157]/1e9,color = 'red')
#text point plot sourced from here https://datascienceparichay.com/article/matplotlib-label-points-on-scatter-plot/
for i in range(39):
    ax_solar_sun_var1.text(rel_x_bs[2*13157*i]/1e9,rel_y_bs[2*13157*i]/1e9,str(1944 + 2*i))
ax_solar_sun_var1.set_xlabel('x[*10^9 m]')
ax_solar_sun_var1.set_ylabel('y[*10^9 m]')
ax_solar_sun_var1.set_aspect('equal')
ax_solar_sun_var1.legend(loc = 'lower right')
ax_solar_sun_var1.grid(True)

#Energy divergence test over 1000 years
N = 500000
Time = 500*Year
second = np.linspace(0, Time, N)
Years = second/Year
t_length = [0,Time]
c = np.array([-7,-8,-9,-10,-11])
for tol in c:
    sub = 10**int(tol)
    sol = solve_ivp(fun = Gen_body,t_span = t_length ,y0 = initial_body, t_eval = second,rtol = sub,atol = sub)
    body = sol.y
    x_pos = body[::6,:]
    y_pos = body[1::6,:]
    z_pos = body[2::6,:]
    x_vel = body[3::6,:]
    y_vel = body[4::6,:]
    z_vel = body[5::6,:]
    KE = np.zeros(N)
    PE = np.zeros(N)
    for i in range(n_body):
	#Summs up the kinetic energy from each body
        KE += (0.5)*Mass_array[i]*(x_vel[i]**2 + y_vel[i]**2 + z_vel[i]**2)
        for j in range(n_body):
            if i != j:
                d_x = x_pos[j] - x_pos[i]
                d_y = y_pos[j] - y_pos[i]
                d_z = z_pos[j] - z_pos[i]
                d_ij = d_x**2 + d_y**2 + d_z**2
	#Summs up the Potential from each body interaction
                PE -= G*Mass_array[i]*Mass_array[j]/(2*np.sqrt(d_ij))
    E_total = KE + PE
    E_change = (E_total - E_total[0])/(E_total[0])
    ax_error.plot(Years,np.abs(E_change))

ax_error.grid(True)
ax_error.set_xlabel('T[yr]')
ax_error.set_ylabel(r'$|\Delta E|$')
#Source for prop size:https://stackoverflow.com/questions/7125009/how-to-change-legend-fontsize-with-matplotlib-pyplot 
ax_error.legend(labels = ['Tolerance = 10^-7','Tolerance = 10^-8','Tolerance = 10^-9','Tolerance = 10^-10','Tolerance = 10^-11'],loc = 'upper left', prop={'size': 8})
ax_error.set_xlim(Years[0],Years[-1])
ax_error.set_yscale("log")
