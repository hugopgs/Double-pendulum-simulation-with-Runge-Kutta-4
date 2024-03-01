import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors

#==========Parameters==========#
M1 = 1.
M2 = 1.
R1 = 1.
R2 = 1.
g = 9.81
n_samples = 1000 #total number of samples
t_lim = 20.  #simulation time in seconds
frame_s = 20  #frames per second default: 60

theta1 = 0.3  #initials conditions rod 1
omega1 = 0.

theta2 = 0.  #initials conditions rod 2
omega2 = 0.

#==========Global_Variables==========#
#Don't change values here
skip_frames = int(n_samples/(t_lim*frame_s))
step = float(t_lim/n_samples)
mod_progression = n_samples/10

R1sq = R1**2
R2sq = R2**2

state = np.array([theta1, omega1, theta2, omega2])  #instantaneous pendulum state
time = np.array([0.])  #list of time_steps
off_Egrav = g*((R2+R1)*M2 + R1*M1)
#==========Functions==========#

def deg_to_rad(deg):
    return(deg*np.pi/180)

def polar_to_cartesian(r, theta=None, offset_x=0, offset_y=0):
    """
    Convert polar to cartesian

    r = scalar.
    return has the same type of theta.
    offset_x = scalar or list.
    offset_y = scalar or list.
    """
    if isinstance(theta, (list, np.ndarray)):
        if isinstance(offset_x , int) and offset_x == 0:
            offset_x = np.zeros(len(theta))
            offset_y = np.zeros(len(theta))
    X = r*np.sin(theta) + offset_x
    Y = -r*np.cos(theta) + offset_y
    return(X, Y)

def f_double_pendulum(state):
    n_state = np.zeros_like(state)
    theta1 = state[0]  #according to RK4 method
    theta2 = state[2]
    omega1 = state[1]
    omega2 = state[3]

    n_state[0] = omega1
    n_state[2] = omega2

    denom = 2*M1+M2-M2*np.cos(2*theta1-2*theta2)

    n_state[1] = (-g*(2*M1 + M2)*np.sin(theta1) - M2*g*np.sin(theta1 - 2*theta2)- 2*np.sin(theta1 - theta2)*M2*((omega2**2)*R2+(omega1**2)*R1*np.cos(theta1-theta2)))/(R1*denom)
    n_state[3] = (2*np.sin(theta1 - theta2)*((omega1**2)*R1*(M1+M2)+g*(M1+M2)*np.cos(theta1)+(omega2**2)*R2*M2*np.cos(theta1-theta2)))/(R2*denom)
    return(n_state)

def f_simple_pendulum(state):
    """
    Simple pendulum parameters are R1 and M1
    """
    n_state = np.zeros_like(state)
    n_state[0] = state[1]
    n_state[1] = - (g/R1)*np.sin(state[0])
    return n_state

def RK4(state, h, F=None):
    k1 = F(state)
    k2 = F(state + 0.5 * h * k1)
    k3 = F(state + 0.5 * h * k2)
    k4 = F(state + h * k3)
    return (state + h * (k1 + 2. * k2 + 2. * k3 + k4) / 6)

def states_to_lists(states):
    if isinstance(states, list):
        n_samples = len(states)
        n_rods = int(len(states[0])/2)
        theta = [np.zeros(n_samples) for _ in range(n_rods)]
        omega = [np.zeros(n_samples) for _ in range(n_rods)]
        for i, state in enumerate(states):
            for j in range(n_rods):
                theta[j][i] = state[(j*2)]
                omega[j][i] = state[(j*2)+1]
    else:
        n_rods = int(len(states)/2)
        theta = [np.zeros(1) for _ in range(n_rods)]
        theta = [np.zeros(0) for _ in range(n_rods)]
        for i in range(n_rods):
            theta[i] = states[(j*2)]
            omega[i] = states[(j*2)+1]
    return(theta, omega)

def norm(XA=0, YA=0, XB=0, YB=0):
    return(np.sqrt((XB-XA)**2 + (YB-YA)**2))

def lagrangian_Ekin(Theta, Omega):
    Ekin = (1./2)*M1*R1sq*Omega[0]**2 + (1./2)*M2*(R1sq*Omega[0]**2 + R2sq*Omega[1]**2 + 2*R1*R2*Omega[0]*Omega[1]*np.cos(Theta[0]-Theta[1]))
    return(np.asarray(Ekin))

def lagrangian_Egrav(Theta):
    Egrav = -(M1 + M2)*g*R1*np.cos(Theta[0])-M2*g*R2*np.cos(Theta[1])
    return(np.asarray(Egrav))

def first_revolution(angle_step, step, ite_max):
    """
    Find time of first second pendulum revolution

    Warning complexity in len(initial_theta1)**2

    Can be easily parallelize by spliting initial angle liste computation on different threads
    """
    energy_needed = M2*2*R2*g

    initial_theta1 = np.arange(-np.pi, np.pi, angle_step) #Can be crop to [0 - np.pi] because of symetry.
    initial_theta2 = np.arange(-np.pi, np.pi, angle_step)
    print(initial_theta1)
    mod_progression = int(len(initial_theta1)/10)
    first_reverse = np.zeros(shape=(len(initial_theta1), len(initial_theta2)))

    for i, theta1 in enumerate(initial_theta1):
        for j, theta2 in enumerate(initial_theta2):
            state = np.array([theta1, 0., theta2, 0.]) #initial omegas are 0
            k = -1
            if enough_energy([theta1, theta2], energy_needed): #optimization, if initial energy is to low to let second pendulum flip
                while (k <= ite_max) and (abs(state[2]) <= np.pi): #once second pendulum flip, compute more state is useless
                    k += 1
                    state = RK4(state, step, f_double_pendulum)
            if k == -1:
                k = ite_max
            first_reverse[i][j] = k

        if not i%mod_progression: #show progression
            print("first revolution searching: ", round(i/len(initial_theta1)*100), "%")

    with open("save_initial.npy", 'wb') as f: #for further treatments 
        np.save(f, first_reverse)
    f.close()
    return(first_reverse)

def enough_energy(Theta, energy_needed):
    return(lagrangian_Egrav(Theta) + off_Egrav >= energy_needed)

def load_save_2D_times(save_file):
    first_flip = np.load(save_file)
    return(first_flip)


#==========Display==========#
def double_pendulum_animation(Theta, T):
    X1, Y1 = polar_to_cartesian(R1, Theta[0])
    X2, Y2 = polar_to_cartesian(R2, Theta[1], X1, Y1)
    fig = plt.figure("Double pendulum animation", figsize=(6, 6))
    scale = -R1-R2
    plt.xlim((scale, -scale))
    plt.ylim((scale, -scale))
    line, = plt.plot([], [], linewidth=2, marker="o", color="blue")
    time_template = "time = %.1fs"
    time_text = plt.text(0.05, 0.9, '')
    def maj(i):
        i *= skip_frames
        time_text.set_text(time_template % T[i])
        line.set_data([0, X1[i], X2[i]], [0, Y1[i], Y2[i]])
        return line, time_text
    ani = animation.FuncAnimation(fig, maj, range(int(n_samples/skip_frames)), interval=(1/frame_s)*1000, blit=True)
    plt.plot(X1, Y1, 'r')
    plt.plot(X2, Y2, 'b')
    plt.show()
    return()

def display_phase_portrait(theta, omega):
    plt.figure("Phases portraits")
    plt.plot(theta[0],omega[0],label='phase portrait first pendulum')
    plt.plot(theta[1],omega[1],label='phase portrait second pendulum')
    plt.xlabel("Theta (rad)")
    plt.ylabel("Omega (rad/s)")
    plt.grid(True)
    plt.legend(loc='best')
    return()

def display_theta(Theta, T):
    plt.figure("Theta")
    plt.plot(T, Theta[0], label="Theta 1")
    plt.plot(T, Theta[1], label="Theta 2")
    plt.xlabel("Time (s)")
    plt.ylabel("Theta (rad)")
    plt.legend()
    return()

def display_graph_energy(Theta, Omega):
    Ekin = lagrangian_Ekin(Theta, Omega)
    Egrav = lagrangian_Egrav(Theta)
    Egrav -= min(Egrav)
    Etot = Egrav + Ekin
    print("E init ", Etot[0])
    print("E_final", Etot[-1])
    plt.figure("Energy")
    plt.subplot(1, 2, 1)
    plt.plot(T, Egrav, label="Potential energy")
    plt.plot(T, Ekin, label="Kinetic energy")
    plt.ylabel("Energy (J)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(T, Etot, label="Total energy")
    plt.hlines(y=Etot[0], xmin=T[0], xmax=T[-1], color="green", label="Inital energy")
    plt.hlines(y=Etot[-1], xmin=T[0], xmax=T[-1], color = "red", label="Final energy")
    plt.ylabel("Energy (J)")
    plt.xlabel("Time (s)")
    plt.yticks(np.linspace(Etot[0], Etot[-1], 5))
    plt.ticklabel_format(useOffset=False)
    plt.legend()
    return()

def plot_time_vs_initials(first_reverse):
    plt.figure("2d")
    plt.title("Initial conditions and first time for flip")
    plt.xlabel("theta 2")
    plt.ylabel("theta 1")
    plt.imshow(first_reverse, cmap="inferno", interpolation='none', extent=[-np.pi, np.pi, -np.pi, np.pi])
    plt.colorbar()
    plt.show()
    return()

#==========Main==========#

##===RK4 solving
list_states = [state]
print("Step: ", step)
for i in range(n_samples):
    state = RK4(state, step, f_double_pendulum)
    list_states.append(state)
    if not i%mod_progression: #show progression
        print("RK4: ",round(i/n_samples*100), "%")

Theta, Omega = states_to_lists(list_states)
##===EndRK4 solving

T = [step*i for i in range(n_samples + 1)]


display_graph_energy(Theta, Omega)
display_phase_portrait(Theta, Omega)
display_theta(Theta, T)
plt.show()
#time_first_flip = first_revolution(deg_to_rad(1), 0.01, 500)
time_first_flip = load_save_2D_times("save_initial.npy")
plot_time_vs_initials(time_first_flip)
double_pendulum_animation(Theta, T)