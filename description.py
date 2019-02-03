import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as mp3d
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def data_description(ds):
    wr = ds[0]
    wl = ds[1]
    V = ds[2]
    omega = ds[3]
    plt.subplot(2,2,1)
    plt.title("$\omega_r$ density")
    plt.hist(wr, density=True, bins=10)
    plt.subplot(2,2,2)
    plt.title("$\omega_l$ density")
    plt.hist(wl, density=True, bins=10)
    plt.subplot(2,2,3)
    plt.title("$V$ density")
    plt.hist(V, density=True, bins=10)
    plt.subplot(2,2,4)
    plt.title("$\Omega$ density")
    plt.hist(omega, density=True, bins=10)

def plot3D(ds, step=4):
    wr = ds[0][::step]
    wl = ds[1][::step]
    V = ds[2][::step]
    omega = ds[3][::step]
    fig_V = plt.figure()
    #ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax_V = Axes3D(fig_V)
    ax_V.scatter(wr, wl, V)
    fig_omega = plt.figure()
    ax_omega = Axes3D(fig_omega)
    ax_omega.scatter(wr, wl, omega)
    ax.set_xlabel("$\omega_r$")
    ax.set_ylabel("$\omega_l$")
    ax.set_zlabel("$V$")

def results_overlaid_on_data(ds, Rr_est, Rl_est, L_est, w_min, w_max, step=4):
    wr = ds[0][::step]
    wl = ds[1][::step]
    V = ds[2][::step]
    omega = ds[3][::step]
    V_regression = [(w_min, w_min,0.5*(Rr_est*w_min+Rl_est*w_min)),
        (w_max, w_min, 0.5*(Rr_est*w_max+Rl_est*w_min)),
        (w_max, w_max, 0.5*(Rr_est*w_max+Rl_est*w_max)),
        (w_min, w_max, 0.5*(Rr_est*w_min+Rl_est*w_max))]
    omega_regression = [(w_min, w_min,0.5*(Rr_est*w_min-Rl_est*w_min)/L_est),
        (w_max, w_min, 0.5*(Rr_est*w_max-Rl_est*w_min)/L_est),
        (w_max, w_max, 0.5*(Rr_est*w_max-Rl_est*w_max)/L_est),
        (w_min, w_max, 0.5*(Rr_est*w_min-Rl_est*w_max)/L_est)]

    #X, Y = np.array([w_min, w_max]), np.array([w_min, w_max])
    #X, Y = np.meshgrid(X, Y)
    #V_predict = 0.5*(Rr_est*X+Rl_est*Y)
    #omega_predict = 0.5*(Rr_est*X-Rl_est*Y)/L_est
    fig = plt.figure()
    ax = Axes3D(fig)
    plane_V = mp3d.art3d.Poly3DCollection([V_regression], alpha=0.5, linewidth=1)
    plane_V.set_alpha = 0.5
    plane_V.set_facecolor("yellow")
    ax.add_collection3d(plane_V)
    ax.scatter(wr, wl, V)
    plane_omega = mp3d.art3d.Poly3DCollection([omega_regression], alpha=0.5, linewidth=1)
    plane_omega.set_alpha = 0.5
    plane_omega.set_facecolor("yellow")
    ax.add_collection3d(plane_omega)
    ax.scatter(wr, wl, omega)
    #surf = ax.plot_surface(X,Y, V_predict, color="yellow", linewidth=0)
