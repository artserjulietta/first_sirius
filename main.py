from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML

def smooth_ramp(x, eps=0.01):
  abs_x = abs(x)
  return (x + abs_x) / (2 * abs_x + eps) * x

elasticity_coef = 2.0e+3
gravity_accel = 9.81

def dynamics(_, state):
  x,y,dx,dy = state
  fx = elasticity_coef * (-smooth_ramp(x - 1) + smooth_ramp(-x - 1))
  fy = elasticity_coef * smooth_ramp(-y - 1)
  ddx = fx
  ddy = -gravity_accel + fy
  return [dx, dy, ddx, ddy]

solver = ode(dynamics)
solver.set_integrator('dopri5', nsteps=20, max_step=1e-3)
solver.set_initial_value([0, 1, 3, 0], 0)

fig,ax = plt.subplots(figsize=(6,5))
plt.axis('equal')
ax.set_xlim((-2, 2))
ax.set_ylim((-2, 2))
circle = plt.Circle((0, 0), 0.2, color='r')

ax.set_facecolor('lightblue')
ax.add_patch(
  plt.Rectangle((-1.3,-1.3), 2.6, 10, color='white')
)

def init():
  ax.add_patch(circle)
  return circle,

def update(frame):
  solver.integrate(timestep * frame)
  x,y,_,_ = solver.y
  circle.center = [x, y]
  return circle,

fps = 60
simtime = 10
timestep = 1/fps
nframes = int(simtime*fps)
anim = animation.FuncAnimation(fig, update, init_func=init,
                               frames=nframes, interval=1000*timestep, blit=True)
# rc('animation', html='jshtml')
# anim
plt.show()