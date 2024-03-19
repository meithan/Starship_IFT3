import csv
import datetime as dtm
import sys
import matplotlib.pyplot as plt
import numpy as np
from math import cos, radians
from scipy.signal import savgol_filter

sys.path.append("/home/meithan/owncloud/orbits")
from ISA1976.ISA1976 import ISA1976
from Orbits import Orbit
from Orbits.SolarSystem import Earth

# ==============================================================================

save = "--save" in sys.argv

combined_plot = False

events = [
  ["Max Q", 60],
  ["MECO", 60*2 + 42],
  ["Stage sep", 60*2 + 48],
  ["SECO1", 60*8 + 21],
  ["SECO2", 60*8 + 35],
]

color_event_line = "0.8"
color_event_label = "0.3"

# ==============================================================================

def parse_float(s):
  try:
    return(float(s))
  except:
    return np.nan

def moving_average(x, w):
  return np.convolve(x, np.ones(w), "same") / w

def impute_missing_values(xs, ys):
  i = 0
  while i < len(xs):
    if np.isnan(ys[i]):
      i0 = i-1; i1 = i; i2 = i
      while np.isnan(ys[i]):
        i2 = i
        i += 1
      i3 = i2 + 1
      deriv = (ys[i3]-ys[i0])/(xs[i3]-xs[i0])
      for j in range(i1, i2+1):
        dx = xs[j] - xs[i0]
        ys[j] = ys[i0] + dx*deriv
    i += 1

def smoothen(values):
  # return moving_average(values, 30)
  return savgol_filter(values, 5, 1)

# ==============================================================================
 
# Load data (own scraping)
fname = "data/IFT3_telemetry.csv"
raw_data = []
with open(fname) as f:
  reader = csv.reader(f)
  header = next(reader)
  # print(header)
  for line in reader:
    mins = parse_float(line[0])
    secs = parse_float(line[1])
    spd = parse_float(line[2]) / 3.6
    alt = parse_float(line[3]) * 1e3
    time = mins*60 + secs
    raw_data.append((time, alt, spd))
_time, _altitude, _speed = zip(*raw_data)
time = np.array(_time)
raw_altitude = np.array(_altitude)
raw_speed = np.array(_speed)

# Impute missing values
impute_missing_values(time, raw_altitude)
impute_missing_values(time, raw_speed)

# Smoothen raw data
altitude = savgol_filter(raw_altitude, 21, 1)
speed = savgol_filter(raw_speed, 3, 1)

# plt.figure()
# plt.title("Altitude")
# plt.plot(raw_altitude, "o")
# plt.plot(altitude, "-")
# plt.figure()
# plt.title("Speed")
# plt.plot(raw_speed, "o", color="k", alpha=0.4, mfc="none")
# plt.plot(speed, "-")
# plt.show()

# Speed in inertial Earth-centered frame
vrot = 465*cos(radians(18))

# Acceleration
accel = np.gradient(savgol_filter(raw_speed, 31, 1), time)
accel = smoothen(accel)

# Speed components
vspeed = np.gradient(savgol_filter(altitude, 31, 1), time)
# vspeed = moving_average(vspeed, 30)
hspeed = np.sqrt(np.clip(speed**2 - moving_average(vspeed, 30)**2, 0, None))
# hspeed = np.sqrt(np.clip(raw_speed**2 - vspeed**2, 0, None))
speed_numer = np.sqrt(hspeed**2 + vspeed**2)

# plt.plot(hspeed)
# plt.plot(vspeed)
# plt.show()

# Acceleration components
haccel = smoothen(np.gradient(savgol_filter(hspeed, 31, 1), time))
vaccel = smoothen(np.gradient(savgol_filter(vspeed, 31, 1), time))
accel_numer = np.sqrt(haccel**2 + vaccel**2)

# plt.plot(accel)
# plt.plot(accel_numer)
# plt.show()

# Horizontal (downrange) distance
hdist = []
tlast = 0
x = 0
for i in range(len(time)):
  t = time[i]; vx = hspeed[i]; h = altitude[i]
  if np.isnan(vx) or np.isnan(h):
    hdist.append(np.nan)
  else:
    x += vx*(t-tlast)
    hdist.append(x)
    tlast = t
hdist = np.array(hdist)

# Mach number and dynamic pressure
atmo = ISA1976()
Mach = []; dynpres = []
for i in range(len(time)):
  h = altitude[i]; v = speed[i]
  if h/1e3 >= 86 or np.isnan(h) or np.isnan(v):
    M = np.nan
    Q = np.nan
  else:
    dens, pres, _ = atmo.get_values(h)
    if dens == 0 or pres == 0:
      M = np.nan
      Q = np.nan
    else:
      cs = atmo.get_sound_speed(h)
      M = v/cs
      Q = 0.5*dens*v**2
  Mach.append(M)
  dynpres.append(Q)
Mach = np.array(Mach)
dynpres = np.array(dynpres)

# Orbital state vectors (position and velocity vectors)
# The trajectory is assumed to be contained in a vertical xy plane
hspeed2 = np.sqrt(np.clip(raw_speed**2 - vspeed**2, 0, None))
pos_orb = []; vel_orb = []
for i in range(len(time)):
  pos_orb.append((0, altitude[i]+Earth.radius, 0))
  vel_orb.append((hspeed2[i]+vrot, vspeed[i], 0))
pos_orb = np.array(pos_orb)
vel_orb = np.array(vel_orb)

# Orbital energy
radius = np.linalg.norm(pos_orb, axis=1)
speed_orb = np.linalg.norm(vel_orb, axis=1)
energy = speed_orb**2/2 - Earth.mu/radius
a = Earth.radius + 150*1e3
E_orbit = -Earth.mu/(2*a)
E_surf = -Earth.mu/Earth.radius
# print(E_orbit/1e6, E_surf/1e6, energy[-1]/1e6)

# Osculating orbits
perigee = []
for i in range(len(time)):
  orb = Orbit(Earth)
  orb.from_state_vectors(pos_orb[i], vel_orb[i], dtm.datetime(2024, 3, 14))
  hpe = np.linalg.norm(orb.get_periapsis())/1e3 - Earth.radius/1e3
  perigee.append(hpe)

print("Final orbit:")
print(orb)

# ==============================================================================
# PLOTS

if combined_plot:
  plt.figure(figsize=(20,12))
  rows = 2
  cols = 3
  subplot = 1

# -----------------------
# Altitude & speed

if combined_plot:
  plt.subplot(rows,cols,subplot); subplot += 1
else:
  plt.figure(figsize=(8,6))

plt.title("Altitude & Speed (telemetry, smoothed)")
handles = []

# Altitude
color = "C0"
ln1, = plt.plot(time, altitude/1e3, color=color, label="Altitude")
plt.xlabel("Time [s]")
plt.ylabel("Altitude [km]")
plt.axhline(0, color="gray", zorder=-10)
plt.gca().spines['left'].set_color(color)
plt.gca().tick_params(axis='y', colors=color)
plt.gca().yaxis.label.set_color(color)
plt.grid(ls=":")

# Events
label_pos = {"Max Q": (0.7, -15), "MECO": (0.7, -15), "Stage sep": (0.7, 3), "SECO1": (0.5, -15), "SECO2": {0.5, 2}}  
for label, t in events:
  y, xoff = label_pos[label]
  plt.axvline(t, ls="--", color=color_event_line, zorder=-10)
  plt.annotate(label, xy=(t, y), xycoords=("data", "axes fraction"), xytext=(xoff, 0), textcoords="offset pixels", rotation=90, color=color_event_label)

# Speed
ax2 = plt.twinx()
color = "C2"
ln2, = plt.plot(time, speed/1e3, color=color, label="Speed")
plt.xlabel("Time [s]")
plt.ylabel("Speed [km/s]")
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_color(color)
plt.gca().tick_params(axis='y', colors=color)
plt.gca().yaxis.label.set_color(color)
plt.legend(handles=[ln1,ln2])

if not combined_plot:
  plt.tight_layout()
  if save:
    fname = "plots/alt_speed.png"
    plt.savefig(fname)
    print("Wrote", fname)

# -----------------------
# Accelerations

if combined_plot:
  plt.subplot(rows,cols,subplot); subplot += 1
else:
  plt.figure(figsize=(8,6))

color = "C3"
plt.plot(time, accel_numer, color=color, label="Total")
# plt.plot(time, accel, color=color, label="Total")
plt.plot(time, haccel, lw=1, color="C0", label="Horizontal")
plt.plot(time, vaccel, lw=1, color="C1", label="Vertical")
plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s²]")
plt.title("Acceleration")
plt.legend()
plt.axhline(0, color="gray", zorder=-10)
plt.gca().spines['left'].set_color(color)
plt.gca().tick_params(axis='y', colors=color)
plt.gca().yaxis.label.set_color(color)
plt.grid(ls=":")
y1, y2 = plt.ylim()

ax2 = plt.twinx()
plt.ylim(y1/9.806, y2/9.806)
plt.gca().spines['right'].set_color(color)
plt.gca().spines['left'].set_visible(False)
plt.gca().tick_params(axis='y', colors=color)
plt.gca().yaxis.label.set_color(color)
plt.ylabel("Acceleration [gees]")

# Events
color = "0.3"
label_pos = {"Max Q": (0.7, -15), "MECO": (0.7, -15), "Stage sep": (0.7, 3), "SECO1": (0.5, -15), "SECO2": {0.5, 2}}
for label, t in events:
  y, xoff = label_pos[label]
  plt.axvline(t, ls="--", color=color_event_line, zorder=-10)
  plt.annotate(label, xy=(t, y), xycoords=("data", "axes fraction"), xytext=(xoff, 0), textcoords="offset pixels", rotation=90, color=color_event_label)

if not combined_plot:
  plt.tight_layout()
  if save:
    fname = "plots/accels.png"
    plt.savefig(fname)
    print("Wrote", fname)

# -----------------------
# Velocity components

if combined_plot:
  plt.subplot(rows,cols,subplot); subplot += 1
else:
  plt.figure(figsize=(8,6))

plt.plot(time, speed_numer/1e3, color="0.5", label="Magnitude")
plt.plot(time, hspeed/1e3, color="C0", label="Horizontal")
plt.plot(time, vspeed/1e3, color="C1", label="Vertical")
y1, y2 = plt.ylim()
plt.legend()
plt.axhline(0, color="gray", zorder=-10)
plt.xlabel("Time [s]")
plt.ylabel("Speed [km/s]")
plt.title("Velocity components")
plt.grid(ls=":")

ax2 = plt.twinx()
plt.ylim(y1*3600, y2*3600)
plt.ylabel("Speed [km/h]")

# Events
color = "0.3"
label_pos = {"Max Q": (0.7, -15), "MECO": (0.7, -15), "Stage sep": (0.7, 3), "SECO1": (0.5, -15), "SECO2": {0.5, 2}}  
for label, t in events:
  y, xoff = label_pos[label]
  plt.axvline(t, ls="--", color=color_event_line, zorder=-10)
  plt.annotate(label, xy=(t, y), xycoords=("data", "axes fraction"), xytext=(xoff, 0), textcoords="offset pixels", rotation=90, color=color_event_label)

if not combined_plot:
  plt.tight_layout()
  if save:
    fname = "plots/velocity.png"
    plt.savefig(fname)
    print("Wrote", fname)

# -----------------------
# Trajectory

if combined_plot:
  plt.subplot(rows,cols,subplot); subplot += 1
else:
  plt.figure(figsize=(8,6))

plt.plot(hdist/1e3, altitude/1e3, color="k")
next_dot = 0; next_text = 60
texts_pos = {60: (5,-5), 120: (5,-2), 180: (5,-2), 240: (5,-2), 300: (5,-7), 360: (-5,-12), 420: (-16,-12), 480: (-15,-12)}
for i in range(len(time)):
  x, y = hdist[i]/1e3, altitude[i]/1e3
  if time[i] >= next_dot:
    s = 20 if time[i] >= next_text else 5
    if time[i] >= next_text:
      text = f"{time[i]:.0f} s"
      pos = texts_pos[next_text]
      plt.annotate(text, xy=(x,y), xytext=pos, textcoords="offset pixels", va="center", fontsize=10, color=color_event_label)
      next_text += 60
    plt.scatter([x], [y], color="k", s=s)
    next_dot += 10

# Events
color = "0.3"
for label, t in events:
  found = False
  for i in range(len(time)-1):
    if time[i] <= t < time[i+1]:
      x, y = hdist[i]/1e3, altitude[i]/1e3
      found = True
      break
  if not found:
    x, y = hdist[-1]/1e3, altitude[-1]/1e3
  xoff = 30; yoff = 0
  if label == "Max Q":
    yoff = 30
  elif label == "SECO1":
    xoff, yoff = -20, -30
  elif label == "SECO2":
    xoff, yoff = -20, -20
  plt.annotate(label, xy=(x,y), xytext=(xoff,yoff), textcoords="offset pixels", va="top", fontsize=10, color=color_event_label, arrowprops=dict(arrowstyle="->", color=color))

plt.annotate("Dots every 10 seconds", xy=(0.01, 0.99), ha="left", va="top", fontsize=10, xycoords="axes fraction", color="0.5")
plt.axhline(0, color="gray", zorder=-10)
x1, x2 = plt.xlim()
plt.ylim(-1, 155)
plt.grid(ls=":")
plt.title("Trajectory profile")
plt.xlabel("Downrange distance [km]")
plt.ylabel("Altitude [km]")
# plt.gca().set_aspect("equal")


if not combined_plot:
  plt.tight_layout()
  if save:
    fname = "plots/trajectory.png"
    plt.savefig(fname)
    print("Wrote", fname)

# -----------------------
# Mach number & dynamic pressure

if combined_plot:
  plt.subplot(rows,cols,subplot); subplot += 1
else:
  plt.figure(figsize=(8,6))

# Mach number
color = "C4"
ln1, = plt.plot(time, Mach, color=color, label="Mach number")
plt.xlabel("Time [s]")
plt.ylabel("Mach number")
plt.title("Mach Number and Dynamic Pressure")
plt.grid(ls=":")
plt.axhline(0, color="gray", zorder=-10)
plt.gca().spines['left'].set_color(color)
plt.gca().tick_params(axis='y', colors=color)
plt.gca().yaxis.label.set_color(color)

# Dynamic pressure
color = "orange"
ax2 = plt.gca().twinx()
ln2, = plt.plot(time, moving_average(dynpres/1e3, 10), color=color, label="Dynamic pressure")
plt.ylabel("Dynamic pressure [kPa]")
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_color(color)
plt.gca().tick_params(axis='y', colors=color)
plt.gca().yaxis.label.set_color(color)
plt.legend(handles=[ln1,ln2], loc="upper right")

# Events
color = "0.3"
label_pos = {"Max Q": (0.5, -15), "MECO": (0.5, -15), "Stage sep": (0.5, 3), "SECO1": (0.5, -15), "SECO2": {0.5, 0}}  
for label, t in events:
  y, xoff = label_pos[label]
  plt.axvline(t, ls="--", color=color_event_line, zorder=-10)
  plt.annotate(label, xy=(t, y), xycoords=("data", "axes fraction"), xytext=(xoff, 0), textcoords="offset pixels", rotation=90, color=color_event_label)

if not combined_plot:
  plt.tight_layout()
  if save:
    fname = "plots/mach_dynpres.png"
    plt.savefig(fname)
    print("Wrote", fname)

# -----------------------
# Specific orbital energy & perigee
if combined_plot:
  plt.subplot(rows,cols,subplot); subplot += 1
else:
  plt.figure(figsize=(8,6))

# Orbital energy
color = "C5"
ln1, = plt.plot(time, energy/1e6, color=color, label="Orbital energy")
plt.axhline(E_surf/1e6, ls="-", color=color)
plt.annotate("Energy at surface", xy=(0.98, E_surf/1e6), xycoords=("axes fraction", "data"), xytext=(0, 5), textcoords="offset pixels", color=color, ha="right")
plt.axhline(E_orbit/1e6, ls="-", color=color)
plt.annotate("Energy for 150 km orbit", xy=(0.98, E_orbit/1e6), xycoords=("axes fraction", "data"), xytext=(0, 5), textcoords="offset pixels", color=color, ha="right")
plt.xlabel("Time [s]")
plt.ylabel("Specific orbital energy [MJ/kg]")
plt.title("Orbital Energy and Perigee")
plt.grid(ls=":")
plt.gca().spines['left'].set_color(color)
plt.gca().tick_params(axis='y', colors=color)
plt.gca().yaxis.label.set_color(color)
# plt.legend(handles=[ln1,ln2])

# Perigee
color = "C6"
ax2 = plt.gca().twinx()
ln2, = plt.plot(time, perigee, color=color, label="Perigee altitude")
plt.ylim(-6700, 400)
plt.ylabel("Perigee altitude [km]")
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_color(color)
plt.gca().tick_params(axis='y', colors=color)
plt.gca().yaxis.label.set_color(color)
plt.legend(handles=[ln1,ln2], loc="upper left")

# Events
color = "0.3"
label_pos = {"Max Q": (0.7, -15), "MECO": (0.7, -15), "Stage sep": (0.7, 3), "SECO1": (0.5, -15), "SECO2": {0.5, 2}}  
for label, t in events:
  y, xoff = label_pos[label]
  plt.axvline(t, ls="--", color=color_event_line, zorder=-10)
  plt.annotate(label, xy=(t, y), xycoords=("data", "axes fraction"), xytext=(xoff, 0), textcoords="offset pixels", rotation=90, color=color_event_label)

if not combined_plot:
  plt.tight_layout()
  if save:
    fname = "plots/energy_perigee.png"
    plt.savefig(fname)
    print("Wrote", fname)

# -----------------------

if save:

  # Save all computed data to file
  out_fname = "data/IFT3_full_data.csv"
  fout = open(out_fname, "w")
  strings = [
    "Time [s]",
    "Raw altitude [km]",
    "Raw speed [m/s]",
    "Smoothed altitude [km]",
    "Smoothed speed [m/s]",
    "Downrange distance [km]",
    "Horizontal speed [m/s]",
    "Vertical speed [m/s]",
    "Total acceleration [m/s^2]",
    "Horizontal acceleration [m/s^2]",
    "Vertical acceleration [m/s^2]",
    "Mach number",
    "Dynamic pressure [kPa]",
    "Specific orbital energy [MJ/kg]",
    "Perigee altitude [km]",
  ]
  fout.write(",".join(strings)+"\n")
  for i in range(len(time)):
    strings = [
      f"{time[i]:.0f}",
      f"{raw_altitude[i]/1e3:.0f}",
      f"{raw_speed[i]:.0f}",
      f"{altitude[i]/1e3:.1f}",
      f"{speed[i]:.1f}",
      f"{hdist[i]/1e3:.1f}",
      f"{hspeed[i]:.1f}",
      f"{vspeed[i]:.1f}",    
      f"{accel_numer[i]:.1f}",
      f"{haccel[i]:.1f}",
      f"{vaccel[i]:.1f}",
      f"{Mach[i]:.1f}",
      f"{dynpres[i]/1e3:.1f}",
      f"{energy[i]/1e6:.1f}",
      f"{perigee[i]:.1f}",
    ]
    fout.write(",".join(strings)+"\n")
  fout.close()
  print("Wrote", out_fname)

# -----------------------

if combined_plot:

  plt.tight_layout()

  plt.annotate("@meithan42", color="0.9", xy=(0.65, 0.5), xycoords="figure fraction", va="center", rotation=90)

  if save:
    fname = "plots/IFT3_combined.png"
    plt.savefig(fname)
    print("Wrote", fname)
  else:
    plt.show()

else:
  
  if not save:
    plt.show()