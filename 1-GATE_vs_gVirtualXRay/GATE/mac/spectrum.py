#!/home/letang/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt

# get the integral nb of photons

f = open('spectrum.mac', 'r')
nbphotons=0.
energy1 = -1.
energy2 = -1.
for line in f:
    line = line.strip()
    columns = line.split()
    if energy1<0:
        energy1 = float(columns[1])
    elif energy2<0:
        energy2 = float(columns[1])
    nbphotons += float(columns[2])
sampling = (energy2-energy1)*1000.
f.close()

# get spectrum

f = open('spectrum.mac', 'r')
data = []
for line in f:
    line = line.strip()
    columns = line.split()
    source = [float(columns[1])*1000.,float(columns[2])/(nbphotons*sampling)]
    data.append(source)
f.close()

# plot spectrum

data_array = np.array(data)

x, y = data_array.T
plt.plot(x,y)
plt.xlabel('energy in keV')
plt.ylabel('probability distribution of photons per keV')

plt.savefig("spectrum.pdf")
