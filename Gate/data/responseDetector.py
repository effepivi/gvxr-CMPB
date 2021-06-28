#!/home/letang/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt

# get the integral nb of photons

f = open('responseDetector.txt', 'r')
data = []
for line in f:
    line = line.strip()
    columns = line.split()
    source = [float(columns[0])*1000.,float(columns[1])*1000.]
    data.append(source)
f.close()

# plot spectrum

data_array = np.array(data)

x, y = data_array.T
plt.plot(x,y)
plt.xlim(0,150)
plt.ylim(0,60)
plt.grid(True,linestyle='dotted')
plt.xlabel('incident energy in keV')
plt.ylabel('absorbed energy in keV')

plt.savefig("detector_response.pdf")
