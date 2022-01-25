#!/usr/bin/python3

import numpy as np
import xraylib as xrl
import matplotlib.pyplot as plt
import matplotlib as mpl
from xpecgen import xpecgen as xg
import csv

E0=85
Emin=3
xrs=xg.calculate_spectrum(E0,11,Emin,E0-Emin+1,epsrel=0.5,monitor=None,z=74)
#Inherent filtration: 1.2mm Al + 100cm Air
fluence_to_dose=xg.get_fluence_to_dose()
xrs.set_norm(value=0.146,weight=fluence_to_dose)
#Attenuation = 0.1mmCu + 1mmAl
Mat_Z=29
Mat_X=0.01 #cm
dMat = xrl.ElementDensity(Mat_Z)
fMat = xrl.AtomicNumberToSymbol(Mat_Z)
xrs.attenuate(Mat_X,xg.get_mu(Mat_Z))
Mat_Z=13
Mat_X=0.1 #cm
dMat = xrl.ElementDensity(Mat_Z)
fMat = xrl.AtomicNumberToSymbol(Mat_Z)
xrs.attenuate(Mat_X,xg.get_mu(Mat_Z))
#Get the figures
Nr_Photons = "%.4g" % (xrs.get_norm())
Average_Energy = "%.2f keV" % (xrs.get_norm(lambda x:x)/xrs.get_norm())

(x2,y2) = xrs.get_points()
axMW = plt.subplot(111)
axMW.plot(x2,y2)
axMW.set_xlim(3,100)
axMW.set_ylim(0,)
plt.xlabel("Énergie [keV]")
plt.ylabel("Nombre de photons par [keV·cm²·mGy] @ 1m")
axMW.grid(which='major', axis='x', linewidth=0.5, linestyle='-', color='0.75')
axMW.grid(which='minor', axis='x', linewidth=0.2, linestyle='-', color='0.85')
axMW.grid(which='major', axis='y', linewidth=0.5, linestyle='-', color='0.75')
axMW.grid(which='minor', axis='y', linewidth=0.2, linestyle='-', color='0.85')
axMW.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%d"))
axMW.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2g"))
axMW.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
axMW.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
axMW.grid(True)

print(Average_Energy)

plt.show(block=True)

x2MeV = [e/1000 for e in x2]
with open('spectrum85kV.txt', 'w') as f:
    f.write("################ InterpolationSpectrum.txt #############\n")
    f.write("3   0\n")
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(zip(x2MeV,y2))
    f.write("###################################################\n")
