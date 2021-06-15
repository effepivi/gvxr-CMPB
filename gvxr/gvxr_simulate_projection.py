#!/usr/bin/env python3

# import os, copy
import numpy as np
# dir_path = os.path.dirname(os.path.realpath(__file__))
#
import math # for pi
# import tomopy
import SimpleITK as sitk # To save the image

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.colors import PowerNorm


import json # Load the JSON file
import gvxrPython3 as gvxr # Simulate X-ray images

# Define the NoneType
NoneType = type(None);


# Print the libraries' version
print (gvxr.getVersionOfSimpleGVXR())
print (gvxr.getVersionOfCoreGVXR())

# Load the JSON file
with open('simulation.json') as f:
  params = json.load(f)

# Create an OpenGL context
window_size = params["WindowSize"];
print("Create an OpenGL context:",
    str(window_size[0]) + "x" + str(window_size[1])
);
gvxr.createWindow();
gvxr.setWindowSize(
    window_size[0],
    window_size[1]
);


# Set up the beam
print("Set up the beam")
source_position = params["Source"]["Position"];
print("\tSource position:", source_position)
gvxr.setSourcePosition(
    source_position[0],
    source_position[1],
    source_position[2],
    source_position[3]
);

source_shape = params["Source"]["Shape"]
print("\tSource shape:", source_shape);

if source_shape == "ParallelBeam":
    gvxr.useParallelBeam();
elif source_shape == "PointSource":
    gvxr.usePointSource();
else:
    raise "Unknown source shape:" + source_shape;

if type(params["Source"]["Beam"]) == list:
    for energy_channel in params["Source"]["Beam"]:
        energy = energy_channel["Energy"];
        unit = energy_channel["Unit"];
        count = energy_channel["PhotonCount"];

        if count == 1:
            print("\t", str(count), "photon of", energy, unit);
        else:
            print("\t", str(count), "photons of", energy, unit);

        gvxr.addEnergyBinToSpectrum(energy, unit, count);
else:
    kvp_in_kV = params["Source"]["Beam"]["kvp"];
    th_in_deg = params["Source"]["Beam"]["tube angle"];
    filter_material = params["Source"]["Beam"]["filter"][0];
    filter_thickness_in_mm = params["Source"]["Beam"]["filter"][1];

    import spekpy as sp

    s = sp.Spek(kvp=kvp_in_kV,th=th_in_deg) # Generate a spectrum (80 kV, 12 degree tube angle)
    s.filter(filter_material, filter_thickness_in_mm) # Filter by 4 mm of Al

    exit();

# Set up the detector
print("Set up the detector");
detector_position = params["Detector"]["Position"];
print("\tDetector position:", detector_position)
gvxr.setDetectorPosition(
    detector_position[0],
    detector_position[1],
    detector_position[2],
    detector_position[3]
);

detector_up = params["Detector"]["UpVector"];
print("\tDetector up vector:", detector_up)
gvxr.setDetectorUpVector(
    detector_up[0],
    detector_up[1],
    detector_up[2]
);

detector_number_of_pixels = params["Detector"]["NumberOfPixels"];
print("\tDetector number of pixels:", detector_number_of_pixels)
gvxr.setDetectorNumberOfPixels(
    detector_number_of_pixels[0],
    detector_number_of_pixels[1]
);

pixel_spacing = params["Detector"]["Spacing"];
print("\tPixel spacing:", pixel_spacing)
gvxr.setDetectorPixelSize(
    pixel_spacing[0],
    pixel_spacing[1],
    pixel_spacing[2]
);

# Load the data
print("Load the 3D data");
colours = list(mcolors.TABLEAU_COLORS);

colour_id = 0;
for mesh in params["Samples"]:

    print("\tLoad", mesh["Label"], "in", mesh["Path"], "using", mesh["Unit"]);

    gvxr.loadMeshFile(
        mesh["Label"],
        mesh["Path"],
        mesh["Unit"]
    );

    material = mesh["Material"];
    if material[0] == "Element":
        gvxr.setElement(
            mesh["Label"],
            material[1]
        );
    elif material[0] == "Mixture":

        if type(material[1]) == str:
            gvxr.setMixture(
                mesh["Label"],
                material[1]
            );
        else:
            elements = [];
            weights = [];
            for Z, weight in zip(material[1][0::2], material[1][1::2]):
                elements.append(Z);
                weights.append(weight);

            gvxr.setMixture(
                mesh["Label"],
                elements,
                weights
            );

    elif material[0] == "Compound":
        gvxr.setCompound(
            mesh["Label"],
            material[1]
        );
    elif material[0] == "HU":
        gvxr.setHounsfieldValue(
            mesh["Label"],
            material[1]
        );
    else:
        raise ("Unknown material type: " + material[0]);

    if "Density" in mesh.keys():
        gvxr.setDensity(
            mesh["Label"],
            mesh["Density"],
            "g/cm3"
        );

    # Add the mesh to the simulation
    if "Type" in mesh.keys():
        if mesh["Type"] == "inner":
            gvxr.addPolygonMeshAsInnerSurface(mesh["Label"]);
        elif mesh["Type"] == "outer":
            gvxr.addPolygonMeshAsOuterSurface(mesh["Label"]);
    else:
        gvxr.addPolygonMeshAsInnerSurface(mesh["Label"]);

    # Change the colour
    colour = mcolors.to_rgb(colours[colour_id]);
    gvxr.setColour(mesh["Label"], colour[0], colour[1], colour[2], 1.0);

    colour_id += 1;
    if colour_id == len(colours):
        colour_id = 0;


# Compute an X-ray image and add it to the list of projections
x_ray_image = gvxr.computeXRayImage();

# Update the 3D visualisation
gvxr.displayScene();

volume = sitk.GetImageFromArray(x_ray_image);
sitk.WriteImage(volume, 'projection.mha');

# Plot the projection
plt.subplot(131)
plt.imshow(x_ray_image, cmap="gray");
plt.colorbar(orientation='horizontal');
plt.title("Using a linear colour scale");

plt.subplot(132)
plt.imshow(x_ray_image, norm=LogNorm(), cmap="gray");
plt.colorbar(orientation='horizontal');
plt.title("Using a log colour scale");

plt.subplot(133)
plt.imshow(x_ray_image, norm=PowerNorm(gamma=1./2.), cmap="gray");
plt.colorbar(orientation='horizontal');
plt.title("Using a Power-law colour scale");

plt.show();

# Display the 3D scene (no event loop)
# Run an interactive loop
# (can rotate the 3D scene and zoom-in)
# Keys are:
# Q/Escape: to quit the event loop (does not close the window)
# B: display/hide the X-ray beam
# W: display the polygon meshes in solid or wireframe
# N: display the X-ray image in negative or positive
# H: display/hide the X-ray detector
gvxr.renderLoop();
