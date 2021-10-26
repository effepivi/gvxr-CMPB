# #!/usr/bin/env python3
#
# # import os, copy
# import numpy as np
# # dir_path = os.path.dirname(os.path.realpath(__file__))
# #
import sys
#
# import math # for pi
# # import tomopy
#
#
import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
#
# import imageio

import json # Load the JSON file
import gvxrPython3 as gvxr # Simulate X-ray images

# Define the NoneType
NoneType = type(None);
params  = None;

# Print the libraries' version
print (gvxr.getVersionOfSimpleGVXR())
print (gvxr.getVersionOfCoreGVXR())

def initGVXR(fname, renderer = "OPENGL"):
    global params;

    # Load the JSON file
    with open(fname) as f:
        params = json.load(f)

    # Create an OpenGL context
    window_size = params["WindowSize"];
    print("Create an OpenGL context:",
        str(window_size[0]) + "x" + str(window_size[1])
    );
    gvxr.createOpenGLContext();
    gvxr.setWindowSize(
        window_size[0],
        window_size[1]
    );

def initSourceGeometry(fname = ""):
    global params;

    # Load the JSON file
    if fname != "":
        with open(fname) as f:
            params = json.load(f)

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

def initSpectrum(fname = "", verbose = 0):
    global params;

    spectrum = {};

    # Load the JSON file
    if fname != "":
        with open(fname) as f:
            params = json.load(f)

    if type(params["Source"]["Beam"]) == list:
        for energy_channel in params["Source"]["Beam"]:
            energy = energy_channel["Energy"];
            unit = energy_channel["Unit"];
            count = energy_channel["PhotonCount"];

            spectrum[energy] = count;

            if verbose > 0:
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
        unit = "keV"
        k, f = s.get_spectrum(edges=True) # Get the spectrum

        min_energy = sys.float_info.max
        max_energy = -sys.float_info.max

        for energy, count in zip(k, f):
            count = round(count)

            if count > 0:

                max_energy = max(max_energy, energy)
                min_energy = min(min_energy, energy)

                if energy in spectrum.keys():
                    spectrum[energy] += count
                else:
                    spectrum[energy] = count

        if verbose > 0:
            print("/gate/source/mybeam/gps/emin", min_energy, "keV")
            print("/gate/source/mybeam/gps/emax", max_energy, "keV")

        for energy in spectrum.keys():
            count = spectrum[energy]
            gvxr.addEnergyBinToSpectrum(energy, unit, count);

            if verbose > 0:
                print("/gate/source/mybeam/gps/histpoint", energy / 1000, count)

    return spectrum, unit, k, f;

def initDetector(fname = ""):
    global params;

    # Load the JSON file
    if fname != "":
        with open(fname) as f:
            params = json.load(f);

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

    if "Spacing" in params["Detector"].keys() == list and "Size" in params["Detector"].keys():
        raise "Cannot use both 'Spacing' and 'Size' for the detector";

    if "Spacing" in params["Detector"].keys():
        pixel_spacing = params["Detector"]["Spacing"];
    elif "Size" in params["Detector"].keys():
        detector_size = params["Detector"]["Size"];
        pixel_spacing = [];
        pixel_spacing.append(detector_size[0] / detector_number_of_pixels[0]);
        pixel_spacing.append(detector_size[1] / detector_number_of_pixels[1]);
        pixel_spacing.append(detector_size[2]);

    print("\tPixel spacing:", pixel_spacing)
    gvxr.setDetectorPixelSize(
        pixel_spacing[0],
        pixel_spacing[1],
        pixel_spacing[2]
    );

def initSamples(fname = "", verbose = 0):
    global params;

    # Load the JSON file
    if fname != "":
        with open(fname) as f:
            params = json.load(f);

    # Load the data
    if verbose > 0:
        print("Load the 3D data\n");

    colours = list(mcolors.TABLEAU_COLORS);
    colour_id = 0;
    for mesh in params["Samples"]:

        if verbose == 1:
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

                if verbose == 2:
                    print(mesh["Label"] + ":",
                          "d="+str(mesh["Density"]), "g/cm3 ;",
                          "n=" + str(len(material[1][0::2])),
                          "; state=solid");

                for Z, weight in zip(material[1][0::2], material[1][1::2]):
                    elements.append(Z);
                    weights.append(weight);

                    if verbose == 2:
                        print("        +el: name="+gvxr.getElementName(Z) + " ; f=" +str(weight) )

                if verbose == 2:
                    print()

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
                gvxr.addPolygonMeshAsInnerSurface(mesh["Label"]);
                # gvxr.addPolygonMeshAsOuterSurface(mesh["Label"]);
        else:
            gvxr.addPolygonMeshAsInnerSurface(mesh["Label"]);

        # Change the colour
        colour = mcolors.to_rgb(colours[colour_id]);
        gvxr.setColour(mesh["Label"], colour[0], colour[1], colour[2], 1.0);

        colour_id += 1;
        if colour_id == len(colours):
            colour_id = 0;
