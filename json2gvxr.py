# #!/usr/bin/env python3
#
# # import os, copy
import numpy as np
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
context_created = False
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
    
    if not context_created:
        if renderer == "OPENGL":
            visibility = True
        else:
            visibility = False

        gvxr.createWindow(-1,
            True,
            renderer)
        
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

    min_energy = sys.float_info.max
    max_energy = -sys.float_info.max

    gvxr.resetBeamSpectrum()
    spectrum = {};
    # Load the JSON file
    if fname != "":
        with open(fname) as f:
            params = json.load(f)
    if type(params["Source"]["Beam"]) == list:
        k = []
        f = []
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
            k.append(energy)
            f.append(count)
        k = np.array(k)
        f = np.array(f)
    else:
        if "GateMacro" in params["Source"]["Beam"]:
            k = []
            f = []
            unit = params["Source"]["Beam"]["Unit"]
            # Read the file
            gate_macro_file = open(params["Source"]["Beam"]["GateMacro"], 'r')
            lines = gate_macro_file.readlines()
            # Process every line
            for line in lines:
                # Check if this is a comment or not
                comment = True
                index_first_non_space_character = len(line) - len(line.lstrip())
                if index_first_non_space_character >= 0 and index_first_non_space_character < len(line):
                    if line[index_first_non_space_character] != '#':
                        comment = False
                # This is not a comment
                if not comment:
                    x = line.split()
                energy = float(x[1])
                count = float(x[2])
                spectrum[energy] = count
                if verbose > 0:
                    if count == 1:
                        print("\t", str(count), "photon of", energy, unit);
                    else:
                        print("\t", str(count), "photons of", energy, unit);
                gvxr.addEnergyBinToSpectrum(energy, unit, count);
                k.append(energy)
                f.append(count)

            k = np.array(k)
            f = np.array(f)
        elif "TextFile" in params["Source"]["Beam"]:
            k = []
            f = []
            unit = params["Source"]["Beam"]["Unit"]

            # Read the file
            gate_macro_file = open(params["Source"]["Beam"]["TextFile"], 'r')
            lines = gate_macro_file.readlines()

            # Process every line
            for line in lines:

                # Check if this is a comment or not
                comment = True
                index_first_non_space_character = len(line) - len(line.lstrip())
                if index_first_non_space_character >= 0 and index_first_non_space_character < len(line):
                    if line[index_first_non_space_character] != '#':
                        comment = False

                # This is not a comment
                if not comment:
                    x = line.split()

                energy = float(x[0])
                count = float(x[1])
                spectrum[energy] = count

                if verbose > 0:
                    if count == 1:
                        print("\t", str(count), "photon of", energy, unit);
                    else:
                        print("\t", str(count), "photons of", energy, unit);

                gvxr.addEnergyBinToSpectrum(energy, unit, count);
                k.append(energy)
                f.append(count)

            k = np.array(k)
            f = np.array(f)
        elif "kvp" in params["Source"]["Beam"]:
            kvp_in_kV = params["Source"]["Beam"]["kvp"];
            th_in_deg = 12

            if "tube angle" in params["Source"]["Beam"]:
                th_in_deg = params["Source"]["Beam"]["tube angle"];

            import spekpy as sp

            if verbose > 0:
                print("kVp (kV):", kvp_in_kV)
                print("tube angle (degrees):", th_in_deg)

            s = sp.Spek(kvp=kvp_in_kV, th=th_in_deg) # Generate a spectrum (80 kV, 12 degree tube angle)

            if "filter" in params["Source"]["Beam"]:
                print('params["Source"]["Beam"]', params["Source"]["Beam"])
                for beam_filter in params["Source"]["Beam"]["filter"]:
                    print(beam_filter)

                    filter_material = beam_filter[0]
                    filter_thickness_in_mm = beam_filter[1]

                    if verbose > 0:
                        print("Filter", filter_thickness_in_mm, "mm of", filter_material)

                    s.filter(filter_material, filter_thickness_in_mm)



            unit = "keV"
            k, f = s.get_spectrum(edges=True) # Get the spectrum

            for energy, count in zip(k, f):
                count = round(count)
                if count > 0:
                    max_energy = max(max_energy, energy)
                    min_energy = min(min_energy, energy)
                    if energy in spectrum.keys():
                        spectrum[energy] += count
                    else:
                        spectrum[energy] = count
        else:
            raise IOError("Invalid beam spectrum in JSON file")
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
        
    if "Energy response" in params["Detector"].keys():
        print("\tEnergy response:", params["Detector"]["Energy response"]["File"], "in", params["Detector"]["Energy response"]["Energy"])
        gvxr.clearDetectorEnergyResponse()
        gvxr.loadDetectorEnergyResponse(params["Detector"]["Energy response"]["File"],
                                        params["Detector"]["Energy response"]["Energy"])
    print("\tPixel spacing:", pixel_spacing)
    gvxr.setDetectorPixelSize(
        pixel_spacing[0],
        pixel_spacing[1],
        pixel_spacing[2]
    );
    
    
def initSamples(fname = "", verbose = 0):
    global params; 
    
    gvxr.removePolygonMeshesFromXRayRenderer()
    gvxr.removePolygonMeshesFromSceneGraph()
    
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
        if "Cube" in mesh:
            if verbose == 1:
                print(mesh["Label"] + " is a cube")
            
            gvxr.makeCube(mesh["Label"], mesh["Cube"][0], mesh["Cube"][1]);
                
        elif "Cylinder" in mesh:
            if verbose == 1:
                print(mesh["Label"] + " is a cylinder")
                
            gvxr.makeCylinder(mesh["Label"], mesh["Cylinder"][0], mesh["Cylinder"][1], mesh["Cylinder"][2], mesh["Cylinder"][3]);
        elif "Path" in mesh:
            if verbose == 1:
                print("\tLoad", mesh["Label"], "in", mesh["Path"], "using", mesh["Unit"]);
            gvxr.loadMeshFile(
                mesh["Label"],
                mesh["Path"],
                mesh["Unit"],
                False
            );
        else:
            raise IOError("Cannot find the geometry of Mesh " + mesh["Label"])

        material = mesh["Material"];
        if material[0].upper() == "ELEMENT":
            gvxr.setElement(
                mesh["Label"],
                material[1]
            );
        elif material[0].upper() == "MIXTURE":
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
        elif material[0].upper() == "COMPOUND":
            gvxr.setCompound(
                mesh["Label"],
                material[1]
            );
        elif material[0].upper() == "HU":
            gvxr.setHounsfieldValue(
                mesh["Label"],
                material[1]
            );
        elif material[0].upper() == "MU":
            gvxr.setLinearAttenuationCoefficient(
                mesh["Label"],
                material[1],
                "cm-1"
            );
        else:
            raise IOError("Unknown material type: " + material[0]);
        if "Density" in mesh.keys():
            gvxr.setDensity(
                mesh["Label"],
                mesh["Density"],
                "g/cm3"
            );

        if "Transform" in mesh.keys():
            for transform in mesh["Transform"]:
                if transform[0] == "Rotation":
                    if len(transform) == 5:
                        gvxr.rotateNode(mesh["Label"],
                                        transform[1],
                                        transform[2],
                                        transform[3],
                                        transform[4])
                    else:
                        raise IOError("Invalid rotation:", transform)
                elif transform[0] == "Translation":
                    if len(transform) == 5:
                        gvxr.translateNode(mesh["Label"],
                                        transform[1],
                                        transform[2],
                                        transform[3],
                                        transform[4])
                    else:
                        raise IOError("Invalid translation:", transform)
                elif transform[0] == "Scaling":
                    if len(transform) == 4:
                        gvxr.scaleNode(mesh["Label"],
                                        transform[1],
                                        transform[2],
                                        transform[3])
                    else:
                        raise IOError("Invalid scaling:", transform)
                else:
                    raise IOError("Invalid transformation:", transform)

            gvxr.applyCurrentLocalTransformation(mesh["Label"])

        # Add the mesh to the simulation
        if "Type" in mesh.keys():
            if mesh["Type"] == "inner":
                gvxr.addPolygonMeshAsInnerSurface(mesh["Label"]);
            elif mesh["Type"] == "outer":
                # gvxr.addPolygonMeshAsInnerSurface(mesh["Label"]);
                gvxr.addPolygonMeshAsOuterSurface(mesh["Label"]);
        else:
            gvxr.addPolygonMeshAsInnerSurface(mesh["Label"]);
        
        # Change the colour
        colour = mcolors.to_rgb(colours[colour_id]);
        
        # Get the opacity
        opacity = 1.0
        if "Opacity" in mesh.keys():
            opacity = mesh["Opacity"]
        
        gvxr.setColour(mesh["Label"], colour[0], colour[1], colour[2], opacity);
        colour_id += 1;
        if colour_id == len(colours):
            colour_id = 0;