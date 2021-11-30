import copy

import matplotlib.pyplot as plt # Plotting
from matplotlib.colors import PowerNorm # Look up table
from matplotlib.colors import LogNorm # Look up table
import numpy as np # Who does not use Numpy?
from skimage.util import compare_images # Checkboard comparison between two images
from tifffile import imread, imsave # Load/Write TIFF files

import SimpleITK as sitk
import vtk

import gvxrPython3 as gvxr # Simulate X-ray images

def interpolate(a_low, a_high, a0, b_low, b_high):
    return b_low + (b_high - b_low) * (a0 - a_low) / (a_high - a_low)

def find_nearest(a, a0, b):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    
    # a[idx] <= a0 <= a[idx+1]
    if a[idx] < a0:
        return interpolate(a[idx], a[idx + 1], a0, b[idx], b[idx + 1])
    # a[idx - 1] <= a0 <= a[idx]
    else:
        return interpolate(a[idx - 1], a[idx], a0, b[idx - 1], b[idx])

def computeXRayImageFromLBuffers(json2gvxr, verbose: bool=False, detector_response: np.array=None, integrate_energy: bool=True, prefix: str=None) -> np.array:
        
    # Dictionanry to hold the L-buffer of every sample
    L_buffer_set = {}

    # Do not register any outer shell
    L_buffer_accumulator = None
    L_buffer_outer = None
    
    image_shape = None

    # Compute the L-buffer of every sample and store it in the dictionary
    for sample in json2gvxr.params["Samples"]:
        
        # Get the label of the sample
        label = sample["Label"]

        # Compute its L-buffer and copy it in the dictionary
        L_buffer_set[label] = np.array(gvxr.computeLBuffer(label))
        image_shape = L_buffer_set[label].shape
        
        # If it is an inner structure, add the L-buffer in the accummulator
        if sample["Type"] == "inner":

            # The accumulator does not exist, create it and copy the L-buffer in the accumulator
            if L_buffer_accumulator is None:
                L_buffer_accumulator = copy.deepcopy(L_buffer_set[label])
            # The accumulator already exists, add the L-buffer in the accummulator
            else:
                L_buffer_accumulator += L_buffer_set[label]
            
            # Save the L-buffer in an image file
            if verbose:
                if prefix is not None:
                    imsave(prefix + "l_buffer-" + label + ".tif", L_buffer_set[label].astype(np.single))
                else:
                    imsave("gVirtualXRay_output_data/l_buffer-" + label + ".tif", L_buffer_set[label].astype(np.single))

        # The structure is an outer structure, keep track of its label
        else:
            L_buffer_outer = label

    # There is an outer structure, subtract the accumulator from its L-buffer
    if L_buffer_outer is not None:
        L_buffer_set[L_buffer_outer] -= L_buffer_accumulator
        
        # Save the L-buffer in an image file
        if verbose:
            if prefix is not None:
                imsave(prefix + "l_buffer-" + L_buffer_outer + ".tif", L_buffer_set[L_buffer_outer].astype(np.single))
            else:
                imsave("gVirtualXRay_output_data/l_buffer-" + L_buffer_outer + ".tif", L_buffer_set[L_buffer_outer].astype(np.single))
        
    # Create an empty X-ray image
    x_ray_image = np.zeros(image_shape)  
    
    # Compute the polychromatic Beer-Lambert law
    for energy, count in zip(gvxr.getEnergyBins("MeV"), gvxr.getPhotonCountEnergyBins()):
        # Create an empty accumulator
        mu_x = np.zeros(image_shape)

        # Accumulate mu * x for every sample
        for sample in json2gvxr.params["Samples"]:
            label = sample["Label"]

            mu = gvxr.getLinearAttenuationCoefficient(label, energy, "MeV")
            mu_x += L_buffer_set[label] * mu

        # Compute the energy fluence
        if integrate_energy:
            # No energy response provided
            if detector_response is None:
                effective_energy = energy
            # Energy response provided
            else:
                effective_energy = find_nearest(detector_response[:,0], energy, detector_response[:,1])
        # Compute the number of photons
        else:
            effective_energy = 1.0
            
        x_ray_image += (effective_energy * count) * np.exp(-mu_x)
            
    # Return the image
    return x_ray_image

def displayLinearPowerScales(image: np.array, caption: str, fname: str, log: bool=False, vmin=0.01, vmax=1.2):
    plt.figure(figsize= (20,10))

    plt.suptitle(caption, y=1.02)

    plt.subplot(121)
    plt.imshow(image, cmap="gray")
    plt.colorbar(orientation='horizontal')
    plt.title("Using a linear colour scale")

    plt.subplot(122)
    if log:
        plt.imshow(image, norm=LogNorm(vmin=vmin, vmax=vmax), cmap="gray")
        plt.title("Using a Log scale")
    else:
        plt.imshow(image, norm=PowerNorm(gamma=1./0.75), cmap="gray")
        plt.title("Using a Power-law colour scale")
    plt.colorbar(orientation='horizontal')

    plt.tight_layout()

    plt.savefig(fname + '.pdf')
    plt.savefig(fname + '.png')

def plotSpectrum(k, f, fname=None, xlim=[0,200]):
    
    plt.figure(figsize= (20,10))

    plt.bar(k, f / f.sum()) # Plot the spectrum
    plt.xlabel('Energy in keV')
    plt.ylabel('Probability distribution of photons per keV')
    plt.title('Photon energy distribution')

    plt.xlim(xlim)

    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname + '.pdf')
        plt.savefig(fname + '.png')
    
def compareImages(gate_image, gvxr_image, caption, fname, threshold=3):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))

    relative_error = 100 * (gate_image - gvxr_image) / gate_image
    comp_equalized = compare_images(gate_image, gvxr_image, method='checkerboard', n_tiles=(15,15))

    im1=axes.flat[0].imshow(comp_equalized, cmap="gray", vmin=0.25, vmax=1)
    axes.flat[0].set_title(caption)
    axes.flat[0].set_xticks([])
    axes.flat[0].set_yticks([])

    im2=axes.flat[1].imshow(relative_error, cmap="RdBu", vmin=-threshold, vmax=threshold)
    axes.flat[1].set_title("Relative error (in \%)")
    axes.flat[1].set_xticks([])
    axes.flat[1].set_yticks([])

    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.2, hspace=0.02)

    # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8

    cb_ax = fig.add_axes([0.83, 0.425, 0.02, 0.15])
    cbar = fig.colorbar(im2, cax=cb_ax)

    # set the colorbar ticks and tick labels
    # cbar.set_ticks(np.arange(0, 1.1, 0.5))
    # cbar.set_ticklabels(['low', 'medium', 'high'])

    # plt.show()

    # plt.tight_layout()
    
    plt.savefig(fname + '.pdf')
    plt.savefig(fname + '.png')


def fullCompareImages(gate_image: np.array, gvxr_image: np.array, title: str, fname: str, log: bool=False, vmin=0.01, vmax=1.2):

    absolute_error = np.abs(gate_image - gvxr_image)
    relative_error = 100 * (gate_image - gvxr_image) / gate_image
    comp_equalized = compare_images(gate_image, gvxr_image, method='checkerboard', n_tiles=(15,15))

    
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 20))

    relative_error = 100 * (gate_image - gvxr_image) / gate_image
    comp_equalized = compare_images(gate_image, gvxr_image, method='checkerboard', n_tiles=(15,15))

    if not log:
        im1 = axes.flat[0].imshow(gate_image, cmap="gray", vmin=0.25, vmax=1)
    else:
        im1 = axes.flat[0].imshow(gate_image, cmap="gray", norm=LogNorm(vmin=vmin, vmax=vmax))
    axes.flat[0].set_title("Gate (ground truth)")
    axes.flat[0].set_xticks([])
    axes.flat[0].set_yticks([])

    if not log:
        im2 = axes.flat[1].imshow(gvxr_image, cmap="gray", vmin=0.25, vmax=1)
    else:
        im2 = axes.flat[1].imshow(gvxr_image, cmap="gray", norm=LogNorm(vmin=vmin, vmax=vmax))
    axes.flat[1].set_title("gVirtualXRay")
    axes.flat[1].set_xticks([])
    axes.flat[1].set_yticks([])

    if not log:
        im3 = axes.flat[2].imshow(comp_equalized, cmap="gray", vmin=0.25, vmax=1)
    else:
        im3 = axes.flat[2].imshow(comp_equalized, cmap="gray", norm=LogNorm(vmin=vmin, vmax=vmax))
    axes.flat[2].set_title("Checkerboard comparison between\nGate \& gVirtualXRay")
    axes.flat[2].set_xticks([])
    axes.flat[2].set_yticks([])

    if not log:
        im4 = axes.flat[3].imshow(relative_error, cmap="RdBu", vmin=-5, vmax=5)
    else:
        im4 = axes.flat[3].imshow(relative_error, cmap="RdBu", norm=LogNorm(vmin=-5, vmax=5))
    axes.flat[3].set_title("Relative error (in \%)")
    axes.flat[3].set_xticks([])
    axes.flat[3].set_yticks([])

    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.2, hspace=0.02)

    # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8
    cb_ax = fig.add_axes([0.83, 0.425, 0.02, 0.15])
    cbar = fig.colorbar(im4, cax=cb_ax)

    plt.savefig(fname + '.pdf')
    plt.savefig(fname + '.png')

    
def plotThreeProfiles(json2gvxr, gate_image, x_ray_image_integration_CPU, x_ray_image_integration_GPU, fname, xlimits=None):

    plt.figure(figsize= (30,20))

    if json2gvxr.params["Source"]["Position"][2] > 0.5:
        offset_line = 20 * (json2gvxr.params["Source"]["Position"][2] - json2gvxr.params["Detector"]["Position"][2]) / json2gvxr.params["Source"]["Position"][2]
    else:
        offset_line = 20 * (json2gvxr.params["Source"]["Position"][1] - json2gvxr.params["Detector"]["Position"][1]) / json2gvxr.params["Source"]["Position"][1]
        
    spacing = json2gvxr.params["Detector"]["Size"][0] / gate_image.shape[0]

    x = np.arange(0.0, json2gvxr.params["Detector"]["Size"][0], spacing)

    x_gate = []
    x_gvxr_integration_CPU = []
    x_gvxr_integration_GPU = []
    temp = []

    use_simulation = 0

    for i in range(x_ray_image_integration_CPU.shape[1]):

        temp.append(i)

        if not i % 5 and i != 0:
            temp.append(i+1)
            if use_simulation == 0:
                x_gate.append(temp)
                use_simulation += 1
            elif use_simulation == 1:
                x_gvxr_integration_CPU.append(temp)
                use_simulation += 1
            elif use_simulation == 2:
                x_gvxr_integration_GPU.append(temp)
                use_simulation = 0

            temp = []
            temp.append(i-1)

    plt.subplot(311)
    plt.title("Profiles (Top line)")
    # plt.yscale("log")
    y_coord = round(gate_image.shape[0] / 2 - offset_line * gate_image.shape[0] / json2gvxr.params["Detector"]["Size"][0])
    y_coord = min(y_coord, gate_image.shape[0] - 1)    
    
    i = 0
    for sub_x in x_gate:
        y = gate_image[y_coord][sub_x]
        if i == 0:
            plt.plot(np.array(sub_x) * spacing, y, label="Gate (noisy)", color='green')
        else:
            plt.plot(np.array(sub_x) * spacing, y, color='green')
        i += 1

    i = 0
    for sub_x in x_gvxr_integration_CPU:
        y = x_ray_image_integration_CPU[y_coord][sub_x]
        if i == 0:
            plt.plot(np.array(sub_x) * spacing, y, label="gVirtualXRay with integration on CPU", color='orange')
        else:
            plt.plot(np.array(sub_x) * spacing, y, color='orange')
        i += 1

    i = 0
    for sub_x in x_gvxr_integration_GPU:
        y = x_ray_image_integration_GPU[y_coord][sub_x]
        if i == 0:
            plt.plot(np.array(sub_x) * spacing, y, label="gVirtualXRay with integration on GPU", color='blue')
        else:
            plt.plot(np.array(sub_x) * spacing, y, color='blue')
        i += 1

    if xlimits is not None:
        plt.xlim(xlimits)

    plt.xlabel("Pixel location (in mm)")
    plt.ylabel("Pixel intensity")
    plt.legend()




    plt.subplot(312)
    plt.title("Profiles (Central line)")
    # plt.yscale("log")
    y_coord = round(gate_image.shape[0] / 2)
    y_coord = min(y_coord, gate_image.shape[0] - 1)

    i = 0
    for sub_x in x_gate:
        y = gate_image[y_coord][sub_x]
        if i == 0:
            plt.plot(np.array(sub_x) * spacing, y, label="Gate (noisy)", color='green')
        else:
            plt.plot(np.array(sub_x) * spacing, y, color='green')
        i += 1

    i = 0
    for sub_x in x_gvxr_integration_CPU:
        y = x_ray_image_integration_CPU[y_coord][sub_x]
        if i == 0:
            plt.plot(np.array(sub_x) * spacing, y, label="gVirtualXRay with integration on CPU", color='orange')
        else:
            plt.plot(np.array(sub_x) * spacing, y, color='orange')
        i += 1

    i = 0
    for sub_x in x_gvxr_integration_GPU:
        y = x_ray_image_integration_GPU[y_coord][sub_x]
        if i == 0:
            plt.plot(np.array(sub_x) * spacing, y, label="gVirtualXRay with integration on GPU", color='blue')
        else:
            plt.plot(np.array(sub_x) * spacing, y, color='blue')
        i += 1

    if xlimits is not None:
        plt.xlim(xlimits)

    plt.xlabel("Pixel location (in mm)")
    plt.ylabel("Pixel intensity")

    plt.subplot(313)
    plt.title("Profiles (Bottom line)")
    # plt.yscale("log")
    y_coord = round(gate_image.shape[0] / 2 + offset_line * gate_image.shape[0] / json2gvxr.params["Detector"]["Size"][0])
    y_coord = min(y_coord, gate_image.shape[0] - 1)

    i = 0
    for sub_x in x_gate:
        y = gate_image[y_coord][sub_x]
        if i == 0:
            plt.plot(np.array(sub_x) * spacing, y, label="Gate (noisy)", color='green')
        else:
            plt.plot(np.array(sub_x) * spacing, y, color='green')
        i += 1

    i = 0
    for sub_x in x_gvxr_integration_CPU:
        y = x_ray_image_integration_CPU[y_coord][sub_x]
        if i == 0:
            plt.plot(np.array(sub_x) * spacing, y, label="gVirtualXRay with integration on CPU", color='orange')
        else:
            plt.plot(np.array(sub_x) * spacing, y, color='orange')
        i += 1

    i = 0
    for sub_x in x_gvxr_integration_GPU:
        y = x_ray_image_integration_GPU[y_coord][sub_x]
        if i == 0:
            plt.plot(np.array(sub_x) * spacing, y, label="gVirtualXRay with integration on GPU", color='blue')
        else:
            plt.plot(np.array(sub_x) * spacing, y, color='blue')
        i += 1

    if xlimits is not None:
        plt.xlim(xlimits)

    plt.xlabel("Pixel location (in mm)")
    plt.ylabel("Pixel intensity")

    plt.tight_layout()

    plt.savefig(fname + ".pdf")
    plt.savefig(fname + ".png")

    
def plotTwoProfiles(json2gvxr, gate_image, x_ray_image_integration_CPU, x_ray_image_integration_GPU, fname, xlimits=None):

    plt.figure(figsize= (20,10))

    if json2gvxr.params["Source"]["Position"][2] > 0.5:
        offset_line = 20 * (json2gvxr.params["Source"]["Position"][2] - json2gvxr.params["Detector"]["Position"][2]) / json2gvxr.params["Source"]["Position"][2]
    else:
        offset_line = 20 * (json2gvxr.params["Source"]["Position"][1] - json2gvxr.params["Detector"]["Position"][1]) / json2gvxr.params["Source"]["Position"][1]
        
    spacing = json2gvxr.params["Detector"]["Size"][0] / gate_image.shape[0]

    x = np.arange(0.0, json2gvxr.params["Detector"]["Size"][0], spacing)

    x_gate = []
    x_gvxr_integration_CPU = []
    x_gvxr_integration_GPU = []
    temp = []

    use_simulation = 0

    for i in range(x_ray_image_integration_CPU.shape[1]):

        temp.append(i)

        if not i % 5 and i != 0:
            temp.append(i+1)
            if use_simulation == 0:
                x_gate.append(temp)
                use_simulation += 1
            elif use_simulation == 1:
                x_gvxr_integration_CPU.append(temp)
                use_simulation += 1
            elif use_simulation == 2:
                x_gvxr_integration_GPU.append(temp)
                use_simulation = 0

            temp = []
            temp.append(i-1)

    plt.subplot(311)
    plt.title("Profiles (Top line)")
    # plt.yscale("log")
    y_coord = round(gate_image.shape[0] / 2 - offset_line * gate_image.shape[0] / json2gvxr.params["Detector"]["Size"][0])
    y_coord = min(y_coord, gate_image.shape[0] - 1)    
    
    i = 0
    for sub_x in x_gate:
        y = gate_image[y_coord][sub_x]
        if i == 0:
            plt.plot(np.array(sub_x) * spacing, y, label="Gate (noisy)", color='green')
        else:
            plt.plot(np.array(sub_x) * spacing, y, color='green')
        i += 1

    i = 0
    for sub_x in x_gvxr_integration_CPU:
        y = x_ray_image_integration_CPU[y_coord][sub_x]
        if i == 0:
            plt.plot(np.array(sub_x) * spacing, y, label="gVirtualXRay with integration on CPU", color='orange')
        else:
            plt.plot(np.array(sub_x) * spacing, y, color='orange')
        i += 1

    i = 0
    for sub_x in x_gvxr_integration_GPU:
        y = x_ray_image_integration_GPU[y_coord][sub_x]
        if i == 0:
            plt.plot(np.array(sub_x) * spacing, y, label="gVirtualXRay with integration on GPU", color='blue')
        else:
            plt.plot(np.array(sub_x) * spacing, y, color='blue')
        i += 1

    if xlimits is not None:
        plt.xlim(xlimits)

    plt.xlabel("Pixel location (in mm)")
    plt.ylabel("Pixel intensity")
    plt.legend()




    plt.subplot(312)
    plt.title("Profiles (Central line)")
    # plt.yscale("log")
    y_coord = round(gate_image.shape[0] / 2)
    y_coord = min(y_coord, gate_image.shape[0] - 1)

    i = 0
    for sub_x in x_gate:
        y = gate_image[y_coord][sub_x]
        if i == 0:
            plt.plot(np.array(sub_x) * spacing, y, label="Gate (noisy)", color='green')
        else:
            plt.plot(np.array(sub_x) * spacing, y, color='green')
        i += 1

    i = 0
    for sub_x in x_gvxr_integration_CPU:
        y = x_ray_image_integration_CPU[y_coord][sub_x]
        if i == 0:
            plt.plot(np.array(sub_x) * spacing, y, label="gVirtualXRay with integration on CPU", color='orange')
        else:
            plt.plot(np.array(sub_x) * spacing, y, color='orange')
        i += 1

    i = 0
    for sub_x in x_gvxr_integration_GPU:
        y = x_ray_image_integration_GPU[y_coord][sub_x]
        if i == 0:
            plt.plot(np.array(sub_x) * spacing, y, label="gVirtualXRay with integration on GPU", color='blue')
        else:
            plt.plot(np.array(sub_x) * spacing, y, color='blue')
        i += 1

    if xlimits is not None:
        plt.xlim(xlimits)

    plt.xlabel("Pixel location (in mm)")
    plt.ylabel("Pixel intensity")

    plt.subplot(313)
    plt.title("Profiles (Bottom line)")
    # plt.yscale("log")
    y_coord = round(gate_image.shape[0] / 2 + offset_line * gate_image.shape[0] / json2gvxr.params["Detector"]["Size"][0])
    y_coord = min(y_coord, gate_image.shape[0] - 1)

    i = 0
    for sub_x in x_gate:
        y = gate_image[y_coord][sub_x]
        if i == 0:
            plt.plot(np.array(sub_x) * spacing, y, label="Gate (noisy)", color='green')
        else:
            plt.plot(np.array(sub_x) * spacing, y, color='green')
        i += 1

    i = 0
    for sub_x in x_gvxr_integration_CPU:
        y = x_ray_image_integration_CPU[y_coord][sub_x]
        if i == 0:
            plt.plot(np.array(sub_x) * spacing, y, label="gVirtualXRay with integration on CPU", color='orange')
        else:
            plt.plot(np.array(sub_x) * spacing, y, color='orange')
        i += 1

    i = 0
    for sub_x in x_gvxr_integration_GPU:
        y = x_ray_image_integration_GPU[y_coord][sub_x]
        if i == 0:
            plt.plot(np.array(sub_x) * spacing, y, label="gVirtualXRay with integration on GPU", color='blue')
        else:
            plt.plot(np.array(sub_x) * spacing, y, color='blue')
        i += 1

    if xlimits is not None:
        plt.xlim(xlimits)

    plt.xlabel("Pixel location (in mm)")
    plt.ylabel("Pixel intensity")

    plt.tight_layout()

    plt.savefig(fname + ".pdf")
    plt.savefig(fname + ".png")

def plotTwoProfiles(json2gvxr, gate_image, x_ray_image_integration_GPU, fname, xlimits=None):

    plt.figure(figsize= (30,20))

    if json2gvxr.params["Source"]["Position"][2] > 0.5:
        offset_line = 20 * (json2gvxr.params["Source"]["Position"][2] - json2gvxr.params["Detector"]["Position"][2]) / json2gvxr.params["Source"]["Position"][2]
    else:
        offset_line = 20 * (json2gvxr.params["Source"]["Position"][1] - json2gvxr.params["Detector"]["Position"][1]) / json2gvxr.params["Source"]["Position"][1]
        
    spacing = json2gvxr.params["Detector"]["Size"][0] / gate_image.shape[0]

    x = np.arange(0.0, json2gvxr.params["Detector"]["Size"][0], spacing)

    plt.subplot(311)
    plt.title("Profiles (Top line)")
    # plt.yscale("log")
    y_coord = round(gate_image.shape[0] / 2 - offset_line * gate_image.shape[0] / json2gvxr.params["Detector"]["Size"][0])
    y_coord = min(y_coord, gate_image.shape[0] - 1)    

    y = gate_image[y_coord]
    plt.plot(x, y, label="Gate (noisy)", color='green')
    
    y = x_ray_image_integration_GPU[y_coord]
    plt.plot(x, y, label="gVirtualXRay", color='blue')

    if xlimits is not None:
        plt.xlim(xlimits)

    plt.xlabel("Pixel location (in mm)")
    plt.ylabel("Pixel intensity")
    plt.legend()




    plt.subplot(312)
    plt.title("Profiles (Central line)")
    # plt.yscale("log")
    y_coord = round(gate_image.shape[0] / 2)
    y_coord = min(y_coord, gate_image.shape[0] - 1)

    y = gate_image[y_coord]
    plt.plot(x, y, label="Gate (noisy)", color='green')
    
    y = x_ray_image_integration_GPU[y_coord]
    plt.plot(x, y, label="gVirtualXRay", color='blue')

    if xlimits is not None:
        plt.xlim(xlimits)

    plt.xlabel("Pixel location (in mm)")
    plt.ylabel("Pixel intensity")

    plt.subplot(313)
    plt.title("Profiles (Bottom line)")
    # plt.yscale("log")
    y_coord = round(gate_image.shape[0] / 2 + offset_line * gate_image.shape[0] / json2gvxr.params["Detector"]["Size"][0])
    y_coord = min(y_coord, gate_image.shape[0] - 1)

    y = gate_image[y_coord]
    plt.plot(x, y, label="Gate (noisy)", color='green')
    
    y = x_ray_image_integration_GPU[y_coord]
    plt.plot(x, y, label="gVirtualXRay", color='blue')

    if xlimits is not None:
        plt.xlim(xlimits)

    plt.xlabel("Pixel location (in mm)")
    plt.ylabel("Pixel intensity")

    plt.tight_layout()

    plt.savefig(fname + ".pdf")
    plt.savefig(fname + ".png")
    
# A function to extract an isosurface from a binary image
def extractSurface(vtk_image, isovalue):

    iso = vtk.vtkContourFilter()
    if vtk.vtkVersion.GetVTKMajorVersion() >= 6:
        iso.SetInputData(vtk_image)
    else:
        iso.SetInput(vtk_image)

    iso.SetValue(0, isovalue)
    iso.Update()
    return iso.GetOutput()

# A function to write STL files
def writeSTL(mesh, name):
    """Write an STL mesh file."""
    try:
        writer = vtk.vtkSTLWriter()
        if vtk.vtkVersion.GetVTKMajorVersion() >= 6:
            writer.SetInputData(mesh)
        else:
            writer.SetInput(mesh)
        writer.SetFileTypeToBinary()
        writer.SetFileName(name)
        writer.Write()
        writer = None
    except BaseException:
        print("STL mesh writer failed")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(
            exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)
    return None
