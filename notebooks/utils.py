import copy

import matplotlib.pyplot as plt # Plotting
from matplotlib.colors import PowerNorm # Look up table
import numpy as np # Who does not use Numpy?
from skimage.util import compare_images # Checkboard comparison between two images
from tifffile import imread, imsave # Load/Write TIFF files

import gvxrPython3 as gvxr # Simulate X-ray images

def computeXRayImageFromLBuffers(json2gvxr, verbose: bool=False) -> np.array:
        
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
                imsave("gVirtualXRay_output_data/l_buffer-" + label + ".tif", L_buffer_set[label].astype(np.single))

        # The structure is an outer structure, keep track of its label
        else:
            L_buffer_outer = label

    # There is an outer structure, subtract the accumulator from its L-buffer
    if L_buffer_outer is not None:
        L_buffer_set[L_buffer_outer] -= L_buffer_accumulator
        
        # Save the L-buffer in an image file
        if verbose:
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

        x_ray_image += (energy * count) * np.exp(-mu_x)
    
    # Return the image
    return x_ray_image

def displayLinearPowerScales(image: np.array, caption: str, fname: str):
    plt.figure(figsize= (20,10))

    plt.suptitle(caption, y=1.02)

    plt.subplot(121)
    plt.imshow(image, cmap="gray")
    plt.colorbar(orientation='horizontal')
    plt.title("Using a linear colour scale")

    plt.subplot(122)
    plt.imshow(image, norm=PowerNorm(gamma=1./0.75), cmap="gray")
    plt.colorbar(orientation='horizontal')
    plt.title("Using a Power-law colour scale")

    plt.tight_layout()

    plt.savefig(fname + '.pdf')
    plt.savefig(fname + '.png')

def plotSpectrum(k, f, fname):
    
    plt.figure(figsize= (20,10))

    plt.bar(k, f / f.sum()) # Plot the spectrum
    plt.xlabel('Energy in keV')
    plt.ylabel('Probability distribution of photons per keV')
    plt.title('Photon energy distribution')

    plt.xlim([0, 200])

    plt.tight_layout()

    plt.savefig(fname + '.pdf')
    plt.savefig(fname + '.png')
    
def compareImages(gate_image, gvxr_image, caption, fname):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))

    relative_error = 100 * (gate_image - gvxr_image) / gate_image
    comp_equalized = compare_images(gate_image, gvxr_image, method='checkerboard', n_tiles=(15,15))

    im1=axes.flat[0].imshow(comp_equalized, cmap="gray", vmin=0.25, vmax=1)
    axes.flat[0].set_title(caption)
    axes.flat[0].set_xticks([])
    axes.flat[0].set_yticks([])

    im2=axes.flat[1].imshow(relative_error, cmap="RdBu", vmin=-3, vmax=3)
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

def fullCompareImages(gate_image, gvxr_image, title, fname):
    absolute_error = np.abs(gate_image - gvxr_image)
    comp_equalized = compare_images(gate_image, gvxr_image, method='checkerboard', n_tiles=(15,15))

    plt.figure(figsize= (20,10))

    plt.subplot(141)
    plt.imshow(gate_image, cmap="gray")#, vmin=0.25, vmax=1)
    plt.title("Gate (ground truth)")

    plt.subplot(142)
    plt.imshow(gvxr_image, cmap="gray")#, vmin=0.25, vmax=1)
    plt.title(title)

    plt.subplot(143)
    plt.imshow(comp_equalized, cmap="gray")#, vmin=0.25, vmax=1)
    plt.title("gVirtualXRay \\& Gate\n (checkerboard comparison)")

    plt.subplot(144)
    plt.imshow(absolute_error, cmap="gray")#, vmin=0.25, vmax=1)
    plt.title("Absolute error")

    plt.tight_layout()

    plt.savefig(fname + '.pdf')
    plt.savefig(fname + '.png')
    
def plotProfiles(json2gvxr, gate_image, x_ray_image_integration_CPU, x_ray_image_integration_GPU, fname, xlimits=None):

    plt.figure(figsize= (30,20))

    offset_line = 20 * (json2gvxr.params["Source"]["Position"][2] - json2gvxr.params["Detector"]["Position"][2]) / json2gvxr.params["Source"]["Position"][2]

    spacing = json2gvxr.params["Detector"]["Size"][0] / json2gvxr.params["Detector"]["NumberOfPixels"][0]

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
    y_coord = round(json2gvxr.params["Detector"]["NumberOfPixels"][0] / 2 - offset_line * json2gvxr.params["Detector"]["NumberOfPixels"][0] / json2gvxr.params["Detector"]["Size"][0])

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
    y_coord = round(json2gvxr.params["Detector"]["NumberOfPixels"][0] / 2)

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
    y_coord = round(json2gvxr.params["Detector"]["NumberOfPixels"][0] / 2 + offset_line * json2gvxr.params["Detector"]["NumberOfPixels"][0] / json2gvxr.params["Detector"]["Size"][0])

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
