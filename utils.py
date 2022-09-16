import copy
import math

import matplotlib.pyplot as plt # Plotting
from matplotlib.colors import PowerNorm # Look up table
from matplotlib.colors import LogNorm # Look up table
import numpy as np # Who does not use Numpy?
from skimage.util import compare_images # Checkboard comparison between two images
from tifffile import imread, imsave # Load/Write TIFF files
from threading import Thread
import os
from convertRaw import *

import multiprocessing
from threading import Thread

# import SimpleITK as sitk
import vtk

import GPUtil
from cpuinfo import get_cpu_info
import platform
import psutil

tomography_backend="tomopy"

# import tigre
# import tigre.algorithms as algs

# import tomopy

# from skimage.transform import iradon
# from skimage.transform import radon

from gvxrPython3 import gvxr # Simulate X-ray images

def printSystemInfo():
    print("OS:")
    # print("\t" + platform.platform())
    # print("\t" + platform.version())
    print("\t" + platform.system(), platform.release())
    # print("\t" + platform.version())
    print("\t" + platform.machine() + "\n")


    print("CPU:\n", "\t" + get_cpu_info()["brand_raw"] + "\n")

    print("RAM:\n\t" + str(round(psutil.virtual_memory().total / (1024.0 **3))), "GB")

    print("GPU:")
    for gpu in GPUtil.getGPUs():
        print("\tName:", gpu.name)
        print("\tDrivers:", gpu.driver)
        print("\tVideo memory:", round(gpu.memoryTotal / 1024), "GB")




def getGeo(mode, filter_name):
    # Using TIGRE
    #Geometry settings
    source_position = gvxr.getSourcePosition("mm")
    detector_position = gvxr.getDetectorPosition("mm")

    nDetector = np.array([gvxr.getDetectorNumberOfPixels()[1], gvxr.getDetectorNumberOfPixels()[0]])
    nVoxel = np.array((nDetector[0], nDetector[1], nDetector[1]))

    print("nDetector", nDetector)
    print("nVoxel", nVoxel)

    geo = tigre.geometry(mode=mode, nVoxel=nVoxel, default=False)

    # Distance Source Origin        (mm)
    geo.DSO = math.sqrt(source_position[0] ** 2 + source_position[1] ** 2 + source_position[2] ** 2)

    # Distance Detector Origin        (mm)
    DDO = math.sqrt(detector_position[0] ** 2 + detector_position[1] ** 2 + detector_position[2] ** 2)

    # Distance Source Detector      (mm)
    geo.DSD = geo.DSO + DDO


    # Detector parameters
    # number of pixels              (px)
    geo.nDetector = nDetector

    # total size of the detector    (mm)
    geo.sDetector = np.array([gvxr.getDetectorSize("mm")[1], gvxr.getDetectorSize("mm")[0]])

    # size of each pixel            (mm)
    geo.dDetector = geo.sDetector / geo.nDetector

    print("geo.sDetector", geo.sDetector)
    print("geo.dDetector", geo.dDetector)

    # Image parameters
    # geo.nVoxel = np.array((geo.nDetector[0], geo.nDetector[1], geo.nDetector[1]))             # number of voxels              (vx)
    geo.sVoxel = np.array([gvxr.getDetectorSize("mm")[1], gvxr.getDetectorSize("mm")[0], gvxr.getDetectorSize("mm")[0]])             # total size of the image       (mm)
    geo.dVoxel = geo.sVoxel / geo.nVoxel               # size of each voxel            (mm)

    # Offsets
    geo.offOrigin = np.array((0, 0, 0))                # Offset of image from origin   (mm)
    geo.offDetector = np.array((0, 0))                 # Offset of Detector            (mm)

    # Auxiliary
    geo.accuracy = 0.5                                 # Accuracy of FWD proj          (vx/sample)

    # Mode
    geo.mode = mode                                  # parallel, cone                ...
    geo.filter = filter_name                       #  None, shepp_logan, cosine, hamming, hann

    return geo


def readPlastimatchImageAsNumpy(path, width, height):
    image_DRR = read_raw(binary_file_name=path,
                 image_size=[width, height],
                 sitk_pixel_type=sitk.sitkFloat32,
                 big_endian="false",
                 verbose=False);
    if (os.path.exists(path) == False):
        print(binary_file_name + " does not exist");
    return np.array(sitk.GetArrayFromImage(image_DRR));

def doLungmanDRRNumpy(iNRMX, iNRMY, iNRMZ,
                      iCX , iCY , in_drr_out_name,
                      iOutW = 725, iOutH = 426,
                      iSID = 65535, iSAD = 65535,
                      iSpacingX = 0.625, iSpacingY = 0.7,
                      in_lungman_path = "lungman_data/lungman.mha"):
    format = 'plastimatch drr --nrm "{iNRMX} {iNRMY} {iNRMZ}" --vup "0 0 1" -r "{iWidth} {iHeight}" -c "{iCX} {iCY}" -z "{iZx} {iZy}" -t raw --sid {sid} --sad {sad} --output Plastimatch_data/{out} {src}';
    iWAdjusted = iOutW * iSpacingX;
    iHAdjusted = iOutH * iSpacingY;
    sParam = format.format(sid=iSID,sad=iSAD,iNRMX=iNRMX, iNRMY=iNRMY,iNRMZ = iNRMZ, iCX = iCX, iCY = iCY, iZx = iWAdjusted, iZy = iHAdjusted, iWidth=iOutW, iHeight=iOutH, src=in_lungman_path, out = in_drr_out_name)
    #print(sParam)
    os.system(sParam);
    format = "Plastimatch_data/{in_name}0000.raw";
    drr= readPlastimatchImageAsNumpy(format.format(in_name = in_drr_out_name), iOutW, iOutH);
    return drr;


def projsFromPhantom(phantom, theta_deg, mode="parallel", proj_lenght_in_pixel=128, number_of_slices=128, backend=tomography_backend):

    # Scikit-image
    if backend == "scikit-image":

        ground_truth_proj = []
        ground_truth_sino = []

        # Compute the Radon transform (DRR)
        for CT_slice in phantom:
            sinogram = radon(CT_slice, theta=theta_deg, circle=False, preserve_range=False)
            ground_truth_sino.append(sinogram.T)

        # Make sure this is a Numpy array
        ground_truth_sino = np.array(ground_truth_sino).astype(np.single)

        # Convert the sinograms into projections
        ground_truth_proj = np.swapaxes(ground_truth_sino, 0, 1).astype(np.single)

    elif backend == "tomopy":
        theta_rad = np.array(theta_deg) * math.pi / 180

        # Compute the Radon transform (DRR)
        ground_truth_proj = tomopy.project(phantom, theta_rad)

        # Convert the projections into sinograms
        ground_truth_sino = np.swapaxes(ground_truth_proj, 0, 1).astype(np.single)

    elif backend == "tigre":
        # theta_rad = np.array(theta_deg) * math.pi / 180
        # geo = getGeo(mode, "hann")
        # ground_truth_proj = tigre.Ax(phantom, geo, theta_rad)
        # print(ground_truth_proj.shape)

        theta_rad = np.array(theta_deg) * math.pi / 180

        # Compute the Radon transform (DRR)
        ground_truth_proj = tomopy.project(phantom, theta_rad)

        # Convert the projections into sinograms
        ground_truth_sino = np.swapaxes(ground_truth_proj, 0, 1).astype(np.single)

    #elif backend == "plastimatch":

    else:
        raise IOError("No Tomgraphy backend chosen")

    # Crop
    half_width_delta = round((ground_truth_proj.shape[2] - proj_lenght_in_pixel) / 2)
    half_height_delta = round((ground_truth_proj.shape[1] - number_of_slices) / 2)

    ground_truth_proj = ground_truth_proj[:,
              half_height_delta:number_of_slices + half_height_delta,
              half_width_delta:proj_lenght_in_pixel + half_width_delta]

    ground_truth_sino = ground_truth_sino[half_height_delta:number_of_slices + half_height_delta,
              :,
              half_width_delta:proj_lenght_in_pixel + half_width_delta]

    return ground_truth_proj, ground_truth_sino

def reconsScikitImage(sinograms, CT_to_append_to, theta, filter_name, iStart, iEnd):
    for i in range(iStart, iEnd):
        CT_to_append_to[i] = iradon(sinograms[i].T, theta=theta, circle=False, filter_name=filter_name);

def recons(proj, theta_deg, mode="parallel", filter_name="hann", slice_cols=128, slice_rows=128, number_of_slices=128, backend=tomography_backend):

    # Scikit-image
    if backend == "scikit-image":

#         CT_volume = []

#         for sinogram in proj:
#             CT_volume.append(iradon(sinogram.T, theta=-theta_deg, circle=True, filter_name=filter_name))

#         CT_volume = np.array(CT_volume).astype(np.single)


        num_projections = len(proj);
        CT_volume = [None] * num_projections;
        handle_recon_threads = [];

        # The number of slices each thread will reconstruct
        num_slices_per_thread = num_projections // multiprocessing.cpu_count()
        remainder = num_projections % multiprocessing.cpu_count()
        first_slice = 0
        last_slice = first_slice + num_slices_per_thread

        for i in range(multiprocessing.cpu_count()):

            if remainder > 0:
                last_slice += 1
                remainder -= 1

            print("Thread", i + 1, "From slice (inclusive)", first_slice, "to", last_slice - 1)

            # Set up the thread
            handle_recon_threads.append(Thread(target=reconsScikitImage,
                                               args=(proj,
                                                     CT_volume,
                                                     theta_deg,
                                                     filter_name,
                                                     first_slice, last_slice)));

            # Start the thread
            handle_recon_threads[-1].start()

            first_slice = last_slice
            last_slice = first_slice + num_slices_per_thread

#             if i % num_slices_per_thread == 0 and iThreadIdx < multiprocessing.cpu_count():

#                 # The last thread will have to pick up the remaining slices
#                 if (iThreadIdx == multiprocessing.cpu_count() - 1): iEnd = num_projections;
#                 else: iEnd = i + num_slices_per_thread;

#                 # Set up the thread
#                 handle_recon_threads.append(Thread(target=reconsScikitImage,
#                                                    args=(proj,
#                                                          CT_volume,
#                                                          theta_deg,
#                                                          filter_name,
#                                                          i, iEnd)));
#                 iThreadIdx+=1;

        # Wait for the threads to complete
        for i in range(len(handle_recon_threads)):
            handle_recon_threads[i].join()

        CT_volume = np.array(CT_volume).astype(np.single)


    elif backend == "tomopy":

        theta_rad = np.array(theta_deg) * math.pi / 180

        rot_centre = proj.shape[2] / 2
        # rot_centre = tomopy.find_center(minus_log_projs, theta_rad, init=rot_centre, ind=0, tol=0.01)
        CT_volume = tomopy.recon(proj,
                             theta_rad,
                             center=rot_centre,
                             algorithm='gridrec',
                             sinogram_order=False,
                             filter_name=filter_name).astype(np.single)


    elif backend == "tigre":

        theta_rad = np.array(theta_deg) * math.pi / 180

        geo = getGeo(mode, filter_name)
        #  None, shepp_logan, cosine, hamming, hann

        #Reconstruction with FDK
        CT_volume = algs.fdk(proj, geo, -theta_rad)

        for i, CT_slice in enumerate(CT_volume):
            CT_volume[i] = np.rot90(CT_slice, 1)

        CT_volume = np.array(CT_volume).astype(np.single)

    else:
        raise IOError("No Tomgraphy backend chosen")

    # Crop
    half_slices_delta = round((CT_volume.shape[0] - number_of_slices) / 2)
    half_rows_delta = round((CT_volume.shape[1] - slice_rows) / 2)
    half_cols_delta = round((CT_volume.shape[2] - slice_cols) / 2)

    CT_volume = CT_volume[half_slices_delta:number_of_slices + half_slices_delta,
              half_rows_delta:slice_rows + half_rows_delta,
              half_cols_delta:slice_cols + half_cols_delta]

    return CT_volume


total_energy = None

def flatFieldCorrection(proj):

    global total_energy

    if total_energy is None:

        # Retrieve the total energy
        total_energy = 0.0;
        energy_bins = gvxr.getEnergyBins("MeV");
        photon_count_per_bin = gvxr.getPhotonCountEnergyBins();

        for energy, count in zip(energy_bins, photon_count_per_bin):
            total_energy += energy * count;

    # Create a mock dark field image
    #dark_field_image = np.zeros(raw_projections.shape);
    dark_field_image = 0.0

    # Create a mock flat field image
    #flat_field_image = np.ones(raw_projections.shape);
    flat_field_image = 1.0

    flat_field_image *= total_energy

    return ((np.array(proj).astype(np.single) - dark_field_image) / (flat_field_image - dark_field_image)).astype(np.single)


def minusLog(proj):

    minus_log_projs = np.copy(proj)

    # Make sure no value is negative or null (because of the log function)
    # It should not be the case, however, when the Laplacian is used to simulate
    # phase contrast, negative values can be generated.
    threshold = 0.000000001
    minus_log_projs[minus_log_projs < threshold] = threshold;

    # Apply the minus log normalisation
    minus_log_projs = -np.log(minus_log_projs);

    # Rescale the data taking into account the pixel size
    pixel_spacing_in_mm = gvxr.getDetectorSize("mm")[0] / gvxr.getDetectorNumberOfPixels()[0]
    minus_log_projs /= pixel_spacing_in_mm * (gvxr.getUnitOfLength("mm") / gvxr.getUnitOfLength("cm"));

    # Make sure the data is in single-precision floating-point numbers
    return minus_log_projs.astype(np.single)



def standardisation(image):
    return (np.array(image).astype(np.single) - np.mean(image)) / np.std(image);

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

    plt.savefig(fname + '.pdf', bbox_inches = 'tight')
    plt.savefig(fname + '.png', bbox_inches = 'tight')

def plotSpectrum(k, f, fname=None, xlim=[0,200]):

    plt.figure(figsize= (20,10))

    plt.bar(k, f / f.sum()) # Plot the spectrum
    plt.xlabel('Energy in keV')
    plt.ylabel('Probability distribution of photons per keV')
    plt.title('Photon energy distribution')

    plt.xlim(xlim)

    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname + '.pdf', bbox_inches = 'tight')
        plt.savefig(fname + '.png', bbox_inches = 'tight')

def compareImages(gate_image, gvxr_image, caption, fname, threshold=3):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))

    relative_error = 100 * (gate_image - gvxr_image) / gate_image
    comp_equalized = compare_images(gate_image, gvxr_image, method='checkerboard', n_tiles=(15,15))

    im1=axes.flat[0].imshow(comp_equalized, cmap="gray", vmin=0.25, vmax=1)
    axes.flat[0].set_title(caption)
    axes.flat[0].set_xticks([])
    axes.flat[0].set_yticks([])

    im2=axes.flat[1].imshow(relative_error, cmap="RdBu", vmin=-threshold, vmax=threshold)
    axes.flat[1].set_title("Relative error (in %)")
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

    plt.savefig(fname + '.pdf', bbox_inches = 'tight')
    plt.savefig(fname + '.png', bbox_inches = 'tight')


def fullCompareImages(gate_image: np.array, gvxr_image: np.array, title: str, fname: str, spacing, log: bool=False, vmin=0.25, vmax=1):


    # offset1 = min(gate_image.min(), gvxr_image.min())
    # offset2 = 0.01 * (gate_image.max() - gate_image.min())
    # offset = offset2 - offset1

    # comp_equalized = compare_images(gate_image.astype(np.single), gvxr_image.astype(np.single), method='diff', n_tiles=(15,15)) / gate_image.astype(np.single)
    comp_equalized = 100 * ((gate_image).astype(np.single) - (gvxr_image).astype(np.single)) / (gate_image).astype(np.single)

    # print("min relative error:", comp_equalized.min(), "%")
    # print("max relative error:", comp_equalized.max(), "%")
    # print("average relative error:", comp_equalized.mean(), "%")
    # print("std relative error:", comp_equalized.std(), "%")

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 20))

    if not log:
        im1 = axes.flat[0].imshow(gate_image, cmap="gray", vmin=vmin, vmax=vmax,
                                 extent=[0,(gate_image.shape[1]-1)*spacing[0],0,(gate_image.shape[0]-1)*spacing[1]])
    else:
        im1 = axes.flat[0].imshow(gate_image, cmap="gray", norm=LogNorm(vmin=0.01, vmax=1.2),
                                 extent=[0,(gate_image.shape[1]-1)*spacing[0],0,(gate_image.shape[0]-1)*spacing[1]])
    axes.flat[0].set_title("Ground truth")
    # axes.flat[0].set_xticks([])
    # axes.flat[0].set_yticks([])


    if not log:
        im2 = axes.flat[1].imshow(gvxr_image, cmap="gray", vmin=vmin, vmax=vmax,
                                 extent=[0,(gvxr_image.shape[1]-1)*spacing[0],0,(gvxr_image.shape[0]-1)*spacing[1]])
    else:
        im2 = axes.flat[1].imshow(gvxr_image, cmap="gray", norm=LogNorm(vmin=0.01, vmax=1.2),
                                 extent=[0,(gvxr_image.shape[1]-1)*spacing[0],0,(gvxr_image.shape[0]-1)*spacing[1]])
    axes.flat[1].set_title("gVirtualXRay")
    # axes.flat[1].set_xticks([])
    # axes.flat[1].set_yticks([])

    im3 = axes.flat[2].imshow(comp_equalized, cmap="RdBu", vmin=-5, vmax=5,
                             extent=[0,(comp_equalized.shape[1]-1)*spacing[0],0,(comp_equalized.shape[0]-1)*spacing[1]])
    axes.flat[2].set_title("Relative error (in %)")
    # axes.flat[2].set_title("Checkerboard comparison between\nGround truth & gVirtualXRay")
    # axes.flat[2].set_xticks([])
    # axes.flat[2].set_yticks([])

    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.2, hspace=0.02)

    # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8
    cb_ax = fig.add_axes([0.83, 0.425, 0.02, 0.15])
    cbar = fig.colorbar(im3, cax=cb_ax)

    axes.flat[1].set_xlabel("Pixel position\n(in mm)")
    axes.flat[0].set_ylabel("Pixel position\n(in mm)")

    plt.savefig(fname + '.pdf', bbox_inches = 'tight')
    plt.savefig(fname + '.png', bbox_inches = 'tight')


# def fullCompareImages(gate_image: np.array, gvxr_image: np.array, title: str, fname: str, log: bool=False, vmin=0.25, vmax=1, avoid_div_0=True):

#     absolute_error = np.abs(gate_image - gvxr_image)

#     if avoid_div_0:
#         offset1 = min(gate_image.min(), gvxr_image.min())
#         offset2 = 0.01 * (gate_image.max() - gate_image.min())
#         offset = offset2 - offset1

#         relative_error = 100 * ((gate_image + offset) - (gvxr_image + offset)) / (gate_image + offset)
#     else:
#         relative_error = 100 * (gate_image - gvxr_image) / gate_image

#     comp_equalized = compare_images(gate_image.astype(np.single), gvxr_image.astype(np.single), method='checkerboard', n_tiles=(15,15))

#     fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 20))

#     if not log:
#         im1 = axes.flat[0].imshow(gate_image, cmap="gray", vmin=vmin, vmax=vmax)
#     else:
#         im1 = axes.flat[0].imshow(gate_image, cmap="gray", norm=LogNorm(vmin=0.01, vmax=1.2))
#     axes.flat[0].set_title("Ground truth")
#     axes.flat[0].set_xticks([])
#     axes.flat[0].set_yticks([])

#     if not log:
#         im2 = axes.flat[1].imshow(gvxr_image, cmap="gray", vmin=vmin, vmax=vmax)
#     else:
#         im2 = axes.flat[1].imshow(gvxr_image, cmap="gray", norm=LogNorm(vmin=0.01, vmax=1.2))
#     axes.flat[1].set_title("gVirtualXRay")
#     axes.flat[1].set_xticks([])
#     axes.flat[1].set_yticks([])

#     if not log:
#         im3 = axes.flat[2].imshow(comp_equalized, cmap="gray", vmin=vmin, vmax=vmax)
#     else:
#         im3 = axes.flat[2].imshow(comp_equalized, cmap="gray", norm=LogNorm(vmin=0.01, vmax=1.2))
#     axes.flat[2].set_title("Checkerboard comparison between\nGround truth & gVirtualXRay")
#     axes.flat[2].set_xticks([])
#     axes.flat[2].set_yticks([])

#     im4 = axes.flat[3].imshow(relative_error, cmap="RdBu", vmin=-5, vmax=5)
#     axes.flat[3].set_title("Relative error (in %)")
#     axes.flat[3].set_xticks([])
#     axes.flat[3].set_yticks([])

#     fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
#                         wspace=0.2, hspace=0.02)

#     # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8
#     cb_ax = fig.add_axes([0.83, 0.425, 0.02, 0.15])
#     cbar = fig.colorbar(im4, cax=cb_ax)

#     plt.savefig(fname + '.pdf', bbox_inches = 'tight')
#     plt.savefig(fname + '.png', bbox_inches = 'tight')


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

    plt.savefig(fname + ".pdf", bbox_inches = 'tight')
    plt.savefig(fname + ".png", bbox_inches = 'tight')


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

    plt.savefig(fname + ".pdf", bbox_inches = 'tight')
    plt.savefig(fname + ".png", bbox_inches = 'tight')

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

    plt.savefig(fname + ".pdf", bbox_inches = 'tight')
    plt.savefig(fname + ".png", bbox_inches = 'tight')

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
