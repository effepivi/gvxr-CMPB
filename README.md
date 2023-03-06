# gvxr-CMPB

**Simulation of X-ray projections on GPU: benchmarking gVirtualXray with clinically realistic phantoms**

Jamie Lea Pointon<sup>a</sup>, Tianci Wen<sup>a</sup>, Jenna Tugwell-Allsup<sup>b</sup>, Aaron S&uacute;jar<sup>c,a</sup>, Jean Michel Létang<sup>d</sup>, Franck Patrick Vidal<sup>a,*</sup>

<i><sup>a</sup> School of Computer Science &amp; Electronic Engineering, Bangor University, UK</i>
<i><sup>b</sup> Radiology Department, Betsi Cadwaladr University Health Board (BCUHB), North Wales, Ysbyty Gwynedd, UK</i>
<i><sup>c</sup> Department of Computer Science, Universidad Rey Juan Carlos, Mostoles, Spain</i>
<i><sup>d</sup> Univ Lyon, INSA-Lyon, Université Claude Bernard Lyon 1, UJM-Saint &Eacute;tienne, CNRS, Inserm, CREATIS UMR 5220, U1294, F-69373, LYON, France</i>
<i><sup>*</sup> Corresponding author</i>

Submitted to Computer Methods and Programs in Biomedicine

## Abstract

*Background and Objectives:* This study provides a quantitative comparison of images created using gVirtualXray (gVXR) to both Monte Carlo (MC) and real images of clinically realistic phantoms. gVirtualXray is an open-source framework that relies on the Beer-Lambert law to simulate X-ray images in realtime on a graphics processor unit (GPU) using triangular meshes.

*Methods:* Images are generated with gVirtualXray and compared with a corresponding ground truth image of an anthropomorphic phantom: (i) an X-ray projection generated using a Monte Carlo simulation code, (ii) real digitally reconstructed radiographs (DRRs), (iii) computed tomography (CT) slices, and (iv) a real radiograph acquired with a clinical X-ray imaging system. When real images are involved, the simulations are used in an image registration framework so that the two images are aligned.

*Results:* The mean absolute percentage error (MAPE) between the images simulated with gVirtualXray and MC is 3.12%, the zero-mean normalised cross-correlation (ZNCC) is 99.96% and the structural similarity index (SSIM) is 0.99. The run-time is 10 days for MC and 23 msec with gVirtualXray. Images simulated using surface models segmented from a CT scan of the Lungman chest phantom were similar to i) DRRs computed from the CT volume and ii) an actual digital radiograph. CT slices reconstructed from images simulated with gVirtualXray were comparable to the corresponding slices of the original CT volume.

*Conclusions:* When scattering can be ignored, accurate images that would take days using MC can be generated in milliseconds with gVirtualXray. This speed of execution enables the use of repetitive simulations with varying parameters, e.g. to generate training data for a deep-learning algorithm, and to minimise the objective function of an optimisation problem in image registration. The use of surface models enables the combination of X-ray simulation with real-time soft-tissue deformation and character animation, which can be deployed in virtual reality applications.

## Keywords

 X-rays; Computed tomography; Simulation; Monte Carlo; GPU programming; Image registration; DRR

## Highlights

- Realistic X-ray simulation from anatomical data
- Registration of simulated X-ray images on real radiographs
- Validation and benchmarking using Monte Carlo simulation, real radiographs and DRRs
- Superior computational performance for VR and high-throughput data applications

## Installation

```bash
conda  env create -f environment.yml
```

## Related software projects

- [gVirtualXray (gVXR)](http://gvirtualxray.sourceforge.io/) provides a programming framework for simulating X-ray images on the graphics processor unit (GPU) using OpenGL. In a nutshell, it computes the polychromatic version of the Beer-Lambert law (the mathematical model that relates the attenuation of X-ray photons to the properties of the material through which the photons are travelling) on the graphics card from polygon meshes.
- [xraylib](https://github.com/tschoonj/xraylib) provides the mass attenuation coefficients used by gVXR. 
- The [Core Imaging Library (CIL](https://ccpi.ac.uk/cil/) is an open-source mainly Python framework for tomographic imaging for cone and parallel beam geometries. It comes with tools for loading, preprocessing, reconstructing and visualising tomographic data.
- [SpekPy](https://bitbucket.org/spekpy/spekpy_release/wiki/Home) is a free software toolkit for calculating and manipulating x-ray tube spectra.
- [Gate](http://www.opengatecollaboration.org/) is an open-source software dedicated to numerical simulations in medical imaging and radiotherapy based on [Geant4](https://geant4.web.cern.ch/), the general-purpose Monte Carlo (MC) code by the European Organization for Nuclear Research (CERN).

## References

- Vidal F.P., Villard P.F. Development and validation of real-time simulation of X-ray imaging with respiratory motion. [Comput Med Imaging Graph](https://www.sciencedirect.com/journal/computerized-medical-imaging-and-graphics). 2016 Apr;49:1-15. doi: [10.1016/j.compmedimag.2015.12.002](https://doi.org/10.1016/j.compmedimag.2015.12.002). Epub 2015 Dec 17. PMID: [26773644](https://pubmed.ncbi.nlm.nih.gov/26773644/).
