# Inhibitory densities of the mouse brain

## Summary
This repository contains the tools used to compute cell densities in the whole mouse brain 
using literature values together with marker expression data from the 
Allen Institute for Brain Science (AIBS) in-situ hybridization (ISH) experiments.
The pipeline is decomposed in 3 steps:
* Fitting of a transfer function from marker intensity to cell density to obtain cell density 
for regions not covered by literature values 
* Filling of the whole mouse brain according to neuron distribution from the BBP Mouse Cell Atlas (CA) 
and previously computed values.
* Correction of the densities to match markers constraints: GAD1 = PV + SST + VIP + Rest

## Installation
### 1. Install Requirements
This repository scripts are based on Python 3.6 

Create the directories *data* and *output* at the root of the project.

### 2. Required input files
The scripts require: 
* The 2017 version of the *annotation atlas* from the AIBS together with the json file containing the region hierarchy description.
* The neuron densities from the CA expressed as neuron/voxel
* The Excel files containing density of inhibitory neurons from Kim et al (2017) and other literature sources.
* Processed marker expression volumes files from the ISH experiments from the AIBS (GAD1, Pvalb, SST, VIP). 
The BBP alignment pipeline should have been applied to the raw brain slices to align the processed marker images to the AIBS Nissl volume. 

The required files should be placed in the *data* folder.

### 3. Excecution
By default, results will be stored in the *output* folder

#### Initialization step:
Before excecuting any of the following notebooks, 
you should run the *Counts&volumes* notebook to obtain the volumes and neuron counts in each region of the brain. 
This script has to be launch only once.

#### Main pipeline:
* First, run the notebooks that perform the *fitting* of the transfer function from region mean intensity to region densities.
* Next, run the notebooks that fill the mouse brain with cell positive to markers and their *densities*. 
* Finally, you can list all the inconsistencies, running the *Combine_markers* notebook

#### Additional notebooks:
* You can display the literature coverage or the neurons distribution in the form of a circle  using the *plot_distrib_neuron* notebook.