# ReGraphCL

This repository contains the code used for Chapters 5 and 6 of my thesis, as well as validation and inference notebooks.

## Directory Structure

```text
TracksterFormation/   # Training code for Chapter 5: Trackster formation
TracksterLinking/     # Training code for Chapter 6: Trackster linking
Validation/           # Inference scripts validation and results generation
```

### TracksterFormation

This folder contains all the training code, configuration files, and helper scripts related to **Chapter 5: Trackster Formation** of the thesis. Each folder within this branch is for the various datasets. Mixed and the 5 pion and 5 electron are used in the thesis. I have also included the 2 pion and 2 electron, which we started with if this project is continued next year.

Within those folders, lie the different training methods tested:
-`Circle`: The Circle loss.
-`Fraction`: Fractional assignments of LC's.
-`Full`: The main folder, used to tain the full NTXENT and HN-NTXENT scripts.
-`NT`: NTXENT
-`NegativeMining`: HN-NTXENT variant described in the thesis.
-`Split`: A novel idea to add another head which predicts if a LC is Mixed or not, similar to fractional assignments.
-`SC`: Supervised Contrastive Learning.

Within each folder contains the files needed to train, submit to HTCondor. However, to reduce the number of files and save storage space, all log files and model scripts are not saved here. If someone is interested, please reach out to me and I can send any model.pt over. 

### TracksterLinking

This folder holds everything required for **Chapter 6: Trackster Linking**, where formed tracksters are linked into full particle trajectories. Similar folders for the various particle types are within this folder.


### Validation

The `Validation` directory provides the scripts you need to run inference with the trained models, generate performance plots, and reproduce the figures in the thesis. This is split into the chapter 5 and chapter 6 folders, as well as an additional folder, which shows the development and some other scripts used along the way. 

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository>
   ```

2. **Install dependencies** (e.g., Python 3.8+, PyTorch, PyG, etc.):
   - my miniconda environement is not saved here, a requirements.txt will be made. Most of this was trained using a custom miniconda environment. TIP: If working with CUDA, make sure the PyG version matches with the PyTorch, otherwise annoying errors will occur. Also by using custom envioronments rather than the CERN source, you might be able to access more GPU's ;).
   - If on imperial HEP Cluster, you can just run:
   ```
   source /vols/cms/mm1221/hgcal/code/code/miniconda/etc/profile.d/conda.sh 
   conda activate env
   ```

3. **Run training**:
   - To train a model, all the condor submit and run.sh files are provided. Check https://www.imperial.ac.uk/computing/people/csg/services/hpc/condor/ for helpful advice.

4. **Validate & plot**:
   - mostly in the form of Jupyter notebooks. pretty self explanitory. 

---

For detailed instructions and parameter descriptions, see the individual `README.md` files in each subdirectory (if present), or refer to the thesis document for theoretical background and experimental setup.


 
