import os
import glob
import random
import subprocess

import numpy as np
import pandas as pd
import h5py
import uproot
import awkward as ak

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import knn_graph

import tqdm
from tqdm import tqdm

import os
import os.path as osp  # This defines 'osp'
import glob



def find_highest_branch(path, base_name):
    with uproot.open(path) as f:
        # Find keys that exactly match the base_name (not containing other variations)
        branches = [k for k in f.keys() if k.startswith(base_name + ';')]
        
        # Sort and select the highest-numbered branch
        sorted_branches = sorted(branches, key=lambda x: int(x.split(';')[-1]))
        return sorted_branches[-1] if sorted_branches else None

class CCV1(Dataset):
    r'''
    Loads trackster-level features and associations for positive/negative edge creation.
    '''

    url = '/dummy/'

    def __init__(self, root, transform=None, max_events=1e8, inp='train'):
        super(CCV1, self).__init__(root, transform)
        self.inp = inp
        self.max_events = max_events
        self.fill_data(max_events)

    def fill_data(self, max_events):
        counter = 0
        print("### Loading tracksters data")


        for path in tqdm(self.raw_paths):
            print(path)
            
            tracksters_path = find_highest_branch(path, 'tracksters')
            associations_path = find_highest_branch(path, 'associations')
            simtrack = find_highest_branch(path, 'simtrackstersCP')
            # Load tracksters features in chunks
            for array in uproot.iterate(
                f"{path}:{tracksters_path}",
                [
                    "time", "raw_energy",
                    "barycenter_x", "barycenter_y", "barycenter_z", 
                    "barycenter_eta", "barycenter_phi",
                    "EV1", "EV2", "EV3",
                    "eVector0_x", "eVector0_y", "eVector0_z",
                    "sigmaPCA1", "sigmaPCA2", "sigmaPCA3", "raw_pt", "vertices_time"
                ],
            ):

                tmp_time = array["time"]
                tmp_raw_energy = array["raw_energy"]
                tmp_bx = array["barycenter_x"]
                tmp_by = array["barycenter_y"]
                tmp_bz = array["barycenter_z"]
                tmp_beta = array["barycenter_eta"]
                tmp_bphi = array["barycenter_phi"]
                tmp_EV1 = array["EV1"]
                tmp_EV2 = array["EV2"]
                tmp_EV3 = array["EV3"]
                tmp_eV0x = array["eVector0_x"]
                tmp_eV0y = array["eVector0_y"]
                tmp_eV0z = array["eVector0_z"]
                tmp_sigma1 = array["sigmaPCA1"]
                tmp_sigma2 = array["sigmaPCA2"]
                tmp_sigma3 = array["sigmaPCA3"]
                tmp_pt = array["raw_pt"]
                tmp_vt = array["vertices_time"]
                
                
                vert_array = []
                for vert_chunk in uproot.iterate(
                    f"{path}:{simtrack}",
                    ["barycenter_x"],
                ):
                    vert_array = vert_chunk["barycenter_x"]
                    break  # Since we have a matching chunk, no need to continue
                

                # Now load the associations for the same events/chunk
                # 'tsCLUE3D_recoToSim_CP' gives association arrays like [[1,0],[0,1],...]
                # Make sure we read from the same events
                tmp_array = []
                score_array = []
                for assoc_chunk in uproot.iterate(
                    f"{path}:{associations_path}",
                    ["tsCLUE3D_recoToSim_CP", "tsCLUE3D_recoToSim_CP_score"],
                ):
                    tmp_array = assoc_chunk["tsCLUE3D_recoToSim_CP"]
                    score_array = assoc_chunk["tsCLUE3D_recoToSim_CP_score"]
                    break  # Since we have a matching chunk, no need to continue
                
                
                skim_mask = []
                for e in vert_array:
                    if len(e) >= 2:
                        skim_mask.append(True)
                    elif len(e) == 0:
                        skim_mask.append(False)

                    else:
                        skim_mask.append(False)



                tmp_time = tmp_time[skim_mask]
                tmp_raw_energy = tmp_raw_energy[skim_mask]
                tmp_bx = tmp_bx[skim_mask]
                tmp_by = tmp_by[skim_mask]
                tmp_bz = tmp_bz[skim_mask]
                tmp_beta = tmp_beta[skim_mask]
                tmp_bphi = tmp_bphi[skim_mask]
                tmp_EV1 = tmp_EV1[skim_mask]
                tmp_EV2 = tmp_EV2[skim_mask]
                tmp_EV3 = tmp_EV3[skim_mask]
                tmp_eV0x = tmp_eV0x[skim_mask]
                tmp_eV0y = tmp_eV0y[skim_mask]
                tmp_eV0z = tmp_eV0z[skim_mask]
                tmp_sigma1 = tmp_sigma1[skim_mask]
                tmp_sigma2 = tmp_sigma2[skim_mask]
                tmp_sigma3 = tmp_sigma3[skim_mask]
                tmp_array = tmp_array[skim_mask]
                tmp_pt = tmp_pt[skim_mask]
                tmp_vt = tmp_vt[skim_mask]
                score_array = score_array[skim_mask]
                
                skim_mask = []
                for e in tmp_array:
                    if 2 <= len(e):
                        skim_mask.append(True)

                    elif len(e) == 0:
                        skim_mask.append(False)

                    else:
                        skim_mask.append(False)

                        
                tmp_time = tmp_time[skim_mask]
                tmp_raw_energy = tmp_raw_energy[skim_mask]
                tmp_bx = tmp_bx[skim_mask]
                tmp_by = tmp_by[skim_mask]
                tmp_bz = tmp_bz[skim_mask]
                tmp_beta = tmp_beta[skim_mask]
                tmp_bphi = tmp_bphi[skim_mask]
                tmp_EV1 = tmp_EV1[skim_mask]
                tmp_EV2 = tmp_EV2[skim_mask]
                tmp_EV3 = tmp_EV3[skim_mask]
                tmp_eV0x = tmp_eV0x[skim_mask]
                tmp_eV0y = tmp_eV0y[skim_mask]
                tmp_eV0z = tmp_eV0z[skim_mask]
                tmp_sigma1 = tmp_sigma1[skim_mask]
                tmp_sigma2 = tmp_sigma2[skim_mask]
                tmp_sigma3 = tmp_sigma3[skim_mask]
                tmp_array = tmp_array[skim_mask]
                tmp_pt = tmp_pt[skim_mask]
                tmp_vt = tmp_vt[skim_mask]
                score_array = score_array[skim_mask]

                
                # Concatenate or initialize storage
                if counter == 0:
                    self.time = tmp_time
                    self.raw_energy = tmp_raw_energy
                    self.bx = tmp_bx
                    self.by = tmp_by
                    self.bz = tmp_bz
                    self.beta = tmp_beta
                    self.bphi = tmp_bphi
                    self.EV1 = tmp_EV1
                    self.EV2 = tmp_EV2
                    self.EV3 = tmp_EV3
                    self.eV0x = tmp_eV0x
                    self.eV0y = tmp_eV0y
                    self.eV0z = tmp_eV0z
                    self.sigma1 = tmp_sigma1
                    self.sigma2 = tmp_sigma2
                    self.sigma3 = tmp_sigma3
                    self.assoc = tmp_array
                    self.pt = tmp_pt
                    self.vt = tmp_vt
                    self.score = score_array
                else:
                    self.time = ak.concatenate((self.time, tmp_time))
                    self.raw_energy = ak.concatenate((self.raw_energy, tmp_raw_energy))
                    self.bx = ak.concatenate((self.bx, tmp_bx))
                    self.by = ak.concatenate((self.by, tmp_by))
                    self.bz = ak.concatenate((self.bz, tmp_bz))
                    self.beta = ak.concatenate((self.beta, tmp_beta))
                    self.bphi = ak.concatenate((self.bphi, tmp_bphi))
                    self.EV1 = ak.concatenate((self.EV1, tmp_EV1))
                    self.EV2 = ak.concatenate((self.EV2, tmp_EV2))
                    self.EV3 = ak.concatenate((self.EV3, tmp_EV3))
                    self.eV0x = ak.concatenate((self.eV0x, tmp_eV0x))
                    self.eV0y = ak.concatenate((self.eV0y, tmp_eV0y))
                    self.eV0z = ak.concatenate((self.eV0z, tmp_eV0z))
                    self.sigma1 = ak.concatenate((self.sigma1, tmp_sigma1))
                    self.sigma2 = ak.concatenate((self.sigma2, tmp_sigma2))
                    self.sigma3 = ak.concatenate((self.sigma3, tmp_sigma3))
                    self.assoc = ak.concatenate((self.assoc, tmp_array))
                    self.pt = ak.concatenate((self.pt, tmp_pt))
                    self.vt = ak.concatenate((self.vt, tmp_vt))
                    self.score = ak.concatenate((self.score, score_array))

                counter += len(tmp_bx)
                if counter >= max_events:
                    print(f"Reached {max_events} events!")
                    break
            if counter >= max_events:
                break

    def download(self):
        raise RuntimeError(
            f'Dataset not found. Please download it from {self.url} and move all '
            f'*.root files to {self.raw_dir}')

    def len(self):
        return len(self.time)

    @property
    def raw_file_names(self):
        raw_files = sorted(glob.glob(osp.join(self.raw_dir, '*.root')))
        return raw_files

    @property
    def processed_file_names(self):
        return []



    def get(self, idx):

        def reconstruct_array(grouped_indices):
            # Finds the maximum index and returns a 1D array listing the group for each index.
            max_index = max(max(indices) for indices in grouped_indices.values())
            reconstructed = [-1] * (max_index + 1)
            for value, indices in grouped_indices.items():
                for idx2 in indices:
                    reconstructed[idx2] = value
            return reconstructed

        # Extract per-event arrays
        event_time = self.time[idx]
        event_raw_energy = self.raw_energy[idx]
        event_bx = self.bx[idx]
        event_by = self.by[idx]
        event_bz = self.bz[idx]
        event_beta = self.beta[idx]
        event_bphi = self.bphi[idx]
        event_EV1 = self.EV1[idx]
        event_EV2 = self.EV2[idx]
        event_EV3 = self.EV3[idx]
        event_eV0x = self.eV0x[idx]
        event_eV0y = self.eV0y[idx]
        event_eV0z = self.eV0z[idx]
        event_sigma1 = self.sigma1[idx]
        event_sigma2 = self.sigma2[idx]
        event_sigma3 = self.sigma3[idx]
        event_assoc = self.assoc[idx]      # associations; e.g. [0, 4, 3, 2]
        event_pt = self.pt[idx]
        event_vt = self.vt[idx]
        event_score = self.score[idx]      # scores; e.g. [0.000, 0.281, 1.0, 1.0]

        # Convert each to NumPy
        event_time = np.array(event_time)
        event_raw_energy = np.array(event_raw_energy)
        event_bx = np.array(event_bx)
        event_by = np.array(event_by)
        event_bz = np.array(event_bz)
        event_beta = np.array(event_beta)
        event_bphi = np.array(event_bphi)
        event_EV1 = np.array(event_EV1)
        event_EV2 = np.array(event_EV2)
        event_EV3 = np.array(event_EV3)
        event_eV0x = np.array(event_eV0x)
        event_eV0y = np.array(event_eV0y)
        event_eV0z = np.array(event_eV0z)
        event_sigma1 = np.array(event_sigma1)
        event_sigma2 = np.array(event_sigma2)
        event_sigma3 = np.array(event_sigma3)
        event_assoc = np.array(event_assoc)   # shape (N, ?) with nested arrays
        event_pt = np.array(event_pt)
        event_score = np.array(event_score)     # shape (N, ?) with nested arrays


        # Stack trackster features into x.
        flat_feats = np.column_stack((
            event_bx, event_by, event_bz, event_raw_energy,
            event_beta, event_bphi,
            event_EV1, event_EV2, event_EV3,
            event_eV0x, event_eV0y, event_eV0z,
            event_sigma1, event_sigma2, event_sigma3,
            event_pt
        ))
        x = torch.from_numpy(flat_feats).float()

        # Convert associations & scores to tensors.
        links_tensor = torch.from_numpy(event_assoc.astype(np.int64))
        scores_tensor = torch.from_numpy(event_score).float()

        # --- Truncate or pad each tensor to 4 columns ---
        def ensure_four_columns(tensor):
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(1)
            nrow, ncol = tensor.shape
            if ncol > 4:
                tensor = tensor[:, :4]
            elif ncol < 4:
                last_col = tensor[:, -1].unsqueeze(1)
                repeat_count = 4 - ncol
                repeated = last_col.repeat(1, repeat_count)
                tensor = torch.cat([tensor, repeated], dim=1)
            return tensor

        scores_tensor = ensure_four_columns(scores_tensor)
        links_tensor = ensure_four_columns(links_tensor)

        

        # Return the Data object with all fields.
        return Data(
            x=x,
            scores=scores_tensor,
            links=links_tensor
        )



