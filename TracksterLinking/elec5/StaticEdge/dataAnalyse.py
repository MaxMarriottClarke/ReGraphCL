import glob
import os.path as osp
import uproot
import awkward as ak
import torch
import numpy as np
import random
import tqdm
from torch_geometric.data import Data, Dataset

import numpy as np
import subprocess
import tqdm
import pandas as pd

import os
import os.path as osp

import glob

import h5py
import uproot

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data import DataLoader

import awkward as ak
import random

import awkward as ak
import random
def find_highest_branch(path, base_name):
    with uproot.open(path) as f:
        # Find keys that exactly match the base_name (not containing other variations)
        branches = [k for k in f.keys() if k.startswith(base_name + ';')]
        
        # Sort and select the highest-numbered branch
        sorted_branches = sorted(branches, key=lambda x: int(x.split(';')[-1]))
        return sorted_branches[-1] if sorted_branches else None

class CCV2(Dataset):
    r'''
    Loads trackster-level features and associations for positive/negative edge creation.
    '''

    url = '/dummy/'

    def __init__(self, root, transform=None, max_events=1e8, inp='train'):
        super(CCV2, self).__init__(root, transform)
        self.inp = inp
        self.max_events = max_events
        self.fill_data(max_events)

    def fill_data(self, max_events):
        counter = 0
        print("### Loading tracksters data")

        # Choose paths depending on input mode

        for path in tqdm.tqdm(self.raw_paths):
            
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
                    "sigmaPCA1", "sigmaPCA2", "sigmaPCA3", "vertices_indexes"
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
                tmp_ind = array["vertices_indexes"]
                
                
                vert_array = []
                GT_ind_array = []
                GT_en_array = []
                for vert_chunk in uproot.iterate(
                    f"{path}:{simtrack}",
                    ["barycenter_x", "vertices_indexes", "vertices_energy"],
                ):
                    vert_array = vert_chunk["barycenter_x"]
                    GT_ind_array = vert_chunk["vertices_indexes"]
                    GT_en_array = vert_chunk["vertices_energy"]
                    break  # Since we have a matching chunk, no need to continue
                    

                
                
                
                assoc_array = []
                RtS_score_array = []
                StR_score_array = []
                
                for assoc_chunk in uproot.iterate(
                    f"{path}:{associations_path}",
                    [
                        "tsCLUE3D_recoToSim_CP",
                        "Mergetracksters_recoToSim_CP_score",
                        "Mergetracksters_simToReco_CP_score"
                    ],
                ):
                    assoc_array = assoc_chunk["tsCLUE3D_recoToSim_CP"]
                    RtS_score_array = assoc_chunk["Mergetracksters_recoToSim_CP_score"]
                    StR_score_array = assoc_chunk["Mergetracksters_simToReco_CP_score"]
                    break

                
                skim_mask = []
                for e in vert_array:
                    if 1 <= len(e) <= 5:
                        skim_mask.append(True)
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
                
                assoc_array = assoc_array[skim_mask]
                RtS_score_array = RtS_score_array[skim_mask]
                StR_score_array = StR_score_array[skim_mask]

                
                
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
                    self.assoc = assoc_array
                    self.RtS_score = RtS_score_array
                    self.StR_score = StR_score_array
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
                    self.assoc = ak.concatenate((self.assoc, assoc_array))
                    self.RtS_score = ak.concatenate((self.RtS_score, RtS_score_array))
                    self.StR_score = ak.concatenate((self.StR_score, StR_score_array))


                counter += len(tmp_time)
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

        event_assoc = self.assoc[idx]  # Shape: [N_tracksters, 2], either [1,0] or [0,1]
        event_RtS_score = np.array(self.RtS_score[idx])
        event_StR_score = np.array(self.StR_score[idx])

        
        
        # Convert to numpy
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
        event_assoc = np.array(event_assoc)  # e.g. [[1,0],[0,1],...

        
        
        
        # Combine features
        flat_feats = np.column_stack((
            event_bx, event_by, event_bz,event_raw_energy,
            event_beta, event_bphi,
            event_EV1, event_EV2, event_EV3,
            event_eV0x, event_eV0y, event_eV0z,
            event_sigma1, event_sigma2, event_sigma3
        ))

        x = torch.from_numpy(flat_feats).float()
        assoc = torch.from_numpy(event_assoc.astype(np.int32)).float()

        RtS_score = torch.from_numpy(event_RtS_score).float()
        StR_score = torch.from_numpy(event_StR_score).float()
        
        total_tracksters = len(event_time)

        # --------------------------------------------------------------------
        # Group tracksters by their association tuple. Two tracksters belong
        # to the same pion group if their association arrays (converted to tuples)
        # match.
        # --------------------------------------------------------------------
        assoc_groups = {}
        for i, assoc in enumerate(event_assoc):
            key = tuple(assoc)  # Convert to tuple so it can be used as a dictionary key.
            if key not in assoc_groups:
                assoc_groups[key] = []
            assoc_groups[key].append(i)

        pos_edges = []
        neg_edges = []

        # --------------------------------------------------------------------
        # For each trackster, pick:
        #  - a positive edge: a trackster from the same group. If no other trackster
        #    is available, allow an edge to itself.
        #  - a negative edge: a trackster from any other group.
        # --------------------------------------------------------------------
        for i in range(total_tracksters):
            key = tuple(event_assoc[i])
            same_group = assoc_groups[key]

            # --- Positive edge ---
            # Find candidates in the same group that are not i.
            pos_candidates = [j for j in same_group if j != i]
            if pos_candidates:
                pos_target = random.choice(pos_candidates)
            else:
                # Since self loops are allowed for positive edges now, use self.
                pos_target = i
            pos_edges.append([i, pos_target])

            # --- Negative edge ---
            # Candidates: tracksters not in the same group.
            neg_candidates = [j for j in range(total_tracksters) if j not in same_group]
            if neg_candidates:
                neg_target = random.choice(neg_candidates)
            else:
                # If, for some reason, no negative candidate exists, you could either
                # allow a self-loop or choose to leave it out. Here we choose self.
                neg_target = i
            neg_edges.append([i, neg_target])

        x_pos_edge = torch.tensor(pos_edges, dtype=torch.long)
        x_neg_edge = torch.tensor(neg_edges, dtype=torch.long)


        
        return Data(x=x, x_pe=x_pos_edge, x_ne=x_neg_edge, assoc= assoc, RtS_score = RtS_score, StR_score = StR_score)