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

        # Choose paths depending on input mode
        if self.inp == 'train':
            tracksters_path = 'tracksters;1'
            associations_path = 'associations;1'
            simtrack = 'simtrackstersCP;3'
        elif self.inp == 'val':
            tracksters_path = 'tracksters;1'
            associations_path = 'associations;1'
            simtrack = 'simtrackstersCP;2'
        else:
            tracksters_path = 'tracksters;1'
            associations_path = 'associations;1'
            simtrack = 'simtrackstersCP;1'

        for path in tqdm.tqdm(self.raw_paths):
            # Load tracksters features in chunks
            for array in uproot.iterate(
                f"{path}:{tracksters_path}",
                [
                    "time", "raw_energy",
                    "barycenter_x", "barycenter_y", "barycenter_z", 
                    "barycenter_eta", "barycenter_phi",
                    "EV1", "EV2", "EV3",
                    "eVector0_x", "eVector0_y", "eVector0_z",
                    "sigmaPCA1", "sigmaPCA2", "sigmaPCA3"
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
                
                
                vert_array = []
                for vert_chunk in uproot.iterate(
                    f"{path}:{simtrack}",
                    ["barycenter_x"],
                ):
                    vert_array = vert_chunk["barycenter_x"]
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
                    if len(e) == 2:
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
        event_assoc = np.array(event_assoc)  # e.g. [[1,0],[0,1],...]
        
        
        
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
        # Create positive and negative edges
        # Identify tracksters from particle A and B
        # A: [1,0], B: [0,1]
        # We'll store the indices of each particle
        particleA = [i for i,assoc in enumerate(event_assoc) if assoc[0] == 1 and assoc[1] == 0]
        particleB = [i for i,assoc in enumerate(event_assoc) if assoc[0] == 0 and assoc[1] == 1]

        pos_edges = []
        neg_edges = []
        total_tracksters = len(event_time)

        # Function to get a random trackster from the same particle (allow self if only one)
        # Function to get a random trackster from the same particle (allow self if only one)
        def get_positive_target(curr_idx, same_particle_list):
            if not same_particle_list:
                # If no tracksters in the same particle (edge case), fall back to self
                return curr_idx
            if len(same_particle_list) == 1:
                # Only one trackster in this particle, must link to itself
                return curr_idx
            else:
                target = random.choice(same_particle_list)
                while target == curr_idx:
                    target = random.choice(same_particle_list)
                return target

        # Function to get a random trackster from the other particle
        def get_negative_target(curr_idx, other_particle_list):
            if not other_particle_list:
                # If no tracksters from the other particle, skip negative edge or link to self
                return curr_idx
            else:
                return random.choice(other_particle_list)

        # Create edges for each trackster
        for i in range(total_tracksters):
            # Determine which particle this trackster belongs to
            if i in particleA:
                pos_target = get_positive_target(i, particleA)
                neg_target = get_negative_target(i, particleB)
            else:
                # belongs to particle B
                pos_target = get_positive_target(i, particleB)
                neg_target = get_negative_target(i, particleA)

            pos_edges.append([i, pos_target])
            neg_edges.append([i, neg_target])

        x_pos_edge = torch.tensor(pos_edges, dtype=torch.long)
        x_neg_edge = torch.tensor(neg_edges, dtype=torch.long)
        
        return Data(x=x, x_pe=x_pos_edge, x_ne=x_neg_edge, assoc= assoc, RtS_score = RtS_score, StR_score = StR_score)