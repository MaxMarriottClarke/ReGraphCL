import numpy as np
import random
import glob
import os.path as osp
import uproot
import awkward as ak
import torch
from torch_geometric.data import Data, Dataset, DataLoader

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

# Function to find the highest numbered branch matching base_name in an uproot file
def find_highest_branch(path, base_name):
    with uproot.open(path) as f:
        branches = [k for k in f.keys() if k.startswith(base_name + ';')]
        sorted_branches = sorted(branches, key=lambda x: int(x.split(';')[-1]))
        return sorted_branches[-1] if sorted_branches else None

# Function to remove duplicates: for each event, only keep the entry with the highest B value for each unique element in A.
def remove_duplicates(A, B):    
    all_masks = []
    for event_idx, event in enumerate(A):
        flat_A = np.array(ak.flatten(A[event_idx]))
        flat_B = np.array(ak.flatten(B[event_idx]))
        mask = np.zeros_like(flat_A, dtype=bool)
        for elem in np.unique(flat_A):
            indices = np.where(flat_A == elem)[0]
            if len(indices) > 1:
                max_index = indices[np.argmax(flat_B[indices])]
                mask[max_index] = True
            else:
                mask[indices[0]] = True
        unflattened_mask = ak.unflatten(mask, ak.num(A[event_idx]))
        all_masks.append(unflattened_mask)
    return ak.Array(all_masks)

class CCV1(Dataset):
    r'''
    Dataset for layer clusters.
    For each event it builds:
      - x: node features, shape (N, D)
      - groups: top-3 contributing group IDs per node (N, 3)
      - fractions: corresponding energy fractions (N, 3)
      - x_pe: positive edge pairs, shape (N, 2)
      - x_ne: negative edge pairs, shape (N, 2)
    '''
    url = '/dummy/'

    def __init__(self, root, transform=None, max_events=1e8, inp='train'):
        super(CCV1, self).__init__(root, transform)
        self.step_size = 500
        self.inp = inp
        self.max_events = max_events

        # These will hold the precomputed arrays (one per event)
        self.precomputed_groups = []
        self.precomputed_fractions = []
        
        self.fill_data(max_events)
        self.precompute_pairings()

    def fill_data(self, max_events):
        counter = 0
        print("### Loading data")
        for fi,path in enumerate(tqdm(self.raw_paths)):
            if self.inp in ['train', 'val']:
                cluster_path = find_highest_branch(path, 'clusters')
                sim_path = find_highest_branch(path, 'simtrackstersCP')
                track_path = find_highest_branch(path, 'tracksters')
            else:
                cluster_path = find_highest_branch(path, 'clusters')
                sim_path = find_highest_branch(path, 'simtrackstersCP')
                track_path = find_highest_branch(path, 'tracksters')
            
            crosstree = uproot.open(path)[cluster_path]
            crosscounter = 0
            
            # Create an iterator for the tracksters branch to load its 'vertices_indexes'
            tracksters_iter = uproot.iterate(
                f"{path}:{track_path}",
                ["vertices_indexes"],
                step_size=self.step_size
            )
            
            for array in uproot.iterate(f"{path}:{sim_path}", 
                                         ["vertices_x", "vertices_y", "vertices_z", 
                                          "vertices_energy", "vertices_multiplicity", "vertices_time", 
                                          "vertices_indexes", "barycenter_x", "barycenter_y", "barycenter_z"],
                                         step_size=self.step_size):
                # Get the tracksters branch data for this iteration
                tmp_tracksters_data = next(tracksters_iter)
                tmp_tracksters_vertices_indexes = tmp_tracksters_data["vertices_indexes"]
                
                # Load the simtrackstersCP branch arrays
                tmp_stsCP_vertices_x = array['vertices_x']
                tmp_stsCP_vertices_y = array['vertices_y']
                tmp_stsCP_vertices_z = array['vertices_z']
                tmp_stsCP_vertices_energy = array['vertices_energy']
                tmp_stsCP_vertices_time = array['vertices_time']
                tmp_stsCP_vertices_indexes = array['vertices_indexes']
                tmp_stsCP_vertices_multiplicity = array['vertices_multiplicity']
                tmp_stsCP_barycenter_x = array['barycenter_x']
                tmp_stsCP_barycenter_y = array['barycenter_y']
                tmp_stsCP_barycenter_z = array['barycenter_z']
                
                self.step_size = min(self.step_size, len(tmp_stsCP_vertices_x))

                tmp_all_vertices_layer_id = crosstree['cluster_layer_id'].array(
                    entry_start=crosscounter*self.step_size,
                    entry_stop=(crosscounter+1)*self.step_size)
                tmp_all_vertices_noh = crosstree['cluster_number_of_hits'].array(
                    entry_start=crosscounter*self.step_size,
                    entry_stop=(crosscounter+1)*self.step_size)
                tmp_all_vertices_eta = crosstree['position_eta'].array(
                    entry_start=crosscounter*self.step_size,
                    entry_stop=(crosscounter+1)*self.step_size)
                tmp_all_vertices_phi = crosstree['position_phi'].array(
                    entry_start=crosscounter*self.step_size,
                    entry_stop=(crosscounter+1)*self.step_size)
                crosscounter += 1

                layer_id_list = []
                noh_list = []
                eta_list = []
                phi_list = []
                for evt_row in range(len(tmp_all_vertices_noh)):
                    layer_id_list_one_event = []
                    noh_list_one_event = []
                    eta_list_one_event = []
                    phi_list_one_event = []
                    for particle in range(len(tmp_stsCP_vertices_indexes[evt_row])):
                        tmp_stsCP_vertices_layer_id_one_particle = tmp_all_vertices_layer_id[evt_row][tmp_stsCP_vertices_indexes[evt_row][particle]]
                        tmp_stsCP_vertices_noh_one_particle = tmp_all_vertices_noh[evt_row][tmp_stsCP_vertices_indexes[evt_row][particle]]
                        tmp_stsCP_vertices_eta_one_particle = tmp_all_vertices_eta[evt_row][tmp_stsCP_vertices_indexes[evt_row][particle]]
                        tmp_stsCP_vertices_phi_one_particle = tmp_all_vertices_phi[evt_row][tmp_stsCP_vertices_indexes[evt_row][particle]]
                        layer_id_list_one_event.append(tmp_stsCP_vertices_layer_id_one_particle)
                        noh_list_one_event.append(tmp_stsCP_vertices_noh_one_particle)
                        eta_list_one_event.append(tmp_stsCP_vertices_eta_one_particle)
                        phi_list_one_event.append(tmp_stsCP_vertices_phi_one_particle)
                    layer_id_list.append(layer_id_list_one_event)
                    noh_list.append(noh_list_one_event)
                    eta_list.append(eta_list_one_event)
                    phi_list.append(phi_list_one_event)
                tmp_stsCP_vertices_layer_id = ak.Array(layer_id_list)
                tmp_stsCP_vertices_noh = ak.Array(noh_list)
                tmp_stsCP_vertices_eta = ak.Array(eta_list)
                tmp_stsCP_vertices_phi = ak.Array(phi_list)
                
                # Apply filter noh > 1 for the LCs
                skim_mask_noh = tmp_stsCP_vertices_noh > 1.0
                tmp_stsCP_vertices_x = tmp_stsCP_vertices_x[skim_mask_noh]
                tmp_stsCP_vertices_y = tmp_stsCP_vertices_y[skim_mask_noh]
                tmp_stsCP_vertices_z = tmp_stsCP_vertices_z[skim_mask_noh]
                tmp_stsCP_vertices_energy = tmp_stsCP_vertices_energy[skim_mask_noh]
                tmp_stsCP_vertices_time = tmp_stsCP_vertices_time[skim_mask_noh]
                tmp_stsCP_vertices_layer_id = tmp_stsCP_vertices_layer_id[skim_mask_noh]
                #tmp_stsCP_vertices_radius = tmp_stsCP_vertices_radius[skim_mask_energyPercent]
                tmp_stsCP_vertices_noh = tmp_stsCP_vertices_noh[skim_mask_noh]
                tmp_stsCP_vertices_eta = tmp_stsCP_vertices_eta[skim_mask_noh]
                tmp_stsCP_vertices_phi = tmp_stsCP_vertices_phi[skim_mask_noh]
                #tmp_stsCP_vertices_indexes_unmasked = tmp_stsCP_vertices_indexes
                tmp_stsCP_vertices_indexes = tmp_stsCP_vertices_indexes[skim_mask_noh]
                tmp_stsCP_vertices_multiplicity = tmp_stsCP_vertices_multiplicity[skim_mask_noh]

                # Further filtering: remove events with fewer than 2 vertices.
                skim_mask = [len(e) >= 2 for e in tmp_stsCP_vertices_x]
                tmp_stsCP_vertices_x = tmp_stsCP_vertices_x[skim_mask]
                tmp_stsCP_vertices_y = tmp_stsCP_vertices_y[skim_mask]
                tmp_stsCP_vertices_z = tmp_stsCP_vertices_z[skim_mask]
                tmp_stsCP_vertices_energy = tmp_stsCP_vertices_energy[skim_mask]
                tmp_stsCP_vertices_time = tmp_stsCP_vertices_time[skim_mask]
                tmp_stsCP_vertices_layer_id = tmp_stsCP_vertices_layer_id[skim_mask]
                tmp_stsCP_vertices_noh = tmp_stsCP_vertices_noh[skim_mask]
                tmp_stsCP_vertices_eta = tmp_stsCP_vertices_eta[skim_mask]
                tmp_stsCP_vertices_phi = tmp_stsCP_vertices_phi[skim_mask]
                tmp_stsCP_vertices_indexes = tmp_stsCP_vertices_indexes[skim_mask]
                tmp_stsCP_vertices_multiplicity = tmp_stsCP_vertices_multiplicity[skim_mask]

                if counter == 0:
                    self.stsCP_vertices_indexes_unfilt = tmp_stsCP_vertices_indexes
                    self.stsCP_vertices_multiplicity_unfilt = tmp_stsCP_vertices_multiplicity
                else:
                    self.stsCP_vertices_indexes_unfilt = ak.concatenate(
                        (self.stsCP_vertices_indexes_unfilt, tmp_stsCP_vertices_indexes))
                    self.stsCP_vertices_multiplicity_unfilt = ak.concatenate(
                        (self.stsCP_vertices_multiplicity_unfilt, tmp_stsCP_vertices_multiplicity))
                
                energyPercent = 1 / tmp_stsCP_vertices_multiplicity
                skim_mask_energyPercent = remove_duplicates(tmp_stsCP_vertices_indexes, energyPercent)
                tmp_stsCP_vertices_x = tmp_stsCP_vertices_x[skim_mask_energyPercent]
                tmp_stsCP_vertices_y = tmp_stsCP_vertices_y[skim_mask_energyPercent]
                tmp_stsCP_vertices_z = tmp_stsCP_vertices_z[skim_mask_energyPercent]
                tmp_stsCP_vertices_energy = tmp_stsCP_vertices_energy[skim_mask_energyPercent]
                tmp_stsCP_vertices_time = tmp_stsCP_vertices_time[skim_mask_energyPercent]
                tmp_stsCP_vertices_layer_id = tmp_stsCP_vertices_layer_id[skim_mask_energyPercent]
                tmp_stsCP_vertices_noh = tmp_stsCP_vertices_noh[skim_mask_energyPercent]
                tmp_stsCP_vertices_eta = tmp_stsCP_vertices_eta[skim_mask_energyPercent]
                tmp_stsCP_vertices_phi = tmp_stsCP_vertices_phi[skim_mask_energyPercent]
                tmp_stsCP_vertices_indexes_filt = tmp_stsCP_vertices_indexes[skim_mask_energyPercent]
                tmp_stsCP_vertices_multiplicity = tmp_stsCP_vertices_multiplicity[skim_mask_energyPercent]
                
                if counter == 0:
                    self.stsCP_vertices_x = tmp_stsCP_vertices_x
                    self.stsCP_vertices_y = tmp_stsCP_vertices_y
                    self.stsCP_vertices_z = tmp_stsCP_vertices_z
                    self.stsCP_vertices_energy = tmp_stsCP_vertices_energy
                    self.stsCP_vertices_time = tmp_stsCP_vertices_time
                    self.stsCP_vertices_layer_id = tmp_stsCP_vertices_layer_id
                    self.stsCP_vertices_noh = tmp_stsCP_vertices_noh
                    self.stsCP_vertices_eta = tmp_stsCP_vertices_eta
                    self.stsCP_vertices_phi = tmp_stsCP_vertices_phi
                    self.stsCP_vertices_indexes = tmp_stsCP_vertices_indexes
                    self.stsCP_barycenter_x = tmp_stsCP_barycenter_x
                    self.stsCP_barycenter_y = tmp_stsCP_barycenter_y
                    self.stsCP_barycenter_z = tmp_stsCP_barycenter_z
                    self.stsCP_vertices_multiplicity = tmp_stsCP_vertices_multiplicity
                    self.stsCP_vertices_indexes_filt = tmp_stsCP_vertices_indexes_filt
                else:
                    self.stsCP_vertices_x = ak.concatenate((self.stsCP_vertices_x, tmp_stsCP_vertices_x))
                    self.stsCP_vertices_y = ak.concatenate((self.stsCP_vertices_y, tmp_stsCP_vertices_y))
                    self.stsCP_vertices_z = ak.concatenate((self.stsCP_vertices_z, tmp_stsCP_vertices_z))
                    self.stsCP_vertices_energy = ak.concatenate((self.stsCP_vertices_energy, tmp_stsCP_vertices_energy))
                    self.stsCP_vertices_time = ak.concatenate((self.stsCP_vertices_time, tmp_stsCP_vertices_time))
                    self.stsCP_vertices_layer_id = ak.concatenate((self.stsCP_vertices_layer_id, tmp_stsCP_vertices_layer_id))
                    self.stsCP_vertices_noh = ak.concatenate((self.stsCP_vertices_noh, tmp_stsCP_vertices_noh))
                    self.stsCP_vertices_eta = ak.concatenate((self.stsCP_vertices_eta, tmp_stsCP_vertices_eta))
                    self.stsCP_vertices_phi = ak.concatenate((self.stsCP_vertices_phi, tmp_stsCP_vertices_phi))
                    self.stsCP_vertices_indexes = ak.concatenate((self.stsCP_vertices_indexes, tmp_stsCP_vertices_indexes))
                    self.stsCP_barycenter_x = ak.concatenate((self.stsCP_barycenter_x, tmp_stsCP_barycenter_x))
                    self.stsCP_barycenter_y = ak.concatenate((self.stsCP_barycenter_y, tmp_stsCP_barycenter_y))
                    self.stsCP_barycenter_z = ak.concatenate((self.stsCP_barycenter_z, tmp_stsCP_barycenter_z))
                    self.stsCP_vertices_multiplicity = ak.concatenate((self.stsCP_vertices_multiplicity, tmp_stsCP_vertices_multiplicity))
                    self.stsCP_vertices_indexes_filt = ak.concatenate((self.stsCP_vertices_indexes_filt, tmp_stsCP_vertices_indexes_filt))
                
                counter += 1
                if len(self.stsCP_vertices_x) > max_events:
                    print(f"Reached {max_events}!")
                    break
            if len(self.stsCP_vertices_x) > max_events:
                break

    def precompute_pairings(self):
        """
        Precompute the groups, fractions, and also the positive/negative edge pairs for each event.
        """
        n_events = len(self.stsCP_vertices_x)
        # (Clear previous lists if necessary)
        self.precomputed_groups = []
        self.precomputed_fractions = []
        self.precomputed_pos_edges = []
        self.precomputed_neg_edges = []
        for idx in range(n_events):
            unfilt_cp_array = self.stsCP_vertices_indexes_unfilt[idx]
            unfilt_mult_array = self.stsCP_vertices_multiplicity_unfilt[idx]
            cluster_contributors = {}
            for cp_id in range(len(unfilt_cp_array)):
                for local_lc_id, cluster_id in enumerate(unfilt_cp_array[cp_id]):
                    frac = 1.0 / unfilt_mult_array[cp_id][local_lc_id]
                    cluster_contributors.setdefault(cluster_id, []).append((cp_id, frac))
            final_cp_array = self.stsCP_vertices_indexes_filt[idx]
            flattened_cluster_ids = ak.flatten(final_cp_array)
            total_lc = len(flattened_cluster_ids)
            groups_np = np.zeros((total_lc, 3), dtype=np.int64)
            fractions_np = np.zeros((total_lc, 3), dtype=np.float32)
            for global_lc, cluster_id in enumerate(flattened_cluster_ids):
                contribs = cluster_contributors.get(cluster_id, [])
                contribs_sorted = sorted(contribs, key=lambda x: x[1], reverse=True)
                if len(contribs_sorted) == 0:
                    top3 = [(0, 0.0)] * 3
                else:
                    top3 = contribs_sorted[:3]
                    while len(top3) < 3:
                        top3.append(top3[-1])
                for i in range(3):
                    groups_np[global_lc, i] = top3[i][0]
                    fractions_np[global_lc, i] = top3[i][1]
            self.precomputed_groups.append(groups_np)
            self.precomputed_fractions.append(fractions_np)
            
            

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download it from {} and move all '
            '*.z files to {}'.format(self.url, self.raw_dir))

    def len(self):
        return len(self.stsCP_vertices_x)

    @property
    def raw_file_names(self):
        return sorted(glob.glob(osp.join(self.raw_dir, '*.root')))

    @property
    def processed_file_names(self):
        return []

    def get(self, idx):
        # 1) Flatten node features for the event
        lc_x = self.stsCP_vertices_x[idx]
        lc_y = self.stsCP_vertices_y[idx]
        lc_z = self.stsCP_vertices_z[idx]
        lc_e = self.stsCP_vertices_energy[idx]
        lc_layer_id = self.stsCP_vertices_layer_id[idx]
        lc_noh = self.stsCP_vertices_noh[idx]
        lc_eta = self.stsCP_vertices_eta[idx]
        lc_phi = self.stsCP_vertices_phi[idx]

        flat_lc_x = np.expand_dims(np.array(ak.flatten(lc_x)), axis=1)
        flat_lc_y = np.expand_dims(np.array(ak.flatten(lc_y)), axis=1)
        flat_lc_z = np.expand_dims(np.array(ak.flatten(lc_z)), axis=1)
        flat_lc_e = np.expand_dims(np.array(ak.flatten(lc_e)), axis=1)
        flat_lc_layer_id = np.expand_dims(np.array(ak.flatten(lc_layer_id)), axis=1)
        flat_lc_noh = np.expand_dims(np.array(ak.flatten(lc_noh)), axis=1)
        flat_lc_eta = np.expand_dims(np.array(ak.flatten(lc_eta)), axis=1)
        flat_lc_phi = np.expand_dims(np.array(ak.flatten(lc_phi)), axis=1)
        flat_lc_feats = np.concatenate(
            (flat_lc_x, flat_lc_y, flat_lc_z, flat_lc_e,
             flat_lc_layer_id, flat_lc_noh, flat_lc_eta, flat_lc_phi),
            axis=-1
        )
        total_lc = flat_lc_feats.shape[0]
        x = torch.from_numpy(flat_lc_feats).float()

        # 2) Retrieve precomputed groups, fractions, and edges
        groups_np = self.precomputed_groups[idx]
        fractions_np = self.precomputed_fractions[idx]


        data = Data(
            x=x,
            groups=torch.from_numpy(groups_np),
            fractions=torch.from_numpy(fractions_np)

        )
        return data
