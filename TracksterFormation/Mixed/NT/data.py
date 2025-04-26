import numpy as np
import subprocess
import tqdm
from tqdm import tqdm
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

#singularity shell --bind /afs/cern.ch/user/p/pkakhand/public/CL/  /afs/cern.ch/user/p/pkakhand/geometricdl.sif

#singularity shell --bind /eos/project/c/contrast/public/solar/  /afs/cern.ch/user/p/pkakhand/geometricdl.sif
#source /cvmfs/sft.cern.ch/lcg/views/LCG_103cuda/x86_64-centos9-gcc11-opt/setup.sh

def find_highest_branch(path, base_name):
    with uproot.open(path) as f:
        # Find keys that exactly match the base_name (not containing other variations)
        branches = [k for k in f.keys() if k.startswith(base_name + ';')]
        
        # Sort and select the highest-numbered branch
        sorted_branches = sorted(branches, key=lambda x: int(x.split(';')[-1]))
        return sorted_branches[-1] if sorted_branches else None

def remove_duplicates(A,B):    
    all_masks = []
    for event_idx, event in enumerate(A):
        flat_A = np.array(ak.flatten(A[event_idx]))
        flat_B = np.array(ak.flatten(B[event_idx]))
        
        # Initialize a mask to keep track of which values to keep
        mask = np.zeros_like(flat_A, dtype=bool)

        # Iterate over the unique elements in A
        for elem in np.unique(flat_A):
            # Get the indices where the element occurs in A
            indices = np.where(flat_A == elem)[0]

            # If there's more than one occurrence, keep the one with the max B value
            if len(indices) > 1:
                max_index = indices[np.argmax(flat_B[indices])]
                mask[max_index] = True
            else:
                # If there's only one occurrence, keep it
                mask[indices[0]] = True

        unflattened_mask = ak.unflatten(mask, ak.num(A[event_idx]))
        all_masks.append(unflattened_mask)
        
    return ak.Array(all_masks)

class CCV1(Dataset):
    r'''
        input: layer clusters

    '''

    url = '/dummy/'

    def __init__(self, root, transform=None, max_events=1e8, inp = 'train'):
        super(CCV1, self).__init__(root, transform)
        self.step_size = 500
        self.inp = inp
        self.max_events = max_events
        self.fill_data(max_events)
    
        # These will hold the precomputed arrays (one per event)
        self.precomputed_groups = []
        self.precomputed_fractions = []
        self.precomputed_pos_edges = []
        self.precomputed_neg_edges = []
        
        self.precompute_pairings()

    def fill_data(self,max_events):
        counter = 0
        arrLens0 = []
        arrLens1 = []

        print("### Loading data")
        for fi,path in enumerate(tqdm(self.raw_paths)):


            if self.inp == 'train':
                cluster_path = find_highest_branch(path, 'clusters')
                sim_path = find_highest_branch(path, 'simtrackstersCP')
                track_path = find_highest_branch(path, 'tracksters')
            elif self.inp == 'val':
                cluster_path = find_highest_branch(path, 'clusters')
                sim_path = find_highest_branch(path, 'simtrackstersCP')
                track_path = find_highest_branch(path, 'tracksters')
            else:
                cluster_path = find_highest_branch(path, 'clusters')
                sim_path = find_highest_branch(path, 'simtrackstersCP')
                track_path = find_highest_branch(path, 'tracksters')
            
            crosstree =  uproot.open(path)[cluster_path]
            crosscounter = 0
            
            # Create an iterator for the tracksters branch to load its 'vertices_indexes'
            tracksters_iter = uproot.iterate(
                f"{path}:{track_path}",
                ["vertices_indexes"],
                step_size=self.step_size
            )
            for array in uproot.iterate(f"{path}:{sim_path}", ["vertices_x", "vertices_y", "vertices_z", 
            "vertices_energy", "vertices_multiplicity", "vertices_time", "vertices_indexes", "barycenter_x", "barycenter_y", "barycenter_z"], step_size=self.step_size):
                
                tmp_tracksters_data = next(tracksters_iter)
                tmp_tracksters_vertices_indexes = tmp_tracksters_data["vertices_indexes"]
            
                tmp_stsCP_vertices_x = array['vertices_x']
                tmp_stsCP_vertices_y = array['vertices_y']
                tmp_stsCP_vertices_z = array['vertices_z']
                tmp_stsCP_vertices_energy = array['vertices_energy']
                tmp_stsCP_vertices_time = array['vertices_time']
                tmp_stsCP_vertices_indexes = array['vertices_indexes']
                tmp_stsCP_barycenter_x = array['barycenter_x']
                tmp_stsCP_barycenter_y = array['barycenter_y']
                tmp_stsCP_barycenter_z = array['barycenter_z']


                tmp_stsCP_vertices_multiplicity = array['vertices_multiplicity']
                
                # weighted energies (A LC appears in its caloparticle assignment array as the energy it contributes not full energy)
                #tmp_stsCP_vertices_energy = tmp_stsCP_vertices_energy * tmp_stsCP_vertices_multiplicity
                
                self.step_size = min(self.step_size,len(tmp_stsCP_vertices_x))


                # Code block for reading from other tree
                tmp_all_vertices_layer_id = crosstree['cluster_layer_id'].array(entry_start=crosscounter*self.step_size,entry_stop=(crosscounter+1)*self.step_size)
                #tmp_all_vertices_radius = crosstree['cluster_radius'].array(entry_start=crosscounter*self.step_size,entry_stop=(crosscounter+1)*self.step_size)
                tmp_all_vertices_noh = crosstree['cluster_number_of_hits'].array(entry_start=crosscounter*self.step_size,entry_stop=(crosscounter+1)*self.step_size)
                tmp_all_vertices_eta = crosstree['position_eta'].array(entry_start=crosscounter*self.step_size,entry_stop=(crosscounter+1)*self.step_size)
                tmp_all_vertices_phi = crosstree['position_phi'].array(entry_start=crosscounter*self.step_size,entry_stop=(crosscounter+1)*self.step_size)
                crosscounter += 1

                layer_id_list = []
                radius_list = []
                noh_list = []
                eta_list = []
                phi_list = []
                for evt_row in range(len(tmp_all_vertices_noh)):
                    #print("Event no: %i"%evt_row)
                    #print("There are %i particles in this event"%len(tmp_stsCP_vertices_indexes[evt_row]))
                    layer_id_list_one_event = []
                    #radius_list_one_event = []
                    noh_list_one_event = []
                    eta_list_one_event = []
                    phi_list_one_event = []
                    for particle in range(len(tmp_stsCP_vertices_indexes[evt_row])):
                        #print("Particle no: %i"%particle)
                        #print("A")
                        #print(np.array(tmp_all_vertices_radius[evt_row]).shape)
                        #print("B")
                        #print(np.array(tmp_stsCP_vertices_indexes[evt_row][particle]).shape)
                        #print("C")
                        tmp_stsCP_vertices_layer_id_one_particle = tmp_all_vertices_layer_id[evt_row][tmp_stsCP_vertices_indexes[evt_row][particle]]
                        #tmp_stsCP_vertices_radius_one_particle = tmp_all_vertices_radius[evt_row][tmp_stsCP_vertices_indexes[evt_row][particle]]
                        tmp_stsCP_vertices_noh_one_particle = tmp_all_vertices_noh[evt_row][tmp_stsCP_vertices_indexes[evt_row][particle]]
                        tmp_stsCP_vertices_eta_one_particle = tmp_all_vertices_eta[evt_row][tmp_stsCP_vertices_indexes[evt_row][particle]]
                        tmp_stsCP_vertices_phi_one_particle = tmp_all_vertices_phi[evt_row][tmp_stsCP_vertices_indexes[evt_row][particle]]
                        #print(tmp_stsCP_vertices_radius_one_particle)
                        layer_id_list_one_event.append(tmp_stsCP_vertices_layer_id_one_particle)
                        #radius_list_one_event.append(tmp_stsCP_vertices_radius_one_particle)
                        noh_list_one_event.append(tmp_stsCP_vertices_noh_one_particle)
                        eta_list_one_event.append(tmp_stsCP_vertices_eta_one_particle)
                        phi_list_one_event.append(tmp_stsCP_vertices_phi_one_particle)
                    layer_id_list.append(layer_id_list_one_event)
                    #radius_list.append(radius_list_one_event)
                    noh_list.append(noh_list_one_event)
                    eta_list.append(eta_list_one_event)
                    phi_list.append(phi_list_one_event)
                tmp_stsCP_vertices_layer_id = ak.Array(layer_id_list)                
                #tmp_stsCP_vertices_radius = ak.Array(radius_list)                
                tmp_stsCP_vertices_noh = ak.Array(noh_list)                
                tmp_stsCP_vertices_eta = ak.Array(eta_list)                
                tmp_stsCP_vertices_phi = ak.Array(phi_list)                
                
                # NEW FILTERING: Remove simtracksters entries whose index is not in any tracksters sub-array.
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
                
                # Remove duplicates by only allowing the caloparticle that contributed the most energy to a LC to actually contribute.
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
            
            # Precompute positive and negative pairs for this event.
            pos_pairs = np.empty(total_lc, dtype=np.int64)
            neg_pairs = np.empty(total_lc, dtype=np.int64)
            for i in range(total_lc):
                pos_candidate = None
                first_group = groups_np[i][0]
                first_frac = fractions_np[i][0]
                if first_frac == 1:
                    # Case 1: Completely dominated by first group.
                    candidates = [j for j in range(total_lc) if j != i and groups_np[j][0] == first_group]
                    if candidates:
                        candidates_exact = [j for j in candidates if fractions_np[j][0] == 1]
                        if candidates_exact:
                            pos_candidate = random.choice(candidates_exact)
                        else:
                            pos_candidate = max(candidates, key=lambda j: fractions_np[j][0])
                elif first_frac > 0.7:
                    # Case 2: Strong but not complete contribution.
                    candidates = [j for j in range(total_lc) if j != i and 
                                  groups_np[j][0] == first_group and fractions_np[j][0] > 0.7]
                    if candidates:
                        candidates_second = [j for j in candidates if groups_np[j][1] == groups_np[i][1]]
                        if candidates_second:
                            pos_candidate = random.choice(candidates_second)
                        else:
                            pos_candidate = random.choice(candidates)
                else:
                    # Case 3: When first fraction is 0.7 or below.
                    candidates = [j for j in range(total_lc) if j != i and set(groups_np[j][:2]) == set(groups_np[i][:2])]
                    if candidates:
                        def diff(j):
                            return abs(fractions_np[j][0] - fractions_np[i][0]) + abs(fractions_np[j][1] - fractions_np[i][1])
                        pos_candidate = min(candidates, key=diff)
                    else:
                        candidates = [j for j in range(total_lc) if j != i and groups_np[j][0] == first_group]
                        if candidates:
                            pos_candidate = random.choice(candidates)
                if pos_candidate is None:
                    pos_candidate = i
                pos_pairs[i] = pos_candidate

                # Negative candidate selection.
                orig_set = set(groups_np[i])
                neg_candidate = None
                candidates = [j for j in range(total_lc) if j != i and 
                              (groups_np[j][0] not in orig_set or fractions_np[j][0] == 0)]
                if candidates:
                    neg_candidate = random.choice(candidates)
                else:
                    candidates = [j for j in range(total_lc) if j != i and groups_np[j][0] != groups_np[i][0]]
                    if candidates:
                        neg_candidate = random.choice(candidates)
                    else:
                        neg_candidate = (i + 1) % total_lc
                neg_pairs[i] = neg_candidate

            pos_edge = torch.from_numpy(np.column_stack((np.arange(total_lc), pos_pairs))).long()
            neg_edge = torch.from_numpy(np.column_stack((np.arange(total_lc), neg_pairs))).long()
            self.precomputed_pos_edges.append(pos_edge)
            self.precomputed_neg_edges.append(neg_edge)
 
            
            
    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download it from {} and move all '
            '*.z files to {}'.format(self.url, self.raw_dir))

    def len(self):
        return len(self.stsCP_vertices_x)

    @property
    def raw_file_names(self):
        raw_files = sorted(glob.glob(osp.join(self.raw_dir, '*.root')))
        
        #raw_files = [osp.join(self.raw_dir, 'step3_NTUPLE.root')]

        return raw_files

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
        pos_edge = self.precomputed_pos_edges[idx]
        neg_edge = self.precomputed_neg_edges[idx]

        data = Data(
            x=x,
            groups=torch.from_numpy(groups_np),
            fractions=torch.from_numpy(fractions_np),
            x_pe=pos_edge,
            x_ne=neg_edge
        )
        return data
