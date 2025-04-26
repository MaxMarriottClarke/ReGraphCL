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

def find_highest_branch(path, base_name):
    with uproot.open(path) as f:
        # Find keys that exactly match the base_name (not containing other variations)
        branches = [k for k in f.keys() if k.startswith(base_name + ';')]
        
        # Sort and select the highest-numbered branch
        sorted_branches = sorted(branches, key=lambda x: int(x.split(';')[-1]))
        return sorted_branches[-1] if sorted_branches else None
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

    def fill_data(self,max_events):
        counter = 0
        arrLens0 = []
        arrLens1 = []

        print("### Loading data")
        for fi,path in enumerate(tqdm.tqdm(self.raw_paths)):


            if self.inp == 'train':
                cluster_path = find_highest_branch(path, 'clusters')
                sim_path = find_highest_branch(path, 'simtrackstersCP')
            elif self.inp == 'val':
                cluster_path = find_highest_branch(path, 'clusters')
                sim_path = find_highest_branch(path, 'simtrackstersCP')
            else:
                cluster_path = find_highest_branch(path, 'clusters')
                sim_path = find_highest_branch(path, 'simtrackstersCP')
            
            crosstree =  uproot.open(path)[cluster_path]
            crosscounter = 0
            for array in uproot.iterate(f"{path}:{sim_path}", ["vertices_x", "vertices_y", "vertices_z", 
            "vertices_energy", "vertices_multiplicity", "vertices_time", "vertices_indexes", "barycenter_x", "barycenter_y", "barycenter_z"], step_size=self.step_size):
            
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
                tmp_stsCP_vertices_energy = tmp_stsCP_vertices_energy * tmp_stsCP_vertices_multiplicity
                
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

                


                

                
                #SHOULD BE LEN(E) >= 2 for MULTI particles
                skim_mask = []
                for e in tmp_stsCP_vertices_x:
                    if 2 <= len(e) <= 5: #<------ only train on samples with > 1 particle
                        skim_mask.append(True)
                    else:
                        skim_mask.append(False)
                tmp_stsCP_vertices_x = tmp_stsCP_vertices_x[skim_mask]
                tmp_stsCP_vertices_y = tmp_stsCP_vertices_y[skim_mask]
                tmp_stsCP_vertices_z = tmp_stsCP_vertices_z[skim_mask]
                tmp_stsCP_vertices_energy = tmp_stsCP_vertices_energy[skim_mask]
                tmp_stsCP_vertices_time = tmp_stsCP_vertices_time[skim_mask]
                tmp_stsCP_vertices_layer_id = tmp_stsCP_vertices_layer_id[skim_mask]
                #tmp_stsCP_vertices_radius = tmp_stsCP_vertices_radius[skim_mask]
                tmp_stsCP_vertices_noh = tmp_stsCP_vertices_noh[skim_mask]
                tmp_stsCP_vertices_eta = tmp_stsCP_vertices_eta[skim_mask]
                tmp_stsCP_vertices_phi = tmp_stsCP_vertices_phi[skim_mask]
                tmp_stsCP_vertices_indexes = tmp_stsCP_vertices_indexes[skim_mask]
                tmp_stsCP_vertices_multiplicity = tmp_stsCP_vertices_multiplicity[skim_mask]

                if counter == 0:
                    self.stsCP_vertices_x = tmp_stsCP_vertices_x
                    self.stsCP_vertices_y = tmp_stsCP_vertices_y
                    self.stsCP_vertices_z = tmp_stsCP_vertices_z
                    self.stsCP_vertices_energy = tmp_stsCP_vertices_energy
                    self.stsCP_vertices_time = tmp_stsCP_vertices_time
                    self.stsCP_vertices_layer_id = tmp_stsCP_vertices_layer_id
                    #self.stsCP_vertices_radius = tmp_stsCP_vertices_radius
                    self.stsCP_vertices_noh = tmp_stsCP_vertices_noh
                    self.stsCP_vertices_eta = tmp_stsCP_vertices_eta
                    self.stsCP_vertices_phi = tmp_stsCP_vertices_phi
                    self.stsCP_vertices_indexes = tmp_stsCP_vertices_indexes
                    self.stsCP_barycenter_x = tmp_stsCP_barycenter_x
                    self.stsCP_barycenter_y = tmp_stsCP_barycenter_y
                    self.stsCP_barycenter_z = tmp_stsCP_barycenter_z
                    self.stsCP_vertices_multiplicity = tmp_stsCP_vertices_multiplicity
                else:
                    self.stsCP_vertices_x = ak.concatenate((self.stsCP_vertices_x,tmp_stsCP_vertices_x))
                    self.stsCP_vertices_y = ak.concatenate((self.stsCP_vertices_y,tmp_stsCP_vertices_y))
                    self.stsCP_vertices_z = ak.concatenate((self.stsCP_vertices_z,tmp_stsCP_vertices_z))
                    self.stsCP_vertices_energy = ak.concatenate((self.stsCP_vertices_energy,tmp_stsCP_vertices_energy))
                    self.stsCP_vertices_time = ak.concatenate((self.stsCP_vertices_time,tmp_stsCP_vertices_time))
                    self.stsCP_vertices_layer_id = ak.concatenate((self.stsCP_vertices_layer_id,tmp_stsCP_vertices_layer_id))
                    #self.stsCP_vertices_radius = ak.concatenate((self.stsCP_vertices_radius,tmp_stsCP_vertices_radius))
                    self.stsCP_vertices_noh = ak.concatenate((self.stsCP_vertices_noh,tmp_stsCP_vertices_noh))
                    self.stsCP_vertices_eta = ak.concatenate((self.stsCP_vertices_eta,tmp_stsCP_vertices_eta))
                    self.stsCP_vertices_phi = ak.concatenate((self.stsCP_vertices_phi,tmp_stsCP_vertices_phi))
                    self.stsCP_vertices_indexes = ak.concatenate((self.stsCP_vertices_indexes,tmp_stsCP_vertices_indexes))
                    self.stsCP_barycenter_x = ak.concatenate((self.stsCP_barycenter_x,tmp_stsCP_barycenter_x))
                    self.stsCP_barycenter_y = ak.concatenate((self.stsCP_barycenter_y,tmp_stsCP_barycenter_y))
                    self.stsCP_barycenter_z = ak.concatenate((self.stsCP_barycenter_z,tmp_stsCP_barycenter_z))
                    self.stsCP_vertices_multiplicity = ak.concatenate((self.stsCP_vertices_multiplicity, tmp_stsCP_vertices_multiplicity))

                #print(len(self.stsCP_vertices_x))
                counter += 1
                if len(self.stsCP_vertices_x) > max_events:
                    print(f"Reached {max_events}!")
                    break
            if len(self.stsCP_vertices_x) > max_events:
                break
     
            
            
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
        def reconstruct_array(grouped_indices):
            # Find the maximum index to determine the array length
            max_index = max(max(indices) for indices in grouped_indices.values())

            # Initialize an array with the correct size, filled with a placeholder (e.g., -1)
            reconstructed = [-1] * (max_index + 1)

            # Populate the array based on the dictionary
            for value, indices in grouped_indices.items():
                for idx in indices:
                    reconstructed[idx] = value

            return reconstructed
        edge_index = torch.empty((2,0), dtype=torch.long)
 
        lc_x = self.stsCP_vertices_x[idx]
        #print(ak.to_numpy(lc_x[0]).shape)
        #print(ak.to_numpy(lc_x[1]).shape)
        flat_lc_x = np.expand_dims(np.array(ak.flatten(lc_x)),axis=1)
        lc_y = self.stsCP_vertices_y[idx]
        flat_lc_y = np.expand_dims(np.array(ak.flatten(lc_y)),axis=1)
        lc_z = self.stsCP_vertices_z[idx]
        flat_lc_z = np.expand_dims(np.array(ak.flatten(lc_z)),axis=1)
        lc_e = self.stsCP_vertices_energy[idx]
        flat_lc_e = np.expand_dims(np.array(ak.flatten(lc_e)),axis=1)     
        lc_t = self.stsCP_vertices_time[idx]
        flat_lc_t = np.expand_dims(np.array(ak.flatten(lc_t)),axis=1)  
        lc_layer_id = self.stsCP_vertices_layer_id[idx]
        flat_lc_layer_id = np.expand_dims(np.array(ak.flatten(lc_layer_id)),axis=1)  
        #lc_radius = self.stsCP_vertices_radius[idx]
        #flat_lc_radius = np.expand_dims(np.array(ak.flatten(lc_radius)),axis=1)  
        lc_noh = self.stsCP_vertices_noh[idx]
        flat_lc_noh = np.expand_dims(np.array(ak.flatten(lc_noh)),axis=1)  
        lc_eta = self.stsCP_vertices_eta[idx]
        flat_lc_eta = np.expand_dims(np.array(ak.flatten(lc_eta)),axis=1)  
        lc_phi = self.stsCP_vertices_phi[idx]
        flat_lc_phi = np.expand_dims(np.array(ak.flatten(lc_phi)),axis=1)  
        
        lc_indexes = self.stsCP_vertices_indexes[idx]
        flat_lc_indexes = np.expand_dims(np.array(ak.flatten(lc_indexes)),axis=1)  
        lc_multiplicity = self.stsCP_vertices_multiplicity[idx]
        flat_lc_multiplicity = np.expand_dims(np.array(ak.flatten(lc_multiplicity)),axis=1) 

        
        mask = np.zeros_like(flat_lc_indexes, dtype=bool)

        # Loop over each unique index.
        for unique_idx in np.unique(flat_lc_indexes):
            # Find the positions where this index occurs.
            positions = np.where(flat_lc_indexes == unique_idx)[0]
            # If there is only one occurrence, keep it.
            if positions.size == 1:
                mask[positions[0]] = True
            else:
                # If repeated, select the position with the minimum multiplicity.
                pos_to_keep = positions[np.argmin(flat_lc_multiplicity[positions])]
                mask[pos_to_keep] = True

        # Now use this mask to filter all arrays:
        flat_lc_x = flat_lc_x[mask].reshape(-1, 1)
        flat_lc_y = flat_lc_y[mask].reshape(-1, 1)
        flat_lc_z = flat_lc_z[mask].reshape(-1, 1)
        flat_lc_e = flat_lc_e[mask].reshape(-1, 1)
        flat_lc_t = flat_lc_t[mask].reshape(-1, 1)
        flat_lc_layer_id = flat_lc_layer_id[mask].reshape(-1, 1)
        flat_lc_noh = flat_lc_noh[mask].reshape(-1, 1)
        flat_lc_eta = flat_lc_eta[mask].reshape(-1, 1)
        flat_lc_phi = flat_lc_phi[mask].reshape(-1, 1)
        flat_lc_indexes = flat_lc_indexes[mask].reshape(-1, 1)
        flat_lc_multiplicity = flat_lc_multiplicity[mask].reshape(-1, 1)
                                                               

        flat_lc_feats = np.concatenate((flat_lc_x,flat_lc_y,flat_lc_z,flat_lc_e,\
                                        flat_lc_layer_id,flat_lc_noh,flat_lc_eta,flat_lc_phi),axis=-1)    

        
        # Determine the maximum vertex index over all sub-arrays
        max_index = max(np.max(arr) for arr in lc_indexes)

        # Initialize the association array and a tracker for the best multiplicity.
        assoc_array = -1 * np.ones(max_index + 1, dtype=int)
        best_mult = np.full(max_index + 1, np.inf)

        # Loop over each sub-array and its multiplicity values.
        for sub_idx, (indexes, mult_values) in enumerate(zip(lc_indexes, lc_multiplicity)):
            for pos, vertex_index in enumerate(indexes):
                candidate_mult = mult_values[pos]
                # If this vertex is not yet assigned or if the current multiplicity is lower, update.
                if assoc_array[vertex_index] == -1 or candidate_mult < best_mult[vertex_index]:
                    assoc_array[vertex_index] = sub_idx
                    best_mult[vertex_index] = candidate_mult
                    
        assoc_groups = {}
        for vertex_index, sub_idx in enumerate(assoc_array):
            # Skip any vertex that was never assigned (i.e. remains -1)
            if sub_idx == -1:
                continue
            # If this sub-array number is not yet a key, create a new list.
            if sub_idx not in assoc_groups:
                assoc_groups[sub_idx] = []
            # Append the current vertex index to the list for that key.
            assoc_groups[sub_idx].append(vertex_index)
            
        assoc_array_final = reconstruct_array(assoc_groups)






        # Create the Data object
        x = torch.from_numpy(flat_lc_feats).float()

        return Data(
            x=x, assoc = assoc_array_final
        )