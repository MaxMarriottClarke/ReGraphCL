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

class CCV3(Dataset):
    r'''
        input: layer clusters
        Loads the data stored in the 'simtrackstersCP (test)' and 'clusters (test)' tree of the ticlNtuplizer.

    '''

    url = '/dummy/'

    def __init__(self, root, transform=None, max_events=1e8):
        super(CCV3, self).__init__(root, transform)
        self.step_size = 100
        self.max_events = max_events
        self.fill_data(max_events)

    def fill_data(self,max_events):
        counter = 0

        print("### Loading data")
        for fi,path in enumerate(tqdm.tqdm(self.raw_paths)): ## only one root file so is there a need for a for loop?

            #if fi > 2:
                #break
            cluster_path = find_highest_branch(path, 'clusters')
            sim_path = find_highest_branch(path, 'simtrackstersCP')
            trackster_path = find_highest_branch(path, 'tracksters')

            crosstree =  uproot.open(path)[cluster_path]
            tracktree = uproot.open(path)[trackster_path]
            crosscounter = 0
            for array in uproot.iterate(f"{path}:{sim_path}", ["vertices_x", "vertices_y", "vertices_z", 
            "vertices_energy", "vertices_multiplicity", "vertices_time", "vertices_indexes", "barycenter_x", "barycenter_y", "barycenter_z", "regressed_energy"], step_size=self.step_size):

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
                tmp_stsCP_regressed_energy = array['regressed_energy']

                self.step_size = min(self.step_size,len(tmp_stsCP_vertices_x))


                # Code block for reading from 'clusters' tree
                tmp_all_vertices_layer_id = crosstree['cluster_layer_id'].array(entry_start=crosscounter*self.step_size,entry_stop=(crosscounter+1)*self.step_size)
                #tmp_all_vertices_radius = crosstree['cluster_radius'].array(entry_start=crosscounter*self.step_size,entry_stop=(crosscounter+1)*self.step_size)
                tmp_all_vertices_noh = crosstree['cluster_number_of_hits'].array(entry_start=crosscounter*self.step_size,entry_stop=(crosscounter+1)*self.step_size)
                tmp_all_vertices_eta = crosstree['position_eta'].array(entry_start=crosscounter*self.step_size,entry_stop=(crosscounter+1)*self.step_size)
                tmp_all_vertices_phi = crosstree['position_phi'].array(entry_start=crosscounter*self.step_size,entry_stop=(crosscounter+1)*self.step_size)
                tmp_all_vertices_energy = crosstree['energy'].array(entry_start=crosscounter*self.step_size,entry_stop=(crosscounter+1)*self.step_size)
                tmp_all_vertices_x = crosstree['position_x'].array(entry_start=crosscounter*self.step_size,entry_stop=(crosscounter+1)*self.step_size)
                tmp_all_vertices_y = crosstree['position_y'].array(entry_start=crosscounter*self.step_size,entry_stop=(crosscounter+1)*self.step_size)
                tmp_all_vertices_z = crosstree['position_z'].array(entry_start=crosscounter*self.step_size,entry_stop=(crosscounter+1)*self.step_size)                

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
                   # radius_list.append(radius_list_one_event)
                    noh_list.append(noh_list_one_event)
                    eta_list.append(eta_list_one_event)
                    phi_list.append(phi_list_one_event)
                tmp_stsCP_vertices_layer_id = ak.Array(layer_id_list)                
                #tmp_stsCP_vertices_radius = ak.Array(radius_list)                
                tmp_stsCP_vertices_noh = ak.Array(noh_list)                
                tmp_stsCP_vertices_eta = ak.Array(eta_list)                
                tmp_stsCP_vertices_phi = ak.Array(phi_list)

                """
                energyPercent = 1/tmp_stsCP_vertices_multiplicity
                skim_mask_energyPercent = energyPercent > 0.5
                tmp_stsCP_vertices_x = tmp_stsCP_vertices_x[skim_mask_energyPercent]
                tmp_stsCP_vertices_y = tmp_stsCP_vertices_y[skim_mask_energyPercent]
                tmp_stsCP_vertices_z = tmp_stsCP_vertices_z[skim_mask_energyPercent]
                tmp_stsCP_vertices_energy = tmp_stsCP_vertices_energy[skim_mask_energyPercent]
                tmp_stsCP_vertices_time = tmp_stsCP_vertices_time[skim_mask_energyPercent]
                tmp_stsCP_vertices_indexes = tmp_stsCP_vertices_indexes[skim_mask_energyPercent]
                tmp_stsCP_vertices_layer_id = tmp_stsCP_vertices_layer_id[skim_mask_energyPercent]
                tmp_stsCP_vertices_radius = tmp_stsCP_vertices_radius[skim_mask_energyPercent]
                tmp_stsCP_vertices_noh = tmp_stsCP_vertices_noh[skim_mask_energyPercent]
                tmp_stsCP_vertices_eta = tmp_stsCP_vertices_eta[skim_mask_energyPercent]
                tmp_stsCP_vertices_phi = tmp_stsCP_vertices_phi[skim_mask_energyPercent]
                """

                # Apply filter noh > 1 for the LCs
                skim_mask_noh = tmp_stsCP_vertices_noh > 1
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
                tmp_stsCP_vertices_multiplicity = tmp_stsCP_vertices_multiplicity[skim_mask_noh] #<---
                
                # Remove duplicates by only allowing the caloparticle that contributed the most energy to a LC to actually contribute.
                # This is so we can define a ground truth
                
                energyPercent = 1/tmp_stsCP_vertices_multiplicity
                skim_mask_energyPercent = remove_duplicates(tmp_stsCP_vertices_indexes,energyPercent)
                tmp_stsCP_vertices_x = tmp_stsCP_vertices_x[skim_mask_energyPercent]
                tmp_stsCP_vertices_y = tmp_stsCP_vertices_y[skim_mask_energyPercent]
                tmp_stsCP_vertices_z = tmp_stsCP_vertices_z[skim_mask_energyPercent]
                tmp_stsCP_vertices_energy = tmp_stsCP_vertices_energy[skim_mask_energyPercent]
                tmp_stsCP_vertices_time = tmp_stsCP_vertices_time[skim_mask_energyPercent]
                tmp_stsCP_vertices_indexes = tmp_stsCP_vertices_indexes[skim_mask_energyPercent]
                tmp_stsCP_vertices_layer_id = tmp_stsCP_vertices_layer_id[skim_mask_energyPercent]
                #tmp_stsCP_vertices_radius = tmp_stsCP_vertices_radius[skim_mask_energyPercent]
                tmp_stsCP_vertices_noh = tmp_stsCP_vertices_noh[skim_mask_energyPercent]
                tmp_stsCP_vertices_eta = tmp_stsCP_vertices_eta[skim_mask_energyPercent]
                tmp_stsCP_vertices_phi = tmp_stsCP_vertices_phi[skim_mask_energyPercent]
                #tmp_stsCP_vertices_indexes_unmasked = tmp_stsCP_vertices_indexes]

                

                
                # Code block for reading from 'tracksters' tree
                tmp_ts_vertices_indexes = tracktree['vertices_indexes'].array(entry_start=crosscounter*self.step_size,entry_stop=(crosscounter+1)*self.step_size)
                tmp_ts_vertices_energy = tracktree['vertices_energy'].array(entry_start=crosscounter*self.step_size,entry_stop=(crosscounter+1)*self.step_size)
                tmp_ts_vertices_multiplicity = tracktree['vertices_multiplicity'].array(entry_start=crosscounter*self.step_size,entry_stop=(crosscounter+1)*self.step_size)
                crosscounter += 1


                
                #SHOULD BE LEN(E) >= 2 for MULTI particles
                skim_mask = []
                for e in tmp_stsCP_vertices_x:
                    if 1 <= len(e):
                        skim_mask.append(True)
                    else:
                        skim_mask.append(False)
                tmp_stsCP_vertices_x = tmp_stsCP_vertices_x[skim_mask]
                tmp_stsCP_vertices_y = tmp_stsCP_vertices_y[skim_mask]
                tmp_stsCP_vertices_z = tmp_stsCP_vertices_z[skim_mask]
                tmp_stsCP_vertices_energy = tmp_stsCP_vertices_energy[skim_mask]
                tmp_stsCP_vertices_time = tmp_stsCP_vertices_time[skim_mask]
                tmp_stsCP_vertices_indexes = tmp_stsCP_vertices_indexes[skim_mask] ## additional filtering
                tmp_stsCP_vertices_layer_id = tmp_stsCP_vertices_layer_id[skim_mask]
                #tmp_stsCP_vertices_radius = tmp_stsCP_vertices_radius[skim_mask]
                tmp_stsCP_vertices_noh = tmp_stsCP_vertices_noh[skim_mask]
                tmp_stsCP_vertices_eta = tmp_stsCP_vertices_eta[skim_mask]
                tmp_stsCP_vertices_phi = tmp_stsCP_vertices_phi[skim_mask]
                tmp_stsCP_barycenter_x = tmp_stsCP_barycenter_x[skim_mask] #<--- additional filtering
                tmp_stsCP_barycenter_y = tmp_stsCP_barycenter_y[skim_mask] #<---- additional filtering
                tmp_stsCP_barycenter_z = tmp_stsCP_barycenter_z[skim_mask] #<----- additional filtering
                tmp_ts_vertices_indexes = tmp_ts_vertices_indexes[skim_mask]
                tmp_ts_vertices_energy = tmp_ts_vertices_energy[skim_mask]
                tmp_all_vertices_energy = tmp_all_vertices_energy[skim_mask]
                



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
                    self.stsCP_regressed_energy = tmp_stsCP_regressed_energy
                    self.ts_vertices_indexes = tmp_ts_vertices_indexes
                    self.ts_vertices_energy = tmp_ts_vertices_energy
                    self.ts_vertices_multiplicity = tmp_ts_vertices_multiplicity
                    self.all_vertices_energy = tmp_all_vertices_energy
                    self.all_vertices_layer_id = tmp_all_vertices_layer_id
                    self.all_vertices_x = tmp_all_vertices_x
                    self.all_vertices_y = tmp_all_vertices_y
                    self.all_vertices_z = tmp_all_vertices_z
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
                    self.stsCP_vertices_multiplicity = ak.concatenate((self.stsCP_vertices_multiplicity,tmp_stsCP_vertices_multiplicity))
                    self.stsCP_regressed_energy = ak.concatenate((self.stsCP_regressed_energy,tmp_stsCP_regressed_energy))
                    self.ts_vertices_indexes = ak.concatenate((self.ts_vertices_indexes,tmp_ts_vertices_indexes))
                    self.ts_vertices_energy = ak.concatenate((self.ts_vertices_energy,tmp_ts_vertices_energy))
                    self.ts_vertices_multiplicity = ak.concatenate((self.ts_vertices_multiplicity,tmp_ts_vertices_multiplicity))
                    self.all_vertices_energy = ak.concatenate((self.all_vertices_energy,tmp_all_vertices_energy))
                    self.all_vertices_layer_id = ak.concatenate((self.all_vertices_layer_id,tmp_all_vertices_layer_id))
                    self.all_vertices_x = ak.concatenate((self.all_vertices_x,tmp_all_vertices_x))
                    self.all_vertices_y = ak.concatenate((self.all_vertices_y,tmp_all_vertices_y))
                    self.all_vertices_z = ak.concatenate((self.all_vertices_z,tmp_all_vertices_z))

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
        # flatten so that we don't know what caloparticle the LC came from
        edge_index = torch.empty((2,0), dtype=torch.long)
        
        lc_id = self.stsCP_vertices_indexes[idx]
        flat_lc_id = np.expand_dims(np.array(ak.flatten(lc_id)),axis=1)
        lc_x = self.stsCP_vertices_x[idx]
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
        lc_noh = self.stsCP_vertices_noh[idx]
        flat_lc_noh = np.expand_dims(np.array(ak.flatten(lc_noh)),axis=1)  
        lc_eta = self.stsCP_vertices_eta[idx]
        flat_lc_eta = np.expand_dims(np.array(ak.flatten(lc_eta)),axis=1)  
        lc_phi = self.stsCP_vertices_phi[idx]
        flat_lc_phi = np.expand_dims(np.array(ak.flatten(lc_phi)),axis=1)
        

        flat_lc_feats = np.concatenate((flat_lc_x,flat_lc_y,flat_lc_z,flat_lc_e,\
                                        flat_lc_layer_id,flat_lc_noh,flat_lc_eta,flat_lc_phi),axis=-1)        
        
        result = np.concatenate([np.full(len(subarr), i) for i, subarr in enumerate(lc_x)])
        result_list = result.tolist() 
   
        x = torch.from_numpy(flat_lc_feats).float()
        x_counts = lc_x
        return Data(x=x, x_counts = x_counts, idx=flat_lc_id, assoc = result_list)
    
    
