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

def remove_duplicates(A, B, threshold=0.9):    
    """
    A: awkward array of cluster IDs (or similar)
    B: awkward array of the same shape indicating 'fractions'
    threshold: fraction cutoff above which to keep the cluster
    """
    all_masks = []
    for event_idx in range(len(A)):
        flat_A = np.array(ak.flatten(A[event_idx]))
        flat_B = np.array(ak.flatten(B[event_idx]))

        # Initialize a mask to keep track of which values to keep
        mask = np.zeros_like(flat_A, dtype=bool)

        # For each unique cluster ID, check if its maximum fraction > threshold
        for elem in np.unique(flat_A):
            indices = np.where(flat_A == elem)[0]
            max_b = np.max(flat_B[indices])
            
            # Only keep one occurrence if max_b > threshold; otherwise, keep none
            if max_b > threshold:
                max_index = indices[np.argmax(flat_B[indices])]
                mask[max_index] = True
        
        # Reshape mask to the original (unflattened) shape
        unflattened_mask = ak.unflatten(mask, ak.num(A[event_idx]))
        all_masks.append(unflattened_mask)
        
    return ak.Array(all_masks)


class CCV3(Dataset):
    r'''
        input: layer clusters

    '''

    url = '/dummy/'

    def __init__(self, root, transform=None, max_events=1e8, inp = 'train'):
        super(CCV3, self).__init__(root, transform)
        self.step_size = 500
        self.inp = inp
        self.max_events = max_events
        self.fill_data(max_events)
    

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
                
                # NEW FILTERING: For each event, remove simtracksters entries whose index is not found
                # in any sub-array of the tracksters branch.
                mask_list = []
                for sim_evt, track_evt in zip(tmp_stsCP_vertices_indexes, tmp_tracksters_vertices_indexes):
                    # Flatten all tracksters indices into a single set
                    track_flat = ak.flatten(track_evt)
                    track_set = set(ak.to_list(track_flat))
                    # Convert the sim event to a nested Python list
                    sim_evt_list = ak.to_list(sim_evt)
                    # Build a nested mask preserving the structure:
                    # For each sub-array in the sim event, check each element for membership in track_set.
                    mask_evt = [[elem in track_set for elem in subarr] for subarr in sim_evt_list]
                    mask_list.append(mask_evt)
                mask_track = ak.Array(mask_list)

                # Apply the new mask to all simtracksters arrays:
                tmp_stsCP_vertices_x = tmp_stsCP_vertices_x[mask_track]
                tmp_stsCP_vertices_y = tmp_stsCP_vertices_y[mask_track]
                tmp_stsCP_vertices_z = tmp_stsCP_vertices_z[mask_track]
                tmp_stsCP_vertices_energy = tmp_stsCP_vertices_energy[mask_track]
                tmp_stsCP_vertices_time = tmp_stsCP_vertices_time[mask_track]
                tmp_stsCP_vertices_layer_id = tmp_stsCP_vertices_layer_id[mask_track]
                tmp_stsCP_vertices_noh = tmp_stsCP_vertices_noh[mask_track]
                tmp_stsCP_vertices_eta = tmp_stsCP_vertices_eta[mask_track]
                tmp_stsCP_vertices_phi = tmp_stsCP_vertices_phi[mask_track]
                tmp_stsCP_vertices_indexes = tmp_stsCP_vertices_indexes[mask_track]
                tmp_stsCP_vertices_multiplicity = tmp_stsCP_vertices_multiplicity[mask_track]
                
                # Remove duplicates by only allowing the caloparticle that contributed the most energy to a LC to actually contribute.
                # Further filtering: remove events with fewer than 2 vertices.
                skim_mask = [len(e) >= 1 for e in tmp_stsCP_vertices_x]
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
        
        result = np.concatenate([np.full(len(subarr), i) for i, subarr in enumerate(lc_x)])
        result_list = result.tolist() 

        data = Data(
            x=x,
            assoc = result_list
        )
        return data
