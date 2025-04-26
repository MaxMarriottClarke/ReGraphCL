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

#singularity shell --bind /afs/cern.ch/user/p/pkakhand/public/CL/  /afs/cern.ch/user/p/pkakhand/geometricdl.sif

#singularity shell --bind /eos/project/c/contrast/public/solar/  /afs/cern.ch/user/p/pkakhand/geometricdl.sif
#source /cvmfs/sft.cern.ch/lcg/views/LCG_103cuda/x86_64-centos9-gcc11-opt/setup.sh



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

            if fi > 2:
                break
            if self.inp == 'train':
                cluster_path = 'clusters;4'
                sim_path = 'simtrackstersCP;3'
            elif self.inp == 'val':
                cluster_path = 'clusters;2'
                sim_path = 'simtrackstersCP;2'
            else:
                cluster_path = 'clusters;2'
                sim_path = 'simtrackstersCP;1'
            
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

                #Filtering all the points where energy percent < 50%
                tmp_stsCP_vertices_multiplicity = array['vertices_multiplicity']
                
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

                energyPercent = 1/tmp_stsCP_vertices_multiplicity
                skim_mask_energyPercent = energyPercent > 0.5
                tmp_stsCP_vertices_x = tmp_stsCP_vertices_x[skim_mask_energyPercent]
                tmp_stsCP_vertices_y = tmp_stsCP_vertices_y[skim_mask_energyPercent]
                tmp_stsCP_vertices_z = tmp_stsCP_vertices_z[skim_mask_energyPercent]
                tmp_stsCP_vertices_energy = tmp_stsCP_vertices_energy[skim_mask_energyPercent]
                tmp_stsCP_vertices_time = tmp_stsCP_vertices_time[skim_mask_energyPercent]
                tmp_stsCP_vertices_layer_id = tmp_stsCP_vertices_layer_id[skim_mask_energyPercent]
                #tmp_stsCP_vertices_radius = tmp_stsCP_vertices_radius[skim_mask_energyPercent]
                tmp_stsCP_vertices_noh = tmp_stsCP_vertices_noh[skim_mask_energyPercent]
                tmp_stsCP_vertices_eta = tmp_stsCP_vertices_eta[skim_mask_energyPercent]
                tmp_stsCP_vertices_phi = tmp_stsCP_vertices_phi[skim_mask_energyPercent]
                

                
                #SHOULD BE LEN(E) >= 2 for MULTI particles
                skim_mask = []
                for e in tmp_stsCP_vertices_x:
                    if len(e) == 2:
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

        #lc_rad = self.stsCP_vertices_radius[idx]
        #flat_lc_rad = np.expand_dims(np.array(ak.flatten(lc_rad)),axis=1)  

        #lc_id = self.stsCP_vertices_id[idx]
        #flat_lc_id = np.expand_dims(np.array(ak.flatten(lc_id)),axis=1)                                                                                          


        #flat_lc_feats = np.concatenate((flat_lc_x,flat_lc_y,flat_lc_z,flat_lc_e),axis=-1)
        flat_lc_feats = np.concatenate((flat_lc_x,flat_lc_y,flat_lc_z,flat_lc_e,\
                                        flat_lc_layer_id,flat_lc_noh,flat_lc_eta,flat_lc_phi),axis=-1)        

        # Loop over number of calo particles to build positive and negative edges
        pos_edges = []
        neg_edges = []
        offset = 0
        idlc = 0

        for cp in range(len(lc_x)):
            n_lc_cp = len(lc_x[cp])

            
            # First the positive edge
            for lc in range(n_lc_cp):
                random_num_pos = int(random.uniform(offset, offset+n_lc_cp))
                random_num_neg = int(random.uniform(0, flat_lc_x.shape[0]))
                while random_num_neg >= offset and random_num_neg < offset+n_lc_cp:
                    random_num_neg = int(random.uniform(0, flat_lc_x.shape[0]))

                #while (idlc == random_num_pos):
                #    random_num_pos = int(random.uniform(offset, offset+n_lc_cp))
                pos_edges.append([idlc,random_num_pos])
                neg_edges.append([idlc,random_num_neg])
                idlc += 1
            offset += n_lc_cp

        x = torch.from_numpy(flat_lc_feats).float()
        x_lc = x
        x_pos_edge = torch.from_numpy(np.array(pos_edges))
        x_neg_edge = torch.from_numpy(np.array(neg_edges))       

        x_counts = lc_x
        y = torch.from_numpy(np.array([0 for u in range(len(flat_lc_feats))])).float()
        return Data(x=x, edge_index=edge_index, y=y,
                        x_lc=x_lc, x_pe=x_pos_edge, x_ne=x_neg_edge, x_counts = x_counts)






def contrastive_loss( start_all, end_all, temperature=0.05):
    xdevice = start_all.get_device()
    z_start = F.normalize(start_all, dim=1 )
    z_end = F.normalize(end_all, dim=1 )
    positives = torch.exp(F.cosine_similarity(z_start[:int(len(z_start)/2)],z_end[:int(len(z_end)/2)],dim=1))
    negatives = torch.exp(F.cosine_similarity(z_start[int(len(z_start)/2):],z_end[int(len(z_end)/2):],dim=1))
    nominator = positives / temperature
    denominator = negatives
    #print(denominator)
    loss = torch.exp(-nominator.sum() / denominator.sum())
    

    #print("Loss:",loss.item())

    return loss

import torch.nn.functional as F
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn import LayerNorm
from torch_geometric.nn.pool import avg_pool_x
from torch.nn import Sequential, Linear
from torch_geometric.nn.norm import BatchNorm
from torch_scatter import scatter


class Net(nn.Module):
    def __init__(self, hidden_dim=64, k_value=24, contrastive_dim=8):
        super(Net, self).__init__()
        
        # Initialize with hyperparameters
        self.hidden_dim = hidden_dim
        self.contrastive_dim = contrastive_dim
        
        self.lc_encode = nn.Sequential(
            nn.Linear(8, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.conv1 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
            ),
            k=k_value
        )
        self.norm1 = LayerNorm(hidden_dim)

        self.conv2 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
            ),
            k=k_value
        )
        self.norm2 = LayerNorm(hidden_dim)

        self.conv3 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
            ),
            k=k_value
        )
        self.norm3 = LayerNorm(hidden_dim)


        self.dropout = nn.Dropout(0.1)

        
        self.output = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(32, self.contrastive_dim),
        )
        
    def forward(self, x_lc, batch_lc):
        x_lc_enc = self.lc_encode(x_lc)
        
        # Layer 1
        residual = x_lc_enc
        feats = self.conv1((x_lc_enc, x_lc_enc), (batch_lc, batch_lc))
        feats = self.norm1(feats + residual)
        feats = self.dropout(feats)
        x = feats

        # Layer 2
        residual = x
        feats = self.conv2((x, x), (batch_lc, batch_lc))
        feats = self.norm2(feats + residual)
        feats = self.dropout(feats)
        x = feats

        # Layer 3
        residual = x
        feats = self.conv3((x, x), (batch_lc, batch_lc))
        feats = self.norm3(feats + residual)
        feats = self.dropout(feats)
        x = feats


        out = self.output(x)

        return out, batch_lc
        


def train(loader, model, optimizer, device, temperature):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x_lc, data.x_lc_batch)

        values, counts = np.unique(data.x_lc_batch.detach().cpu().numpy(), return_counts=True)
        losses = []
        for e in range(len(counts)):
            dummy_tensor = torch.randn(200,200, device='cpu')
            dummy_result = torch.matmul(dummy_tensor, dummy_tensor)

            lower_edge = 0 if e == 0 else np.sum(counts[:e])
            upper_edge = lower_edge + counts[e]

            start_pos = out[0][lower_edge:upper_edge][data.x_pe[lower_edge:upper_edge, 0]]
            end_pos = out[0][lower_edge:upper_edge][data.x_pe[lower_edge:upper_edge, 1]]
            start_neg = out[0][lower_edge:upper_edge][data.x_ne[lower_edge:upper_edge, 0]]
            end_neg = out[0][lower_edge:upper_edge][data.x_ne[lower_edge:upper_edge, 1]]
            start_all = torch.cat((start_pos, start_neg), 0)
            end_all = torch.cat((end_pos, end_neg), 0)

            if len(losses) == 0:
                losses.append(contrastive_loss(start_all, end_all, temperature))
            else:
                losses.append(losses[-1] + contrastive_loss(start_all, end_all, temperature))

        loss = losses[-1]
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(loader, model, device, temperature):
    model.eval()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        out = model(data.x_lc, data.x_lc_batch)
        
        values, counts = np.unique(data.x_lc_batch.detach().cpu().numpy(), return_counts=True)
        losses = []
        for e in range(len(counts)):
            dummy_tensor = torch.randn(200,200, device='cpu')
            dummy_result = torch.matmul(dummy_tensor, dummy_tensor)

            lower_edge = 0 if e == 0 else np.sum(counts[:e])
            upper_edge = lower_edge + counts[e]

            start_pos = out[0][lower_edge:upper_edge][data.x_pe[lower_edge:upper_edge, 0]]
            end_pos = out[0][lower_edge:upper_edge][data.x_pe[lower_edge:upper_edge, 1]]
            start_neg = out[0][lower_edge:upper_edge][data.x_ne[lower_edge:upper_edge, 0]]
            end_neg = out[0][lower_edge:upper_edge][data.x_ne[lower_edge:upper_edge, 1]]
            start_all = torch.cat((start_pos, start_neg), 0)
            end_all = torch.cat((end_pos, end_neg), 0)

            if len(losses) == 0:
                losses.append(contrastive_loss(start_all, end_all, temperature))
            else:
                losses.append(losses[-1] + contrastive_loss(start_all, end_all, temperature))

        loss = losses[-1]
        total_loss += loss.item()

    return total_loss / len(loader.dataset)