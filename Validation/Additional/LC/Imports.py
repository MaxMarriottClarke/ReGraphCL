import numpy as np
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from tqdm import tqdm

def Aggloremative(all_predictions,  threshold = 0.7, metric = 'cosine', linkage = 'average'):
    all_cluster_labels = []             

    for i, pred in enumerate(all_predictions):

        if len(pred) < 2:
            cluster_labels = np.ones(len(pred), dtype=int) 
        else:
            agglomerative = AgglomerativeClustering(
                n_clusters=None,                 
                distance_threshold=threshold,
                linkage=linkage,
                metric=metric,
                compute_distances=True
            )
            cluster_labels = agglomerative.fit_predict(pred) 

        all_cluster_labels.append(cluster_labels)

    all_cluster_labels = np.array(all_cluster_labels)
    return all_cluster_labels

def calculate_sim_to_reco_score(CaloParticle, energies_indices, ReconstructedTrackster, track_mult, calo_mult):
    """
    Calculate the sim-to-reco score for a given CaloParticle and ReconstructedTrackster.
    
    Parameters:
    - CaloParticle: array of Layer Clusters in the CaloParticle.
    - Multiplicity: array of Multiplicity for layer clusters in CP
    - energies_indices: array of energies associated with all LC (indexed by LC).
    - ReconstructedTrackster: array of LC in the reconstructed Trackster.
    
    Returns:
    - sim_to_reco_score: the calculated sim-to-reco score.
    """
    numerator = 0.0
    denominator = 0.0

    # Iterate over all DetIds in the CaloParticle
    for i, det_id in enumerate(CaloParticle):
        energy_k = energies_indices[det_id]  # Energy for the current DetId in CaloParticle
        
        # Fraction of energy in the Trackster (fr_k^TST)
        if det_id in ReconstructedTrackster:
            index = np.where(ReconstructedTrackster == det_id)[0][0]
            fr_tst_k = 1 / track_mult[index]
            
        else:
            fr_tst_k = 0 # binary function also for CaloParticle

        # Fraction of energy in the CaloParticle (fr_k^SC)
        fr_sc_k = 1 / calo_mult[i]

        # Update numerator using the min function
        numerator += min(
            (fr_sc_k - fr_tst_k) ** 2,  # First term in the min function
            fr_sc_k ** 2                # Second term in the min function
        ) * (energy_k ** 2)

        # Update denominator
        denominator += (fr_sc_k ** 2) * (energy_k ** 2)


    # Calculate score
    sim_to_reco_score = numerator / denominator if denominator != 0 else 1.0
    return sim_to_reco_score

def calculate_reco_to_sim_score_and_sharedE(ReconstructedTrackster, energies_indices, CaloParticle, track_mult, calo_mult):
    """
    Calculate the reco-to-sim score for a given ReconstructedTrackster and CaloParticle.

    Parameters:
    - ReconstructedTrackster: array of DetIds in the ReconstructedTrackster.
    - energies_indices: array of energies associated with all DetIds (indexed by DetId).
    - CaloParticle: array of DetIds in the CaloParticle.

    Returns:
    - reco_to_sim_score: the calculated reco-to-sim score.
    """
    numerator = 0.0
    denominator = 0.0
    sharedEnergy = 0.0

    # Iterate over all DetIds in the ReconstructedTrackster
    for i, det_id in enumerate(ReconstructedTrackster):
        energy_k = energies_indices[det_id]  # Energy for the current DetId in the Trackster
        
        # Fraction of energy in the Trackster (fr_k^TST)
        fr_tst_k = 1 / track_mult[i]

        #Fraction of energy in the caloparticle
        if det_id in CaloParticle:
            index = np.where(CaloParticle == det_id)[0][0]
            fr_sc_k = 1 / calo_mult[index]
            
        else:
            fr_sc_k = 0 # binary function also for CaloParticle
            
        # Update numerator using the min function
        numerator += ((fr_tst_k - fr_sc_k) ** 2) * (energy_k ** 2)

        # Update denominator
        denominator += (fr_tst_k ** 2) * (energy_k ** 2)
        
        #shared_energy calculation
        recosharedEnergy = energy_k * fr_tst_k
        simsharedEnergy = energy_k * fr_sc_k
        sharedEnergy += min(simsharedEnergy,recosharedEnergy)
        
        

    # Calculate score
    reco_to_sim_score = numerator / denominator if denominator != 0 else 1.0
    return reco_to_sim_score, sharedEnergy


def calculate_all_event_scores(GT_ind, GT_mult, GT_regE, energies, recon_ind, recon_mult, num_events = 100):
    """
    Calculate sim-to-reco and reco-to-sim scores for all CaloParticle and ReconstructedTrackster combinations across all events.

    Parameters:
    - GT_ind: List of CaloParticle indices for all events.
    - energies: List of energy arrays for all events.
    - recon_ind: List of ReconstructedTrackster indices for all events.
    - LC_x, LC_y, LC_z, LC_eta: Lists of x, y, z positions and eta values for all DetIds across events.

    Returns:
    - DataFrame containing scores and additional features for each CaloParticle-Trackster combination across all events.
    """
    # Initialize an empty list to store results
    all_results = []

    # Loop over all events with a progress bar
    for event_index in range(num_events):
        caloparticles = GT_ind[event_index]  # Indices for all CaloParticles in the event
        tracksters = recon_ind[event_index]  # Indices for all ReconstructedTracksters in the event
        event_energies = energies[event_index]  # Energies for this event
        event_GT_mult = GT_mult[event_index]
        event_recon_mult = recon_mult[event_index]
        event_GT_regE = GT_regE[event_index]

        
        # Loop over all CaloParticles
        for calo_idx, caloparticle in enumerate(caloparticles):
            calo_mult = event_GT_mult[calo_idx]
            cp_raw_energy_lc = event_energies[caloparticle] / calo_mult
            cp_raw_energy = np.sum(cp_raw_energy_lc)
            cp_regressed_energy = event_GT_regE[calo_idx]
            
            for trackster_idx, trackster in enumerate(tracksters):
                track_mult = event_recon_mult[trackster_idx]
                
                # Calculate sim-to-reco score
                sim_to_reco_score = calculate_sim_to_reco_score(caloparticle, event_energies, trackster, track_mult, calo_mult)
                
                # Calculate reco-to-sim score
                reco_to_sim_score, shared_energy = calculate_reco_to_sim_score_and_sharedE(trackster, event_energies, caloparticle, track_mult, calo_mult)
                # Calculate trackster energy
                trackster_energy_lc = event_energies[trackster] / track_mult
                trackster_energy = np.sum(trackster_energy_lc)

                # Append results
                all_results.append({
                    "event_index": event_index,
                    "cp_id": calo_idx,
                    "trackster_id": trackster_idx,
                    "reco_to_sim_score": reco_to_sim_score,
                    "cp_raw_energy": cp_raw_energy,
                    "cp_regressed_energy": cp_regressed_energy,
                    "trackster_energy": trackster_energy,
                    "shared_energy": shared_energy
                })

    # Convert results to a DataFrame
    df = pd.DataFrame(all_results)
    return df