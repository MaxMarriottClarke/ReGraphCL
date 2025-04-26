#0: imports

import uproot 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from dataAnalyse import CCV2
from torch_geometric.data import DataLoader 
from model import Net
from torch_geometric.nn import knn_graph
import awkward as ak

import numpy as np
import time
from Imports import Aggloremative, calculate_reco_to_sim_score, calculate_sim_to_reco_score, calculate_all_event_scores

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

#1: Load Data + Model
#1.1: Load Data
testpath = "/vols/cms/mm1221/Data/5pi/test/"  
data_test = CCV2(testpath, max_events=12000, inp = 'test')

test_loader = DataLoader(data_test, batch_size=1, shuffle=False, follow_batch=['x'])

data_path = '/vols/cms/mm1221/Data/5pi/test/raw/test.root'
data_file = uproot.open(data_path)

Track_ind = data_file['tracksters;1']['vertices_indexes'].array()

GT_ind = data_file['simtrackstersCP;1']['vertices_indexes'].array()
GT_mult = data_file['simtrackstersCP;1']['vertices_multiplicity'].array()

GT_bc = data_file['simtrackstersCP;1']['barycenter_x'].array()
energies = data_file['clusters;2']['energy'].array()
LC_x = data_file['clusters;2']['position_x'].array()
LC_y = data_file['clusters;2']['position_y'].array()
LC_z = data_file['clusters;2']['position_z'].array()
LC_eta = data_file['clusters;2']['position_eta'].array()
MT_ind = data_file['trackstersMerged;1']['vertices_indexes'].array()

#1.3 Filter so the same indexes are included for fair comparison

skim_mask = []
for e in GT_bc:
    if 1 <= len(e) <= 5:
        skim_mask.append(True)
    else:
        skim_mask.append(False)

Track_ind = Track_ind[skim_mask]
GT_ind = GT_ind[skim_mask]
GT_mult = GT_mult[skim_mask]

energies = energies[skim_mask]
LC_x = LC_x[skim_mask]
LC_y = LC_y[skim_mask]
LC_z = LC_z[skim_mask]
LC_eta = LC_eta[skim_mask]
MT_ind = MT_ind[skim_mask]



model = Net(128, 4, 0.1, 8)
checkpoint= torch.load('/vols/cms/mm1221/hgcal/pion5/Track/StaticEdge/results/Run2/results_lr0.0001_bs64_hd128_nl4_do0.1_k8_cd8/best_model.pt',  map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)  
model.eval()  

"""
# Create new lists to store the filtered results
filtered_GT_ind = []
filtered_GT_mult = []
filtered_MT_ind = []

# Iterate over each event
for event_idx, track_indices in enumerate(Track_ind):
    # Flatten the current event's track indices and convert to a set
    track_flat = set(ak.flatten(track_indices).tolist())  # Ensure it contains only integers
    
    # Filter GT_ind and GT_mult for the current event, preserving structure
    event_GT_ind = GT_ind[event_idx]
    event_GT_mult = GT_mult[event_idx]
    filtered_event_GT_ind = []
    filtered_event_GT_mult = []
    for sublist_ind, sublist_mult in zip(event_GT_ind, event_GT_mult):
        filtered_sublist_ind = [idx for idx in sublist_ind if idx in track_flat]
        filtered_sublist_mult = [mult for idx, mult in zip(sublist_ind, sublist_mult) if idx in track_flat]
        filtered_event_GT_ind.append(filtered_sublist_ind)
        filtered_event_GT_mult.append(filtered_sublist_mult)

    # Filter MT_ind for the current event, preserving structure
    event_MT_ind = MT_ind[event_idx]
    filtered_event_MT_ind = []
    for sublist in event_MT_ind:
        filtered_sublist = [idx for idx in sublist if idx in track_flat]
        filtered_event_MT_ind.append(filtered_sublist)

    # Append filtered results
    filtered_GT_ind.append(filtered_event_GT_ind)
    filtered_GT_mult.append(filtered_event_GT_mult)
    filtered_MT_ind.append(filtered_event_MT_ind)

# Convert the filtered results back to awkward arrays
GT_ind_filt = ak.Array(filtered_GT_ind)
GT_mult_filt = ak.Array(filtered_GT_mult)
MT_ind_filt = ak.Array(filtered_MT_ind)
"""

#3: Make Predictions + Cluster -> Calculate the inference time
#3.1: Make Predictions

all_predictions = []  
total_times = []
start_time = time.time()

for i, data in enumerate(data_test):
    edge_index = knn_graph(data.x, k=8)  
    predictions = model(data.x, edge_index, 1)
    all_predictions.append(predictions[0].detach().cpu().numpy())  

all_predictions = np.array(all_predictions)

#3.2: Cluster using threshold found in Script A

all_cluster_labels = Aggloremative(all_predictions, threshold = 0.2)

end_time = time.time()

#3.3: Calculate average inference time

time_diff = end_time - start_time
inference_time = time_diff/len(all_cluster_labels)
print("average inference time:", inference_time)

#4: Calculate Scores and create DF for our model and TICL

#4.1: Turn the cluster labels into our reconstructed tracksters

recon_ind = []

for event_idx, labels in enumerate(all_cluster_labels):
    event_clusters = {} 
    
    for cluster_idx, cluster_label in enumerate(labels):
        if cluster_label not in event_clusters:
            event_clusters[cluster_label] = []
        event_clusters[cluster_label].extend(Track_ind[event_idx][cluster_idx])
    
    recon_ind.append([event_clusters[label] for label in sorted(event_clusters.keys())])

#4.2 Make DF from our model and CERN

df_CL = calculate_all_event_scores(GT_ind, energies, recon_ind, LC_x, LC_y, LC_z, LC_eta, GT_mult, num_events = 50)
df_TICL = calculate_all_event_scores(GT_ind, energies, MT_ind, LC_x, LC_y, LC_z, LC_eta, GT_mult, num_events = 50)

print(df_CL)

#5: Print metrics

def calculate_metrics(df, model_name):
    # ----- Efficiency Calculation -----
    cp_valid = df.dropna(subset=['cp_id']).copy()
    cp_grouped = cp_valid.groupby(['event_index', 'cp_id'])
    cp_associated = cp_grouped['sim_to_reco_score'].min() < 0.2
    num_associated_cp = cp_associated.sum()
    total_cp = cp_associated.count()
    efficiency = num_associated_cp / total_cp if total_cp > 0 else 0

    # ----- Purity Calculation -----
    tst_valid = df.dropna(subset=['trackster_id']).copy()
    tst_grouped = tst_valid.groupby(['event_index', 'trackster_id'])
    tst_associated = tst_grouped['reco_to_sim_score'].min() < 0.2
    num_associated_tst = tst_associated.sum()
    total_tst = tst_associated.count()
    purity = num_associated_tst / total_tst if total_tst > 0 else 0

    # ----- Average Energy Ratio Calculation -----
    low_score_mask = df['sim_to_reco_score'] < 0.2
    low_score_events = df[low_score_mask]
    if not low_score_events.empty:
        avg_energy_ratio = (low_score_events['trackster_energy'] / low_score_events['cp_energy']).mean()
    else:
        avg_energy_ratio = 0

    # Print results for the model
    print(f"\nModel: {model_name}")
    print(f"Efficiency: {efficiency:.4f} ({num_associated_cp} associated CPs out of {total_cp} total CPs)")
    print(f"Purity: {purity:.4f} ({num_associated_tst} associated Tracksters out of {total_tst} total Tracksters)")
    print(f"Num tracksters ratio: {total_tst / total_cp if total_cp > 0 else 0:.4f}")
    print(f"Average Energy Ratio: {avg_energy_ratio:.4f}")

    return {
        'efficiency': efficiency,
        'purity': purity,
        'avg_energy_ratio': avg_energy_ratio,
    }

# Example usage
your_model_metrics = calculate_metrics(df_CL, "Your Model")
cern_model_metrics = calculate_metrics(df_TICL, "CERN Model")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Preprocessing
# -----------------------------
# Convert relevant columns to numeric.
for df in [df_CL, df_TICL]:
    df['cp_id'] = pd.to_numeric(df['cp_id'], errors='coerce')
    df['sim_to_reco_score'] = pd.to_numeric(df['sim_to_reco_score'], errors='coerce')
    df['cp_energy'] = pd.to_numeric(df['cp_energy'], errors='coerce')

# -----------------------------
# Prepare Caloparticle-Level Data
# -----------------------------
def prepare_cp_data(df):
    """
    Group the DataFrame by ['event_index', 'cp_id'] so that each caloparticle is counted once.
    For each group:
      - Take the first cp_energy (they are assumed to be identical).
      - Take the minimum sim_to_reco_score.
      - Mark the caloparticle as 'reconstructed' if min(sim_to_reco_score) < 0.2.
    """
    grouped = df.groupby(['event_index', 'cp_id']).agg({
        'cp_energy': 'first',          # Use the first cp_energy value.
        'sim_to_reco_score': 'min'       # Minimum score among the rows for that cp.
    }).reset_index()
    
    # Mark as reconstructed if any sim_to_reco_score < 0.2.
    grouped['reco'] = (grouped['sim_to_reco_score'] < 0.2).astype(int)
    return grouped

# Prepare the caloparticle data for both DataFrames.
df_CL_cp = prepare_cp_data(df_CL)
df_TICL_cp   = prepare_cp_data(df_TICL)
# -----------------------------
# Bin Caloparticles by Energy
# -----------------------------
# Define energy bins based on the range of cp_energy from df_CL.
min_energy = df_CL_cp['cp_energy'].min()
max_energy = df_CL_cp['cp_energy'].max()
n_bins = 10  # Adjust the number of bins if desired.
# Create n_bins bins (n_bins+1 edges).
energy_bins = np.linspace(min_energy, max_energy, n_bins + 1)

# Assign each caloparticle to an energy bin.
df_CL_cp['energy_bin'] = pd.cut(df_CL_cp['cp_energy'], bins=energy_bins, labels=False, include_lowest=True)
df_TICL_cp['energy_bin']   = pd.cut(df_TICL_cp['cp_energy'],   bins=energy_bins, labels=False, include_lowest=True)


# -----------------------------
# Calculate Efficiency per Energy Bin
# -----------------------------
def aggregate_efficiency(df):
    """
    For each energy bin, calculate:
      - The total number of caloparticles in the bin.
      - The number of reconstructed caloparticles.
      - Efficiency = (number of reconstructed) / (total number).
    """
    agg = df.groupby('energy_bin').agg(
        total_cp=('cp_energy', 'count'),
        reco_cp=('reco', 'sum')
    ).reset_index()
    agg['efficiency'] = agg['reco_cp'] / agg['total_cp']
    return agg

agg_CL = aggregate_efficiency(df_CL_cp)
agg_TICL   = aggregate_efficiency(df_TICL_cp)


# -----------------------------
# Plot Efficiency vs Energy with Histogram Overlay
# -----------------------------
# Compute bin centers for plotting: average of adjacent bin edges.
bin_centers = (energy_bins[:-1] + energy_bins[1:]) / 2
bar_width = energy_bins[1] - energy_bins[0]

# Create a figure with two y-axes.
fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()

# Plot the efficiency curves on ax1.
ax1.plot(bin_centers, agg_CL['efficiency'], marker='o', linestyle='-', label='Efficiency (df_CL)')
ax1.plot(bin_centers, agg_TICL['efficiency'],   marker='x', linestyle='--', label='Efficiency (df_TICL)')
ax1.set_xlabel('Caloparticle Energy')
ax1.set_ylabel('Efficiency')
ax1.set_ylim(0, 1)
ax1.legend(loc='upper left')
ax1.grid(True)

# Plot the histogram (number of caloparticles per energy bin) on ax2.
# Here we use the counts from df_CL.
ax2.bar(bin_centers, agg_CL['total_cp'], width=bar_width, 
        color='lightblue', alpha=0.4, label='Event Count')
ax2.set_ylim(0, agg_CL['total_cp'].max()*2)
ax2.set_ylabel('Number of Caloparticles')
ax2.legend(loc='upper right')

plt.title('Efficiency vs Caloparticle Energy with Event Count Histogram')
plt.savefig("plots/Efficiency.png")
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Preprocessing
# -----------------------------
# Convert relevant columns to numeric.
for df in [df_CL, df_TICL]:
    df['trackster_id'] = pd.to_numeric(df['trackster_id'], errors='coerce')
    df['reco_to_sim_score'] = pd.to_numeric(df['reco_to_sim_score'], errors='coerce')
    df['trackster_energy'] = pd.to_numeric(df['trackster_energy'], errors='coerce')

# -----------------------------
# Prepare Trackster-Level Data for Purity
# -----------------------------
def prepare_trackster_data(df):
    """
    Group the DataFrame by ['event_index', 'trackster_id'] so that each trackster is counted once.
    For each group:
      - Take the first trackster_energy (assumed to be identical across rows),
      - Take the minimum reco_to_sim_score,
      - Mark the trackster as 'associated' if the minimum reco_to_sim_score is < 0.2.
    """
    grouped = df.groupby(['event_index', 'trackster_id']).agg({
        'trackster_energy': 'first',       # Use the first trackster_energy value.
        'reco_to_sim_score': 'min'           # Minimum score among the rows for that trackster.
    }).reset_index()
    
    # Mark as associated if reco_to_sim_score is < 0.2.
    grouped['assoc'] = (grouped['reco_to_sim_score'] < 0.2).astype(int)
    return grouped

# Prepare the trackster-level data for both DataFrames.
df_CL_ts = prepare_trackster_data(df_CL)
df_TICL_ts   = prepare_trackster_data(df_TICL)

# -----------------------------
# Bin Tracksters by Energy
# -----------------------------
# Define energy bins based on the range of trackster_energy from df_CL.
min_energy_ts = df_CL_ts['trackster_energy'].min()
max_energy_ts = df_CL_ts['trackster_energy'].max()
n_bins_ts = 10  # Adjust the number of bins if desired.
energy_bins_ts = np.linspace(min_energy_ts, max_energy_ts, n_bins_ts + 1)

# Assign each trackster to an energy bin.
df_CL_ts['energy_bin'] = pd.cut(df_CL_ts['trackster_energy'],
                                    bins=energy_bins_ts, labels=False, include_lowest=True)
df_TICL_ts['energy_bin']   = pd.cut(df_TICL_ts['trackster_energy'],
                                    bins=energy_bins_ts, labels=False, include_lowest=True)

# -----------------------------
# Calculate Purity per Energy Bin
# -----------------------------
def aggregate_purity(df):
    """
    For each energy bin, calculate:
      - Total number of tracksters,
      - Number of associated tracksters (with reco_to_sim_score < 0.2),
      - Purity = (number of associated tracksters) / (total number).
    """
    agg = df.groupby('energy_bin').agg(
        total_ts = ('trackster_energy', 'count'),
        assoc_ts = ('assoc', 'sum')
    ).reset_index()
    agg['purity'] = agg['assoc_ts'] / agg['total_ts']
    return agg

agg_CL_ts = aggregate_purity(df_CL_ts)
agg_TICL_ts   = aggregate_purity(df_TICL_ts)

# Reindex both aggregated DataFrames so that they have one row per energy bin (0 to n_bins_ts-1)
agg_CL_ts = agg_CL_ts.set_index('energy_bin').reindex(range(n_bins_ts), fill_value=np.nan).reset_index()
agg_TICL_ts   = agg_TICL_ts.set_index('energy_bin').reindex(range(n_bins_ts), fill_value=np.nan).reset_index()

# -----------------------------
# Plot Purity vs Trackster Energy with Histogram Overlay
# -----------------------------
# Compute bin centers for plotting: average of adjacent bin edges.
bin_centers_ts = (energy_bins_ts[:-1] + energy_bins_ts[1:]) / 2
bar_width_ts = energy_bins_ts[1] - energy_bins_ts[0]

# Create a figure with two y-axes.
fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()

# Plot purity curves on the primary y-axis.
ax1.plot(bin_centers_ts, agg_CL_ts['purity'], marker='o', linestyle='-', label='Purity (df_CL)')
ax1.plot(bin_centers_ts, agg_TICL_ts['purity'], marker='x', linestyle='--', label='Purity (df_TICL)')
ax1.set_xlabel('Trackster Energy')
ax1.set_ylabel('Purity')
ax1.set_ylim(0, 1)
ax1.legend(loc='upper left')
ax1.grid(True)

# Plot histogram (trackster count per energy bin) on the secondary y-axis.
ax2.bar(bin_centers_ts, agg_CL_ts['total_ts'], width=bar_width_ts,
        color='lightblue', alpha=0.4, label='Trackster Count')
ax2.set_ylabel('Number of Tracksters')
ax2.set_ylim(0, (agg_CL_ts['total_ts'].max() or 1) * 1.5)
ax2.legend(loc='upper right')

plt.title('Purity vs Trackster Energy with Trackster Count Histogram')
plt.savefig("plots/Purity.png")
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ====================================================
# 1. Preprocessing and Association Selection for Calo Particles
# ====================================================
def process_cp_dataframe(df):
    """
    Convert relevant columns to numeric, then for each calo particle (identified by event_index and cp_id):
      - Group the data and take the first cp_energy value (assumed constant)
      - Compute the minimum sim_to_reco_score (i.e. best association)
      - Also take the energy_ratio from the best-associated trackster.
    Only keep those calo particles for which the minimum sim_to_reco_score < 0.2.
    """
    # Convert columns to numeric.
    for col in ['cp_id', 'sim_to_reco_score', 'cp_energy', 'energy_ratio']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Keep only rows with a sim_to_reco_score < 0.2
    df_cp = df[df['sim_to_reco_score'] < 0.2].copy()
    
    # For each calo particle (cp_id per event) keep the row with the smallest sim_to_reco_score.
    df_cp = (df_cp.groupby(['event_index', 'cp_id'], as_index=False)
             .apply(lambda g: g.loc[g['sim_to_reco_score'].idxmin()])
             .reset_index(drop=True))
    return df_cp

# Process both dataframes
df_CL_cp   = process_cp_dataframe(df_CL)
df_TICL_cp = process_cp_dataframe(df_TICL)

# ====================================================
# 2. Binning by Calo Particle Energy (cp_energy)
# ====================================================
n_bins = 10
# Determine energy bins from the CL model (you can change this if needed)
min_energy = df_CL_cp['cp_energy'].min()
max_energy = df_CL_cp['cp_energy'].max()
energy_bins = np.linspace(min_energy, max_energy, n_bins + 1)
bin_centers = (energy_bins[:-1] + energy_bins[1:]) / 2

# Assign each calo particle to an energy bin based on cp_energy.
df_CL_cp['energy_bin']   = pd.cut(df_CL_cp['cp_energy'], bins=energy_bins, labels=False, include_lowest=True)
df_TICL_cp['energy_bin'] = pd.cut(df_TICL_cp['cp_energy'], bins=energy_bins, labels=False, include_lowest=True)

# ====================================================
# 3. Gaussian Fit in Each Energy Bin & Count Aggregation
# ====================================================
def fit_bins(df, n_bins):
    """
    For each energy bin, take the energy_ratio values from the processed calo particles,
    fit a Gaussian (if data exists), and record the fitted mean (response) and sigma (resolution).
    Also record the number of calo particles in each bin.
    """
    fitted_mean = []
    fitted_sigma = []
    counts = []  # number of calo particles in the bin.
    for b in range(n_bins):
        bin_data = df.loc[df['energy_bin'] == b, 'energy_ratio'].dropna()
        counts.append(len(bin_data))
        if len(bin_data) > 0:
            mu, sigma = norm.fit(bin_data)
        else:
            mu, sigma = np.nan, np.nan
        fitted_mean.append(mu)
        fitted_sigma.append(sigma)
    return np.array(fitted_mean), np.array(fitted_sigma), np.array(counts)

fitted_mean_CL, fitted_sigma_CL, counts_CL     = fit_bins(df_CL_cp, n_bins)
fitted_mean_TICL, fitted_sigma_TICL, counts_TICL = fit_bins(df_TICL_cp, n_bins)

# ====================================================
# 4. Plot Side-by-Side Histograms for Selected Bins (First, Middle, Last)
# ====================================================
selected_bins = [0, n_bins // 2, n_bins - 1]
n_sel = len(selected_bins)
fig_hist, axs = plt.subplots(n_sel, 2, figsize=(12, 4 * n_sel), constrained_layout=True)
xlim_range = (0, 2)  # fixed x-axis

for i, b in enumerate(selected_bins):
    # For df_CL histogram:
    data_CL = df_CL_cp.loc[df_CL_cp['energy_bin'] == b, 'energy_ratio'].dropna()
    ax_left = axs[i, 0] if n_sel > 1 else axs[0]
    n_CL, bins_CL, _ = ax_left.hist(data_CL, bins=20, density=True,
                                    color='lightblue', alpha=0.6, edgecolor='k')
    ax_left.set_xlim(xlim_range)
    if len(data_CL) > 0:
        mu_CL = fitted_mean_CL[b]
        sigma_CL = fitted_sigma_CL[b]
        x_fit = np.linspace(xlim_range[0], xlim_range[1], 100)
        p_CL = norm.pdf(x_fit, mu_CL, sigma_CL)
        ax_left.plot(x_fit, p_CL, 'r--', linewidth=2,
                     label=f'Fit: μ={mu_CL:.2f}, σ={sigma_CL:.2f}')
    ax_left.set_title(f'Our Model, Bin {b} (cp Energy={bin_centers[b]:.2f}); Count={len(data_CL)}')
    ax_left.set_xlabel('Energy Ratio')
    ax_left.set_ylabel('Density')
    ax_left.legend(fontsize=10)
    
    # For df_TICL histogram:
    data_TICL = df_TICL_cp.loc[df_TICL_cp['energy_bin'] == b, 'energy_ratio'].dropna()
    ax_right = axs[i, 1] if n_sel > 1 else axs[1]
    n_TICL, bins_TICL, _ = ax_right.hist(data_TICL, bins=20, density=True,
                                          color='lightgreen', alpha=0.6, edgecolor='k')
    ax_right.set_xlim(xlim_range)
    if len(data_TICL) > 0:
        mu_TICL = fitted_mean_TICL[b]
        sigma_TICL = fitted_sigma_TICL[b]
        x_fit = np.linspace(xlim_range[0], xlim_range[1], 100)
        p_TICL = norm.pdf(x_fit, mu_TICL, sigma_TICL)
        ax_right.plot(x_fit, p_TICL, 'r--', linewidth=2,
                      label=f'Fit: μ={mu_TICL:.2f}, σ={sigma_TICL:.2f}')
    ax_right.set_title(f'TICL, Bin {b} (cp Energy={bin_centers[b]:.2f}); Count={len(data_TICL)}')
    ax_right.set_xlabel('Energy Ratio')
    ax_right.set_ylabel('Density')
    ax_right.legend(fontsize=10)

plt.suptitle('Side-by-Side Histograms of energy_ratio in Selected cp Energy Bins', fontsize=14)
plt.savefig("plots/histogram.png")
plt.show()

# ====================================================
# 5. Plot Response and Resolution vs. cp Energy Bin Centers
#     (with the df_CL counts as a background histogram)
# ====================================================
bar_width = energy_bins[1] - energy_bins[0]

fig_params, (ax_resp, ax_res) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

# Plot Response (fitted mean) vs cp_energy bin centers.
ax_resp.plot(bin_centers, fitted_mean_CL, marker='o', linestyle='-', color='blue', label='Our Model')
ax_resp.plot(bin_centers, fitted_mean_TICL, marker='s', linestyle='-', color='green', label='TICL')
ax_resp.axhline(1, color='k', linestyle='--', label='Reference: 1')
ax_resp.set_ylabel('Response')
ax_resp.set_title('Response + Resolution vs Calo Particle Energy', fontsize=12)
ax_resp.grid(True)
ax_resp.legend(loc='upper left', fontsize=10)
ax_resp.set_ylim(0, fitted_mean_CL.max() + 0.5)

# Twin axis to display the number of calo particles (from df_CL) as background.
ax_resp2 = ax_resp.twinx()
ax_resp2.bar(bin_centers, counts_CL, width=bar_width, color='lightblue', alpha=0.3)
ax_resp2.set_ylabel('Number of Calo Particles')
ax_resp2.set_ylim(0, 3*counts_CL.max())

# Plot Resolution (fitted sigma) vs cp_energy bin centers.
ax_res.plot(bin_centers, fitted_sigma_CL, marker='o', linestyle='-', color='blue', label='Our Model')
ax_res.plot(bin_centers, fitted_sigma_TICL, marker='s', linestyle='-', color='green', label='TICL')
ax_res.set_xlabel('Calo Particle Energy (Bin Center)')
ax_res.set_ylabel('Resolution')
ax_res.grid(True)
ax_res.legend(loc='upper left', fontsize=10)
ax_res.set_ylim(0, fitted_sigma_CL.max() + 0.1)

# Twin axis for df_CL counts.
ax_res2 = ax_res.twinx()
ax_res2.bar(bin_centers, counts_CL, width=bar_width, color='lightblue', alpha=0.3)
ax_res2.set_ylabel('Number of Calo Particles')
ax_res2.set_ylim(0, 3*counts_CL.max())
plt.savefig("plots/ResponseResolution.png")
plt.show()