###################### Adding Regularization + Using Balanced Datasets + ACE ###############################################

import nni
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import logging
import argparse
import csv
import os
import random 

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

from functorch import vmap, jacfwd, hessian

import pandas as pd
import numpy as np
import smogn
# import shap
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")


logger = logging.getLogger()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
################################################### Functions and classes ##################################################

def set_random_seed(seed_val):
    os.environ['PYTHONHASHSEED'] = str(seed_val)
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
def map_act_func(af_name):
    activation_functions = {
        "RELU": torch.nn.ReLU(),
        "LeakyReLU": torch.nn.LeakyReLU(),
        "Sigmoid": torch.nn.Sigmoid(),
        "Tanh": torch.nn.Tanh(),
        "Softplus": torch.nn.Softplus()
    }
    act_func = activation_functions.get(af_name)
    
    if act_func:
        return act_func
    else:
        sys.exit(f"Invalid activation function:{af_name}")
        
def map_optimizer(opt_name,net_params, lr):
    optimizers = {
        "SGD": optim.SGD,
        "Adam": optim.Adam
    }
    optimizer = optimizers.get(opt_name)
    if optimizer:
        return optimizer(net_params, lr=lr)
    else:
        sys.exit(f"Invalid optimizer: {opt_name}")


class MLP(nn.Module):
    def __init__(self, input_size, params):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.layer_size = params['layer_size']
        hidden_layer_num = params['hidden_layer_number']
        self.dropout_rate = params['dropout_rate']  # Get dropout rate

        self.first_layer = nn.Linear(input_size, self.layer_size)
        self.batch_norm1 = nn.BatchNorm1d(self.layer_size)  # Batch Normalization for first layer
        self.dropout1 = nn.Dropout(self.dropout_rate)  # Dropout for first layer
        self.act_func = map_act_func(params['act_func'])

        if params['modify_weights']:
            self.first_layer.weight.data = self.weight_dist_gen(params['ates'])
            self.first_layer.bias.data.fill_(0.0)

        if hidden_layer_num >= 1:
            self.hidden_layers = nn.ModuleList([
                nn.Linear(self.layer_size, self.layer_size) for _ in range(hidden_layer_num)
            ])
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(self.layer_size) for _ in range(hidden_layer_num)
            ])
            self.dropouts = nn.ModuleList([
                nn.Dropout(self.dropout_rate) for _ in range(hidden_layer_num)
            ])

        self.final_layer = nn.Linear(self.layer_size, 1)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.batch_norm1(x)  # Apply batch normalization
        x = self.act_func(x)
        x = self.dropout1(x)  # Apply dropout

        if hasattr(self, 'hidden_layers'):
            for hidden_layer, batch_norm, dropout in zip(self.hidden_layers, self.batch_norms, self.dropouts):
                x = hidden_layer(x)
                x = batch_norm(x)  # Apply batch normalization
                x = self.act_func(x)
                x = dropout(x)  # Apply dropout

        x = self.final_layer(x)
        return x

    def weight_dist_gen(self, ate_vals):
        random_weights = np.zeros((self.input_size, self.layer_size))
        for i in range(self.input_size):
            random_weights[i, :] = np.random.normal(ate_vals[i], 0.1, params['layer_size'])
        return torch.from_numpy(random_weights.astype('float32')).T

def gen_save_feature_attributions(model, X_val_tensor, X_train_tensor, feature_names, fold, output_dir="feature_attributions"):
    """
    Generates feature attribution charts using Captum and saves them to CSV files.

    Args:
        model (torch.nn.Module): The trained neural network model.
        X_val_tensor (torch.Tensor): Validation input data tensor.
        X_train_tensor (torch.Tensor): Training input data tensor (used for GradientShap reference).
        feature_names (list): List of strings for feature names.
        fold (int): The current K-Fold number, used for naming output files.
        output_dir (str): Directory to save the attribution CSV files.
    """
    model.eval()

    ig = IntegratedGradients(model)
    ig_nt = NoiseTunnel(ig)
    dl = DeepLift(model)
    gs = GradientShap(model)
    fa = FeatureAblation(model)
    
    with torch.no_grad(): # Use no_grad for inference where gradients are not needed
        # Integrated Gradients
        ig_attr = ig.attribute(X_val_tensor, target=0, n_steps=50) # Assuming single output (target=0)
        ig_nt_attr = ig_nt.attribute(X_val_tensor, target=0, n_steps=50) # target=0 for single output
    
        # DeepLift (uncomment if you add a proper reference baseline)
        # You'll likely need a baseline for DeepLift, e.g., a zero tensor or mean of training data
        # dl_attr = dl.attribute(X_val_tensor, baselines=torch.zeros_like(X_val_tensor), target=0)
    
        # GradientShap requires baselines (e.g., a random sample from training data)
        # Using X_train_tensor as the baseline reference, adjust if you need a smaller subset
        gs_attr = gs.attribute(X_val_tensor, baselines=X_train_tensor, target=0)
    
        # Feature Ablation
        fa_attr = fa.attribute(X_val_tensor, target=0) # target=0 for single output
    

    # Prepare attributions for visualization (sum across samples and normalize)
    x_axis_data = np.arange(X_val_tensor.shape[1])
    x_axis_data_labels = list(map(lambda idx: feature_names[idx], x_axis_data))

    # Integrated Gradients
    ig_attr_sum = ig_attr.detach().cpu().numpy().sum(axis=0)
    ig_attr_norm_sum = ig_attr_sum / (np.linalg.norm(ig_attr_sum, ord=1) + 1e-8) # Add small epsilon to prevent div by zero

    # Integrated Gradients with NoiseTunnel (SmoothGrad)
    ig_nt_attr_sum = ig_nt_attr.detach().cpu().numpy().sum(axis=0)
    ig_nt_attr_norm_sum = ig_nt_attr_sum / (np.linalg.norm(ig_nt_attr_sum, ord=1) + 1e-8)

    # DeepLift (uncomment if used)
    # dl_attr_sum = dl_attr.detach().cpu().numpy().sum(axis=0)
    # dl_attr_norm_sum = dl_attr_sum / (np.linalg.norm(dl_attr_sum, ord=1) + 1e-8)

    # GradientShap
    gs_attr_sum = gs_attr.detach().cpu().numpy().sum(axis=0)
    gs_attr_norm_sum = gs_attr_sum / (np.linalg.norm(gs_attr_sum, ord=1) + 1e-8)

    # Feature Ablation
    fa_attr_sum = fa_attr.detach().cpu().numpy().sum(axis=0)
    fa_attr_norm_sum = fa_attr_sum / (np.linalg.norm(fa_attr_sum, ord=1) + 1e-8)

    # Get the normalized weights from the first layer
    # model.first_layer.weight has shape (output_features, input_features)
    # For a single output, model.first_layer.weight[0] gives the weights for the first output neuron
    lin_weight = model.first_layer.weight[0].detach().cpu().numpy()
    y_axis_lin_weight = lin_weight / (np.linalg.norm(lin_weight, ord=1) + 1e-8)

    # --- Plotting ---
    width = 0.14
    legends = ['Int Grads', 'Int Grads w/SmoothGrad', 'GradientSHAP', 'Feature Ablation', 'Learned Weights (First Layer)']

    plt.figure(figsize=(20, 10))
    ax = plt.subplot()
    ax.set_title(f'Feature Importances (Fold {fold + 1})', fontsize=20)
    ax.set_ylabel('Normalized Attributions', fontsize=18)

    # Set font sizes
    FONT_SIZE_LABELS = 16
    FONT_SIZE_TICKS = 14
    FONT_SIZE_LEGEND = 12

    plt.rc('font', size=FONT_SIZE_LABELS)
    plt.rc('axes', titlesize=FONT_SIZE_LABELS)
    plt.rc('axes', labelsize=FONT_SIZE_LABELS)
    plt.rc('legend', fontsize=FONT_SIZE_LEGEND)
    plt.tick_params(axis='x', labelsize=FONT_SIZE_TICKS)
    plt.tick_params(axis='y', labelsize=FONT_SIZE_TICKS)

    # Plot bars
    ax.bar(x_axis_data, ig_attr_norm_sum, width, align='center', alpha=0.8, color='#eb5e7c', label='Int Grads')
    ax.bar(x_axis_data + width, ig_nt_attr_norm_sum, width, align='center', alpha=0.7, color='#A90000', label='Int Grads w/SmoothGrad')
    # ax.bar(x_axis_data + 2 * width, dl_attr_norm_sum, width, align='center', alpha=0.6, color='#34b8e0', label='DeepLift')
    ax.bar(x_axis_data + 2 * width, gs_attr_norm_sum, width, align='center',  alpha=0.8, color='#4260f5', label='GradientSHAP')
    ax.bar(x_axis_data + 3 * width, fa_attr_norm_sum, width, align='center', alpha=1.0, color='#49ba81', label='Feature Ablation')
    ax.bar(x_axis_data + 4 * width, y_axis_lin_weight, width, align='center', alpha=1.0, color='grey', label='Learned Weights') # Shifted by 4*width

    ax.autoscale_view()
    plt.tight_layout()

    # Adjust x-axis ticks to be centered relative to the groups of bars
    ax.set_xticks(x_axis_data + (len(legends)-1) * width / 2) # Adjust center for the number of bars
    ax.set_xticklabels(x_axis_data_labels)

    plt.legend(loc='best', frameon=True, fancybox=True, shadow=True, borderpad=1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # --- Save to DataFrame and CSV ---
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    data = {
        'Feature': x_axis_data_labels,
        'Int Grads': ig_attr_norm_sum,
        'Int Grads w/SmoothGrad': ig_nt_attr_norm_sum,
        # 'DeepLift': dl_attr_norm_sum, # Uncomment if DeepLift is used
        'GradientSHAP': gs_attr_norm_sum,
        'Feature Ablation': fa_attr_norm_sum,
        'Weights (First Layer)': y_axis_lin_weight
    }

    # Create a DataFrame
    df_att = pd.DataFrame(data)

    # Define the file name for the CSV file
    csv_file_name = os.path.join(output_dir, f"feature_importances_fold_{fold+1}.csv")

    # Save the DataFrame to a CSV file
    df_att.to_csv(csv_file_name, index=False)
    print(f"Attributions saved to: {csv_file_name}")
    
################################################ Initialize Stuff ###################################################
random_seed = 18
set_random_seed(random_seed)

# Load data
# df = pd.read_csv("Causal Learning_Train_Regression.csv")
x = np.random.randn(100)
y = np.random.randn(100)
z = np.sin(x) + y**2 + 0.1*np.random.randn(100)
w = 0.7*np.exp(z) + 0.1*np.random.randn(100)
df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'w': w})

## If you have to balance the dataset this is what we used
## SMOGN - Synthetic Minority over Sampling Technique for regression with Gaussian Noise
## There are other approaches but we liked this because it does over and under sampling at the same time to give a good training set
# Removed it since we wont be needing it for this example

"""
# Specify the relevance matrix
rg_matrix = [
    [0.1,  0, 0],
    [0.15,  1, 0],
    [0.2,  1, 0]
]

df = smogn.smoter(
    data = df,
    y = 'w',
    pert = 0.04,
    samp_method = 'balance',
    replace = False,
    rel_thres = 0.1,
    rel_method='manual',
    rel_ctrl_pts_rg = rg_mtrx
)
"""

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)
X = StandardScaler().fit_transform(X)

# Default params for the MLP to initiate NNI training 
params = {
    'ates':[0, 0.47, 0.16], # Sample ATEs from DECI
    'modify_weights':1,
    'lambda_1':0.5,
    'hidden_layer_number': 2,
    'layer_size': 32,
    'act_func': 'Tanh',
    'learning_rate': 0.05,
    'optimizer': 'Adam',
    'dropout_rate': 0.3
}

# Update params with NNI-generated values
params.update(nni.get_next_parameter())

# Get the experiment name
experiment_id = nni.get_experiment_id()

# File for storing results for all experiments
results_csv = experiment_id +'_results_.csv'

# Storage 
train_loss = []
test_loss = []
fold_losses=[] 
rmse_list = []
r2_list = []
spearman_list = []
results = []

fold = 5
epochs = 500

# Early Stopping
patience = 20 # Experiment with value
best_val_loss = float('inf')
epochs_without_improvement = 0

############################################### Model Stuff ######################################################

kf = KFold(n_splits=fold, shuffle=True, random_state=42) # Initialize KFold

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")
    
    # Split data into training and validation sets
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    input_size = X_train.shape[1]
    model = MLP(input_size,params).to(device)
    optimizer = map_optimizer(params['optimizer'], model.parameters(), params['learning_rate'])

    
    # Convert data to PyTorch tensors and move to GPU - might need to change
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)              # inputs - before numpy 
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)  # targets
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)                  # data - before numpy                                
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)      # label - before numpy 


    # Train
    for epoch in range(1, epochs+1):

        model.train()
        optimizer.zero_grad()
        predictions = model(X_train_tensor)

        # Loss criterion
        
        if params['lambda_1'] > 0:
            target_jacobian_vals = torch.tensor([params['ates']],device=device) # target derivative vector 
            batch_jacobian = vmap(jacfwd(model, argnums=0), in_dims=0)
            model_jacobians = batch_jacobian(X_train_tensor).to(device)
            jacobian_deviation = model_jacobians - target_jacobian_vals # Difference between model and target
            L1_norm_per_sample = torch.norm(jacobian_deviation, p=1,dim=2) # P1 = L1 norm
            tolerance = 0.1 # Tolerance for deviation
            penalized_deviation = torch.max(L1_norm_per_sample - tolerance, torch.zeros_like(L1_norm_per_sample))
            causal_reg_loss_term = params['lambda_1'] * torch.mean(penalized_deviation)
            total_loss =  torch.nn.MSELoss()(predictions, y_train_tensor) + torch.mean(causal_reg_loss_term)
            
        else: 
            loss = torch.nn.MSELoss()(predictions, y_train_tensor)
            
        loss.backward()
        optimizer.step()

        # Evaluate for Early Stoping
        if epoch %10 == 0 or epoch == epochs - 1:
            model.eval()

            with torch.no_grad():
                val_predictions_np = model(X_val_tensor).cpu().numpy()
                y_val_np = y_val_tensor.cpu().numpy()
                rmse = np.sqrt(mean_squared_error(y_val_np, val_predictions_np))  # Move data back to CPU for sklearn #Rmse used for model evaluation

                # Checking using rmse as the loss function
                if rmse < best_val_loss:
                    best_val_loss = rmse
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    # print(f"Early stopping at epoch {epoch + 1} with validation loss: {val_loss}")
                    break
                    
        # Evaluate the model one last time for final results
        model.eval()
        with torch.no_grad():
            val_predictions_np = model(X_val_tensor).cpu().numpy()
            y_val_np = y_val_tensor.cpu().numpy()
            rmse = np.sqrt(mean_squared_error(y_val_np, val_predictions_np))  # Move data back to CPU for sklearn 
            r2 = r2_score(y_val_np, val_predictions_np)
            spearman_corr, _ = spearmanr(y_val_np.flatten(), val_predictions_np.flatten())

            print(f"Validation Loss for Fold {fold + 1}: {rmse}")
            print(f"Fold {fold + 1} - RMSE: {rmse}, R²: {r2}, Spearman Correlation: {spearman_corr}")

            # Fold Results 
            fold_losses.append(rmse)
            rmse_list.append(rmse)
            r2_list.append(r2)
            spearman_list.append(spearman_corr)
            
        # nni.report_intermediate_result(rmse)
        
# Compute the averages and store experiment results - we used averages but that can be unsrable and some ppl prefer max or min
avg_rmse_loss = np.mean(fold_losses)
avg_rmse = np.mean(rmse_list)
avg_r2 = np.mean(r2_list)
avg_spearman = np.mean(spearman_list)

results_df = pd.DataFrame([{
    "RMSE": avg_rmse, 
    "R_squared": avg_r2, 
    "Spearman_Correlation": avg_spearman
}]
                         )

params_df = pd.DataFrame([params])
params_df = params_df.drop(["ates"], axis=1) # ates is the same through out so has no value
final_df = pd.concat([params_df, results_df],axis=1)
                    
if os.path.exists(results_csv):
    final_df.to_csv(results_csv, mode='a',header=False, index=False)
else:
    final_df.to_csv(results_csv, index=False)

nni.report_final_result(avg_rmse_loss)


        

        


    
    
    






















