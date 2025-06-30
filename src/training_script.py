import numpy as np
import os
import torch
from  models.sae import BatchTopKSAE, JumpReLUSAE, VanillaSAE
from models.model import SAE
from tqdm import tqdm
import matplotlib.pyplot as plt
from evaluate_feature import calculate_AUC_matrix, calculate_precision_matrix
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper
import argparse


def load_data(data_path, batch_size=2048, eval_size=1000):
    data = np.load(data_path)
    # Flatten the data
    data = data.reshape(-1, data.shape[-1])
    # Data Loader
    dataset = torch.tensor(data, dtype=torch.float32)
    # Create DataLoader for training and validation
    train_data, val_data = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    train_wrapper = IterableWrapper(train_data)
    val_wrapper = IterableWrapper(val_data)
    # Create DataLoader objects
    train_loader = DataLoader(train_wrapper, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_wrapper, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(IterableWrapper(dataset), batch_size=1000, shuffle=False)
    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    print(f"Evaluation data size: {len(dataset)}")
    
    return train_loader, val_loader, eval_loader

def load_ground_truth(gt_path, eval_size=1000):
    gt = np.load(gt_path)
    # Flatten the ground truth
    gt = gt.reshape(-1, 58)[:,57].reshape(-1, 1)
    print(f"Ground truth data size: {gt.shape}")
    gt_wrapper = IterableWrapper(gt)
    gt_loader = DataLoader(gt_wrapper, batch_size=eval_size, shuffle=False)
    return gt_loader

def train_model(model, train_loader, val_loader, optimizer, loss_function, cfg,num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()
            outputs = model(batch)
            if model.__class__.__name__ == 'BatchTopKSAE' or model.__class__.__name__ == 'JumpReLUSAE' or model.__class__.__name__ == 'VanillaSAE':
                loss = outputs['loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                model.make_decoder_weights_and_grad_unit_norm()
                optimizer.step()
                
            else:
                loss = loss_function(outputs, batch)
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(train_loader)}")

        model.eval()
        val_epoch_loss = 0
        best_loss = float('inf')
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch)
                if model.__class__.__name__ == 'BatchTopKSAE' or model.__class__.__name__ == 'JumpReLUSAE' or model.__class__.__name__ == 'VanillaSAE':
                    loss = outputs['loss']
                else:
                    loss = loss_function(outputs, batch)
                val_epoch_loss += loss.item()
        print(f"Validation Loss after Epoch {epoch + 1}: {epoch_loss / len(val_loader)}")
        if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                best_model = model.state_dict()


    return model, best_model

def save_model(model, model_path):
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), f"{model_path}/model.pt")

def evaluate_model(model, data_loader, ground_truth):
    model.eval()
    
    for data_sample, gt in zip(data_loader, ground_truth):
        if torch.sum(gt) != 0:
            print("Found non-zero ground truth data.")
            break
    data_sample = data_sample.numpy()
    gt = gt.numpy()
    
    with torch.no_grad():
        output = model(torch.tensor(data_sample, dtype=torch.float32))
        if model.__class__.__name__ == 'BatchTopKSAE' or model.__class__.__name__ == 'JumpReLUSAE' or model.__class__.__name__ == 'VanillaSAE':
            feature_acts = output["feature_acts"].cpu().numpy()
        else:
            feature_acts = output
    
    print(f"Feature activations shape: {feature_acts.shape}")
    print(f"Ground truth shape: {gt.shape}")
    
    auc_matrix = calculate_AUC_matrix(feature_acts,gt)
        
    precision_matrix = calculate_precision_matrix(feature_acts, gt)

    best_activations_auc = feature_acts[:, np.argsort(auc_matrix, axis=0)[-10:].flatten()]
    best_activations_precision = feature_acts[:, np.argsort(precision_matrix, axis=0)[-10:].flatten()]

    best_activations_auc_indices = np.argsort(auc_matrix, axis=0)[-10:].flatten()
    best_activations_precision_indices = np.argsort(precision_matrix, axis=0)[-10:].flatten()
    
    return auc_matrix, precision_matrix, best_activations_auc, best_activations_precision, best_activations_auc_indices, best_activations_precision_indices, gt


def plot_best_aucs(auc_matrix, save_path):
    best_aucs = np.sort(auc_matrix, axis=0)[-10:]
    # Plot Bar for AUC of each PC
    plt.figure(figsize=(25, 6))
    plt.style.use('seaborn-v0_8-deep')
    plt.bar(range(1, best_aucs.shape[0] + 1), best_aucs.flatten(), color='lightblue', edgecolor='black', width=0.6, alpha=0.7)
    plt.axhline(y=0.5, color='red', linestyle='--', label='Above Random')
    plt.xlabel('Best Activation Features')
    plt.ylabel('AUC')
    plt.title('AUC for 10 best Activation Features of BatchTopKSAE')

    plt.ylim(0, 1)
    plt.xticks(range(1, 33))
    # color the bar with higer AUC in green
    max_auc = np.max(best_aucs)
    max_auc_index = np.argmax(best_aucs)
    plt.bar(max_auc_index + 1, best_aucs.flatten()[max_auc_index], color='lightgreen',width=0.6, alpha=0.7, label=f'Max AUC with {max_auc:.2f}')
    plt.legend()
    plt.savefig(f"{save_path}/AUC_plot.png")


def plot_best_precision(precision_matrix, save_path):
    best_precision = np.sort(precision_matrix, axis=0)[-10:]
    # Plot Bar for Precision of each PC
    plt.figure(figsize=(25, 6))
    plt.style.use('seaborn-v0_8-deep')
    plt.bar(range(1, best_precision.shape[0] + 1), best_precision.flatten(), color='lightblue', edgecolor='black', width=0.6, alpha=0.7)
    plt.axhline(y=0.5, color='red', linestyle='--', label='Above Random')
    plt.xlabel('Best Activation Features')
    plt.ylabel('Precision')
    plt.title('Precision for 10 best Activation Features of BatchTopKSAE')

    plt.ylim(0, 1)
    plt.xticks(range(1, 33))
    # color the bar with higer Precision in green
    max_precision = np.max(best_precision)
    max_precision_index = np.argmax(best_precision)
    plt.bar(max_precision_index + 1, best_precision.flatten()[max_precision_index], color='lightgreen',width=0.6, alpha=0.7, label=f'Max Precision with {max_precision:.2f}')
    plt.legend()
    plt.savefig(f"{save_path}/Precision_plot.png")

def plot_genomic_track(ground_truth, feature, feature_name, save_path):
    plt.figure(figsize=(25, 5))
    plt.suptitle(f'Ground Truth vs {feature_name} for evaluated embeddings', fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.style.use('seaborn-v0_8')
    continuous_track = feature.flatten()
    ground_truth_track = ground_truth.flatten()
    print(ground_truth_track.shape, np.sum(ground_truth_track))
    continuous_track = (continuous_track - continuous_track.min()) / (continuous_track.max() - continuous_track.min())
    plt.plot(continuous_track, label=f'{feature_name}', color='blue')
    plt.bar(range(len(ground_truth_track)), ground_truth_track, color='blue', alpha=0.5, label='Ground Truth')
    plt.xlabel('Position')
    plt.ylabel('Intensity')
    plt.xlim(500, 1000)
    plt.savefig(f"{save_path}/{feature_name}_genomic_track.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate BatchTopKSAE model")
    parser.add_argument('--data_path', type=str, default='../data/layer_11_embeddings_30subset.npy', help='Path to the data')
    parser.add_argument('--gt_path', type=str, default='../data/chip_exo_57_TF_binding_sites_30subset.npy', help='Path to the ground truth data')
    parser.add_argument('--model_path', type=str, help='Path to save the trained model')
    parser.add_argument("--model-class", type=str, default='BatchTopKSAE', choices=['BatchTopKSAE', 'JumpReLUSAE', 'VanillaSAE','Ellie_SAE'], help='Model class to use')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--loss_function', type=str, default='mse', choices=['mse', 'bce'], help='Loss function to use')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer to use')
    parser.add_argument('--max_grad_norm', type=float, default=100000, help='Maximum gradient norm for clipping')
    parser.add_argument('--eval_size', type=int, default=1000, help='Size of the evaluation set')
    parser.add_argument('--save_path', type=str, default='../data/models', help='Path to save the model and plots')
    parser.add_argument('--embeddings_num', type=int, default=1030*100, help='Number of embeddings to plot')
    parser.add_argument('--l1_coeff', type=float, default=1e-5, help='L1 regularization coefficient for BatchTopKSAE')
    parser.add_argument('--latent_space_multiplier', type=int, default=10, help='Dimensionality of the latent space for BatchTopKSAE')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--top_k', type=int, default=128, help='Top K features to select for BatchTopKSAE')
    parser.add_argument('--top_k_aux', type=int, default=512, help='Top K auxiliary features to select for BatchTopKSAE')
    parser.add_argument('--aux_penalty', type=float, default=(1/32), help='Auxiliary penalty for BatchTopKSAE')
    parser.add_argument('--num_batches_in_buffer', type=int, default=10, help='Number of batches in buffer for BatchTopKSAE')
    parser.add_argument('--n_batches_to_dead', type=int, default=5, help='Number of batches to dead for BatchTopKSAE')
    parser.add_argument('--input_unit_norm', action='store_true', help='Whether to normalize input units for BatchTopKSAE')
    parser.add_argument('--evalmode', action='store_true', help='Whether to run in evaluation mode')
    parser.add_argument('--nowarnings', action='store_true', help='Suppress warnings during training')



    args = parser.parse_args()

    if args.nowarnings:
        import warnings
        warnings.filterwarnings("ignore")
        warnings.filterwarnings("ignore", module="torch")

    if args.evalmode and (args.model_path is None or args.latent_space_multiplier is None):
        raise ValueError("You must provide --model_path and --latent_space_multiplier when running in evaluation mode (--evalmode).")

    cfg = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lr": args.learning_rate,
        "l1_coeff": args.l1_coeff,
        "dtype": torch.float32,
        "act_size": 768,
        "dict_size": 768*args.latent_space_multiplier,
        "device": "cpu" if not torch.cuda.is_available() else "cuda",
        "num_batches_in_buffer": args.num_batches_in_buffer,
        "input_unit_norm": args.input_unit_norm,
        "max_grad_norm": args.max_grad_norm,
        "n_batches_to_dead": args.n_batches_to_dead,
        "n_epochs": args.num_epochs,

        # (Batch)TopKSAE specific
        "top_k": args.top_k,
        "top_k_aux": args.top_k_aux,
        "aux_penalty": args.aux_penalty,

    }

    train_loader, val_loader, data_loader = load_data(args.data_path, cfg["batch_size"], args.eval_size)
    gt_loader = load_ground_truth(args.gt_path, args.eval_size)

    if args.model_path is not None:
        if args.model_class == 'BatchTopKSAE':
            model = BatchTopKSAE(cfg)
        elif args.model_class == 'JumpReLUSAE':
            model = JumpReLUSAE(cfg)
        elif args.model_class == 'VanillaSAE':
            model = VanillaSAE(cfg)
        else:
            raise ValueError(f"Unknown model class: {args.model_class}")
        
        state_dict = torch.load(args.model_path, map_location=cfg["device"])
        
        
        model.eval()
    

    else:
        if args.model_class == 'BatchTopKSAE':
            model = BatchTopKSAE(cfg)
        elif args.model_class == 'JumpReLUSAE':
            model = JumpReLUSAE(cfg)
        elif args.model_class == 'VanillaSAE':
            model = VanillaSAE(cfg)
        else:
            raise ValueError(f"Unknown model class: {args.model_class}")
        
        if args.loss_function == 'mse':
            loss_function = torch.nn.MSELoss()
        elif args.loss_function == 'bce':
            loss_function = torch.nn.BCELoss()
        else:
            raise ValueError(f"Unknown loss function: {args.loss_function}")
        
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9)
            
        model.to(cfg["device"])
        torch.manual_seed(cfg["seed"])

    

    if not args.evalmode:
        model, best_model = train_model(model, train_loader, val_loader, optimizer, loss_function,cfg, cfg["n_epochs"])
        save_model(model, args.save_path)

    auc_matrix, precision_matrix, best_activations_auc, best_activations_precision, best_activations_auc_indeces, best_activations_precision_indeces, ground_truth_track_batch = evaluate_model(model, data_loader, gt_loader)

    plot_best_aucs(auc_matrix, args.save_path)
    plot_best_precision(precision_matrix, args.save_path)

    print("Best AUC Features:")
    print(best_activations_auc.shape)
    print("Best Precision Features:")
    print(best_activations_precision.shape)


    for i in range(10):
        plot_genomic_track(ground_truth_track_batch, best_activations_auc[:,i], f'Best_AUC_Feature_{best_activations_auc_indeces[i]}', args.save_path)
        plot_genomic_track(ground_truth_track_batch, best_activations_precision[:,i], f'Best_Precision_Feature_{best_activations_precision_indeces[i]}', args.save_path)
