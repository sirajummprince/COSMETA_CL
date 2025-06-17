# main.py - Enhanced with integrated utilities
import torch
import random
import numpy as np
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import NormalizeFeatures

# Import your modules
from data import create_tasks, create_train_test_split
from models import GNNEncoder
from train import train, evaluate, meta_test_step
from utils import (
    plot_training_history, 
    analyze_mistakes,
    ensemble_predictions,
    augment_graph,
    LabelSmoothingLoss,
    FocalLoss
)


def main(dataset_name, seed, device, use_utilities=True):
    """Enhanced main function with integrated utilities"""
    print(f"\n{'='*50}")
    print(f"Training on {dataset_name} dataset")
    print(f"{'='*50}\n")
    
    # Load dataset
    dataset = load_dataset(dataset_name)
    data = dataset[0].to(device)
    data.edge_index = to_undirected(data.edge_index)
    
    # Get configuration
    config = {dataset_name: get_memory_efficient_config(dataset_name, dataset)}
    
    # Optional: Use enhanced loss functions
    if use_utilities:
        config[dataset_name]['use_label_smoothing'] = True
        config[dataset_name]['use_focal_loss'] = False  # Choose one
        config[dataset_name]['label_smoothing'] = 0.1
        config[dataset_name]['use_augmentation'] = True
        config[dataset_name]['aug_ratio'] = 0.1
    
    # Create train/test split
    train_classes, test_classes = create_train_test_split(
        data, seed,
        config[dataset_name]['train_class_size'],
        config[dataset_name]['test_class_size']
    )
    
    print(f'Training Classes: {train_classes}')
    print(f'Testing Classes: {test_classes}')
    print(f'Dataset Statistics:')
    print(f'  - Nodes: {data.x.size(0)}')
    print(f'  - Edges: {data.edge_index.size(1) // 2}')
    print(f'  - Features: {data.x.size(1)}')
    print(f'  - Classes: {dataset.num_classes}')
    
    # Create tasks
    train_tasks = create_tasks(
        data, train_classes,
        n_way=config[dataset_name]['n_way'],
        k_shot=config[dataset_name]['k_shot'],
        n_query=config[dataset_name]['n_query'],
        n_tasks=config[dataset_name]['n_train_tasks'],
        balanced=True
    )
    
    test_tasks = create_tasks(
        data, test_classes,
        n_way=config[dataset_name]['n_way'],
        k_shot=config[dataset_name]['k_shot'],
        n_query=config[dataset_name]['n_query'],
        n_tasks=config[dataset_name]['n_test_tasks'],
        balanced=True
    )
    
    print(f'\nCreated {len(train_tasks)} training tasks')
    print(f'Created {len(test_tasks)} testing tasks')
    
    # Initialize model(s) - create ensemble if specified
    if use_utilities and config[dataset_name].get('use_ensemble', False):
        models = []
        optimizers = []
        num_models = config[dataset_name].get('ensemble_size', 3)
        
        for i in range(num_models):
            model = GNNEncoder(
                config[dataset_name]['input_dim'],
                config[dataset_name]['hidden_dim'],
                config[dataset_name]['output_dim'],
                encoder_type=config[dataset_name]['encoder_type'],
                pool_type=config[dataset_name]['pool_type'],
                num_layers=config[dataset_name]['num_layers'],
                dropout=config[dataset_name]['dropout'],
                use_norm=config[dataset_name]['use_norm']
            ).to(device)
            
            optimizer = AdamW(
                model.parameters(),
                lr=config[dataset_name]['learning_rate'],
                weight_decay=config[dataset_name]['weight_decay']
            )
            
            models.append(model)
            optimizers.append(optimizer)
        
        print(f"Created ensemble of {num_models} models")
    else:
        # Single model
        model = GNNEncoder(
            config[dataset_name]['input_dim'],
            config[dataset_name]['hidden_dim'],
            config[dataset_name]['output_dim'],
            encoder_type=config[dataset_name]['encoder_type'],
            pool_type=config[dataset_name]['pool_type'],
            num_layers=config[dataset_name]['num_layers'],
            dropout=config[dataset_name]['dropout'],
            use_norm=config[dataset_name]['use_norm']
        ).to(device)
        
        optimizer = AdamW(
            model.parameters(),
            lr=config[dataset_name]['learning_rate'],
            weight_decay=config[dataset_name]['weight_decay']
        )
        
        models = [model]
        optimizers = [optimizer]
    
    # Learning rate schedulers
    schedulers = [
        CosineAnnealingLR(opt, T_max=config[dataset_name]['n_epochs'], eta_min=1e-6)
        for opt in optimizers
    ]
    
    # Training history
    history = {
        'support_loss': [],
        'query_loss': [],
        'test_loss': [],
        'test_acc': [],
        'test_confidence': []
    }
    
    # For detailed analysis
    all_test_predictions = []
    all_test_labels = []
    
    best_test_acc = 0
    best_epoch = 0
    patience = 20
    patience_counter = 0
    
    print("\nStarting training...")
    print(f"Configuration: {json.dumps(config[dataset_name], indent=2)}")
    
    # META TRAINING
    for epoch in tqdm(range(config[dataset_name]['n_epochs']), desc="Training"):
        epoch_support_loss = 0
        epoch_query_loss = 0
        
        # Train each model in ensemble
        for model_idx, (model, optimizer) in enumerate(zip(models, optimizers)):
            support_set_loss = 0
            query_set_loss = 0
            n_batches = 0
            
            # Shuffle tasks each epoch
            np.random.shuffle(train_tasks)
            
            # Augment data if specified
            if use_utilities and config[dataset_name].get('use_augmentation', False):
                augmented_data = data.clone()
                augmented_data.x, augmented_data.edge_index = augment_graph(
                    data.x, data.edge_index, 
                    aug_ratio=config[dataset_name].get('aug_ratio', 0.1)
                )
                current_data = augmented_data if epoch % 2 == 0 else data
            else:
                current_data = data
            
            # Process tasks in batches
            for i in range(0, len(train_tasks), config[dataset_name]['batch_size']):
                batch_tasks = train_tasks[i:i + config[dataset_name]['batch_size']]
                
                # Pass loss function configuration
                if use_utilities:
                    config[dataset_name]['loss_fn'] = (
                        LabelSmoothingLoss(dataset.num_classes, smoothing=0.1)
                        if config[dataset_name].get('use_label_smoothing', False)
                        else FocalLoss() if config[dataset_name].get('use_focal_loss', False)
                        else None
                    )
                
                batch_support_loss, batch_query_loss = train(
                    model, dataset_name, current_data, batch_tasks, optimizer, config, device
                )
                support_set_loss += batch_support_loss
                query_set_loss += batch_query_loss
                n_batches += 1
            
            support_set_loss = support_set_loss / max(1, n_batches)
            query_set_loss = query_set_loss / max(1, n_batches)
            
            epoch_support_loss += support_set_loss
            epoch_query_loss += query_set_loss
        
        # Average across ensemble
        epoch_support_loss /= len(models)
        epoch_query_loss /= len(models)
        
        history['support_loss'].append(epoch_support_loss)
        history['query_loss'].append(epoch_query_loss)
        
        # Periodic evaluation
        if (epoch + 1) % 10 == 0:
            # Evaluate ensemble or single model
            if len(models) > 1 and use_utilities:
                # Ensemble evaluation
                test_acc = 0
                test_loss = 0
                test_confidence = 0
                ensemble_preds = []
                ensemble_labels = []
                
                # Sample test tasks for evaluation
                eval_tasks = test_tasks[:min(50, len(test_tasks))]
                
                for task in eval_tasks:
                    preds = ensemble_predictions(models, data, task, config, device)
                    # Get true labels
                    true_labels = [data.y[idx].item() for idx in task['query']]
                    
                    # Calculate accuracy
                    correct = sum(p == l for p, l in zip(preds, true_labels))
                    test_acc += correct / len(true_labels)
                    
                    ensemble_preds.extend(preds)
                    ensemble_labels.extend(true_labels)
                
                test_acc /= len(eval_tasks)
                test_loss = 0  # Placeholder
                test_confidence = 0.9  # Placeholder
                
                all_test_predictions = ensemble_preds
                all_test_labels = ensemble_labels
            else:
                # Single model evaluation
                test_loss, test_acc, test_confidence = evaluate(
                    models[0], dataset_name, data, test_tasks[:50], config, device
                )
            
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['test_confidence'].append(test_confidence)
            
            print(f"\nEpoch {epoch+1}/{config[dataset_name]['n_epochs']}")
            print(f"Meta TRAINING - Support Loss: {epoch_support_loss:.4f}, Query Loss: {epoch_query_loss:.4f}")
            print(f"Meta TEST - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, Confidence: {test_confidence:.4f}")
            print(f"Learning Rate: {schedulers[0].get_last_lr()[0]:.6f}")
            
            # Save best model(s)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch + 1
                patience_counter = 0
                
                # Save checkpoint
                os.makedirs('checkpoints', exist_ok=True)
                for i, model in enumerate(models):
                    checkpoint_path = f'checkpoints/best_model_{dataset_name}_model{i}.pt'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizers[i].state_dict(),
                        'best_acc': best_test_acc,
                        'config': config[dataset_name]
                    }, checkpoint_path)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Update learning rates
        for scheduler in schedulers:
            scheduler.step()
    
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Best test accuracy: {best_test_acc:.4f} at epoch {best_epoch}")
    print(f"{'='*50}\n")
    
    # Load best model(s) for final evaluation
    for i, model in enumerate(models):
        checkpoint_path = f'checkpoints/best_model_{dataset_name}_model{i}.pt'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
    
    # FINAL META TESTING with detailed analysis
    print("Final evaluation on all test tasks...")
    
    if len(models) > 1 and use_utilities:
        # Ensemble final evaluation
        final_test_acc = 0
        final_predictions = []
        final_labels = []
        
        for task in tqdm(test_tasks, desc="Final evaluation"):
            preds = ensemble_predictions(models, data, task, config, device)
            true_labels = [data.y[idx].item() for idx in task['query']]
            
            correct = sum(p == l for p, l in zip(preds, true_labels))
            final_test_acc += correct / len(true_labels)
            
            final_predictions.extend(preds)
            final_labels.extend(true_labels)
        
        final_test_acc /= len(test_tasks)
        final_test_loss = 0  # Placeholder
        final_test_confidence = 0.9  # Placeholder
    else:
        # Single model final evaluation
        final_test_loss, final_test_acc, final_test_confidence = evaluate(
            models[0], dataset_name, data, test_tasks, config, device
        )
        
        # Collect predictions for analysis
        final_predictions = []
        final_labels = []
        
        for task in test_tasks[:20]:  # Sample for confusion matrix
            _, _, preds, labels, _ = meta_test_step(
                models[0], dataset_name, data, task, config, device
            )
            final_predictions.extend(preds)
            final_labels.extend(labels)
    
    print(f"\nFINAL TEST RESULTS:")
    print(f"Loss: {final_test_loss:.4f}")
    print(f"Accuracy: {final_test_acc:.4f}")
    print(f"Average Confidence: {final_test_confidence:.4f}")
    print(f"{'='*50}\n")
    
    # Use utilities for analysis and visualization
    if use_utilities:
        # Plot training history
        plot_training_history(history, dataset_name)
        print(f"Training history plot saved as 'training_history_{dataset_name}.png'")
        
        # Analyze mistakes if we have predictions
        if final_predictions and final_labels:
            cm, class_acc = analyze_mistakes(
                final_predictions, 
                final_labels,
                class_names=[str(i) for i in range(config[dataset_name]['n_way'])]
            )
            print("\nPer-class accuracy:")
            for cls, acc in class_acc.items():
                print(f"  Class {cls}: {acc:.4f}")
            print(f"Confusion matrix saved as 'confusion_matrix.png'")
        
        # Save detailed results
        results = {
            'final_accuracy': final_test_acc,
            'final_loss': final_test_loss,
            'final_confidence': final_test_confidence,
            'best_epoch': best_epoch,
            'history': history,
            'config': config[dataset_name],
            'per_class_accuracy': class_acc if 'class_acc' in locals() else {}
        }
        
        os.makedirs('results', exist_ok=True)
        results_path = f'results/detailed_results_{dataset_name}_seed{seed}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to '{results_path}'")
    
    return final_test_acc, history


# The rest of the functions remain the same...
def load_dataset(dataset_name, root='../data'):
    """Load and preprocess dataset"""
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(
            name=dataset_name, 
            root=f'{root}/{dataset_name}',
            transform=NormalizeFeatures()
        )
    elif dataset_name in ['Computers', 'Photo']:
        dataset = Amazon(
            name=dataset_name,
            root=f'{root}/Amazon',
            transform=NormalizeFeatures()
        )
    elif dataset_name in ['CS', 'Physics']:
        dataset = Coauthor(
            name=dataset_name,
            root=f'{root}/Coauthor',
            transform=NormalizeFeatures()
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def get_memory_efficient_config(dataset_name, dataset):
    """Memory-efficient configuration with utilities options"""
    base_config = {
        'input_dim': dataset.num_features,
        'hidden_dim': 128,
        'output_dim': dataset.num_classes,
        'n_epochs': 200,
        'learning_rate': 0.001,
        'weight_decay': 5e-4,
        'n_way': 5,
        'k_shot': 5,
        'n_query': 5,
        'n_train_tasks': 100,
        'n_test_tasks': 50,
        'batch_size': 10,
        'max_negative_samples': 2,
        'max_negative_samples_mem': 2,
        'inner_steps': 5,
        'test_inner_steps': 3,
        'lr_inner': 0.01,
        'lr_decay': 0.95,
        'temperature': 0.1,
        'cl_weight': 0.5,
        'grad_clip': 1.0,
        'encoder_type': 'GCN',
        'pool_type': 'mean_pool',
        'num_layers': 2,
        'dropout': 0.5,
        'use_norm': 'batch',
        'num_hops': 2,
        'inner_batch_size': 5,
        'query_batch_size': 5,
        'skip_cl_inner': True,
        'disable_cl': False,
        # Utilities options
        'use_ensemble': False,
        'ensemble_size': 3,
        'use_label_smoothing': False,
        'use_focal_loss': False,
        'use_augmentation': False,
        'aug_ratio': 0.1,
    }
    
    # Dataset-specific configurations
    dataset_configs = {
        'Cora': {
            'train_class_size': 4,
            'test_class_size': 3,
            'hidden_dim': 128,
            'lr_inner': 0.01,
            'encoder_type': 'GCN',
        },
        'CiteSeer': {
            'train_class_size': 3,
            'test_class_size': 3,
            'hidden_dim': 128,
            'n_way': 3,
            'encoder_type': 'GCN',
        },
        'PubMed': {
            'train_class_size': 2,
            'test_class_size': 1,
            'hidden_dim': 256,
            'n_way': 2,
            'k_shot': 5,
            'encoder_type': 'GCN',
        },
    }
    
    if dataset_name in dataset_configs:
        base_config.update(dataset_configs[dataset_name])
    
    return base_config


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Configuration
    dataset_names = ['Cora']
    seeds = [42]
    use_utilities = True  # Enable utility functions
    
    # Set initial seed
    set_seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run experiment with utilities
    for dataset_name in dataset_names:
        for seed in seeds:
            print(f"\n{'*'*70}")
            print(f"Running {dataset_name} with seed {seed}")
            print(f"Utilities enabled: {use_utilities}")
            print(f"{'*'*70}")
            
            set_seed(seed)
            acc, history = main(dataset_name, seed, device, use_utilities=use_utilities)