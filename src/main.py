import torch
import random
import numpy as np
from tqdm import tqdm
import json
import os

from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import NormalizeFeatures

from data import create_tasks, create_train_test_split
from models import GNNEncoder
from train import train, evaluate

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


def get_config(dataset_name, dataset):
    """Get optimized configuration for each dataset"""
    base_config = {
        'input_dim': dataset.num_features,
        'hidden_dim': 256,
        'output_dim': dataset.num_classes,
        'n_epochs': 200,
        'learning_rate': 0.001,
        'weight_decay': 5e-4,
        'n_way': 5,
        'k_shot': 5,
        'n_query': 10,
        'n_train_tasks': 200,
        'n_test_tasks': 100,
        'batch_size': 20,
        'max_negative_samples': 5,
        'inner_steps': 10,
        'lr_inner': 0.01,
        'lr_decay': 0.95,
        'temperature': 0.1,
        'cl_weight': 0.5,
        'grad_clip': 1.0,
        'encoder_type': 'GCN',
        'pool_type': 'mean_pool',
        'num_layers': 3,
        'dropout': 0.5,
        'use_norm': 'batch'
    }
    
    # Dataset-specific configurations
    dataset_configs = {
        'Cora': {
            'train_class_size': 4,
            'test_class_size': 3,
            'hidden_dim': 256,
            'lr_inner': 0.01,
            'encoder_type': 'GAT',
        },
        'CiteSeer': {
            'train_class_size': 3,
            'test_class_size': 3,
            'hidden_dim': 256,
            'n_way': 3,
            'encoder_type': 'GraphSAGE',
        },
        'PubMed': {
            'train_class_size': 2,
            'test_class_size': 1,
            'hidden_dim': 512,
            'n_way': 2,
            'k_shot': 10,
            'encoder_type': 'GCN',
        },
        'Computers': {
            'train_class_size': 7,
            'test_class_size': 3,
            'hidden_dim': 512,
            'n_way': 7,
            'batch_size': 10,
            'encoder_type': 'GIN',
        },
        'Photo': {
            'train_class_size': 5,
            'test_class_size': 3,
            'hidden_dim': 512,
            'n_way': 5,
            'encoder_type': 'GIN',
        }
    }
    
    # Update base config with dataset-specific settings
    if dataset_name in dataset_configs:
        base_config.update(dataset_configs[dataset_name])
    
    return base_config


def main(dataset_name, seed, device):
    print(f"\n{'='*50}")
    print(f"Training on {dataset_name} dataset")
    print(f"{'='*50}\n")
    
    # Load dataset
    dataset = load_dataset(dataset_name)
    data = dataset[0].to(device)
    data.edge_index = to_undirected(data.edge_index)
    
    # Get configuration
    config = {dataset_name: get_config(dataset_name, dataset)}
    
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
    
    # Initialize model
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
    
    # Initialize optimizer with weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=config[dataset_name]['learning_rate'],
        weight_decay=config[dataset_name]['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=config[dataset_name]['n_epochs'],
        eta_min=1e-6
    )
    
    # Training history
    history = {
        'support_loss': [],
        'query_loss': [],
        'test_loss': [],
        'test_acc': [],
        'test_confidence': []
    }
    
    best_test_acc = 0
    best_epoch = 0
    patience = 20
    patience_counter = 0
    
    print("\nStarting training...")
    print(f"Configuration: {json.dumps(config[dataset_name], indent=2)}")
    
    # META TRAINING
    for epoch in tqdm(range(config[dataset_name]['n_epochs']), desc="Training"):
        support_set_loss = 0
        query_set_loss = 0
        n_batches = 0
        
        # Shuffle tasks each epoch
        np.random.shuffle(train_tasks)
        
        # Process tasks in batches
        for i in range(0, len(train_tasks), config[dataset_name]['batch_size']):
            batch_tasks = train_tasks[i:i + config[dataset_name]['batch_size']]
            batch_support_loss, batch_query_loss = train(
                model, dataset_name, data, batch_tasks, optimizer, config, device
            )
            support_set_loss += batch_support_loss
            query_set_loss += batch_query_loss
            n_batches += 1
        
        support_set_loss = support_set_loss / max(1, n_batches)
        query_set_loss = query_set_loss / max(1, n_batches)
        
        history['support_loss'].append(support_set_loss)
        history['query_loss'].append(query_set_loss)
        
        # Periodic evaluation
        if (epoch + 1) % 10 == 0:
            # Evaluate on test tasks
            test_loss, test_acc, test_confidence = evaluate(
                model, dataset_name, data, test_tasks[:50], config, device
            )
            
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['test_confidence'].append(test_confidence)
            
            print(f"\nEpoch {epoch+1}/{config[dataset_name]['n_epochs']}")
            print(f"Meta TRAINING - Support Loss: {support_set_loss:.4f}, Query Loss: {query_set_loss:.4f}")
            print(f"Meta TEST - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, Confidence: {test_confidence:.4f}")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch + 1
                patience_counter = 0
                # Save checkpoint
                checkpoint_path = f'checkpoints/best_model_{dataset_name}.pt'
                os.makedirs('checkpoints', exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_test_acc,
                    'config': config[dataset_name]
                }, checkpoint_path)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Update learning rate
        scheduler.step()
    
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Best test accuracy: {best_test_acc:.4f} at epoch {best_epoch}")
    print(f"{'='*50}\n")
    
    # Load best model for final evaluation
    checkpoint_path = f'checkpoints/best_model_{dataset_name}.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # FINAL META TESTING
    print("Final evaluation on all test tasks...")
    final_test_loss, final_test_acc, final_test_confidence = evaluate(
        model, dataset_name, data, test_tasks, config, device
    )
    
    print(f"\nFINAL TEST RESULTS:")
    print(f"Loss: {final_test_loss:.4f}")
    print(f"Accuracy: {final_test_acc:.4f}")
    print(f"Average Confidence: {final_test_confidence:.4f}")
    print(f"{'='*50}\n")
    
    return final_test_acc, history


def run_experiments(dataset_names, seeds, device):
    """Run experiments across multiple datasets and seeds"""
    results = {}
    
    for dataset_name in dataset_names:
        results[dataset_name] = []
        
        for seed in seeds:
            print(f"\n{'*'*70}")
            print(f"Running {dataset_name} with seed {seed}")
            print(f"{'*'*70}")
            
            set_seed(seed)
            acc, history = main(dataset_name, seed, device)
            results[dataset_name].append(acc)
            
            # Save results
            os.makedirs('results', exist_ok=True)
            torch.save({
                'accuracy': acc,
                'history': history,
                'seed': seed
            }, f'results/results_{dataset_name}_seed{seed}.pt')
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    for dataset_name, accs in results.items():
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"{dataset_name}: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  Individual runs: {[f'{acc:.4f}' for acc in accs]}")
    
    return results


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
    seeds = [42, 123, 456] 
    
    # Set initial seed
    set_seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run single experiment
    if len(dataset_names) == 1 and len(seeds) == 1:
        main(dataset_names[0], seeds[0], device)
    else:
        # Run multiple experiments
        run_experiments(dataset_names, seeds, device)