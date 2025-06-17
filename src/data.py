import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.utils import degree


def create_train_test_split(data, seed, train_size, test_size):
    """Stratified split of classes"""
    unique_classes = torch.unique(data.y).cpu().numpy()
    
    # Ensure we have enough samples per class
    class_counts = {}
    for cls in unique_classes:
        class_counts[cls] = (data.y == cls).sum().item()
    
    # Filter classes with sufficient samples
    valid_classes = [cls for cls, count in class_counts.items() if count >= 20]
    
    if len(valid_classes) < train_size + test_size:
        print(f"Warning: Not enough valid classes. Using {len(valid_classes)} classes.")
        train_size = int(len(valid_classes) * 0.7)
        test_size = len(valid_classes) - train_size
    
    np.random.seed(seed)
    np.random.shuffle(valid_classes)
    
    train_classes = valid_classes[:train_size]
    test_classes = valid_classes[train_size:train_size + test_size]
    
    return train_classes, test_classes


def create_tasks(data, classes, n_way, k_shot, n_query, n_tasks, balanced=True):
    """Create episodic tasks with optional balancing"""
    tasks = []
    
    n_way = min(len(classes), n_way)
    
    for _ in range(n_tasks):
        if balanced:
            # Ensure balanced sampling across classes
            selected_classes = np.random.choice(classes, n_way, replace=False)
        else:
            # Weight by class frequency
            class_weights = []
            for cls in classes:
                count = (data.y == cls).sum().item()
                class_weights.append(count)
            class_weights = np.array(class_weights)
            class_weights = class_weights / class_weights.sum()
            selected_classes = np.random.choice(
                classes, n_way, replace=False, p=class_weights
            )
        
        support_examples = []
        query_examples = []
        
        for cls in selected_classes:
            cls_indices = torch.where(data.y.cpu() == cls)[0].numpy()
            
            # Ensure we have enough examples
            if len(cls_indices) < k_shot + n_query:
                continue
            
            # Stratified sampling based on node degree
            node_degrees = degree(data.edge_index[0], num_nodes=data.x.size(0))
            cls_node_degrees = node_degrees[cls_indices].cpu()
            
            # Sort by degree and select diverse samples
            sorted_indices = cls_indices[torch.argsort(cls_node_degrees).numpy()]
            step = max(1, len(sorted_indices) // (k_shot + n_query))
            selected_indices = sorted_indices[::step][:k_shot + n_query]
            
            if len(selected_indices) < k_shot + n_query:
                # Fall back to random sampling
                selected_indices = np.random.choice(
                    cls_indices, k_shot + n_query, replace=False
                )
            
            support_examples.extend(selected_indices[:k_shot])
            query_examples.extend(selected_indices[k_shot:k_shot + n_query])
        
        if len(support_examples) > 0 and len(query_examples) > 0:
            tasks.append({
                'support': torch.tensor(support_examples, device=data.x.device),
                'query': torch.tensor(query_examples, device=data.x.device),
                'classes': selected_classes
            })
    
    return tasks