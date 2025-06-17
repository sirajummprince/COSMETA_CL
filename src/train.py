import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.func import functional_call
from torch_geometric.utils import k_hop_subgraph


def get_subgraph(node_idx, edge_index, x, num_hops=2):
    """Extract k-hop subgraph around node"""
    node_idx = torch.tensor([node_idx], device=edge_index.device)
    subset, sub_edge_index, _, _ = k_hop_subgraph(
        node_idx, 
        num_hops, 
        edge_index,
        relabel_nodes=True,
        num_nodes=x.size(0)
    )
    return sub_edge_index, subset


def compute_contrastive_loss(anchor_emb, positive_emb, negative_embs, temperature=0.1):
    """InfoNCE contrastive loss"""
    # Normalize embeddings
    anchor_emb = F.normalize(anchor_emb, dim=-1)
    positive_emb = F.normalize(positive_emb, dim=-1)
    negative_embs = F.normalize(negative_embs, dim=-1)
    
    # Compute similarities
    pos_sim = torch.sum(anchor_emb * positive_emb) / temperature
    neg_sims = torch.matmul(negative_embs, anchor_emb.T) / temperature
    
    # Compute loss
    logits = torch.cat([pos_sim.unsqueeze(0), neg_sims.squeeze()])
    labels = torch.zeros(1, dtype=torch.long, device=anchor_emb.device)
    
    return F.cross_entropy(logits.unsqueeze(0), labels)


def compute_loss_functional(fmodel, params, buffers, dataset_name, data, idx, 
                          task_idx, config, device, return_pred=False):
    """Enhanced loss computation with multiple objectives"""
    ce_criterion = nn.CrossEntropyLoss()
    
    # Extract subgraph
    anchor_edge_index, subset = get_subgraph(idx, data.edge_index, data.x, num_hops=3)
    
    # Forward pass
    anchor_node_embedding, anchor_graph_embedding = functional_call(
        fmodel, (params, buffers), (data.x.to(device), anchor_edge_index.to(device))
    )
    
    # Classification loss
    ce_loss = ce_criterion(anchor_node_embedding[idx].unsqueeze(0), data.y[idx].unsqueeze(0).to(device))
    
    if return_pred:
        prediction = anchor_node_embedding[idx].argmax(dim=-1).item()
        label = data.y[idx].item()
        return ce_loss, prediction, label
    
    # Contrastive learning
    loss = ce_loss
    
    # Find positive examples (same class)
    positive_indices = task_idx[data.y[task_idx] == data.y[idx]]
    positive_indices = positive_indices[positive_indices != idx]
    
    if len(positive_indices) > 0:
        positive_idx = positive_indices[torch.randperm(len(positive_indices))[0]]
        
        pos_edge_index, _ = get_subgraph(positive_idx, data.edge_index, data.x)
        _, positive_graph_embedding = functional_call(
            fmodel, (params, buffers), (data.x.to(device), pos_edge_index.to(device))
        )
        
        # Find negative examples (different classes)
        negative_indices = task_idx[data.y[task_idx] != data.y[idx]]
        n_neg = min(len(negative_indices), config[dataset_name]['max_negative_samples'])
        
        if n_neg > 0:
            neg_indices = np.random.choice(negative_indices.cpu(), n_neg, replace=False)
            negative_graph_embeddings = []
            
            for neg_idx in neg_indices:
                neg_edge_index, _ = get_subgraph(neg_idx, data.edge_index, data.x)
                _, neg_graph_embedding = functional_call(
                    fmodel, (params, buffers), (data.x.to(device), neg_edge_index.to(device))
                )
                negative_graph_embeddings.append(neg_graph_embedding)
            
            negative_graph_embeddings = torch.vstack(negative_graph_embeddings)
            
            # Compute contrastive loss
            cl_loss = compute_contrastive_loss(
                anchor_graph_embedding, 
                positive_graph_embedding, 
                negative_graph_embeddings,
                temperature=config[dataset_name].get('temperature', 0.1)
            )
            
            loss = ce_loss + config[dataset_name].get('cl_weight', 0.5) * cl_loss
    
    return loss


def meta_learning_step(model, dataset_name, data, task, optimizer, config, device):
    """Enhanced meta-learning with adaptive learning rates"""
    task_support_loss = 0
    task_query_loss = 0
    
    # Prepare model for functional use
    params = OrderedDict(model.named_parameters())
    buffers = OrderedDict(model.named_buffers())
    
    support_idx = task['support']
    query_idx = task['query']
    
    fast_params = {name: p.clone() for name, p in params.items()}
    lr_inner = config[dataset_name]['lr_inner']
    inner_steps = config[dataset_name]['inner_steps']
    
    # Adaptive inner loop learning rate
    lr_decay = config[dataset_name].get('lr_decay', 0.9)
    
    # Inner loop optimization
    for step in range(inner_steps):
        accumulated_grads = {name: torch.zeros_like(p) for name, p in fast_params.items()}
        total_loss = 0
        
        # Compute gradients for support set
        for idx in support_idx:
            loss = compute_loss_functional(
                model, fast_params, buffers, dataset_name, 
                data, idx, support_idx, config, device
            )
            
            grads = torch.autograd.grad(
                loss, fast_params.values(), 
                create_graph=True,  # Enable higher-order gradients
                allow_unused=True
            )
            
            grads = [
                g if g is not None else torch.zeros_like(p)
                for g, p in zip(grads, fast_params.values())
            ]
            
            for (name, _), g in zip(fast_params.items(), grads):
                accumulated_grads[name] += g
            
            total_loss += loss.item()
        
        # Average gradients
        avg_grads = {name: g / len(support_idx) for name, g in accumulated_grads.items()}
        
        # Update with decaying learning rate
        current_lr = lr_inner * (lr_decay ** step)
        fast_params = OrderedDict({
            name: p - current_lr * avg_grads[name]
            for name, p in fast_params.items()
        })
        
        task_support_loss += total_loss / len(support_idx)
    
    task_support_loss = task_support_loss / max(1, inner_steps)
    
    # Outer loop on query set
    query_losses = []
    for idx in query_idx:
        query_loss = compute_loss_functional(
            model, fast_params, buffers, dataset_name, 
            data, idx, query_idx, config, device
        )
        query_losses.append(query_loss)
        task_query_loss += query_loss.item()
    
    # Compute meta-gradient
    meta_loss = torch.stack(query_losses).mean()
    
    optimizer.zero_grad()
    meta_loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), config[dataset_name].get('grad_clip', 1.0))
    
    optimizer.step()
    
    task_query_loss = task_query_loss / max(1, len(query_idx))
    
    return task_support_loss, task_query_loss


def train(model, dataset_name, data, batch_tasks, optimizer, config, device):
    """Training with multiple tasks"""
    model.train()
    total_support_loss = 0
    total_query_loss = 0
    n_batches = 0
    
    for task in batch_tasks:
        batch_support_loss, batch_query_loss = meta_learning_step(
            model, dataset_name, data, task, optimizer, config, device
        )
        total_support_loss += batch_support_loss
        total_query_loss += batch_query_loss
        n_batches += 1
    
    return total_support_loss / max(1, n_batches), total_query_loss / max(1, n_batches)


def meta_test_step(model, dataset_name, data, task, config, device):
    """Enhanced testing with confidence scores"""
    model.eval()
    
    params = OrderedDict(model.named_parameters())
    buffers = OrderedDict(model.named_buffers())
    
    support_idx = task['support']
    query_idx = task['query']
    
    fast_params = {name: p.clone() for name, p in params.items()}
    lr_inner = config[dataset_name]['lr_inner'] * 0.5  # Lower LR for test
    inner_steps = config[dataset_name]['inner_steps']
    
    # Adapt to support set
    with torch.enable_grad():
        for _ in range(inner_steps):
            accumulated_grads = {name: torch.zeros_like(p) for name, p in fast_params.items()}
            
            for idx in support_idx:
                loss = compute_loss_functional(
                    model, fast_params, buffers, dataset_name, 
                    data, idx, support_idx, config, device
                )
                
                grads = torch.autograd.grad(
                    loss, fast_params.values(), 
                    create_graph=False, allow_unused=True
                )
                
                grads = [
                    g if g is not None else torch.zeros_like(p)
                    for g, p in zip(grads, fast_params.values())
                ]
                
                for (name, _), g in zip(fast_params.items(), grads):
                    accumulated_grads[name] += g
            
            avg_grads = {name: g / len(support_idx) for name, g in accumulated_grads.items()}
            fast_params = OrderedDict({
                name: p - lr_inner * avg_grads[name]
                for name, p in fast_params.items()
            })
    
    # Evaluate on query set
    total_loss = 0
    correct = 0
    predictions = []
    true_labels = []
    confidences = []
    
    with torch.no_grad():
        for idx in query_idx:
            loss, pred, label = compute_loss_functional(
                model, fast_params, buffers, dataset_name, 
                data, idx, query_idx, config, device, return_pred=True
            )
            
            total_loss += loss.item()
            
            if pred == label:
                correct += 1
            
            predictions.append(pred)
            true_labels.append(label)
            
            # Get confidence
            edge_index, _ = get_subgraph(idx, data.edge_index, data.x)
            logits, _ = functional_call(
                model, (fast_params, buffers), 
                (data.x.to(device), edge_index.to(device))
            )
            probs = F.softmax(logits[idx], dim=-1)
            confidences.append(probs.max().item())
    
    accuracy = correct / max(1, len(query_idx))
    avg_confidence = np.mean(confidences)
    
    return total_loss / len(query_idx), accuracy, predictions, true_labels, avg_confidence


def evaluate(model, dataset_name, data, batch_tasks, config, device):
    """Comprehensive evaluation"""
    avg_loss = 0
    avg_acc = 0
    avg_confidence = 0
    n_batches = 0
    
    all_predictions = []
    all_labels = []
    
    for task in batch_tasks:
        loss, acc, predictions, labels, confidence = meta_test_step(
            model, dataset_name, data, task, config, device
        )
        avg_loss += loss
        avg_acc += acc
        avg_confidence += confidence
        n_batches += 1
        
        all_predictions.extend(predictions)
        all_labels.extend(labels)
    
    avg_loss /= n_batches
    avg_acc /= n_batches
    avg_confidence /= n_batches
    
    return avg_loss, avg_acc, avg_confidence