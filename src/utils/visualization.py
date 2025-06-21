import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import traceback

def plot_resnet_architecture(spectrograms_dir):
    try:
        dot = graphviz.Digraph(comment='ResNet Architecture', format='png')
        dot.attr(rankdir='TB', size='8,12')
        
        with dot.subgraph(name='cluster_resnet18') as c:
            c.attr(label='ResNet18 (1-channel input)')
            c.node('input18', 'Input\n75x50x1', shape='box')
            c.node('conv18', 'Conv2D\n64 filters, 7x7, stride=2', shape='box')
            c.node('bn18', 'BatchNorm', shape='box')
            c.node('pool18', 'MaxPool\n3x3, stride=2', shape='box')
            c.node('block18_1', 'ResBlock\n64 filters x2', shape='box')
            c.node('block18_2', 'ResBlock\n128 filters x2, stride=2', shape='box')
            c.node('block18_3', 'ResBlock\n256 filters x2, stride=2', shape='box')
            c.node('block18_4', 'ResBlock\n512 filters x2, stride=2', shape='box')
            c.node('gap18', 'GlobalAvgPool', shape='box')
            c.node('dense18_1', 'Dense\n512, ReLU', shape='box')
            c.node('drop18', 'Dropout\n0.5', shape='box')
            c.node('output18', 'Dense\n4, Softmax', shape='box')
            
            c.edges([('input18', 'conv18'), ('conv18', 'bn18'), ('bn18', 'pool18'),
                     ('pool18', 'block18_1'), ('block18_1', 'block18_2'),
                     ('block18_2', 'block18_3'), ('block18_3', 'block18_4'),
                     ('block18_4', 'gap18'), ('gap18', 'dense18_1'),
                     ('dense18_1', 'drop18'), ('drop18', 'output18')])
        
        with dot.subgraph(name='cluster_resnet50') as c:
            c.attr(label='ResNet50 (3-channel input)')
            c.node('input50', 'Input\n75x50x3', shape='box')
            c.node('resnet50', 'ResNet50\nPretrained, fine-tune last 40 layers', shape='box')
            c.node('gap50', 'GlobalAvgPool', shape='box')
            c.node('dense50_1', 'Dense\n512, ReLU', shape='box')
            c.node('drop50', 'Dropout\n0.5', shape='box')
            c.node('output50', 'Dense\n4, Softmax', shape='box')
            
            c.edges([('input50', 'resnet50'), ('resnet50', 'gap50'),
                     ('gap50', 'dense50_1'), ('dense50_1', 'drop50'),
                     ('drop50', 'output50')])
        
        dot.render(os.path.join(spectrograms_dir, 'figure1_resnet_architecture'), view=False)
    except Exception as e:
        print(f"Error generating Figure 1: {e}. Ensure Graphviz is installed and added to PATH.")
        traceback.print_exc()

def plot_spectrograms(spectrograms_resized, labels, spectrograms_dir):
    classes = ['Normal', 'Crackles', 'Wheezes', 'Both']
    plt.figure(figsize=(12, 8))
    for i, cls in enumerate([0, 1, 2, 3]):
        idx = np.where(labels == cls)[0][0]
        plt.subplot(2, 2, i+1)
        plt.imshow(spectrograms_resized[idx, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
        plt.title(f'({"abcd"[i]}) {classes[cls]}')
        plt.xlabel('Time (0-2.7s)')
        plt.ylabel('Frequency (0-2kHz)')
        plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(spectrograms_dir, 'figure2_spectrograms.png'))
    plt.close()

def plot_confusion_matrices(metrics, spectrograms_dir):
    plt.figure(figsize=(12, 5))
    classes = ['Normal', 'Crackles', 'Wheezes', 'Both']
    for i, model_name in enumerate(['ResNet18', 'ResNet50']):
        avg_cm = np.mean(metrics[model_name]['cm'], axis=0)
        avg_cm = avg_cm / avg_cm.sum(axis=1, keepdims=True) * 100
        plt.subplot(1, 2, i+1)
        sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues', cbar=False,
                    xticklabels=classes, yticklabels=classes)
        plt.title(f'({"ab"[i]}) {model_name} Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(spectrograms_dir, 'figure3_confusion_matrices.png'))
    plt.close()

def plot_performance_comparison(metrics, spectrograms_dir):
    metrics_names = ['Sensitivity', 'Specificity', 'Score', 'Accuracy']
    resnet18_vals = [np.mean(metrics['ResNet18'][m.lower()]) for m in metrics_names]
    resnet50_vals = [np.mean(metrics['ResNet50'][m.lower()]) for m in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    plt.figure(figsize=(8, 6))
    plt.bar(x - width/2, resnet18_vals, width, label='ResNet18', color='blue')
    plt.bar(x + width/2, resnet50_vals, width, label='ResNet50', color='orange')
    plt.xticks(x, metrics_names)
    plt.ylabel('Value')
    plt.title('Performance Comparison')
    plt.ylim(0, 1)
    plt.legend()
    for i, v in enumerate(resnet18_vals):
        plt.text(i - width/2, v + 0.02, f'{v:.4f}', ha='center')
    for i, v in enumerate(resnet50_vals):
        plt.text(i + width/2, v + 0.02, f'{v:.4f}', ha='center')
    plt.savefig(os.path.join(spectrograms_dir, 'figure4_performance_comparison.png'))
    plt.close()

def plot_training_curves(histories, spectrograms_dir):
    plt.figure(figsize=(12, 5))
    
    for i, metric in enumerate(['loss', 'accuracy']):
        plt.subplot(1, 2, i+1)
        for model_name in ['ResNet18', 'ResNet50']:
            min_epochs = min(len(h['accuracy']) for h in histories[model_name])
            train_metric = np.mean([h[metric][:min_epochs] for h in histories[model_name]], axis=0)
            val_metric = np.mean([h[f'val_{metric}'][:min_epochs] for h in histories[model_name]], axis=0)
            
            plt.plot(train_metric, label=f'{model_name} Train', linestyle='-' if model_name == 'ResNet18' else '--')
            plt.plot(val_metric, label=f'{model_name} Val', linestyle='-' if model_name == 'ResNet18' else '--')
        
        plt.title(f'({"ab"[i]}) {"Loss" if metric == "loss" else "Accuracy"} Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss' if metric == 'loss' else 'Accuracy')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(spectrograms_dir, 'figure5_training_curves.png'))
    plt.close()

def plot_classwise_performance(metrics, spectrograms_dir):
    classes = ['Normal', 'Crackles', 'Wheezes', 'Both']
    angles = np.linspace(0, 2 * np.pi, len(classes), endpoint=False).tolist()
    angles += angles[:1]
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    for model_name in ['ResNet18', 'ResNet50']:
        f1_scores = np.mean(metrics[model_name]['f1'], axis=0)
        f1_scores = np.append(f1_scores, f1_scores[0])
        ax.plot(angles, f1_scores, label=model_name, linestyle='-' if model_name == 'ResNet18' else '--')
        ax.fill(angles, f1_scores, alpha=0.1)
    
    ax.set_thetagrids(np.degrees(angles[:-1]), classes)
    ax.set_title('Class-wise F1-Score Comparison')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.savefig(os.path.join(spectrograms_dir, 'figure6_classwise_performance.png'))
    plt.close()

def plot_model_size_tradeoff(models, metrics, spectrograms_dir):
    model_sizes = {
        'ResNet18': sum(np.prod(p.shape) for p in models['ResNet18'].trainable_weights) / 1e6,
        'ResNet50': sum(np.prod(p.shape) for p in models['ResNet50'].trainable_weights) / 1e6
    }
    scores = {
        'ResNet18': np.mean(metrics['ResNet18']['score']),
        'ResNet50': np.mean(metrics['ResNet50']['score'])
    }
    
    plt.figure(figsize=(8, 6))
    for model_name in ['ResNet18', 'ResNet50']:
        plt.scatter(model_sizes[model_name], scores[model_name], label=model_name)
        plt.annotate(model_name, (model_sizes[model_name], scores[model_name]))
    
    plt.xlabel('Model Size (Million Parameters)')
    plt.ylabel('Score')
    plt.title('Model Size vs. Performance Trade-off')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(spectrograms_dir, 'figure7_model_size_tradeoff.png'))
    plt.close()
