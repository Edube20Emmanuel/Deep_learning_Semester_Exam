"""
Plot learning curves for model comparison.
This script plots training and validation metrics for Baseline CNN, ResNet50, and InceptionV3 models.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pickle

def load_training_history(filepath):
    """Load training history from a JSON or pickle file."""
    if not os.path.exists(filepath):
        return None
    
    try:
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                return json.load(f)
        elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def check_variable_exists(var_name, namespace):
    """Check if a variable exists in the given namespace."""
    return var_name in namespace

def plot_learning_curves(
    baseline_train_losses=None, baseline_val_losses=None,
    baseline_train_accs=None, baseline_val_accs=None,
    resnet_train_losses=None, resnet_val_losses=None,
    resnet_train_accs=None, resnet_val_accs=None,
    inception_train_losses=None, inception_val_losses=None,
    inception_train_accs=None, inception_val_accs=None,
    training_times=None,
    save_path=None
):
    """
    Plot learning curves for model comparison.
    
    Parameters:
    -----------
    baseline_train_losses, baseline_val_losses, etc.: Lists or arrays of metrics
    training_times: List of training epochs for each model [baseline, resnet, inception]
    save_path: Optional path to save the figure
    """
    
    # Check which models have data
    has_baseline = (baseline_train_losses is not None and baseline_val_losses is not None)
    has_resnet = (resnet_train_losses is not None and resnet_val_losses is not None)
    has_inception = (inception_train_losses is not None and inception_val_losses is not None)
    
    if not (has_baseline or has_resnet or has_inception):
        raise ValueError("No training data provided. Please provide at least one model's training history.")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Loss curves
    if has_baseline:
        epochs_baseline = range(1, len(baseline_train_losses) + 1)
        axes[0, 0].plot(epochs_baseline, baseline_train_losses, label='Baseline CNN - Train', linewidth=2)
        axes[0, 0].plot(epochs_baseline, baseline_val_losses, label='Baseline CNN - Val', linewidth=2, linestyle='--')
    
    if has_resnet:
        epochs_resnet = range(1, len(resnet_train_losses) + 1)
        axes[0, 0].plot(epochs_resnet, resnet_train_losses, label='ResNet50 - Train', linewidth=2)
        axes[0, 0].plot(epochs_resnet, resnet_val_losses, label='ResNet50 - Val', linewidth=2, linestyle='--')
    
    if has_inception:
        epochs_inception = range(1, len(inception_train_losses) + 1)
        axes[0, 0].plot(epochs_inception, inception_train_losses, label='InceptionV3 - Train', linewidth=2)
        axes[0, 0].plot(epochs_inception, inception_val_losses, label='InceptionV3 - Val', linewidth=2, linestyle='--')
    
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Accuracy curves
    if has_baseline and baseline_train_accs is not None and baseline_val_accs is not None:
        epochs_baseline = range(1, len(baseline_train_accs) + 1)
        axes[0, 1].plot(epochs_baseline, baseline_train_accs, label='Baseline CNN - Train', linewidth=2)
        axes[0, 1].plot(epochs_baseline, baseline_val_accs, label='Baseline CNN - Val', linewidth=2, linestyle='--')
    
    if has_resnet and resnet_train_accs is not None and resnet_val_accs is not None:
        epochs_resnet = range(1, len(resnet_train_accs) + 1)
        axes[0, 1].plot(epochs_resnet, resnet_train_accs, label='ResNet50 - Train', linewidth=2)
        axes[0, 1].plot(epochs_resnet, resnet_val_accs, label='ResNet50 - Val', linewidth=2, linestyle='--')
    
    if has_inception and inception_train_accs is not None and inception_val_accs is not None:
        epochs_inception = range(1, len(inception_train_accs) + 1)
        axes[0, 1].plot(epochs_inception, inception_train_accs, label='InceptionV3 - Train', linewidth=2)
        axes[0, 1].plot(epochs_inception, inception_val_accs, label='InceptionV3 - Val', linewidth=2, linestyle='--')
    
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Model comparison - Final validation accuracy
    models = []
    final_val_accs = []
    colors_bar = []
    
    if has_baseline and baseline_val_accs is not None:
        models.append('Baseline CNN')
        final_val_accs.append(max(baseline_val_accs))
        colors_bar.append('#3498db')
    
    if has_resnet and resnet_val_accs is not None:
        models.append('ResNet50')
        final_val_accs.append(max(resnet_val_accs))
        colors_bar.append('#2ecc71')
    
    if has_inception and inception_val_accs is not None:
        models.append('InceptionV3')
        final_val_accs.append(max(inception_val_accs))
        colors_bar.append('#9b59b6')
    
    if models:
        axes[1, 0].bar(models, final_val_accs, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
        axes[1, 0].set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Model Comparison - Best Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(final_val_accs):
            axes[1, 0].text(i, v + 1, f'{v:.2f}%', ha='center', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylim([0, max(final_val_accs) * 1.2 if final_val_accs else 105])
    
    # Model comparison - Training time (approximate)
    if training_times is None:
        # Use number of epochs as proxy if training_times not provided
        training_times = []
        if has_baseline and baseline_train_losses is not None:
            training_times.append(len(baseline_train_losses))
        if has_resnet and resnet_train_losses is not None:
            training_times.append(len(resnet_train_losses))
        if has_inception and inception_train_losses is not None:
            training_times.append(len(inception_train_losses))
    
    if models and training_times:
        # Match training_times to models
        if len(training_times) < len(models):
            # Pad with last value or use epoch counts
            while len(training_times) < len(models):
                training_times.append(training_times[-1] if training_times else 30)
        elif len(training_times) > len(models):
            training_times = training_times[:len(models)]
        
        axes[1, 1].bar(models, training_times, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
        axes[1, 1].set_ylabel('Training Epochs', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Training Duration Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(training_times):
            axes[1, 1].text(i, v + 0.5, f'{v}', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    return fig


# Main execution - try to get variables from global namespace or load from files
if __name__ == "__main__":
    # Try to get variables from the calling namespace
    import inspect
    frame = inspect.currentframe()
    try:
        # Get the caller's namespace
        caller_frame = frame.f_back
        caller_globals = caller_frame.f_globals if caller_frame else globals()
        caller_locals = caller_frame.f_locals if caller_frame else locals()
        
        # Combine globals and locals
        namespace = {**caller_globals, **caller_locals}
        
        # Extract variables if they exist
        baseline_train_losses = namespace.get('baseline_train_losses')
        baseline_val_losses = namespace.get('baseline_val_losses')
        baseline_train_accs = namespace.get('baseline_train_accs')
        baseline_val_accs = namespace.get('baseline_val_accs')
        
        resnet_train_losses = namespace.get('resnet_train_losses')
        resnet_val_losses = namespace.get('resnet_val_losses')
        resnet_train_accs = namespace.get('resnet_train_accs')
        resnet_val_accs = namespace.get('resnet_val_accs')
        
        inception_train_losses = namespace.get('inception_train_losses')
        inception_val_losses = namespace.get('inception_val_losses')
        inception_train_accs = namespace.get('inception_train_accs')
        inception_val_accs = namespace.get('inception_val_accs')
        
        training_times = namespace.get('training_times')
        
    except:
        # If we can't access caller namespace, set to None
        baseline_train_losses = baseline_val_losses = None
        baseline_train_accs = baseline_val_accs = None
        resnet_train_losses = resnet_val_losses = None
        resnet_train_accs = resnet_val_accs = None
        inception_train_losses = inception_val_losses = None
        inception_train_accs = inception_val_accs = None
        training_times = None
    
    # Try to load from saved files if variables don't exist
    if baseline_train_losses is None:
        history = load_training_history('baseline_history.json') or load_training_history('baseline_history.pkl')
        if history:
            baseline_train_losses = history.get('train_loss', history.get('loss'))
            baseline_val_losses = history.get('val_loss')
            baseline_train_accs = history.get('train_acc', history.get('accuracy'))
            baseline_val_accs = history.get('val_acc', history.get('val_accuracy'))
    
    if resnet_train_losses is None:
        history = load_training_history('resnet_history.json') or load_training_history('resnet_history.pkl')
        if history:
            resnet_train_losses = history.get('train_loss', history.get('loss'))
            resnet_val_losses = history.get('val_loss')
            resnet_train_accs = history.get('train_acc', history.get('accuracy'))
            resnet_val_accs = history.get('val_acc', history.get('val_accuracy'))
    
    if inception_train_losses is None:
        history = load_training_history('inception_history.json') or load_training_history('inception_history.pkl')
        if history:
            inception_train_losses = history.get('train_loss', history.get('loss'))
            inception_val_losses = history.get('val_loss')
            inception_train_accs = history.get('train_acc', history.get('accuracy'))
            inception_val_accs = history.get('val_acc', history.get('val_accuracy'))
    
    # Plot the curves
    try:
        plot_learning_curves(
            baseline_train_losses=baseline_train_losses,
            baseline_val_losses=baseline_val_losses,
            baseline_train_accs=baseline_train_accs,
            baseline_val_accs=baseline_val_accs,
            resnet_train_losses=resnet_train_losses,
            resnet_val_losses=resnet_val_losses,
            resnet_train_accs=resnet_train_accs,
            resnet_val_accs=resnet_val_accs,
            inception_train_losses=inception_train_losses,
            inception_val_losses=inception_val_losses,
            inception_train_accs=inception_train_accs,
            inception_val_accs=inception_val_accs,
            training_times=training_times
        )
    except ValueError as e:
        print(f"Error: {e}")
        print("\nTo use this script, you need to either:")
        print("1. Define the training history variables in your code before calling this script")
        print("2. Save training history to JSON/PKL files (baseline_history.json, resnet_history.json, inception_history.json)")
        print("3. Pass the variables directly to the plot_learning_curves() function")



