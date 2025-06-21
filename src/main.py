import os
import numpy as np
from utils.models import build_resnet18, build_resnet50, focal_loss
from utils.metrics import compute_metrics
from utils.visualization import (
    plot_resnet_architecture, plot_spectrograms, plot_confusion_matrices,
    plot_performance_comparison, plot_training_curves, plot_classwise_performance,
    plot_model_size_tradeoff
)
from sklearn.model_selection import GroupKFold
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Paths
base_dir = r"C:\Dataset\ICBHI_final_database"
spectrograms_dir = os.path.join(base_dir, "spectrograms")

# Load spectrograms and labels
try:
    spectrograms_resized = np.load(os.path.join(spectrograms_dir, "spectrograms_resized.npy"))
    labels = np.load(os.path.join(spectrograms_dir, "labels.npy"))
    patient_ids = np.load(os.path.join(spectrograms_dir, "patient_ids.npy"))
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure spectrograms_resized.npy, labels.npy, and patient_ids.npy exist in {spectrograms_dir}")
    raise

# Convert 1-channel spectrograms to 3 channels for ResNet50
spectrograms_resized_3ch = np.repeat(spectrograms_resized, 3, axis=-1)

# Validate data
print(f"Spectrograms shape: {spectrograms_resized.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Patient IDs shape: {patient_ids.shape}")
if spectrograms_resized.shape[0] != labels.shape[0] or spectrograms_resized.shape[0] != patient_ids.shape[0]:
    raise ValueError("Mismatch in data sizes")
if spectrograms_resized.shape[0] != 6898:
    print(f"Warning: Expected 6898 samples, got {spectrograms_resized.shape[0]}")

# Class weights
class_counts = np.bincount(labels)
class_weights = {i: max(class_counts) / count for i, count in enumerate(class_counts)}
print("Class distribution:", class_counts)
print("Class weights:", class_weights)

# Prepare data
labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=4)
gkf = GroupKFold(n_splits=2)
folds = list(gkf.split(spectrograms_resized, labels, groups=patient_ids))

# Train and evaluate
models = {'ResNet18': build_resnet18(), 'ResNet50': build_resnet50()}
histories = {'ResNet18': [], 'ResNet50': []}
metrics = {'ResNet18': {'sensitivity': [], 'specificity': [], 'score': [], 'accuracy': [], 'cm': [], 'f1': []},
           'ResNet50': {'sensitivity': [], 'specificity': [], 'score': [], 'accuracy': [], 'cm': [], 'f1': []}}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=focal_loss(gamma=2.0), metrics=['accuracy'])
    model.summary()
    
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        X_train = spectrograms_resized[train_idx] if model_name == 'ResNet18' else spectrograms_resized_3ch[train_idx]
        X_test = spectrograms_resized[test_idx] if model_name == 'ResNet18' else spectrograms_resized_3ch[test_idx]
        y_train, y_test = labels_one_hot[train_idx], labels_one_hot[test_idx]
        
        print(f"{model_name} - Training fold {fold_idx + 1}/2")
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1, class_weight=class_weights)
        histories[model_name].append(history.history)
        
        y_pred = model.predict(X_test)
        sensitivity, specificity, score, accuracy, cm, f1 = compute_metrics(y_test, y_pred)
        
        metrics[model_name]['sensitivity'].append(sensitivity)
        metrics[model_name]['specificity'].append(specificity)
        metrics[model_name]['score'].append(score)
        metrics[model_name]['accuracy'].append(accuracy)
        metrics[model_name]['cm'].append(cm)
        metrics[model_name]['f1'].append(f1)
    
    print(f"\n{model_name} Average Results:")
    print(f"Average Sensitivity: {np.mean(metrics[model_name]['sensitivity']):.4f}")
    print(f"Average Specificity: {np.mean(metrics[model_name]['specificity']):.4f}")
    print(f"Average Score: {np.mean(metrics[model_name]['score']):.4f}")
    print(f"Average Accuracy: {np.mean(metrics[model_name]['accuracy']):.4f}")

# Generate visualizations
plot_resnet_architecture()
plot_spectrograms()
plot_confusion_matrices()
plot_performance_comparison()
plot_training_curves()
plot_classwise_performance()
plot_model_size_tradeoff()
