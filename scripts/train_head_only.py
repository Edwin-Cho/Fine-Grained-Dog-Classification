#!/usr/bin/env python3
"""
Head-Only Training Script

Ablation baseline: only the classification head (GAP + BN + Dense layers)
is trainable. The entire ResNet-50 backbone is frozen (~0.5M trainable params).

Author: HyunHeum Cho
Date: 2026.05.18
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import matplotlib.pyplot as plt

# GPU Setup
print("🔍 Checking GPU availability...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU detected: {len(gpus)} GPU(s) available")
    except RuntimeError as e:
        print(f"⚠️  GPU setup error: {e}")
else:
    print("⚠️  No GPU detected - running on CPU")

# ==========================
# Configuration
# ==========================

# Model hyperparameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 35
LEARNING_RATE = 0.0001
PATIENCE = 10

# Paths
# Set DATASET_PATH environment variable or use default relative path
DATASET_PATH = os.environ.get('DATASET_PATH', '../Dataset_Stanford/Stanford_Images')
OUTPUT_DIR = '../ablation_results/head_only'

# Data augmentation parameters
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'zoom_range': 0.2,
    'shear_range': 0.2,
    'fill_mode': 'nearest'
}

print("="*80)
print("🧪 ABLATION STUDY: Head-Only (Lower Bound)")
print("="*80)
print(f"📊 Configuration:")
print(f"  - Strategy: Head-Only (Backbone fully frozen)")
print(f"  - Image size: {IMAGE_SIZE}")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Epochs: {EPOCHS}")
print(f"  - Learning rate: {LEARNING_RATE}")
print()

def create_head_only_model(num_classes: int) -> tf.keras.Model:
    """Create Head-Only model with backbone fully frozen.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model with only classification head trainable
    """
    print("🔨 Creating Head-Only model...")
    
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3)
    )
    
    # Freeze entire backbone
    base_model.trainable = False
    
    print(f"✅ Backbone frozen — only classification head is trainable")
    
    # Classification head
    model = tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Count parameters
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    total_params = sum([tf.keras.backend.count_params(w) for w in model.weights])
    
    print(f"📊 Model parameters:")
    print(f"  - Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"  - Total: {total_params:,}")
    print()
    
    return model

# Data generators
def create_data_generators():
    """Create training and validation data generators"""
    print("📦 Creating data generators...")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    train_gen = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_gen = val_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    num_classes = len(train_gen.class_indices)
    
    print(f"✅ Data ready")
    print(f"  - Classes: {num_classes}")
    print(f"  - Training samples: {train_gen.samples}")
    print(f"  - Validation samples: {val_gen.samples}")
    print()
    
    return train_gen, val_gen, num_classes

# Training history plot
def plot_training_history(history):
    """Plot and save training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title('Head-Only - Accuracy', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Head-Only - Loss', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
    print(f"💾 Saved training curves: {OUTPUT_DIR}training_history.png")
    plt.close()

# Main execution
if __name__ == '__main__':
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create data generators
    train_gen, val_gen, num_classes = create_data_generators()
    
    # Save class names
    class_names = np.array(list(train_gen.class_indices.keys()))
    np.save(os.path.join(OUTPUT_DIR, 'class_names.npy'), class_names)
    
    # Create model
    model = create_head_only_model(num_classes)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(OUTPUT_DIR, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print("🏋️  Starting training...")
    print("="*80)
    
    # Train
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    print()
    print("="*80)
    print("✅ Training completed!")
    print("="*80)
    
    # Results
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    
    print(f"📊 Results:")
    print(f"  - Final Train Acc: {final_train_acc*100:.2f}%")
    print(f"  - Final Val Acc: {final_val_acc*100:.2f}%")
    print(f"  - Best Val Acc: {best_val_acc*100:.2f}%")
    print()
    
    # Save results
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    total_params = sum([tf.keras.backend.count_params(w) for w in model.weights])
    results = {
        'strategy': 'Head-Only (Lower Bound)',
        'final_train_acc': float(final_train_acc),
        'final_val_acc': float(final_val_acc),
        'best_val_acc': float(best_val_acc),
        'trainable_params': int(trainable_params),
        'total_params': int(total_params),
        'epochs_trained': len(history.history['accuracy'])
    }
    
    np.save(os.path.join(OUTPUT_DIR, 'results.npy'), results)
    
    history_path = os.path.join(OUTPUT_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=2)
    print(f"💾 Saved training history JSON: {history_path}")
    
    # Plot
    plot_training_history(history)
    
    print(f"💾 Results saved to: {OUTPUT_DIR}")
    print("="*80)
