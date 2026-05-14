#!/usr/bin/env python3
"""
BN-Only Fine-tuning Training Script

Proposed method for ablation study. Only BatchNormalization layers are trainable
while Conv layers remain frozen, reducing trainable parameters by 95.3%.

Author: Edwin R. Cho
Date: 2025.11.08
"""

import os
import json
import numpy as np
import tensorflow as tf

# GPU Setup
print("🔍 Checking GPU availability...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for all GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU detected: {len(gpus)} GPU(s) available")
        print(f"   GPU devices: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"⚠️  GPU setup error: {e}")
else:
    print("⚠️  No GPU detected - running on CPU (slower)")
print()
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

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
OUTPUT_DIR = '../ablation_results/bn_only'

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


def create_bn_only_model(num_classes: int) -> tf.keras.Model:
    """Create BN-Only fine-tuning model.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model with only BN layers trainable
    """
    # Load ResNet50 base model with ImageNet weights
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3)
    )
    
    # Enable trainable mode first
    base_model.trainable = True
    
    # Only BatchNormalization layers trainable
    trainable_count = 0
    frozen_count = 0
    
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
            trainable_count += 1
        else:
            layer.trainable = False
            frozen_count += 1
    
    print(f"  - Trainable BN layers: {trainable_count}")
    print(f"  - Frozen layers: {frozen_count}")
    
    # Add custom classification head
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
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_data_generators():
    """Create training and validation data generators"""
    # Training data with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        **AUGMENTATION_CONFIG
    )
    
    # Validation data without augmentation
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
    
    return train_gen, val_gen, num_classes


def plot_training_history(history, save_path):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Accuracy Trend', fontsize=14)
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Loss Trend', fontsize=14)
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("="*80)
    print("🔬 ABLATION STUDY: BN-Only Fine-tuning (Proposed Method)")
    print("="*80)
    
    # Check dataset
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found: {DATASET_PATH}")
        return
    
    print(f"📁 Dataset: {DATASET_PATH}")
    print(f"📊 Image size: {IMAGE_SIZE}")
    print(f"🔢 Batch size: {BATCH_SIZE}")
    print(f"📈 Epochs: {EPOCHS}")
    print()
    
    # Create data generators
    print("📦 Creating data generators...")
    train_gen, val_gen, num_classes = create_data_generators()
    
    print(f"✅ Data ready")
    print(f"  - Classes: {num_classes}")
    print(f"  - Training samples: {train_gen.samples}")
    print(f"  - Validation samples: {val_gen.samples}")
    print()
    
    # Get class names
    class_names = np.array(list(train_gen.class_indices.keys()))
    
    # Create model
    print("🔨 Creating BN-Only model...")
    model = create_bn_only_model(num_classes)
    
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    total_params = sum([tf.keras.backend.count_params(w) for w in model.weights])
    
    print(f"✅ Model created")
    print(f"  - Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"  - Total: {total_params:,}")
    print()
    
    # Setup output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, 'bn_only_best.h5')
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    ]
    
    print("🏋️  Starting training...")
    print(f"📂 Model will be saved to: {model_path}")
    print("="*80)
    print()
    
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
    
    # Save artifacts
    class_names_path = os.path.join(OUTPUT_DIR, 'class_names.npy')
    np.save(class_names_path, class_names)
    print(f"💾 Class names: {class_names_path}")
    
    print("📈 Generating training curve...")
    history_path = os.path.join(OUTPUT_DIR, 'training_history.png')
    plot_training_history(history, history_path)
    print(f"💾 Training curve: {history_path}")
    
    # Save epoch-level history as JSON (for paper figures)
    hist_json = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist_json_path = os.path.join(OUTPUT_DIR, 'training_history.json')
    with open(hist_json_path, 'w') as f:
        json.dump(hist_json, f, indent=2)
    print(f"💾 Training history JSON: {hist_json_path}")
    
    # Save results summary for ablation study
    results = {
        'strategy': 'BN-Only (Proposed)',
        'final_train_acc': float(final_train_acc),
        'final_val_acc': float(final_val_acc),
        'best_val_acc': float(best_val_acc),
        'trainable_params': int(trainable_params),
        'total_params': int(total_params),
        'epochs_trained': len(history.history['accuracy'])
    }
    results_path = os.path.join(OUTPUT_DIR, 'results.npy')
    np.save(results_path, results)
    print(f"💾 Results summary: {results_path}")
    
    print()
    print("="*80)
    print("🎉 Training completed successfully!")
    print(f"📂 Results saved to: {OUTPUT_DIR}/")
    print("="*80)


if __name__ == '__main__':
    main()
