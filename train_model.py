import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import json
import tensorflowjs as tfjs
from glob import glob

def load_dataset(data_path):
 
    image_paths = sorted(glob(os.path.join(data_path, 'images', '*.jpg')))
    if not image_paths:
        image_paths = sorted(glob(os.path.join(data_path, 'images', '*.png')))
    
    images = []
    class_masks = []
    instance_masks = []
    labels = []
    
    for img_path in image_paths:
        try:
           
            filename = os.path.basename(img_path)
            base_filename = os.path.splitext(filename)[0]
            
            class_mask_path = os.path.join(data_path, 'class_masks', f'{base_filename}.png')
            instance_mask_path = os.path.join(data_path, 'instance_masks', f'{base_filename}.png')
            
            if os.path.exists(class_mask_path) and os.path.exists(instance_mask_path):
            
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Could not load image: {img_path}")
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img = img / 255.0
                
                
                class_mask = cv2.imread(class_mask_path, cv2.IMREAD_GRAYSCALE)
                if class_mask is None:
                    print(f"Could not load class mask: {class_mask_path}")
                    continue
                class_mask = cv2.resize(class_mask, (224, 224))
                class_mask = class_mask / 255.0
                class_mask = np.expand_dims(class_mask, axis=-1)
                
                instance_mask = cv2.imread(instance_mask_path, cv2.IMREAD_GRAYSCALE)
                if instance_mask is None:
                    print(f"Could not load instance mask: {instance_mask_path}")
                    continue
                instance_mask = cv2.resize(instance_mask, (224, 224))
                instance_mask = instance_mask / 255.0
                instance_mask = np.expand_dims(instance_mask, axis=-1)
                
                images.append(img)
                class_masks.append(class_mask)
                instance_masks.append(instance_mask)
                
                label = int(np.median(class_mask) * 255)  
                labels.append(label)
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
 
    images = np.array(images)
    class_masks = np.array(class_masks)
    instance_masks = np.array(instance_masks)
    labels = np.array(labels)
    
    
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    
    label_map = {old_label: new_label for new_label, old_label in enumerate(sorted(unique_classes))}
    labels = np.array([label_map[label] for label in labels])
    
    class_names = [f"class_{i}" for i in range(num_classes)]  # You can update these names later
    
    return images, class_masks, instance_masks, labels, class_names, num_classes

def create_multi_input_model(num_classes, input_shape=(224, 224, 3)):
    
    image_input = Input(shape=input_shape, name='image_input')
    class_mask_input = Input(shape=(input_shape[0], input_shape[1], 1), name='class_mask_input')
    instance_mask_input = Input(shape=(input_shape[0], input_shape[1], 1), name='instance_mask_input')
    
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=image_input
    )
    base_model.trainable = False
    
    mask_concat = layers.Concatenate()([class_mask_input, instance_mask_input])
    mask_conv = layers.Conv2D(64, 3, padding='same', activation='relu')(mask_concat)
    mask_pool = layers.MaxPooling2D()(mask_conv)
    mask_conv2 = layers.Conv2D(128, 3, padding='same', activation='relu')(mask_pool)
    mask_pool2 = layers.GlobalAveragePooling2D()(mask_conv2)
    
    image_features = layers.GlobalAveragePooling2D()(base_model.output)
    combined = layers.Concatenate()([image_features, mask_pool2])
    
    dense1 = layers.Dense(512, activation='relu')(combined)
    dropout1 = layers.Dropout(0.5)(dense1)
    dense2 = layers.Dense(256, activation='relu')(dropout1)
    dropout2 = layers.Dropout(0.3)(dense2)
    outputs = layers.Dense(num_classes, activation='softmax')(dropout2)

    model = Model(
        inputs=[image_input, class_mask_input, instance_mask_input],
        outputs=outputs
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_data, val_data, epochs=50, batch_size=16):
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

    history = model.fit(
        [train_data[0], train_data[1], train_data[2]], 
        train_data[3],
        validation_data=([val_data[0], val_data[1], val_data[2]], val_data[3]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    return history

def save_model_for_web(model, class_names, save_dir="web_model"):
    
    os.makedirs(save_dir, exist_ok=True)
    
    
    with open(os.path.join(save_dir, 'class_names.json'), 'w') as f:
        json.dump(class_names, f)
    
    
    tfjs.converters.save_keras_model(model, save_dir)
    print(f"Model saved to {save_dir}")

if __name__ == "__main__":
   
    DATA_PATH = r"C:\Users\abdou\Desktop\aggriapp\seeds_data"
    
    
    print("Loading dataset...")
    images, class_masks, instance_masks, labels, class_names, num_classes = load_dataset(DATA_PATH)
    
    print("Splitting dataset...")
    train_idx, val_idx = train_test_split(range(len(images)), test_size=0.2, random_state=42)
    
    train_data = (images[train_idx], class_masks[train_idx], 
                 instance_masks[train_idx], labels[train_idx])
    val_data = (images[val_idx], class_masks[val_idx], 
                instance_masks[val_idx], labels[val_idx])
    
    print(f"Creating model with {num_classes} classes...")
    model = create_multi_input_model(num_classes)
    
    print("Training model...")
    history = train_model(model, train_data, val_data)
    
    
    print("Saving model...")
    save_model_for_web(model, class_names)
    
    print("Training complete!")
