import torch
import torch.nn as nn
import torch.optim as optim
import csv
import random
import matplotlib.pyplot as plt
from rsc_module.embedding_classifier import RSCSystem

# Use GPU if available; otherwise, use CPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize the RSC system.
rsc_system = RSCSystem(device=device)

# Freeze the embedder's parameters (pretrained).
for param in rsc_system.embedder.parameters():
    param.requires_grad = False

def load_dataset(filename="synthetic_dataset.csv"):
    dataset = []
    with open(filename, mode="r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dataset.append({
                "query": row["query"],
                "result": row["result"],
                "role": row["role"],
                "label": float(row["label"])  # Smoothed float label.
            })
    return dataset

dataset = load_dataset("synthetic_dataset.csv")
print(f"Loaded {len(dataset)} records from dataset.")

# Shuffle and split dataset: 80% training, 20% validation.
random.shuffle(dataset)
split_index = int(0.8 * len(dataset))
train_data = dataset[:split_index]
val_data = dataset[split_index:]
print(f"Training records: {len(train_data)}, Validation records: {len(val_data)}")

# Prepare optimizer: update only fusion, dimension reducer, and classifier.
trainable_params = list(rsc_system.fusion.parameters()) + \
                   list(rsc_system.dim_reducer.parameters()) + \
                   list(rsc_system.classifier.parameters())
optimizer = optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-5)

# Use Binary Cross Entropy Loss.
criterion = nn.BCELoss()

# Learning rate scheduler.
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

num_epochs = 10
best_val_loss = float('inf')
early_stop_patience = 5  # Patience for validation loss.
epochs_without_improvement = 0

# Set a threshold for training loss; if the average training loss falls below this value, stop immediately.
TRAIN_LOSS_THRESHOLD = 0.0001

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    rsc_system.fusion.train()
    rsc_system.dim_reducer.train()
    rsc_system.classifier.train()
    
    epoch_train_loss = 0.0
    random.shuffle(train_data)
    for sample in train_data:
        query_text = sample["query"]
        query_result = sample["result"]
        user_role = sample["role"]
        target = torch.tensor([sample["label"]], dtype=torch.float, device=device)
        
        optimizer.zero_grad()
        composite_embedding = rsc_system.get_composite_embedding(query_text, query_result, user_role)
        output = rsc_system.classifier(composite_embedding)
        loss = criterion(output.view(-1), target.view(-1))
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item()
    avg_train_loss = epoch_train_loss / len(train_data)
    train_losses.append(avg_train_loss)
    
    # Check training loss threshold.
    if avg_train_loss < TRAIN_LOSS_THRESHOLD:
        print(f"Early stopping triggered: Training loss {avg_train_loss:.6f} is below threshold {TRAIN_LOSS_THRESHOLD}.")
        break

    rsc_system.fusion.eval()
    rsc_system.dim_reducer.eval()
    rsc_system.classifier.eval()
    
    epoch_val_loss = 0.0
    with torch.no_grad():
        for sample in val_data:
            query_text = sample["query"]
            query_result = sample["result"]
            user_role = sample["role"]
            target = torch.tensor([sample["label"]], dtype=torch.float, device=device)
            
            composite_embedding = rsc_system.get_composite_embedding(query_text, query_result, user_role)
            output = rsc_system.classifier(composite_embedding)
            loss = criterion(output.view(-1), target.view(-1))
            epoch_val_loss += loss.item()
    avg_val_loss = epoch_val_loss / len(val_data)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    scheduler.step(avg_val_loss)
    
    # Early stopping based on validation loss.
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
    if epochs_without_improvement >= early_stop_patience:
        print("Early stopping triggered based on validation loss.")
        break

torch.save({
    "fusion_state_dict": rsc_system.fusion.state_dict(),
    "dim_reducer_state_dict": rsc_system.dim_reducer.state_dict(),
    "classifier_state_dict": rsc_system.classifier.state_dict(),
}, "rsc_system_weights.pth")
print("Training complete. Model weights saved as 'rsc_system_weights.pth'.")

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses)+1), train_losses, label="Training Loss", marker='o')
plt.plot(range(1, len(val_losses)+1), val_losses, label="Validation Loss", marker='o')
plt.title("Training and Validation Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_vs_epochs.png')  # Saves the figure to a file.
plt.show()
