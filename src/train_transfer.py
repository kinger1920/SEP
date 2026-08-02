import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from src.model import TransferLearningNetwork
from src.dataset import CustomImageDataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate model with fine-tuning option (freeze_backbone=False)
model = TransferLearningNetwork(num_classes=20, pretrained=True, freeze_backbone=False).to(device)

num_epochs = 20
batch_size = 16

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
criterion = torch.nn.CrossEntropyLoss().to(device)

full_dataset = CustomImageDataset("../images/labels.csv", "../images")

# Load stratified split information without modifying dataset.py
try:
    split_df = pd.read_csv("../dataset_final_split.csv")
    train_indices = split_df[split_df['split'] == 'train'].index.tolist()
    val_indices = split_df[split_df['split'] == 'val'].index.tolist()
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    print(f"Loaded stratified split: {len(train_dataset)} train, {len(val_dataset)} val samples.")
except Exception as e:
    print(f"Could not load stratified split, falling back to random split: {e}")
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

best_val_acc = 0.0

for epoch in range(num_epochs):
    print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
    model.train()
    running_loss = 0.0
    
    # Training phase
    for step, (example, label) in enumerate(tqdm(train_loader, desc='Training Batch')):
        optimizer.zero_grad()
        
        example = example.to(device)
        label = label.to(device)
        
        prediction = model(example)
        loss = criterion(prediction, label)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    epoch_train_loss = running_loss / len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for val_example, val_label in tqdm(val_loader, desc="Validation Batch"):
            val_example = val_example.to(device)
            val_label = val_label.to(device)
            
            val_pred = model(val_example)
            loss = criterion(val_pred, val_label)
            val_loss += loss.item()
            
            _, predicted = torch.max(val_pred, dim=1)
            total += val_label.size(0)
            correct += (predicted == val_label).sum().item()
            
    epoch_val_loss = val_loss / len(val_loader)
    epoch_val_acc = 100 * correct / total
    
    print(f"Epoch {epoch + 1} Summary -> "
          f"Train Loss: {epoch_train_loss:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f} | "
          f"Val Acc: {epoch_val_acc:.2f}%")
    
    # Save the best model
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), "best_model_transfer.pth")
        print(f"Saved new best model with Validation Accuracy: {best_val_acc:.2f}%")
        
    scheduler.step()

print(f"\nTraining completed! Best Validation Accuracy: {best_val_acc:.2f}%")
