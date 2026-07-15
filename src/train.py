import torch
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from src.model import ConvolutionalNetwork
from src.dataset import CustomImageDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ConvolutionalNetwork().to(device)

num_epochs = 20

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
losses = []

criterion = torch.nn.CrossEntropyLoss().to(device)


# Load the dataset and split 80/20
full_dataset = CustomImageDataset("../images/labels.csv", "../images")
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

state_dict = torch.load("model_v4.pth", map_location=device)
model.load_state_dict(state_dict)

for epoch in tqdm(range(num_epochs), desc='Epoch'):
    for step, [example, label] in enumerate(tqdm(train_loader, desc='Batch')):
        optimizer.zero_grad()

        example = example.view(-1, 3, 224, 224).to(device)
        label = label.to(device)
        prediction = model.forward(example)

        loss = criterion.forward(prediction, label)
        losses.append(loss.item())

        loss.backward()
        
        optimizer.step()

    # Now validate on the validation set
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for val_example, val_label in tqdm(val_loader, desc="Validation Batch"):

            val_example = val_example.view(-1, 3, 224, 224).to(device)
            val_label = val_label.to(device)

            val_pred = model.forward(val_example)
            total_loss += criterion.forward(val_pred, val_label).item()

            _, predicted = torch.max(val_pred, dim=1)
            total += val_label.size(0)
            correct += (predicted == val_label).sum().item()

    epoch_val_loss = total_loss / len(val_loader)
    epoch_accuracy = 100 * correct / total
    print(f"\nEpoch {epoch + 1} summary -> Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_accuracy:.2f}%")

    model.train()
    scheduler.step()

# plt.plot(losses)
# plt.xlabel('Step')
# plt.ylabel('Loss')
# plt.show()

torch.save(model.state_dict(), "model_v5.pth")