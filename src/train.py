import torch
from tqdm import tqdm
from src.model import ConvolutionalNetwork
from src.dataset import get_dataloader

use_gpu = True


model = ConvolutionalNetwork()

optimizer = torch.optim.Adam(model.parameters(), eps=1e-4)
losses = []

criterion = torch.nn.CrossEntropyLoss()
train_loader = get_dataloader("../images/labels.csv", "../images")

if use_gpu:
    model.cuda()
    criterion.cuda()

for epoch in tqdm(range(3), desc='Epoch'):
    for step, [example, label] in enumerate(tqdm(train_loader, desc='Batch')):
        if use_gpu:
            example = example.cuda()
            label = label.cuda()

        optimizer.zero_grad()

        example = example.view(-1, 3, 224, 224)
        prediction = model.forward(example)

        loss = criterion.forward(prediction, label)
        losses.append(loss.item())

        loss.backward()
        
        optimizer.step()

    # Now validate on the whole test set
    # TODO
    # P.eval()
    #
    # total_loss = 0.0
    # correct = 0
    # total = 0
    #
    # for val_example, val_label in tqdm(val_loader, desc="Batch"):
    #     if use_gpu:
    #         val_example = val_example.cuda()
    #         val_label = val_label.cuda()
    #
    #     val_example = val_example.view(-1, 3, 224, 224)
    #
    #     val_pred = P.forward(val_example)
    #     total_loss += criterion.forward(val_pred, val_label).item()
    #
    #     _, predicted = torch.max(val_pred, dim=1)
    #     total += val_label.size(0)
    #     correct += (predicted == val_label).sum().item()
    #
    # epoch_val_loss = total_loss / total
    # epoch_accuracy = 100 * correct / total
    # print(f"Epoch {epoch + 1} summary -> Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_accuracy:.2f}%")
    #
    # P.train()

# plt.plot(losses)
# plt.xlabel('Step')
# plt.ylabel('Loss')
# plt.show()

torch.save(model.state_dict(), "best_model.pth")