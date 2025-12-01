import torch
from torch import nn, optim
from torchvision import datasets, transforms

# i am running on the A100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

batch_size = 128

# load the MINST data
# transform MINST data to a tensor using ToTensor and then normlaize using Normalize
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# define the training set from the MINST data
trainset = datasets.MNIST(root="~/MNIST_data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                          num_workers=2, pin_memory=True)

# define the test set from the MINST data
testset = datasets.MNIST(root="~/MNIST_data", train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                         num_workers=2, pin_memory=True)

# confirm length of train/test sets
len(trainset), len(testset)

# Define mlp 
#later tune: Hidden sizes, Dropout rate, Learning rate, Batch size, Number of epochs
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.dropout(torch.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Establish a training loop that has loss logging so we can ensure loss is going down
import time

epochs = 10
train_losses = []
val_losses = []

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    
    for images, labels in trainloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(trainloader.dataset)
    train_losses.append(epoch_loss)
    
    elapsed = time.time() - start_time
    print(f"Epoch {epoch}/{epochs} - Training loss: {epoch_loss:.4f} - {elapsed:.1f}s")

print("Training complete.")

"""
# plot the loss just too see it dropping visually even though its obvious from print
import matplotlib.pyplot as plt

plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("MLP training loss on MNIST")
plt.show()
"""

# evaluate the model on the MINST test set now
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

test_loss, test_acc = evaluate(model, testloader, device)
print(f"MNIST Test loss: {test_loss:.4f}, accuracy: {100*test_acc:.2f}%")

# load the classes test digit data
from pathlib import Path
from PIL import Image
import numpy as np

def ProjectDataLoader(digits_dir="../digits"):
    digits_path = Path(digits_dir)
    image_list = []
    label_list = []

    for png_path in sorted(digits_path.glob("*.png")):
        fname = png_path.stem
        parts = fname.split("-")
        if len(parts) < 1:
            continue
        try:
            label = int(parts[0])
        except ValueError:
            continue

        img = Image.open(png_path).convert("L")
        img = img.resize((28, 28))

        arr = np.array(img, dtype=np.float32)
        image_list.append(arr)
        label_list.append(label)

    images = np.stack(image_list, axis=0) if image_list else np.empty((0, 28, 28), dtype=np.float32)
    labels = np.array(label_list, dtype=np.int64)
    return images, labels

images_np, labels_np = ProjectDataLoader("../digits")
images_np.shape, labels_np.shape, np.unique(labels_np, return_counts=True)

