"""
Differences from Dylans code:
- i kept the overall design of:
  * an mlp with 3 hidden layers
  * a ProjectDataLoader that reads pngs and extracts labels fromt he filenames
  * visualizing the predicitons like the professor does witht he image and bar chart
- i changed a couple things:
  * relative "../digits" path so we can all use the code
  * i have additions so i can run the code on the schoole A100 machine,
    but should run fine still locally
  * i use a Dataset and DataLoader fro the class digits so i can use the same
    evaluation code for the class digits that is used for MNIST
  * added dropout and regularization
- we can still add:
  * preprocessing for the class digits so they look more like MNISt
    - centering the digits and resizing them (cropping)
    - make cure they are black background and white digit even tho i think shen did this
  * add more hidden units in the hidden layer
  * train for more epochs
  * decrease the learnign rate
-- these are all things we should vary and then present how they resulted in over/underfitting
   or how they agve us the perfect medium for getting the best accuracy
  * change the transform for just the training data so it accepts some rotations/shifts during
    training so the model isnt so sensitive to this when testing
"""

import torch, time 
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np

# visualization helper adapted from Dylans code:
# - given a single image tensor and the probability vector over each digit:
# -- plot the image on the left
# -- plot the horizontal bar chart of probabilities on the right
def plot_prediction(image_tensor, probs, filename):
    
    image_np = image_tensor.squeeze().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # left: the digit image
    ax1.imshow(image_np, cmap="viridis")
    ax1.axis("off")

    # right: Probability bars
    y_pos = np.arange(10)
    ax2.barh(y_pos, probs)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([str(d) for d in range(10)])
    ax2.set_xlim(0, 1)
    ax2.set_title("Class Probability")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)

# i am running on the A100
# use the GPU here if it is available, if not use CPU like usual
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# load the classes test digit data
# Helper to load the class digits dataset:
# - reads all the png files
# - extracts the labels form filenames
# - return N x 28 x 28 grayscale images and labels
# use the "../digits" path so it works for anyone
def ProjectDataLoader(digits_dir="../digits"):
    digits_path = Path(digits_dir)
    image_list = []
    label_list = []
    
    # sort for deterministic ordering so we can recreate runs
    # good for debugging and alaysis in our report
    for png_path in sorted(digits_path.glob("*.png")):
        fname = png_path.stem
        parts = fname.split("-")
        if len(parts) < 1:
            # skip any file that doesnt follow the naming convention
            continue
        try:
            # first part of filename is the digit label we want
            label = int(parts[0])
        except ValueError:
            # skip if this isnt an integer
            continue

        # open image and force it to grayscale("L") to match MINST format
        img = Image.open(png_path).convert("L")
        # ensure that the size is actualy 28 x 28
        img = img.resize((28, 28))

        # convert to flost32 np array
        arr = np.array(img, dtype=np.float32)
        image_list.append(arr)
        label_list.append(label)

    # stack into a single np array of shape (N, 28, 28)
    # return empty array if there are no images with this smae shape
    images = np.stack(image_list, axis=0) if image_list else np.empty((0, 28, 28), dtype=np.float32)
    labels = np.array(label_list, dtype=np.int64)
    return images, labels

# Create the dataset of the classes digits pngs to use in testing the mlp
# class datset wrapper for the ProjectDataLoader function above
# lets us plug in the class digit set into a PyTorch DataLoader
# where we can apply the same transforms (ToTensor/Normalize) that we do to MINST
class ProjectDigitsDataset(torch.utils.data.Dataset):

    def __init__(self, digits_dir="../digits", transform=None):
        # load all the images abd labels intp np arrays
        self.images_np, self.labels_np = ProjectDataLoader(digits_dir)
        self.transform = transform

    def __len__(self):
        return len(self.labels_np)

    def __getitem__(self, idx):
        # get the 28 x 28 np image and label oh shapr (28, 28)
        img_arr = self.images_np[idx]
        label = int(self.labels_np[idx])

        # convert the np array back to a PIL image so that we can apply
        # torchvision transforms (ToTensor/Normalize)
        # using mode="L" is just 8-bit grayscale
        img = Image.fromarray(img_arr.astype(np.uint8), mode="L")

        # apply the same transform pipeline as MNIST to img
        if self.transform is not None:
            img = self.transform(img)

        return img, label

# Define mlp 
# later tune: Hidden sizes, Dropout rate, Learning rate, Batch size, Number of epochs
# mlp model architecture: fuuly connected feed forward network
# - input: 28*28 = 784 features (flattened image)
# - hidden layers: 256, 128, 64 with ReLU
# - dropout: 0.2 to reduce the amount of overfitting
# - output: 10 logits (one per digit)
class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # flatten from (batch, 1, 28, 28) to (batch, 784)
        x = x.view(x.size(0), -1)
        # apply fully connected layers with ReLU and dropout
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.dropout(torch.relu(self.fc3(x)))
        # final layer outputs raw logits
        # CroddEntropyLoss handles the softmax internally
        x = self.fc4(x)
        return x

# evaluate the model on the MINST test set now
# Evaluation helper used for both MNIST and class data
# computes average loss and accuracy over the given DataLoader
# keep this funcitong eneric by being able to pass in "criterion"
def evaluate(model, dataloader, device, criterion):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            # move batch to GPU/CPU if s=using the A100 like i am
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # forward pass
            logits = model(images)
            loss = criterion(logits, labels)
            # accumulate total (sum of loss * batch size)
            running_loss += loss.item() * images.size(0)

            # compute predictions and count correct ones
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # avoid divion by zero is dataloader is empty
    if total == 0:
        return float("nan"), float("nan")

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# main training and evaluation calls
# Main funciton:
# - load and preprocesss MNIST train and test data
# - train the MLP on MNISt data
# - evaluate on MNIST test set
# - load class digits set, preprocess it, and evaluate it on MLP
def main():

    batch_size = 128

    # define transforms for image preprocessing:
    # - ToTensor: [0, 255] -> [0, 1]
    # - Normalize(mean=0.5, std=0.5): [0, 1] -> [-1, 1]
    # use these same transforms for both MNISt and class data so its consistent for the MLP
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # download MNIST training set: used for training the MLP
    trainset = datasets.MNIST(
            root="~/MNIST_data",
            train=True,
            download=True,
            transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
    )

    # download MNISt test set: used to evaluate loss/accuracy
    testset = datasets.MNIST(
            root="~/MNIST_data",
            train=False,
            download=True,
            transform=transform
    )
    testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
    )

    # initalize model, loss, and optimizer
    # - MLP() is the fully connected network
    # - CrossEntropyLoss = softmax + log loss
    # - Adam optimizer with lr=1e-3 (this is pretty common)
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # training loop:
    # - run for fixed number of epochs (we should change this and graph results)
    # - for each epoch: iterate iver all the training batches
    # - compute loss, backprop, and update weights
    # - track and print the average training loss per epoch: shows us if the model is improving
    epochs = 10
    train_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()
    
        for images, labels in trainloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        
            # zero out gradients from previous step
            optimizer.zero_grad()
            # forward pass
            logits = model(images)
            # compute training loss
            loss = criterion(logits, labels)
            # backward pass
            loss.backward()
            # update parameters
            optimizer.step()
         
            # accumulate the total loss
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(trainloader.dataset)
        train_losses.append(epoch_loss)
    
        elapsed = time.time() - start_time
        print(f"Epoch {epoch}/{epochs} - Training loss: {epoch_loss:.4f} - {elapsed:.1f}s")

    print("Training complete.")

    """
    # plot the loss just too see it dropping visually even though its obvious from print

    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("MLP training loss on MNIST")
    plt.show()
    """

    # Evaluate the trained MLP on the MNISt test set
    # this shows us the baseline performance on the stndardized data
    # then we will have to compare this to the class digits performance 
    test_loss, test_acc = evaluate(model, testloader, device, criterion)
    print(f"MNIST Test loss: {test_loss:.4f}, accuracy: {100*test_acc:.2f}%")

    # load the class digits from "../digits" and check shapes/labels
    images_np, labels_np = ProjectDataLoader("../digits")
    print("Project digits shapes:", images_np.shape, labels_np.shape)
    print("Label distribution (digit, count):", np.unique(labels_np, return_counts=True))

    # build Dataset and DataLoader for the class digits using the same transforms as MNIST
    # now we cna reuse the same evaluation code that we sued for MNIST
    project_dataset = ProjectDigitsDataset("../digits", transform=transform)
    project_loader = torch.utils.data.DataLoader(
        project_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    print("Project digits dataset size:", len(project_dataset))

    """
    accuracy is lower on real handwritten digits:
    - the classes digits vary a lot
    - MNIST is very standardized and our handwriting is not
    - MLP struggles with image distortions
    - No translation invariance
    - No convolutional filters
    --> So off center digits or unusual writing styles confuse it

    Some submitted images are low quality/too faint/weird background/wrong color inversion/not thick
    and Dataset is small and small test datasets produce bad accuracy
    """
    # evaluate the smae trained MLP on the class digits
    proj_loss, proj_acc = evaluate(model, project_loader, device, criterion)
    print(f"Class handwritten digits - loss: {proj_loss:.4f}, accuracy: {100*proj_acc:.2f}%")

    # analysis/visualization:
    # - build a tensor of all project images
    # - run the model once to get logits
    # - convert logits to probabilities using softmax
    # - for each image, print true vs predicted and show the probability bar chart and  image
    if len(project_dataset) > 0:
        
        # build a single big batch of all project images
        imgs_tensor = torch.stack([project_dataset[i][0] for i in range(len(project_dataset))])
        lbls_array = np.array([project_dataset[i][1] for i in range(len(project_dataset))])

        imgs_tensor_device = imgs_tensor.to(device)
        with torch.no_grad():
            logits_all = model(imgs_tensor_device)
            probs_all = torch.softmax(logits_all, dim=1).cpu().numpy()

        preds_array = probs_all.argmax(axis=1)
        overall_correct = (preds_array == lbls_array).sum()
        print(f"[Per-image check] Team Images Accuracy: {overall_correct / len(lbls_array):.2f}")

        # plot a some examples
        output_dir = "predictions"
        Path(output_dir).mkdir(exist_ok=True)
        
        for i in range(len(imgs_tensor)):
            img_t = imgs_tensor[i]
            p = probs_all[i]
            true_label = lbls_array[i]
            pred_label = preds_array[i]
            print(f"Image {i} --- True: {true_label}, Predicted: {pred_label}")
            filename = f"{output_dir}/digit_{i}_true{true_label}_pred{pred_label}.png"
            plot_prediction(img_t, p, filename)

# entry point
if __name__ == "__main__":
    main()
