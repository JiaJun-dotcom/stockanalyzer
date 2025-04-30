import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split

dataset_path = "data/.cache/kagglehub/datasets/mustaphaelbakai/stock-chart-patterns/versions/5/Patterns"

transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize image
    transforms.ToTensor(), 
    # Convert image to PyTorch tensors for processing
    transforms.Normalize(
        [0.485, 0.456, 0.406], # ImageNet mean
        [0.229, 0.224, 0.225] # ImageNet std
    )
])

dataset = datasets.ImageFolder(
    root=dataset_path, 
    transform=transform
)

# 70% train, 15% validation, 15% test
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = models.efficientnet_b0(pretrained=True)
# Structure of efficientnet_b0 model:
# Sequential(
#  (0): Dropout(p=0.2, inplace=True)
#  (1): Linear(in_features=1280, out_features=1000, bias=True)
# )

num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2) 
# Replace decision layer(Set output features to be just 2, Double Top and 
# Double Bottom)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10

def train_model(model, data, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad() # Clear gradients from previous batch, prevent them from accumulating.
        predictions = model(images)

        loss = criterion(predictions, labels)
        loss.backward() # Backpropagation to compute gradients

        optimizer.step() # Learning step to update model weights using gradients

        running_loss += loss.item()
        _, predicted = predictions.max(1) 
        # This step obtains the indices of the predictions, 
        # Sample output of a prediction from efficientnet:
        # tensor([[1.2, 0.3],
        # [0.1, 2.5],
        # [2.1, 0.9],
        # [0.2, 3.0]])
        # Each column represents either class(DT vs DB), each row of values are logits(think of them as confidence levels)
        # Higher logit means the model predicts that particular class more(higher probability of that class for that image)
        # This step would output to become [0, 1, 0, 1]: indices of the predicted classes(DT or DB)
        
        total += labels.size(0)
        
        correct += predicted.eq(labels).sum().item()
        # predicted.eq(labels) compares the predictions to the ground truth from labels eg:
        # predicted = tensor([0, 1, 0, 1])
        # labels    = tensor([0, 1, 1, 1])
        # predicted.eq(labels) → tensor([True, True, False, True])
        # .sum() then adds up the "true" values of the predictions: 3, accumulating to correct for total correct predictions.
    
    accuracy = 100 * correct / total
    
    print(f"Epoch [{epoch+1}] Loss: {running_loss/len(train_loader):.4f}, "
    f"Accuracy: {accuracy:.2f}%")

def validate_model(model, data, criterion, device):
    model.eval()  # Evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # No gradients needed for validation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

for epoch in range(num_epochs):
    train_model(model=model, data=train_loader, criterion=criterion, optimizer=optimizer, device=device, epoch=epoch)
    validate_model(model=model, data=val_loader, criterion=criterion, device=device)
