import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm.auto import tqdm

REPO_DIR = "mobile_former"

current_dir = os.getcwd()
try:
    os.chdir(REPO_DIR)
    sys.path.append(os.getcwd())
    from mobile_former import mobile_former_26m

    print("Successfully imported mobile_former_26m")

    # Initialize model with specific dropout
    model = mobile_former_26m(drop_rate=0.17)

except ImportError as e:
    print(f"Import failed: {e}")
    raise e
finally:
    os.chdir(current_dir)


class MobileFormerAdapter(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.head = nn.LazyLinear(num_classes)
        self.debug_shape = True

    def forward(self, inputs):
        # Unpack tuple: x is spatial features, z is global tokens
        x, z = inputs

        batch_size = x.shape[0]

        # Ensure tokens are shaped [Batch, Tokens, Dim]
        if z.shape[0] != batch_size and z.shape[1] == batch_size:
            z = z.permute(1, 0, 2)

        # Global average pooling over tokens creates the final embedding
        out = z.mean(dim=1)

        return self.head(out)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 100
IMAGE_SIZE = 224

print(f"Adapting model for {NUM_CLASSES} classes on {DEVICE}...")
model.classifier = MobileFormerAdapter(NUM_CLASSES)
model = model.to(DEVICE)

# Essential: Run dummy data to initialize LazyLinear weights based on input shape
print("Running dummy pass to initialize parameters...")
with torch.no_grad():
    dummy_input = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    _ = model(dummy_input)
print("✅ Model Initialized.")


print("Loading Mini-ImageNet dataset...")
dataset = load_dataset("timm/mini-imagenet")

# Standard ImageNet normalization statistics
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
)

val_transform = transforms.Compose(
    [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), normalize]
)


def apply_train_transforms(examples):
    examples["pixel_values"] = [
        train_transform(image.convert("RGB")) for image in examples["image"]
    ]
    return examples


def apply_val_transforms(examples):
    examples["pixel_values"] = [
        val_transform(image.convert("RGB")) for image in examples["image"]
    ]
    return examples


train_ds = dataset["train"].with_transform(apply_train_transforms)
val_ds = dataset["validation"].with_transform(apply_val_transforms)
test_ds = dataset["test"].with_transform(apply_val_transforms)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return pixel_values, labels


BATCH_SIZE = 32
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=7.5e-4, weight_decay=0.075)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Training", leave=False)
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100.0 * correct / total


EPOCHS = 225
print(f"Starting Training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)

    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

    # Periodically test and save checkpoints
    if (epoch + 1) % 25 == 0:
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    print("-" * 40)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        torch.save(model, f"./saved_models/{epoch+1}_model.pt")

print("Testing...")
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"Final Test Accuracy: {test_acc:.2f}%")
