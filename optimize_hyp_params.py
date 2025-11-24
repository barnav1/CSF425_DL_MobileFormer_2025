import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm.auto import tqdm
import optuna

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 100
BATCH_SIZE = 32
IMAGE_SIZE = 224
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
        x, z = inputs
        batch_size = x.size(0)

        # Permute if output shape is [Tokens, Batch, Dim]
        if z.shape[0] != batch_size and z.shape[1] == batch_size:
            z = z.permute(1, 0, 2)

        out = z.mean(dim=1)
        return self.head(out)


dataset = load_dataset("timm/mini-imagenet")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_tf = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
)

val_tf = transforms.Compose(
    [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), normalize]
)


def apply_train_tf(batch):
    batch["pixel_values"] = [train_tf(img.convert("RGB")) for img in batch["image"]]
    return batch


def apply_val_tf(batch):
    batch["pixel_values"] = [val_tf(img.convert("RGB")) for img in batch["image"]]
    return batch


train_ds = dataset["train"].with_transform(apply_train_tf)
val_ds = dataset["validation"].with_transform(apply_val_tf)
test_ds = dataset["test"].with_transform(apply_val_tf)


def collate_fn(examples):
    px = torch.stack([ex["pixel_values"] for ex in examples])
    labels = torch.tensor([ex["label"] for ex in examples])
    return px, labels


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    correct = total = 0
    losses = 0

    for images, labels in tqdm(loader, leave=False, desc="Train"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses += loss.item()
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()

    return losses / len(loader), 100 * correct / total


def evaluate(model, loader, criterion):
    model.eval()
    correct = total = 0
    losses = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False, desc="Eval"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses += loss.item()

            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

    return losses / len(loader), 100 * correct / total


def objective(trial):
    lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 8e-3, 1e-1, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    model = mobile_former_26m(drop_rate=dropout)
    model.classifier = MobileFormerAdapter(NUM_CLASSES)
    model = model.to(DEVICE)

    # Required dry run to initialize LazyLinear parameters before optimizer creation
    with torch.no_grad():
        dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
        model(dummy)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    print(f"Trying lr: {lr}, wd: {weight_decay}, d: {dropout}")

    for epoch in range(20):
        train_one_epoch(model, train_loader, optimizer, criterion)
        _, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1} Val {val_acc:.2f}%")

        # Pruning stops unpromising trials early
        trial.report(val_acc, epoch)
        if trial.should_prune():
            print(f"Pruned lr: {lr}, wd: {weight_decay}, d: {dropout}")
            raise optuna.TrialPruned()

    return val_acc


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15)

print("Best Parameters:", study.best_params)
print("Best Accuracy:", study.best_value)
