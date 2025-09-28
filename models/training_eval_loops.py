import torch


def training(model, train_loader, criterion, optimizer, device):
    
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): PyTorch model to train.
        train_loader (DataLoader): Training data loader.
        criterion (Loss): Loss function.
        optimizer (Optimizer): Optimizer instance.
        device (torch.device): Device to run on (CPU/GPU).

    Returns:
        train_loss (float): Average loss over training batches.
        train_acc (float): Training accuracy percentage.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc = 100 * correct / total
    return train_loss, train_acc


def evaluation(model, val_loader, criterion, device):
    """
    Evaluates the model on validation data.

    Args:
        model (nn.Module): PyTorch model to evaluate.
        val_loader (DataLoader): Validation data loader.
        criterion (Loss): Loss function.
        device (torch.device): Device to run on (CPU/GPU).

    Returns:
        val_loss (float): Average validation loss.
        val_acc (float): Validation accuracy percentage.
        all_labels (list): True labels collected over validation set.
        all_preds (list): Predicted labels collected over validation set.
        all_probs (list): Predicted class probabilities collected over validation set.
    """
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            val_correct += (preds == y).sum().item()
            val_total += y.size(0)

    val_loss /= len(val_loader)
    val_acc = 100 * val_correct / val_total
    return val_loss, val_acc, all_labels, all_preds, all_probs