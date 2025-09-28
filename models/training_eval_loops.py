import torch


def training(model, train_loader, criterion, optimizer, device):
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