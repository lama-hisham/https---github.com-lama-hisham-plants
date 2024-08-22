import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f'Batch {batch_idx}, Loss: {loss.item()}')

        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                
                if batch_idx % 10 == 0:  # Print every 10 batches
                    print(f'Val Batch {batch_idx}, Val Loss: {loss.item()}')

        accuracy = correct / len(val_loader.dataset) * 100  # Convert to percentage
        print(f'Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}, Val Acc: {accuracy:.2f}%')

