import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from models.driver_state_model import DriverStateCNNLSTM
from dataset import DriverStateDataset

def train_model(train_path, val_path, checkpoint_dir="checkpoints", epochs=30, batch_size=16, lr=1e-4, device="cuda"):
    """
    Main training function for the DriverStateCNNLSTM model.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # 1. Device configuration
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Datasets & Dataloaders
    print("Loading datasets...")
    train_dataset = DriverStateDataset(train_path)
    val_dataset = DriverStateDataset(val_path)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Model initialization
    # Using default sizes: cnn_feat=64, geom=10, lstm_hidden=128
    model = DriverStateCNNLSTM(cnn_feature_dim=64, geom_dim=10, lstm_hidden_dim=128, lstm_layers=2)
    model.to(device)
    
    # 4. Loss & Optimizer
    criterion = nn.BCELoss() # Since model has Sigmoid activation
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 5. Training loop
    best_val_loss = float("inf")
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }
    
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (el, er, m, geom, label) in enumerate(train_loader):
            el = el.to(device)
            er = er.to(device)
            m = m.to(device)
            geom = geom.to(device)
            label = label.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(el, er, m, geom)
            loss = criterion(outputs, label)
            
            # Backward pass & Optimize
            loss.backward()
            optimizer.step()
            
            # Calculate statistics
            train_loss += loss.item() * el.size(0)
            preds = (outputs >= 0.5).float()
            train_correct += (preds == label).sum().item()
            train_total += el.size(0)
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for el, er, m, geom, label in val_loader:
                el = el.to(device)
                er = er.to(device)
                m = m.to(device)
                geom = geom.to(device)
                label = label.to(device)
                
                outputs = model(el, er, m, geom)
                loss = criterion(outputs, label)
                
                val_loss += loss.item() * el.size(0)
                preds = (outputs >= 0.5).float()
                val_correct += (preds == label).sum().item()
                val_total += el.size(0)
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        # Step LR Scheduler
        scheduler.step(epoch_val_loss)
        
        # Save history
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)
        
        print(f"Epoch [{epoch}/{epochs}] "
              f"| Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc * 100:.2f}% "
              f"| Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc * 100:.2f}%")
        
        # Save best model checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': epoch_val_loss,
                'val_acc': epoch_val_acc
            }, checkpoint_path)
            print(f"  --> Saved new best model checkpoint to {checkpoint_path}")
            
        # Also save latest checkpoint
        latest_path = os.path.join(checkpoint_dir, "latest_model.pth")
        torch.save(model.state_dict(), latest_path)

    # 6. Plotting learning curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history["train_loss"], label="Train Loss")
    plt.plot(range(1, epochs + 1), history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history["train_acc"], label="Train Acc")
    plt.plot(range(1, epochs + 1), history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("reports/training_curves.png")
    print("Training complete. Learning curves saved to reports/training_curves.png")

if __name__ == "__main__":
    # For testing, we look for data/train_dummy.npz, if not present we generate it
    train_data = "data/train_dummy.npz"
    val_data = "data/val_dummy.npz"
    
    if not os.path.exists(train_data) or not os.path.exists(val_data):
        print("Data files not found. Creating simulated dummy data...")
        from generate_dummy_data import generate_dummy_dataset
        os.makedirs("data", exist_ok=True)
        generate_dummy_dataset(train_data, num_samples=160)
        generate_dummy_dataset(val_data, num_samples=40)
        
    train_model(train_data, val_data, epochs=20, lr=2e-4, batch_size=16)
