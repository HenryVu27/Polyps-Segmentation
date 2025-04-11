import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from config import *
import os
from tqdm.auto import tqdm
import wandb
import gc
import psutil
from torchvision.utils import make_grid

def train_model(model, train_loader, val_loader, device, output_dir, num_epochs=EPOCHS, 
                project_name="ultrasound-segmentation", run_name=None):
    # Initialize wandb
    wandb.init(project=project_name, name=run_name, config={
        "learning_rate": LEARNING_RATE,
        "epochs": num_epochs,
        "batch_size": BATCH_SIZE,
        "model_type": model.__class__.__name__,
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "loss_function": "BCELoss",
        "image_size": IMAGE_SIZE,
        "device": device
    })
    
    # Watch the model to track parameters and gradients
    wandb.watch(model, log="all", log_freq=100)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # For Dice coefficient
    def dice_score(preds, targets):
        smooth = 1.0
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        intersection = (preds_flat * targets_flat).sum()
        return (2. * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    epoch_pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        # ===== TRAINING =====
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", 
                          leave=False, unit="batch")
        for batch_idx, (images, masks) in enumerate(train_pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            # forward
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate Dice coefficient
            batch_dice = dice_score((outputs > 0.5).float(), masks)
            train_dice += batch_dice.item() * images.size(0)
            
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
            # Log batch metrics (every 20 batches)
            if batch_idx % 20 == 0:
                # Log memory usage
                gpu_memory_allocated = torch.cuda.memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0
                gpu_memory_reserved = torch.cuda.memory_reserved(device) / 1024**2 if torch.cuda.is_available() else 0
                
                process = psutil.Process(os.getpid())
                ram_usage = process.memory_info().rss / 1024**2  # MB
                
                wandb.log({
                    "batch": epoch * len(train_loader) + batch_idx,
                    "batch_train_loss": loss.item(),
                    "batch_train_dice": batch_dice.item(),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "gpu_memory_allocated_MB": gpu_memory_allocated,
                    "gpu_memory_reserved_MB": gpu_memory_reserved,
                    "ram_usage_MB": ram_usage
                })
        
        train_loss = train_loss / len(train_loader.dataset)
        train_dice = train_dice / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # ===== VALIDATION =====
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        example_images = []
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]", 
                        leave=False, unit="batch")
        with torch.no_grad():
            for i, (images, masks) in enumerate(val_pbar):
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # Calculate Dice coefficient
                batch_dice = dice_score((outputs > 0.5).float(), masks)
                val_dice += batch_dice.item() * images.size(0)
                
                val_loss += loss.item() * images.size(0)
                
                # Save a few example predictions for visualization
                if i == 0:
                    for j in range(min(4, images.shape[0])):
                        example_images.append(
                            wandb.Image(
                                images[j].cpu(),
                                masks={
                                    "ground_truth": {
                                        "mask_data": masks[j, 0].cpu().numpy(),
                                        "class_labels": {0: "background", 1: "target"}
                                    },
                                    "prediction": {
                                        "mask_data": (outputs[j, 0] > 0.5).cpu().numpy(),
                                        "class_labels": {0: "background", 1: "target"}
                                    }
                                }
                            )
                        )
                
        val_loss = val_loss / len(val_loader.dataset)
        val_dice = val_dice / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # update lr
        scheduler.step(val_loss)
        
        # Log epoch metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_dice": train_dice,
            "val_dice": val_dice, 
            "example_predictions": example_images,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            # Also save to wandb
            wandb.save(os.path.join(output_dir, 'best_model.pth'))
            print(f"Model saved at epoch {epoch+1}")
        
        # Explicitly collect garbage to free memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.3f}, Train Dice: {train_dice:.3f}, "
              f"Val Loss: {val_loss:.3f}, Val Dice: {val_dice:.3f}")
    
    # plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir,'training_history.png'))
    wandb.log({"training_curve": wandb.Image(plt)})
    plt.close()
    
    # Close wandb run
    wandb.finish()
    
    return model