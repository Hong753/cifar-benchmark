import os
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from models.vqmae import VQMAE

#----------------------------------------------------------------------------
# DATA

def get_loaders(data_dir, loader_kwargs):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    train_transforms = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean, std)
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean, std)
    ])
    
    train_dataset = datasets.CIFAR10(os.path.join(data_dir, "cifar-10"), train=True, transform=train_transforms, download=False)
    test_dataset = datasets.CIFAR10(os.path.join(data_dir, "cifar-10"), train=False, transform=test_transforms, download=False)
    
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    
    return train_loader, test_loader

#----------------------------------------------------------------------------
# TRAINING FUNCTION

def train_one_epoch(epoch, train_loader, model, optimizer, device):
    model.train()
    total, total_mae_loss, total_commit_loss = 0, 0, 0
    
    with tqdm(train_loader, leave=False) as pbar:
        for batch in pbar:
            pbar.set_description(f"[TRAIN] Epoch {epoch + 1}")
            
            images = batch[0].to(device)
    
            optimizer.zero_grad()
            pred_images, mask, commit_loss = model(images)
            mae_loss = torch.mean((pred_images - images) ** 2 * mask) / 0.75
            (mae_loss + commit_loss).backward()
            optimizer.step()
    
            n = len(images)
            total += n
            total_mae_loss += n * mae_loss.item()
            total_commit_loss += n * commit_loss.item()
            
            results = {
                "train_mae_loss": total_mae_loss / total, 
                "train_commit_loss": total_commit_loss / total, 
                "train_count": model.count.item()
            }
            pbar.set_postfix(results)
    
    model.reset_count()
    
    return results

#----------------------------------------------------------------------------
# EVAL FUNCTION

@torch.no_grad()
def evaluate(epoch, test_loader, model, device):
    model.eval()
    total, total_mae_loss, total_commit_loss = 0, 0, 0
    with tqdm(test_loader, leave=False) as pbar:
        for batch in pbar:
            pbar.set_description(f"[TEST] Epoch {epoch + 1}")
            
            images = batch[0].to(device)
    
            pred_images, mask, commit_loss = model(images)
            mae_loss = torch.mean((pred_images - images) ** 2 * mask) / 0.75
    
            n = len(images)
            total += n
            total_mae_loss += n * mae_loss.item()
            total_commit_loss += n * commit_loss.item()
            
            results = {
                "test_mae_loss": total_mae_loss / total, 
                "test_commit_loss": total_commit_loss / total, 
                "test_count": model.count.item()
            }
            pbar.set_postfix(results)
    
    model.reset_count()

    return results

#----------------------------------------------------------------------------
# RUN

def run(data_dir, device_list, batch_size, warmup_epochs, num_epochs):
    # LOADERS
    loader_kwargs = {"batch_size": batch_size, "num_workers": 4, "pin_memory": True}
    train_loader, test_loader = get_loaders(data_dir, loader_kwargs)
    
    # MODEL
    model = VQMAE(img_size=32, patch_size=2, mask_ratio=0.75)
    
    # CUDA SETTINGS
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda", device_list[0])
    if len(device_list) == 1:
        model = model.to(device)
    elif len(device_list) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_list)
    
    # OPTIM
    base_lr = 0.00015
    lr = base_lr * batch_size / 256
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    
    # SCHEDULER
    lr_func = lambda epoch: min((epoch + 1) / (warmup_epochs + 1e-8), 0.5 * (math.cos(epoch / num_epochs * math.pi) + 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=False)
    
    # RUN
    loss, best_loss = np.infty, np.infty
    with tqdm(range(num_epochs), leave=True) as pbar:
        for epoch in pbar:
            pbar.set_description("[MAIN]")
            
            train_results = train_one_epoch(epoch, train_loader, model, optimizer, device)
            test_results = evaluate(epoch, test_loader, model, device)
            scheduler.step()
            
            loss = test_results["test_mae_loss"]
            if loss < best_loss:
                best_loss = loss
                if len(device_list) == 1:
                    model_state_dict = model.state_dict()
                elif len(device_list) > 1:
                    model_state_dict = model.module.state_dict()
                checkpoint = {"best_loss": best_loss, "model_state_dict": model_state_dict}
                torch.save(checkpoint, "./checkpoints/pretrain_vqmae_down2_patch2_32.pth")
            
            all_results = {**train_results, **test_results, "best_loss": best_loss}
            pbar.set_postfix(all_results)
            
            if (epoch + 1) % 100 == 0:
                visualize_reconstructions(test_loader.dataset, model, device, epoch + 1)

#----------------------------------------------------------------------------

def visualize_reconstructions(test_dataset, model, device, steps):
    random_indexes = np.random.choice(len(test_dataset), size=(16, ), replace=False)
    real_images = torch.stack([test_dataset[i][0] for i in random_indexes]).to(device)
    pred_images, mask, commit_loss = model(real_images)
    pred_images = pred_images * mask + real_images * (1 - mask)
    save_path = f"./images/vqmae_reconstructions_{steps}.png"
    images = torch.cat([real_images * (1 - mask), pred_images, real_images], dim=0)
    images = ((images + 1) / 2).clamp(0.0, 1.0)
    save_image(images, save_path, nrow=16, padding=0, normalize=False)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    data_dir = "/home/hong/datasets"
    device_list = [0]
    batch_size = 128
    warmup_epochs = 200
    num_epochs = 2000
    run(data_dir, device_list, batch_size, warmup_epochs, num_epochs)
