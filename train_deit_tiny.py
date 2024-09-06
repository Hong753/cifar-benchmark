import os
import math
import torch
import torch.nn as nn

from functools import partial
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.vision_transformer import VisionTransformer

#----------------------------------------------------------------------------
# DATA

def get_loaders(data_dir, loader_kwargs):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=3, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    train_dataset = datasets.CIFAR10(os.path.join(data_dir, "cifar-10"), train=True, transform=train_transforms, download=False)
    test_dataset = datasets.CIFAR10(os.path.join(data_dir, "cifar-10"), train=False, transform=test_transforms, download=False)
    
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    
    return train_loader, test_loader

#----------------------------------------------------------------------------
# TRAINING FUNCTION

def train_one_epoch(epoch, train_loader, model, optimizer, criterion, device):
    model.train()
    total, total_loss, total_correct = 0, 0, 0
    
    with tqdm(train_loader, leave=False) as pbar:
        for batch in pbar:
            pbar.set_description(f"[TRAIN] Epoch {epoch + 1}")
            
            images = batch[0].to(device)
            targets = batch[1].to(device)
    
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
    
            n = len(targets)
            total += n
            total_loss += n * loss.item()
            total_correct += torch.eq(logits.argmax(dim=1), targets).sum().item()
            
            results = {"train_loss": total_loss / total, "train_acc": total_correct / total}
            pbar.set_postfix(results)
    
    return results

#----------------------------------------------------------------------------
# EVAL FUNCTION

@torch.no_grad()
def evaluate(epoch, test_loader, model, criterion, device):
    model.eval()
    total, total_loss, total_correct = 0, 0, 0
    with tqdm(test_loader, leave=False) as pbar:
        for batch in pbar:
            pbar.set_description(f"[TEST] Epoch {epoch + 1}")
            
            images = batch[0].to(device)
            targets = batch[1].to(device)
    
            logits = model(images)
            loss = criterion(logits, targets)
    
            n = len(targets)
            total += n
            total_loss += n * loss.item()
            total_correct += torch.eq(logits.argmax(dim=1), targets).sum().item()
            
            results = {"test_loss": total_loss / total, "test_acc": total_correct / total}
            pbar.set_postfix(results)

    return results

#----------------------------------------------------------------------------
# RUN

def run(data_dir, device_list, batch_size, warmup_epochs, num_epochs):
    # LOADERS
    loader_kwargs = {"batch_size": batch_size, "num_workers": 4, "pin_memory": True}
    train_loader, test_loader = get_loaders(data_dir, loader_kwargs)
    
    # MODEL
    model_kwargs = {
        "img_size": 32,
        "patch_size": 4,
        "num_classes": 10,
        "embed_dim": 192,
        "depth": 12,
        "num_heads": 3,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
    }
    model = VisionTransformer(**model_kwargs)
    
    # CUDA SETTINGS
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda", device_list[0])
    if len(device_list) == 1:
        model = model.to(device)
    elif len(device_list) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_list)
    
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("Trainable parameters: {:d} ({:.1f}M)".format(n_params, n_params / 1e6))
    
    # OPTIM
    base_lr = 0.005
    lr = base_lr * batch_size / 256
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.05)
    
    # SCHEDULER
    lr_func = lambda epoch: min((epoch + 1) / (warmup_epochs + 1e-8), 0.5 * (math.cos(epoch / num_epochs * math.pi) + 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=False)
    
    # CRITERION
    criterion = torch.nn.CrossEntropyLoss()
    
    # RUN
    acc, best_acc = 0, 0
    with tqdm(range(num_epochs), leave=True) as pbar:
        for epoch in pbar:
            pbar.set_description("[MAIN]")
            
            train_results = train_one_epoch(epoch, train_loader, model, optimizer, criterion, device)
            test_results = evaluate(epoch, test_loader, model, criterion, device)
            scheduler.step()
            
            acc = test_results["test_acc"]
            if acc > best_acc:
                best_acc = acc
                if len(device_list) == 1:
                    model_state_dict = model.state_dict()
                elif len(device_list) > 1:
                    model_state_dict = model.module.state_dict()
                checkpoint = {"best_acc": best_acc, "model_state_dict": model_state_dict}
                torch.save(checkpoint, "./checkpoints/vit_tiny_patch2_32.pth")
            
            all_results = {**train_results, **test_results, "best_acc": best_acc}
            pbar.set_postfix(all_results)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    data_dir = "/workspace/datasets"
    device_list = [0]
    batch_size = 128
    warmup_epochs = 5
    num_epochs = 100
    run(data_dir, device_list, batch_size, warmup_epochs, num_epochs)
    
    #  batch | 128
    #  patch | [8, 8]
    #    acc | 81.6%
    # params | 5.4M
    #   VRAM | 7.05 GB
    #   time | 89 mins