import os
import time
import torch
import torch_tensorrt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.vssm import VSSM

#----------------------------------------------------------------------------
# DATA

def get_loaders(data_dir, loader_kwargs):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    
    test_transforms = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean, std)
    ])
    test_dataset = datasets.CIFAR10(os.path.join(data_dir, "cifar-10"), train=False, transform=test_transforms, download=False)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    
    return test_loader

#----------------------------------------------------------------------------
# EVAL FUNCTION

@torch.no_grad()
def evaluate(test_loader, model, dtype, device, warmup_iters=50):
    total, total_correct = 0, 0
    timings = []
    with tqdm(test_loader, leave=False) as pbar:
        for batch_idx, batch in enumerate(pbar):
            images = batch[0].to(dtype=dtype, device=device)
            targets = batch[1].to(dtype=dtype, device=device)
            
            if batch_idx >= warmup_iters:
                torch.cuda.current_stream().synchronize()
                t_start = time.perf_counter()
                logits = model(images)
                torch.cuda.current_stream().synchronize()
                t_end = time.perf_counter()
                timings += [t_end - t_start]
            else:
                logits = model(images)
            
            n = len(targets)
            total += n
            total_correct += torch.eq(logits.argmax(dim=1), targets).sum().item()
            
            results = {"test_acc": total_correct / total}
            pbar.set_postfix(results)
    
    results.update({"timings": timings})
    return results

#----------------------------------------------------------------------------
# RUN

def run(data_dir, device_list, use_fp16=False):
    # LOADERS
    loader_kwargs = {"batch_size": 1, "num_workers": 4, "pin_memory": True}
    test_loader = get_loaders(data_dir, loader_kwargs)
    
    # MODEL
    model_kwargs = {
        "img_size": 32,
        "patch_size": 4,
        "num_classes": 10,
        "dims": [96, 96, 96, 96],
        "tensorrt": True,
    }
    model = VSSM(**model_kwargs)
    ckpt_path = "checkpoints/vssm_tiny.pth"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # CUDA SETTINGS
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda", device_list[0])
    model = model.to(device)
    if len(device_list) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_list)
    
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("Trainable parameters: {:d} ({:.1f}M)".format(n_params, n_params / 1e6))
    
    # Compile with TensorRT
    dtype = torch.float32
    if use_fp16:
        dtype = torch.float16
        model.half()
    model.eval()
    trt_model = torch_tensorrt.compile(
        model,
        inputs=[torch.rand((1, 3, 32, 32), dtype=dtype, device=device)],
        ir="dynamo",
        enabled_precisions={dtype},
        debug=False,
    )
    # import pdb; pdb.set_trace()
    
    # RUN
    test_results = evaluate(test_loader, trt_model, dtype, device)
    print("Test accuracy: {:.2f}%".format(test_results["test_acc"] * 100.0))
    
    model_times = test_results["timings"]
    latency = sum(model_times) / (len(model_times) / 1000)
    fps = len(model_times) / sum(model_times)
    print("- {:>20s}:\t{:.9f} ms\t{:.9f} FPS".format("model", latency, fps))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    data_dir = "/workspace/datasets"
    device_list = [0]
    run(data_dir, device_list, use_fp16=True)
    
    # VMamba-tiny (PyTorch-FP32)
    #   patch | [8, 8]
    #     acc | 86.2%
    #  params | 1.7M
    #    VRAM | 0.31 GB
    # latency | 11.45 ms
    #     fps | 87.35
    
    # VMamba-tiny (TensorRT-FP32)
    #   patch | [8, 8]
    #     acc | 86.2%
    #  params | 1.7M
    #    VRAM | 0.17 GB
    # latency | 4.56 ms
    #     fps | 219.48
    
    # VMamba-tiny (TensorRT-FP16) => implement fp16 selective_scan
    #   patch | [8, 8]
    #     acc | 86.2%
    #  params | 1.7M
    #    VRAM | 0.41 GB
    # latency | 6.60 ms
    #     fps | 151.48