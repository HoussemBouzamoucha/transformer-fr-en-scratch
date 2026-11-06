import sys
import subprocess

def check_gpu():
    import torch

    print("=== Python ===")
    print(f"Version: {sys.version}")

    print("\n=== PyTorch ===")
    print(f"Version:        {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("No GPU found — that is fine for now, training will run on Kaggle.")
        return

    print(f"Device name:    {torch.cuda.get_device_name(0)}")
    print(f"Device count:   {torch.cuda.device_count()}")
    print(f"CUDA version:   {torch.version.cuda}")

    print("\n=== Memory ===")
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Total GPU memory: {total:.1f} GB")

    print("\n=== nvcc ===")
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        print(result.stdout.strip())
    except FileNotFoundError:
        print("nvcc not found.")

    print("\n=== Quick tensor test ===")
    x = torch.randn(4, 4).cuda()
    print(f"Tensor on GPU: {x.device} — OK")


if __name__ == "__main__":
    check_gpu()