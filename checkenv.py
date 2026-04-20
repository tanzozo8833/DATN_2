import torch
import torch_directml
import sys

print("--- KIỂM TRẠ HỆ THỐNG (Bản 2.2.1) ---")
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")

try:
    # Khởi tạo thiết bị DirectML
    device = torch_directml.device()
    print(f"Thiết bị DirectML nhận diện: {device}")

    # Chạy thử một phép tính trên GPU
    a = torch.ones(2, 2).to(device)
    b = a + a
    print(f"Tính toán trên GPU AMD: OK (1+1={b[0][0].item()})")
    print("\n>>> CHÚC MƯNG: Mọi thứ đã khớp! Bạn có thể làm DATN rồi.")
except Exception as e:
    print(f"\n>>> LỖI CÒN SÓT LẠI: {e}")