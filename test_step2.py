import torch
import torch_directml
from modelling.CoordinateModel import CoordinateTransformer

# 1. Khởi tạo thiết bị AMD
device = torch_directml.device()
print(f"--- BẮT ĐẦU KIỂM TRA BƯỚC 2 ---")
print(f"Đang chạy trên thiết bị: {device}")

# 2. Giả lập dữ liệu đầu vào từ Dataset (Batch=2 video, Time=86 frames, Dim=2212)
dummy_input = torch.randn(2, 86, 2212).to(device)

# 3. Khởi tạo Model và đẩy lên GPU
model = CoordinateTransformer(input_dim=2212, num_classes=1124).to(device)

# 4. Chạy thử (Forward pass)
try:
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"1. Model nhận dữ liệu thành công.")
    print(f"2. Hình dạng đầu ra (Time, Batch, Classes): {output.shape}") 
    # Kỳ vọng: torch.Size([86, 2, 1124])

    if output.shape == (86, 2, 1124):
        print("\n>>> KẾT QUẢ: ĐÚNG! Model đã sẵn sàng nhận dữ liệu thật.")
    else:
        print("\n>>> KẾT QUẢ: SAI kích thước đầu ra.")
        
except Exception as e:
    print(f"LỖI KHI CHẠY MODEL: {e}")