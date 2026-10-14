# Chạy lab_day10.ipynb đã điền code

Notebook này chạy E0–E4 theo các mục 0–8 của mẫu lab, với thuật toán hiển thị trực tiếp
trong cell. Có cell train và evaluate riêng cho E1/E2/E3. E5/GRPO vẫn là bonus chưa triển khai.

1. Upload `lab_day10.ipynb` lên Colab. Chọn Runtime → Change runtime type → T4 GPU.
2. Chọn nguồn dữ liệu trong cell cấu hình:
   - Upload `archive.zip` vào Files của Colab, đặt `DATA_ZIP="/content/archive.zip"`;
   - hoặc đặt `RAW_DATA_DIR` vào thư mục có `images/` và `labels-YOLO/`;
   - hoặc giữ cả hai là None, tạo Secrets `KAGGLE_USERNAME` và `KAGGLE_KEY`, cấp quyền
     cho notebook. API key chỉ đọc vào biến môi trường, không ghi vào source/file output.
3. `MOUNT_DRIVE=True` là mặc định. Cho phép mount khi Colab hỏi; checkpoint và kết quả
   được backup vào `/content/drive/MyDrive/lab_day10_backup`. Nếu không dùng Drive,
   đặt False; file chỉ tồn tại trong session, cần tải ZIP/checkpoint trước khi session hết.
4. Chạy các cell từ trên xuống, hoặc Runtime → Run all. Bootstrap có sẵn công cụ đề,
   không cần clone repo. Dataset và model cache đặt trong `/content` để tránh đọc Drive chậm.
5. Cell cuối hiển thị bảng kết quả, backup Drive và tải `/content/lab_day10_results.zip`.

Cùng cấu hình QLoRA, mask loss, checkpoint projector và chấm điểm như bản Kaggle.
Giữ torch/torchvision của Colab, cài các bản thư viện đã ghim và gỡ backend TorchAO
không dùng. Batch 1, grad_accum 8; rank 16, alpha 32; max_steps 300; inference limit 512.

`save_weights` backup adapter/projector vào Drive mỗi 100 bước. Kết thúc mỗi cell train/evaluate,
backup toàn bộ work/report và ZIP để giữ log và output thô. Không backup ảnh dataset hoặc
model gốc. Drive IO được tính vào thời gian train khi diễn ra trong checkpoint callback.

`RESUME_SOURCE` có thể trỏ vào folder backup của lượt **đã hoàn thành**. Cần cùng SHA model,
dataset_version và code/config. Checkpoint giữa lượt chỉ chứa trọng số; không resume chính xác
optimizer/RNG. Chạy lại cell train dùng checkpoint đầy đủ cùng config nếu đã có.

Code và luồng cell được kiểm tra trên CPU; chưa chạy lại full QLoRA trên Colab T4.
Báo cáo đã hoàn thành trong repo được giữ nguyên. Bootstrap tạo bản REPORT mẫu riêng cho runtime
mới; không dùng lại số liệu trong báo cáo cũ làm kết quả của một lượt train mới.

Rebuild hai notebook:

```powershell
.\.venv\Scripts\python.exe build_colab_notebook.py
```

Mẫu ban đầu: `lab_day10_template.ipynb`. Các công cụ `scripts/` và `tests/` của đề giữ nguyên.
