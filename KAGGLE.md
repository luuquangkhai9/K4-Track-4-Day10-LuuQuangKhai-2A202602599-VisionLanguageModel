# Chạy Lab Day 10 bằng Kaggle Save Version

E5 bonus có notebook riêng `lab_day10_kaggle_e5.ipynb`; xem [E5.md](E5.md).
Add Input Output checkpoint của lượt SFT cũ để tiếp tục, không cần chạy lại E0–E4.

Notebook: `lab_day10_kaggle.ipynb`. Đây là notebook tự chứa mã nguồn và chia cell theo
mục 0–8 của `lab_day10.ipynb`: dữ liệu, nhãn JSON, E0, mask loss, LoRA, train,
đánh giá E1–E3 và E4. Thuật toán hiển thị trực tiếp trong cell; không gọi `--task all`.
Không cần clone repo, Kaggle API key, hay Colab Secrets.
`scripts/` và `tests/` gốc không bị sửa. `lab_day10.ipynb` cũng đã được điền cho Colab;
bản khung ban đầu được giữ tại `lab_day10_template.ipynb`.

Bản cũ dùng một cell gọi toàn bộ pipeline được giữ tại `lab_day10_kaggle_pipeline.ipynb`.
Phiên Kaggle đang chạy bản cũ không bị ảnh hưởng bởi thay đổi file trên máy.

## Thiết lập

1. Tạo Kaggle Notebook, dùng Import Notebook để upload `lab_day10_kaggle.ipynb`.
2. Settings: chọn GPU T4 và bật Internet. Nếu có T4 x2, mã chỉ dùng GPU 0;
   không cộng bộ nhớ hai GPU thành 32 GB. P100 có thể dùng fp16 nhưng chưa được kiểm chứng.
3. Add Input dataset `lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes`.
   Có thể upload `archive.zip` thành private dataset rồi Add Input thay thế.
4. Giữ cấu hình mặc định, chọn Save Version → Save & Run All. Chờ phiên chạy hoàn tất.
5. Trong Output, tải `lab_day10_results.zip`. Gửi file này để phân tích và viết báo cáo.

Ô cài đặt giữ nguyên torch/torchvision/torchaudio của Kaggle và gỡ backend tùy chọn
`torchao` nếu có. Pipeline dùng bitsandbytes NF4; TorchAO không được sử dụng.
PEFT 0.21.2 có thể dừng ngay khi gắn LoRA nếu phát hiện TorchAO cũ hơn 0.16.0.
Không bỏ qua test E1/E2/E3 để tiếp tục khi gặp lỗi này.

Kaggle Save & Run All chạy một phiên mới từ đầu; các file cần xuất được ghi vào
`/kaggle/working`. Tài liệu: https://www.kaggle.com/docs/notebooks
Thời gian thực tế phụ thuộc GPU, tải model, số box, giới hạn sinh và quota tài khoản;
không mặc định rằng toàn bộ pipeline sẽ xong trong 4 giờ. Kiểm tra quota/session trong UI.
Đề ghi Colab T4: nên xác nhận với giảng viên việc thay nền tảng chạy.

## Pipeline và cấu hình

- Chuẩn bị split bằng công cụ gốc; kiểm tra session không giao nhau, vẽ nhãn.
- Dataset thật có một nhãn crack train tại `vlcsnap-00058` với height=0.
  Bỏ object suy biến này khỏi đáp án huấn luyện để không dạy model xuất box JSON không hợp lệ;
  giữ nguyên split và GT của đề. Lưu trường hợp này trong `report/data_checks.json`.
- Ghim cùng một SHA model cho mọi lượt chạy; model gốc tải về `/tmp`, không lưu vào Output.
- E0 zero-shot; E1 LoRA LLM; E2 thêm projector full; E3 thêm LoRA ViT.
- NF4 4-bit cho LLM, `visual` và `lm_head` không lượng tử hóa.
  Trong Transformers 5.19.0, dùng tên đầy đủ `model.visual` trong
  `llm_int8_skip_modules`; tên ngắn `visual` không loại trừ các module con.
  Test thực hiện chuyển Linear sang bitsandbytes và xác nhận chỉ 252 module LLM được chuyển.
- Các tham số trainable ở fp32, tính toán autocast fp16 và GradScaler. ViT đóng băng giữ fp16.
- AdamW lr=2e-4, warmup 5%, cosine, 300 **bước optimizer thực sự**;
  AMP overflow bị bỏ qua không tính thành bước. Dừng nếu loss không hữu hạn.
- Batch=1, grad_accum=8 cho **tất cả** lượt train, effective batch=8 như cấu hình đề.
- Rank=16, alpha=32, dropout=0.05, seed=42. SDPA, gradient checkpointing non-reentrant.
- Inference batch=2, tự giảm khi OOM; greedy, `max_new_tokens=512` cho mọi setup
  (đề gợi ý 256; tăng để giảm nguy cơ cắt JSON trên ảnh nhiều box).
- Train từ model gốc độc lập, save adapter + projector mỗi 100 bước; đánh giá sau khi
  nạp lại checkpoint vào base mới. Giữ checkpoint mới nhất, không giữ 3 bản đầy đủ.
- E0 chạy ở mục 3. Mục 6 có ba cell train riêng E1/E2/E3; mục 7 có ba cell đánh giá riêng.
  Mục 8 có cell chọn baseline, train E4 và đánh giá E4. Save & Run All chạy tuần tự các cell.
  Chạy lại cell train sẽ tái dùng checkpoint train đầy đủ cùng config nếu có;
  checkpoint chưa đủ 300 bước được train lại từ đầu, không resume optimizer giữa lượt.
  Train/evaluate dùng kernel notebook và giải phóng GPU trong finally sau từng setup.
- Chấm đủ val và test; chọn setup E4 theo **full validation micro-F1**, tie ưu tiên E1 rồi E2.
- E4 đổi đúng `train_fraction`: 10% vs 100%, cùng 300 bước, cùng seed và cấu hình còn lại.
  Mốc 100% dùng lại kết quả setup được chọn; tập 10% lấy mẫu cố định bằng seed 42.
  Giữ số bước có nghĩa 10% được lặp lại nhiều hơn: nêu điều này khi phân tích ablation.
- E5/GRPO là bonus, không triển khai trong phiên bản này.

Triển khai tham khảo API chính thức:
https://huggingface.co/docs/transformers/v5.19.0/model_doc/qwen3_vl
https://huggingface.co/docs/peft/main/en/developer_guides/quantization
https://docs.pytorch.org/docs/stable/notes/amp_examples.html

## File kết quả

`/kaggle/working/lab_day10_results.zip` chứa số liệu và dữ liệu phục vụ phân tích:

- `report/evidence/e0.json`–`e3.json`, `e4_data_size_10.json`, `e4_data_size_100.json`.
- `report/validation/`, `summary.csv`, `environment.json`, `quality_checks.json`.
- `report/e4_selection.json`, kiểm tra dữ liệu, phân tích sai thang tọa độ.
- `report/figures/`: 6 ảnh nhãn train và 6 ảnh dự đoán mỗi setup, GT xanh/pred đỏ,
  kèm TP/FP/FN và `captions.json`. Chọn ba ảnh ít lỗi và ba ảnh nhiều lỗi, không giả định
  nhóm ít lỗi luôn là dự đoán hoàn hảo.
- `report/errors/`: output thô và số lỗi từng ảnh, cờ cắt ở giới hạn token.
- Log train ở `work/<experiment>/train_log.jsonl`; log console nằm trong notebook.
  Bản pipeline cũ còn có `report/logs/` từ subprocess.
- `work/*.jsonl`: toàn bộ output thô val/test; log loss, số tham số, module tree,
  kiểm tra mask, vision token, xác nhận nạp lại projector và cấu hình thực chạy.
  E0 có train_minutes=0, thời gian suy luận ghi riêng ở run.json/inference.json.
- `data/splits/*.jsonl`, `meta.json`: nhãn và split để phân tích mà không tải lại dataset.

ZIP phân tích không chứa trọng số hay ảnh dataset gốc. Checkpoint được giữ riêng ở
`day10/work/<name>/adapter/` và `projector.pt` cho E2/E3. Download riêng nếu cần dùng tiếp.
Không commit `data/`, `work/`, file ZIP kết quả, khóa, hoặc trọng số lên repo.

`quality_checks.json` ghi ngưỡng rubric thật. Không tự sửa output khi F1/JSON chưa đạt.
`submission_check.txt` vẫn có FAIL vì REPORT chưa điền và không có lịch sử Git trong
notebook nhúng; hoàn thành báo cáo và ≥5 commit trong repo trước khi kiểm tra bài nộp.
Các CI từ script gốc bootstrap theo ảnh; không phải CI theo session hoặc kiểm định paired.

## Kiểm tra nhanh và tiếp tục phiên khác

- Có thể đặt `MAX_STEPS=2` để kiểm tra pipeline. `quality_checks.json` đánh dấu `smoke_run`;
  các kết quả đó không dùng làm bài nộp. Chạy lại bản mặc định 300 bước để lấy số liệu thật.
- Nếu cell train/evaluate lỗi, ghi `report/failure.json` kèm traceback và tạo lại ZIP
  chứa phần kết quả đã có. Xem Output của cell tương ứng để tìm bước gây lỗi.
- Để tái dùng lượt **đã hoàn thành**, Add Input Output của phiên cũ, đặt `RESUME_SOURCE`
  trỏ vào thư mục `day10` ở Input; đặt `MODEL_REVISION` bằng SHA trong
  `kaggle_run_config.json` của phiên cũ. Cấu hình, dataset_version, mã nguồn phải giống nhau.
- Chỉ lượt có `completed.json` được tái dùng. Checkpoint giữa lượt lưu trọng số phục vụ
  phục hồi thủ công; không lưu optimizer/RNG, nên không hỗ trợ resume chính xác giữa lượt.
- Thay cấu hình, đổi code, hoặc từ smoke run sang full run: bỏ RESUME_SOURCE.

## Cập nhật notebook sau khi sửa mã

```powershell
.\.venv\Scripts\python.exe build_kaggle_notebook.py
```

Builder đưa các hàm Python đã kiểm tra vào cell theo từng bước; bootstrap đóng gói công cụ
đề và bản mã nguồn cho unit test/đối chiếu. Không nhúng archive.zip, dữ liệu, môi trường,
trọng số hay khóa. Không sửa trực tiếp chuỗi PAYLOAD trong notebook.
Bootstrap dùng `REPORT_template.md` và `lab_day10_template.ipynb` để tránh đưa báo cáo
đã hoàn thành của lượt trước vào runtime mới hoặc nhúng notebook lặp vô hạn khi rebuild.
