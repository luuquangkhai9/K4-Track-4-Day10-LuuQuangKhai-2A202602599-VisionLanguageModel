# Kiểm tra kết quả và báo cáo

Nguồn: `lab_day10_results.zip` do học viên cung cấp. SHA256 của ZIP và mã nguồn thực
chạy được lưu trong `report/provenance/archive.json`. Bản mã thực chạy nằm dưới
`report/provenance/`; không ghi đè mã hiện tại của repo bằng bản trong ZIP.

## Đã kiểm tra

- Mã nguồn trong ZIP khớp `code_sha256` của cấu hình lượt chạy.
- Output val/test đủ image_id, không trùng hoặc thiếu; split và dataset_version khớp.
- Chấm lại E0, E1, E2, E3, E4 10% trên cả val/test bằng `scripts.score.evaluate`:
  mọi metric và CI95 khớp evidence, sai số so sánh tối đa 1e-12.
- E4 100% dùng lại E3, không phải lượt train thứ hai; metric khớp E3.
- Các lượt trained có đủ 300 bước optimizer thành công; số tham số, VRAM và thời gian
  trong run.json khớp meta của evidence. E3 có 2, E4 10% có 4 attempt AMP bỏ qua.
- Các session không giao nhau; subset E4 gồm 142 ảnh thực sự thuộc train.
- Đã xem các ảnh minh họa E3, ảnh pothole bổ sung và hai biểu đồ phân tích.
- Các liên kết file/ảnh trong REPORT.md đã được kiểm tra tồn tại.
- `git diff -- scripts tests` rỗng: không sửa công cụ gốc của đề.

Phân tích bổ sung dùng paired bootstrap theo ảnh 10.000 lần, seed 42; không thay thế
CI chính thức và không mô hình hóa tương quan frame trong cùng session hay biến thiên
seed huấn luyện. Các kết luận trong báo cáo nêu rõ giới hạn này.

## Kiểm tra mã và bài nộp

Lệnh kiểm tra các test còn áp dụng sau khi điền báo cáo:

```bash
python -m pytest -q tests kaggle_checks local_checks -k 'not report_template_has_student_markers_and_ten_questions'
```

Kết quả lượt kiểm tra E0–E4 trên môi trường local CPU tối giản: **78 passed, 4 skipped, 1 deselected**.
Test `test_report_template_has_student_markers_and_ten_questions` được loại khỏi lượt
kiểm tra này vì nó yêu cầu REPORT.md còn chuỗi placeholder của mẫu; báo cáo đã hoàn
thành thì yêu cầu đó không còn phù hợp. Không sửa hay vô hiệu hóa test trong file đề.
Các test phụ thuộc thư viện mô hình không có trong `.venv` tối giản được bỏ qua.

`python scripts/check_submission.py` tại thời điểm hoàn thành báo cáo E0–E4 cho **1 FAIL, 0 WARN**:
chỉ FAIL `git commits — 1/5`. Tất cả kiểm tra evidence, meta, dataset, báo cáo và
file bị cấm đều PASS. Lịch sử Git chưa được thay đổi bởi bước phân tích này.

`report/submission_check.txt` là snapshot gốc từ Kaggle, trước khi viết báo cáo;
`report/analysis/submission_check_local.txt` là snapshot local khi phân tích E5.
Các snapshot giữ trạng thái Git tại thời điểm ghi log, không phản ánh các commit
được tạo sau đó. Kiểm tra trạng thái bài nộp hiện tại bằng:

```bash
python scripts/check_submission.py
git rev-list --count HEAD
git status --short
```

## Tái tạo phân tích

```bash
python -m pip install numpy matplotlib pillow
python analyze_results.py --archive lab_day10_results.zip
```

Script nhập các artifact, kiểm tra số liệu, lưu training evidence gọn dưới
`report/resources/`, tạo `verified_analysis.json` và hai biểu đồ. Không huấn luyện lại
model hoặc thay evidence. ZIP, nhãn dưới data/ và output dưới work/ được Git bỏ qua.

## Bổ sung kiểm chứng E5

Nguồn `lab_day10_e5_results.zip`; SHA256 và bản mã thực chạy lưu riêng ở
`report/provenance/e5/`, không ghi đè provenance SFT hay mã nguồn hiện tại.

- Chấm lại E3 và E5 trên val/test, gồm CI95 bootstrap chính thức: khớp evidence.
- Dataset/split trong ZIP E5 khớp về nội dung với dữ liệu local từ lượt E0–E4.
  Input checkpoint trên Kaggle không chứa split gốc; log warm start ghi rõ điều đó
  và đã đối chiếu cached prediction/score val/test. Tại local có đủ hai ZIP để kiểm tra.
- Chấm lại toàn bộ 240 reward từ raw completion và GT train: khớp mọi trường audit.
  Có 60 nhóm, mỗi nhóm 4 completion của cùng ảnh; 60 ảnh train khác nhau.
- Đối chiếu optimizer log: 60 global step, 58 cập nhật, AMP bỏ qua step 17 và 51.
- Precision/projector check đều ghi verified=true. ZIP phân tích không có trọng số,
  nên kiểm tra giá trị tensor trước/sau dựa vào log đã ghi trong lượt GPU.
- Recompute paired ΔF1 CI95 10.000 lần: [-0.0241674, 0.0295185], khớp file comparison.
  CI chứa 0, chưa chứng minh lợi ích F1 của GRPO. Đây vẫn là bootstrap theo ảnh.
- Xem ảnh E5 đúng/sai và biểu đồ; giữ nguyên các file evidence gốc trong ZIP.

```bash
python analyze_e5_results.py --archive lab_day10_e5_results.zip
```

Script không nạp model GPU; lưu training evidence dưới `report/resources/e5/`, tạo
`e5_verified_analysis.json`, `e5_comparison.png` và `summary_with_e5.csv`.
REPORT.md được cập nhật từ số liệu này, gồm phân tích reward hacking và giới hạn CI.

Lượt kiểm tra sau khi bổ sung E5:

```bash
python -m pytest -q tests e5_checks local_checks -k 'not report_template_has_student_markers_and_ten_questions'
```

Kết quả: **95 passed, 3 skipped, 1 deselected**. Tại lượt kiểm tra E5 đó còn
**1 FAIL, 0 WARN**, chỉ `git commits — 2/5`; các mục kết quả và báo cáo đều PASS.
Checker gốc không có mục bonus E5, nên các đối chiếu bổ sung ở trên kiểm tra riêng E5.
