# Lưu dữ liệu JIN trên ổ D

## Vị trí hiện tại

Project: `D:\Code folder\Jin New`.

| Nội dung | Thư mục bên trong project |
|---|---|
| Corpus, dataset tải/chuyển đổi, prepared data, memory | `data/` |
| Parquet gốc / HF downloader | `cache/hf/` |
| Token cache | `cache/tokens_*` |
| Checkpoint / optimizer state | `checkpoints/` |
| Training manifests / reports | `runs/`, `bench_results/` |
| File tạm của Python và subprocess | `cache/tmp/` |
| Hugging Face library cache | `cache/huggingface/` |
| Torch / Inductor, CUDA, Triton, Numba | `cache/torch/`, `cache/cuda/`, `cache/triton/`, `cache/numba/` |
| pip download/build cache | `cache/pip/` |
| Môi trường Python của project | `.venv/` |

Xem [báo cáo đường dẫn thực tế](../bench_results/storage_report.json).

## Chạy lệnh theo cách bảo đảm môi trường ngay từ đầu

Trong PowerShell tại project:

```powershell
# Xem đường dẫn và dung lượng trống, không tải/train.
.\.venv\Scripts\python.exe scripts/python_d.py --storage-report

# Thay lệnh python fetch_datasets.py ... bằng:
.\.venv\Scripts\python.exe scripts/python_d.py fetch_datasets.py --catalog hf_datasets.json --only fineweb2-vi --sample-rows 1000 --total-download-mb 64

# Kiểm tra prepared data:
.\.venv\Scripts\python.exe scripts/python_d.py data_pipeline.py --verify --output-dir data/prepared_v4

# Khi thực sự cần cài requirements, chủ động chạy (chưa chạy trong phiên này):
.\.venv\Scripts\python.exe scripts/python_d.py -m pip install -r requirements.txt
```

Git Bash: dùng `./.venv/Scripts/python.exe` thay đường dẫn PowerShell. Sau khi activate `.venv`, cũng có thể dùng `python scripts/python_d.py ...`.

Launcher dùng chính Python đã gọi nó, chạy từ project root và đặt biến môi trường cache/temp trước khi Python con import thư viện. **Không dùng system Python hoặc `pip --user`** nếu muốn package cài vào `.venv` trên D. Bare pip chạy ở terminal khác không tự nhận cấu hình của launcher.

## Các chốt an toàn

[storage.py](../storage.py) áp dụng chính sách:

- Windows: đầu ra được bảo vệ phải resolve trên **D:**. Linux/Kaggle không ép ký tự ổ đĩa Windows.
- Đường dẫn tương đối gắn với project root, không phụ thuộc terminal đang đứng ở C hay D.
- Resolve symlink/junction trước khi ghi: đường dẫn nhìn như D nhưng trỏ sang C bị từ chối.
- Các đường ghi lớn của downloader, prepared data, corpus builders, token caches, checkpoint saves/exports và launcher training được kiểm tra. `--output-dir C:/...`, checkpoint hoặc token-cache override trên C sẽ báo lỗi thay vì ghi âm thầm.
- `config.py` cấu hình cache và `TEMP/TMP/TMPDIR` cho tiến trình JIN; `tempfile.tempdir` cũng được cập nhật để không giữ C:/Temp đã cache từ trước.
- `scripts/python_d.py` là cách khuyến nghị để cấu hình **trước tất cả imports**, kể cả pip/compiler hoặc entrypoint import torch trước config.
- Chỉ đổi môi trường của tiến trình và tiến trình con. **Không chỉnh registry, không đổi TEMP toàn Windows, không di chuyển/xóa cache cũ của app khác.**

Không phải sandbox cấp hệ điều hành: chương trình bên ngoài, native tool bỏ qua biến môi trường, trình duyệt tải vào Downloads, crash dumps hoặc pagefile của Windows vẫn có thể ghi C. Không sửa Windows pagefile trong thay đổi này. Process/IDE đang mở trước thay đổi phải khởi động lại bằng launcher để nhận môi trường.

Dữ liệu lớn hiện có trong project đã ở D nên không cần sao chép lại. Không tự dò/xóa file ngoài project. Nếu muốn chuyển cache cũ bên ngoài, cần kiểm kê và xác nhận riêng trước.

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe scripts/check_project.py --report bench_results/storage_tests.json
```

[Tests](../tests/test_storage.py) kiểm tra D/case-insensitive, chặn C/E/UNC, junction trỏ sang C, anchoring đường dẫn tương đối, cache/temp process-local, và từ chối ghi/tải trước khi bắt đầu I/O lớn. Xem [kết quả regression](../bench_results/storage_tests.json). Không có static typechecker cấu hình; chạy compile cú pháp và executable tests.
