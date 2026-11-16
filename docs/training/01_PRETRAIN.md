# 01 — Pretrain / continued pretrain

[Quay về mục lục](README.md) · [Dữ liệu tổng quát](DATA_GENERALIST.md)

**Học gì:** dự đoán token kế tiếp từ văn bản/code; tăng độ phủ ngôn ngữ và kiến thức. Không chỉ học một danh sách sự thật. Continued pretrain có thể làm giảm khả năng chat, cần benchmark và có thể SFT phục hồi sau.

**Dữ liệu:** tài liệu nguyên vẹn, sạch, ranh giới EOT, train/val khác tài liệu; mixture đã materialize và verify. Parquet mới tải là raw, chưa được đưa trực tiếp vào trainer. [pipeline hiện có](../../data_pipeline.py) đọc kho cũ mặc định; thêm `--download-report data/downloads/general_v1/download_report.json` để đọc batch parquet mới đã hoàn tất/hash đúng. Origin 70/30 chưa được thực thi bởi domain mixture hiện tại.

**Checkpoint:** chọn parent sau benchmark/metadata, không tự chọn `jingpt_latest.pt` chỉ vì tên. Không rebuild BPE hoặc tăng kiến trúc giữa tiếp weights. Trainer pretrain hiện chỉ có khởi tạo từ checkpoint qua resume; launcher copy parent sang output MỚI rồi resume. Phải xem optimizer/config/tokenizer tương thích, không đổi phase rồi gọi là resume chuẩn. Scheduler LR tính lại theo target, validation khác thì best_loss cũ không so được.

## Chuẩn bị batch có sẵn (máy nhà, không train)

Tận dụng kho cũ, tạo version mới; lệnh có thể mất nhiều phút và KHÔNG bảo đảm tỷ lệ nguồn 70/30:

```powershell
.\.venv\Scripts\python.exe scripts/python_d.py data_pipeline.py --prepare --output-dir data/prepared_general_v1 --mb-per-domain 128 --sft-per-source 10000
.\.venv\Scripts\python.exe scripts/python_d.py data_pipeline.py --verify --output-dir data/prepared_general_v1
```

Hai lệnh trên **đã chạy xong cho `data/prepared_general_v1`** (manifest `ready`, verify pass 2026-10-01; 519.660.893 byte pretrain, 54.091 hội thoại SFT train). Nguồn mới cần bước nhập/prepare có provenance riêng trước, không copy đè corpus hay giả vờ pipeline đã đọc hết downloads.

## Kaggle: continued pretrain smoke trước

Sau khi parent được chọn và metadata xác nhận, pack bộ verified mới:

```powershell
# Thay PARENT.pt bằng training checkpoint đã kiểm tra, không dùng inference-only.
.\.venv\Scripts\python.exe scripts/python_d.py scripts/pack_kaggle.py --prepared data/prepared_general_v1 --checkpoint 'checkpoints/PARENT.pt' --output-dir kaggle_upload/pretrain_general_v1
```

**Gói hiện có (2026-10-01):** `kaggle_upload/pretrain_general_v1` (2 ZIP + upload_manifest.json), parent pin `checkpoints/kaggle_v4_60020/kaggle_v4_instruct.pt` (step 60.020, AdamW, tokenizer khớp `bpe.json`).

Upload hai ZIP + upload_manifest + đúng parent riêng vào PRIVATE Dataset. Import [notebook](../JIN_KAGGLE_V4.ipynb), chọn **`MODE='pretrain'`, `RUN_LONG=False`**. Tên notebook v4 không giới hạn prepared path: đọc `prepared_relative` từ upload manifest. Ô 3 kiểm tra tokenizer và nhận dạng optimizer, ô 4 chạy smoke 20 bước trên bản copy output mới; không lấy inference-only checkpoint.

Xem [Kaggle guide](../KAGGLE_V4_VI.md) cho setup. Chỉ chạy tiếp khi smoke pass, không bỏ asserts. **Bộ ước tính epoch SFT ở ô 5 không áp dụng pretrain**; số bước phải tính từ token được đo, không từ hội thoại hay GB. Mục tiêu token mới và giới hạn session quyết định ngân sách.

## Test / đo

- `check_project.py --group pretrain --group shared --skip-preset-shapes`: test dedupe, mixture, optimizer và hạ tầng.
- Trong run: validation loss cùng mixture; theo dõi từng domain nếu có.
- Sau run: `bench.py --checkpoint ... --device cuda --no-train-bench`; cùng frozen eval, xem forgetting VI/chat/code. Dùng `--compare A.json B.json` với đường dẫn báo cáo thật.
- Không chạy `bench.py --init-eval` để thử checkpoint; lệnh đó sửa corpus.

**Không train thêm chỉ vì loss train giảm.** Giữ parent + candidate + manifests, rồi quyết định có cần SFT tiếp hay rollback.
