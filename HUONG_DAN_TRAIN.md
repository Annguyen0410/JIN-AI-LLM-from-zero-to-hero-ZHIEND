# HƯỚNG DẪN RUN TỪNG BƯỚC — sau đợt sửa lỗi MemoryError + URL 404

> Mọi file tải về, checkpoint, cache **đều nằm trong `D:\Code folder\Jin New`**.
> Thứ từng bị ghi vào ổ C là **cache của pip** (file cài torch/pyarrow, nặng vài GB) —
> Bước 1 sẽ dời nó sang D.

---

## Bước 0 — Mở PowerShell + kích hoạt venv + kiểm tra code

```powershell
cd "D:\Code folder\Jin New"
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests
```

- Kết quả mong đợi: `Ran 149 tests ... OK` (chạy ~1 phút, CPU).
- Nếu OK → sang Bước 1.

---

## Bước 1 — Dời pip cache sang ổ D (giải phóng ổ C)

```powershell
# Xoá cache cũ trên C (có thể đã chiếm 2-5GB do wheel torch/pyarrow)
pip cache purge

# Từ giờ pip lưu cache vào D
pip config set global.cache-dir "D:\pip-cache"
```

Kiểm tra C còn trống bao nhiêu: `Get-PSDrive C`

---

## Bước 2 — Cài PyTorch bản CUDA (nếu chưa)

```powershell
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

- In `CUDA: True` → tốt.
- In `False` → cập nhật NVIDIA driver (NVIDIA App hoặc nvidia.com/drivers) rồi chạy lại.

---

## Bước 3 — Tải nốt dữ liệu bị 404 (đã sửa URL)

Lần trước 404 vì: Cosmopedia đã chuyển repo (`smollm-corpus`), Wikipedia VI có **4 shard** (không phải 2). Đã sửa trong `hf_datasets.json`.

```powershell
python fetch_datasets.py --catalog hf_datasets.json --only cosmopedia-v2-sample wikipedia-vi-sample wikipedia-vi-shard2 wikipedia-vi-shard3 wikipedia-vi-shard4
```

- ~30-60 phút tuỳ mạng. Tải thêm ~3GB vào `cache\hf\` (ổ D).
- File convert nằm ở `data\raw\hf-*` (pretrain) — vẫn ổ D.
- Lỗi 1 nguồn không ảnh hưởng nguồn khác.

---

## Bước 4 — Build corpus pretrain (đã stream, hết MemoryError)

```powershell
python build_dataset.py --append-existing --backup-existing --snapshot --report
```

- Không load file lớn vào RAM nữa → TinyStories 2.2GB chạy thru bình thường.
- **Không cần chạy `data_quality.py` trên corpus lớn nữa** (nó sẽ chặn input >1GB).
  Contamination check đã tự chạy trong lệnh trên (từng file).
- Nghiêm cửa sổ Chrome/trình duyệt khi chạy để có nhiều RAM trống.

### Tuỳ chọn: giới hạn corpus nếu GPU yếu

Corpus đầy đủ ~5-7GB (~1.5B tokens) — train 1 epoch trên RTX 2060 mất ~10+ tiếng.
Muốn nhanh hơn, build lại với cap mỗi file (VD 400MB):

```powershell
python build_dataset.py --backup-existing --snapshot --report --max-mb-per-file 400
```

→ Corpus ~2-3GB, vẫn 20× lớn hơn corpus cũ 76MB.

---

## Bước 5 — Carve held-out set (chạy 1 lần, trước khi train)

```powershell
python bench.py --init-eval
```

Tách ~2% corpus cuối ra `data\eval\heldout.txt` để đo val loss trung thực.

---

## Bước 6 — Train BPE tokenizer (sample, không cần load cả corpus)

```powershell
python bpe_tokenizer.py --data data/train.txt --out bpe.json --vocab-size 16384
```

- Tự sample 60MB rải đều khắp corpus (kể cả phần tiếng Việt) — RAM nhẹ.
- Muốn sample lớn hơn: thêm `--sample-mb 120`.

> ⚠️ BPE mới = vocab mới → checkpoint cũ không load được. Train lại từ đầu là đúng.

---

## Bước 7 — Pretrain (stream-encode, memmap cache — RAM ổn định)

```powershell

python train.py --device cuda --optimizer muon --batch-size 1 --grad-accum 16
```

- Lần đầu sẽ thấy `[Cache] Streaming encode ... (constant memory, one-time)` —
  encode 1 lần (~20-60 phút tuỳ corpus) rồi cache ra `cache\*.stream.u16`, lần sau hit.
- Muốn ép CPU (không có GPU): bỏ `--device cuda` và dùng preset nhỏ:
  `$env:JINGPT_SCALE='tiny'; python train.py`

---

## Bước 8 — Build chat corpus + Instruct tuning

```powershell
python build_instruct_dataset.py --no-literary
python train.py --mode instruct --steps 5000 --device cuda --batch-size 1 --grad-accum 32
```

- Chat corpus sẽ mix: OpenHermes + MetaMathQA + Bactrian-X VI + OASST2 + Alpaca…
  (mỗi nguồn lấy tối đa 20k hội thoại — đã cap sẵn).
- Het VRAM → thêm `--fast` (giảm block-size còn 256).

---

## Bước 9 — Đo kết quả

```powershell
python fluency_check.py
python bench.py --checkpoint checkpoints/jingpt_chat.pt
python bench.py --compare bench_results/baseline_v25.json bench_results/<file_mới>.json
```

So sánh probe pass rate (cũ 10.9%) + held-out loss (cũ 11.86) + fluency (cũ 0.768).

---

## Bước 10 (tuỳ chọn) — Kaggle GPU miễn phí 16GB

> 📘 **Hướng dẫn siêu chi tiết nằm ở file `KAGGLE_GUIDE.md`** (cùng thư mục) —
> làm được từ đầu tới cuối kể cả lần đầu dùng Kaggle: giữ ổ C không đầy,
> đóng gói + upload bằng kaggle CLI, cell code cho cả preset `small` và `medium`,
> Save & Run All (tắt máy về ngủ), vòng lặp resume nhiều buổi cho `medium`,
> và bảng bẫy thường gặp.

Tóm tắt: kaggle.com → verify SĐT → upload dataset (code + train.txt) →
New Notebook + GPU T4 x2 → chạy `train.py --device cuda --optimizer muon` →
tải checkpoint về `checkpoints\`. Quota 30h GPU/tuần, session ≤ 12h.

---

## Troubleshooting

| Triệu chứng | Cách xử lý |
|---|---|
| `CUDA out of memory` | `--batch-size 1 --grad-accum 32`, thêm `--fast`, hoặc `--low-vram` |
| `torch.cuda.is_available() is False` | Cập nhật driver NVIDIA ≥ 552.x, cài lại wheel cu124 |
| Corpus build chậm | Bình thường với vài GB — mọi thứ stream ra đĩa, theo dõi file `data\train.txt.building` lớn dần |
| Muốn dataset khác | Sửa `enabled` trong `hf_datasets.json` rồi `python fetch_datasets.py --catalog hf_datasets.json` |

## Những gì đã sửa trong đợt này

1. **URL 404**: Cosmopedia → repo `smollm-corpus`; Wikipedia VI → đủ 4 shard; bỏ dump abstracts-2 (không tồn tại).
2. **MemoryError**: `build_dataset.py`, `build_instruct_dataset.py`, `data_quality.py`, `bpe_tokenizer.py`, `train.py` đều chuyển sang **streaming/ sample** — không load corpus nhiều GB vào RAM.
3. **data_quality bị cắt corpus 76MB → 493KB**: doc dài bị đánh dấu `too_long` → giờ segment thay vì bỏ; chặn input >1GB (dùng cho chat corpus thôi).
4. Token cache corpus lớn lưu **uint16 memmap** (RAM thấp), convert per-batch.
5. OpenHermes dùng converter `json_messages` stream (file 1.9GB JSON).
6. pip cache dời sang `D:\pip-cache` (Bước 1).
