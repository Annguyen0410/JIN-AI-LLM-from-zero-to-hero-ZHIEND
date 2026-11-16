# Kaggle — current workflow

**Cho bộ prepared_v4:** dùng [hướng dẫn từng bước](docs/KAGGLE_V4_VI.md) và [notebook](docs/JIN_KAGGLE_V4.ipynb). Bắt đầu smoke 20 steps, sau đó run có chủ đích; upload/download cục bộ trên D. Notebook hỗ trợ Kaggle auto-unzip và tái pin mixture Windows sang Linux sau khi kiểm tra hash.

## Generic hash-pinned launcher (2026-09-30)

Use [the hash-pinned recipe template](docs/training_run.example.json) and
[launcher](scripts/run_training.py). Fill actual paths/SHA256 values; preflight
refuses missing or mismatched inputs. Copy code to `/kaggle/working/jin` and run:

```bash
python scripts/run_training.py --config training_run.json
python scripts/run_training.py --config training_run.json --execute
```

No remote execution was performed in this upgrade. See
[verified scope and caveats](docs/UPGRADE_2026.md). Legacy explicit token-cache
overrides now require a cache manifest; opt into `JINGPT_ALLOW_UNVERIFIED_CACHE=1`
only after independently confirming tokenizer identity. Do not casually reuse
an arbitrary first glob match.

---

# Historical runbook — archived context, not current process state

The PIDs, "already running" claims, junctions and old local paths below are
historical. They do **not** describe processes currently running after restart.

# 🌙 KAGGLE RUNBOOK — ĐÊM NAY (historical)

> 🏠 **NHÀ MỚI (16/09):** mọi thứ đã dồn vào `D:\JINAI Main` — bản đồ đầy đủ ở
> `D:\JINAI Main\README.md`. Các đường dẫn cũ `D:\jin-kaggle`, `D:\jin-ckpt`,
> `D:\jin-tokencache`, `D:\kaggle`, `D:\jin-loop-dl` giờ là **junction** trỏ vào đó,
> nên mọi lệnh dưới đây vẫn chạy y nguyên, không phải sửa dòng nào.
>
> Thay đổi thật sự nằm ở chỗ khác: rác (`out46k`, `out_latest`, `out34k`, `trash`,
> checkpoint lạ) đã được đưa **ra khỏi** `D:\jin-kaggle`. Trước đây
> `kaggle datasets version -p D:\jin-kaggle` sẽ đẩy thêm ~7 GB rác lên
> `jin-train-data`, vì lệnh này upload TOÀN BỘ file trong folder. Giờ folder đó
> chỉ còn đúng payload: `dataset-metadata.json` + `jin-code.zip` + `train.txt` +
> `jingpt_10k.pt` — khớp chính xác thứ Kaggle đang giữ.
>
> Đường dẫn canonical: `D:\JINAI Main\kaggle\jin-train-data` |
> `D:\JINAI Main\kaggle\jin-ckpt` | `D:\JINAI Main\kaggle\jin-tokencache` |
> `D:\JINAI Main\kaggle\config`. Checkpoint/scratch tải về nằm ở
> `D:\JINAI Main\scratch\downloads\`.

> **Mục tiêu đêm nay:** Kaggle chạy **pretrain** 10,000 → 22,000 steps (~6h, an toàn trong 12h).
> Máy nhà chạy **SFT grounded** song song. Sáng dậy có cả base mới + chatbot biết tra mạng.
>
> **Sự thật về upload (đã kiểm chứng 13/09):** `kaggle datasets version` của CLI 2.x
> **gửi lại TOÀN BỘ file trong folder** — không có "diff only" như tôi từng nói.
> Lần này = 5.9GB (train.txt 4.94G + ckpt 1.16G + zip 0.7M) ≈ **~2 tiếng** với mạng ~800kB/s.
> ⚠️ ĐỪNG bấm `datasets version` nhiều lần — mỗi lần là resend hết.
> **Bài học:** mai mốt tách dataset — `jin-corpus` (train.txt, up 1 lần rồi để đó)
> + `jin-code` (vài MB, version thoả thích). Checkpoint về nhà qua Output tab, khỏi version.

---

## BƯỚC 1 — Một lệnh ở NHÀ (đang chạy sẵn — chỉ giám sát)

Folder `D:\jin-kaggle` đã dọn sạch: `dataset-metadata.json` + `jin-code.zip` (code MỚI)
+ `jingpt_10k.pt` + `train.txt`. Lệnh upload (PID 1824 — **đừng kill**):

```powershell
$env:KAGGLE_CONFIG_DIR="D:\kaggle"
kaggle datasets version -p D:\jin-kaggle -m "new code + ckpt10k"
```

**Theo dõi tiến độ (mọi lúc, không chặn):**
```powershell
Get-Content D:\tmp\uplog.txt -Tail 3          # thấy "Upload successful: train.txt" = XONG
Get-Process kaggle -EA SilentlyContinue       # hết hiện = tiến trình đã kết thúc
```

Sau khi XONG → Kaggle: Input → `jin-train-data` → **Version ▾ → mới nhất** → restart.

> ❗ Nếu bạn KHÔNG muốn upload gì hết: vẫn chạy được bằng cell patch bên dưới (đánh dấu 🅱️),
> nhưng **chỉ khi** BƯỚC 2 cho thấy checkpoint + cache còn nguyên trên Kaggle.

---

## BƯỚC 2 — Trên Kaggle: xem persistence còn giữ được gì (CPU session cũng được)

Mở notebook. ⚠️ **Input → jin-train-data → Version ▾ → chọn bản MỚI NHẤT** (bạn vừa push) → cho restart.

**CELL 0:**
```python
!ls -lh /kaggle/working/jin/data/train.txt /kaggle/working/jin/cache/*.u16 /kaggle/working/jin/checkpoints/*.pt 2>&1
```

| Kết quả | Ý nghĩa |
|---|---|
| Có `.u16` (2.4GB) | Cache token còn → khỏi encode 1.5h 🎉 (code mới có env override đảm bảo dùng được nó) |
| Có `jingpt_latest.pt` | Checkpoint còn → không cần copy từ input |
| Có `train.txt` | Giữ nguyên, **đừng copy đè** (mtime đổi là fingerprint đổi) |
| Thiếu gì | Cell 1 bên dưới tự bù từ dataset — chỉ cần bạn đã làm BƯỚC 1 |

---

## BƯỚC 3 — Dán 5 cell theo thứ tự

**CELL 1 — Dựng workspace (tự bù mọi thứ thiếu, không phá thứ còn):**
```python
!mkdir -p /kaggle/working/jin/data /kaggle/working/jin/checkpoints
!cp -r "$(dirname "$(find /kaggle/input -name bpe.json | head -1)")"/. /kaggle/working/jin/
!test -f /kaggle/working/jin/data/train.txt || cp -p "$(find /kaggle/input -name train.txt | head -1)" /kaggle/working/jin/data/train.txt
!test -f /kaggle/working/jin/checkpoints/jingpt_latest.pt || cp "$(find /kaggle/input -name jingpt_10k.pt | head -1)" /kaggle/working/jin/checkpoints/jingpt_latest.pt
%cd /kaggle/working/jin
!ls -lh data/train.txt checkpoints/
```
✅ Thấy `train.txt ~5.0G` + `jingpt_latest.pt ~1.2G` là đạt.
(Nếu `cp jingpt_10k` lỗi "cannot stat" = bạn chưa làm BƯỚC 1 mà checkpoint trên Kaggle cũng mất → hoặc upload, hoặc 🅱️ hết đường, phải làm BƯỚC 1.)

**CELL 2 — GPU check:**
```python
import torch
print("GPU:", torch.cuda.get_device_name(0))
```

**CELL 3 — 🅱️ PATCH (chỉ khi bạn KHÔNG upload code mới — code mới sẽ in "Already fixed"):**
```python
p = "/kaggle/working/jin/train.py"
src = open(p).read()
if "Honor --batch-size" not in src:
    anchor = '    print(f"[Train] Mode: pretrain'
    fix = ("    # Honor --batch-size / --grad-accum in PRETRAIN\n"
           "    config.BATCH_SIZE = args.batch_size\n"
           "    config.GRAD_ACCUM = args.grad_accum\n")
    open(p, "w").write(src.replace(anchor, fix + anchor))
    print("Patched!")
else:
    print("Already fixed")
```

**CELL 4 — 🔥 PRETRAIN (cell chính, ~5.5-7h):**
```python
import glob, os
c = glob.glob("/kaggle/working/jin/cache/*.u16")
if c: os.environ["JINGPT_TOKEN_CACHE"] = c[0]
!python train.py --device cuda --optimizer muon --batch-size 8 --grad-accum 2 --resume --run-iters 12000
```

**5 dòng vàng trong log (thấy hết = mọi thứ đúng):**
```
[Cache] Hit -> ...stream.u16 (1,280,141,491 tokens)   ← không encode lại (nếu [Encode] chạy thì cũng OK, ~1.5h)
[Resume] Continuing from iteration 10,000             ← kế thừa công sức Kaggle cũ
[Train] Mode: pretrain | batch 8x2 | ... target 22,000 ← batch fix生效
step 10,000 | ... loss ~5.x
[Checkpoint] Saved -> ...                             ← mỗi 1000 steps, atomic, hết chết vì lock
```

**CELL 5 — 🌅 Sáng hôm sau: gom thành quả về dl folder (chỉ khi chạy interactive; commit thì vào thẳng Output tab):**
```python
!mkdir -p /kaggle/working/dl
!cp /kaggle/working/jin/checkpoints/*.pt /kaggle/working/dl/
!ls -lh /kaggle/working/dl/
```

---

## BƯỚC 4 — Bấm nút quan trọng nhất

Sau khi Cell 1-4 chạy tay OK (đặc biệt Cell 4 thấy `batch 8x2`):
→ **Save Version → Save & Run All (Commit) → Save** → **đóng laptop đi ngủ**.
Session mới sẽ chạy lại từ đầu các cell: cache hit tức tốc (Cell 4 có env override) → train 6h → **tự lưu Output**.

Sáng: tab **Output** → tải `jin/checkpoints/jingpt_latest.pt` (≈22,000) về `D:\jin-kaggle\`:
```powershell
Move-Item "$env:USERPROFILE\Downloads\jingpt_latest.pt" "D:\jin-kaggle\jingpt_22k.pt" -Force
```
Hoặc CLI: `kaggle kernels output annguyen0410/notebookc6ea0ca54d -p "D:\jin-kaggle\out"`

---

## 🏠 SONG SONG ở nhà (GPU nhà + Kaggle = 2 máy 2 việc)

```powershell
cd "D:\Code folder\Jin New"; .\.venv\Scripts\Activate.ps1
# 1) Sinh data "đọc web trả lời" (~1h, search thật, chạy nền được)
python build_grounded_data.py --max-questions 1500
# 2) SFT grounded (nối từ step 9,200)
python train.py --mode instruct --resume --steps 4000 --device cuda --fast --batch-size 1 --grad-accum 8 --max-convs 60000 2>&1 | Tee-Object -FilePath train_log.txt
echo "EXIT=$LASTEXITCODE"
# 3) Test web: Web search = ON → hỏi "What is the price of RTX 5060?" → phải ra $ + [1]
python app.py --checkpoint "checkpoints\jingpt_chat.pt"
```

---

## 🔄 VÒNG LẶP ĐÊM — 22k → 34k → 46k → ... 100k+

> Mỗi đêm = **1 commit ~7h** (84 phút encode + ~5.5h train 12k steps).
> Vấn đề của cách thô: download 1.2GB rồi up lại vào dataset 5.9GB = resend cả train.txt!
> **Giải pháp: 2 dataset NHỎ riêng** (đã tạo sẵn metadata ở nhà):

### Setup 1 lần — sáng mai sau khi commit 22k xong

```powershell
# 1. Tải jingpt_latest.pt (base 22k) + file cache từ Output về, xếp vào folder:
New-Item -ItemType Directory -Force "D:\jin-ckpt","D:\jin-tokencache" | Out-Null
Move-Item "$env:USERPROFILE\Downloads\jingpt_latest.pt" "D:\jin-ckpt\jingpt_latest.pt"
Move-Item "$env:USERPROFILE\Downloads\tokens_*.stream.u16" "D:\jin-tokencache\" -EA SilentlyContinue
# 2. Tạo 2 dataset (lần đầu dùng create; các đêm sau chỉ version folder jin-ckpt)
$env:KAGGLE_CONFIG_DIR="D:\kaggle"
kaggle datasets create -p D:\jin-ckpt
kaggle datasets create -p D:\jin-tokencache
```

### 🌙 Người gác đêm (`D:\tmp\night_loop.ps1`)

- Watcher chạy **trên máy nhà** → cần máy THỨC mới làm việc. Máy tắt cũng **không sao**: Kaggle tự train xong và Output chờ nguyên đó.
- Sáng bật máy → nhắn tôi "sáng rồi" hoặc tự chạy 4 lệnh catch-up: `kernels status` (phải COMPLETE) → `kernels output -o --file-pattern "jingpt_latest\.pt"` → `Copy-Item` vào `D:\jin-ckpt\jingpt_latest.pt` → `datasets version -p D:\jin-ckpt`.
- Muốn watcher tự chạy đêm: Settings → Power → *When plugged in, PC sleeps* = **None**.
- Watcher chỉ chạy 1 đêm/1 lần bật; đêm sau nhắn tôi mồi lại.

### Công thức mỗi đêm

1. **Sáng:** commit đêm trước xong → tải `jingpt_latest.pt` đè vào `D:\jin-ckpt\`
2. **Tối:** `kaggle datasets version -p D:\jin-ckpt -m "step 34k"` (chỉ 1.2GB — folder này không có train.txt!) → vào notebook: **cả 3 dataset đều Update Version ▾**
3. Chạy Cell B' → E → Save & Run All → ngủ

**Cell B' (bản thay thế cho các đêm loop — copy ckpt + cache từ 2 dataset nhỏ):**
```python
!mkdir -p /kaggle/working/jin/data /kaggle/working/jin/checkpoints /kaggle/working/jin/cache
!cp -r "$(dirname "$(find /kaggle/input -name bpe.json | head -1)")"/. /kaggle/working/jin/
!cp "$(find /kaggle/input -path '*jin-train-data*' -name train.txt | head -1)" /kaggle/working/jin/data/train.txt
!cp "$(find /kaggle/input -path '*jin-ckpt*' -name jingpt_latest.pt | head -1)" /kaggle/working/jin/checkpoints/jingpt_latest.pt
!cp "$(find /kaggle/input -path '*jin-tokencache*' -name '*.u16' | head -1)" /kaggle/working/jin/cache/ 2>/dev/null || true
%cd /kaggle/working/jin
!ls -lh checkpoints/ cache/
```
Cell E giữ nguyên (`--resume --run-iters 12000`). Cache .u16 copy vào → Cell E glob thấy → `JINGPT_TOKEN_CACHE` hit → **khỏi encode 84 phút** 🎉 → mỗi đêm chỉ tốn ~5.5h train.

> ℹ️ LR mỗi đêm restart từ đỉnh 2e-4 rồi decay — đó là nhịp **re-warm + re-decay** có chủ đích
> (continual learning), không phải bug.

### Mỗi +10-12k base steps — "hồi sinh chat" ở nhà (30 phút)

```powershell
# base mới vừa tải về (VD 34k): re-SFT nhanh + đo
python train.py --mode instruct --init-from "D:\jin-kaggle\jingpt_34k.pt" --steps 3000 --device cuda --fast --batch-size 1 --grad-accum 8 --max-convs 60000
python bench.py --checkpoint "checkpoints\jingpt_chat.pt"
python bench.py --compare bench_results\baseline_v25.json bench_results\<file_moi>.json
```

### 🛑 Dấu hiệu nên DỪNG pretrain & đổi chiến thuật

- val best **không cải thiện 3 đêm liên tiếp** → thêm data chất lượng (teacher distillation) có ích hơn thêm steps
- tự sinh 5 câu test thấy văn tệ hơn đêm trước → Ctrl+C pretrain, giữ checkpoint best
- chạm **val ~4.0** → model 147M gần bão hòa data hiện tại → chuyển sang preset `medium` (360M) hoặc distill từ base khác


---

## ⛑ NẾU CÓ SỰ CỐ

| Triệu chứng | Xử lý |
|---|---|
| Session Kaggle chết giữ step X (timeout/kill) | **Không mất gì** nếu là COMMIT đã "Run successfully" (Output có hết). Nếu bị kill giữa chừng → sáng chạy lại cell 1→4, `--resume` từ checkpoint mới nhất trong working |
| `[Cache] Miss` + encode chạy | Bình thường — ~1.5h trên T4, trong ngân sách 12h |
| `batch 1x2` trong log | Cell 3 chưa chạy / chưa upload code — quay lại |
| `cp: jingpt_10k cannot stat` | Chưa BƯỚC 1 → làm đi (1.2GB, không phải 5.8GB) |
| Upload version ở nhà lỗi giữa đường | Chạy lại y nguyên lệnh BƯỚC 1 — CLI **không resume**, sẽ gửi lại từ đầu (đừng bấm lặp lại liên tiếp khi nó đang chạy) |
| Đào tạo tụt về 10k / encode lại 84 phút | Dán NHẦM cell 2 bản cũ (`jingpt_10k.pt`) — từ đêm 2 phải dùng **Cell B'** copy từ dataset `jin-ckpt` + `jin-tokencache` |
| `kaggle datasets create` báo `Expecting value: line 1 column 1` | **Bug CLI 2.2.4 với dataset CHƯA tồn tại** (bước kiểm tra trả 403, CLI parse JSON nổ). Không phải lỗi token! Cách giải quyết: tạo dataset rỗng 1 lần qua **web** (New Dataset + file placeholder), từ đó về sau chỉ dùng `kaggle datasets version -p <folder> -m "..."` (loại này chạy hoàn hảo, đã test 5.9GB) |
| Token KGAT_ chỉ đọc: mọi lệnh ghi báo 403 `Forbidden`, đọc vẫn OK | Xóa/đổi tên `access_token` trong `KAGGLE_CONFIG_DIR` để CLI rơi về OAuth cache toàn quyền; hoặc tạo key mới có scope write |
| `kaggle datasets version` báo `Expecting value: line 1 column 1 (char 0)` NGAY KHI CHẠY | **`dataset-metadata.json` nhiễm BOM** (PowerShell `Set-Content -Encoding UTF8` thêm 3 byte `EF BB BF` đầu file, Python json không đọc được). Fix: viết lại bằng `-Encoding ASCII`. Kiểm tra: 3 byte đầu phải là `123 34 116` (`{"t`) |
| `cp: cannot stat jingpt_latest.pt` trong Cell B' | Quên Add Input 2 dataset mới, hoặc quên Update Version ▾ sau khi `datasets version` |
| Không còn quota GPU | Đổi sang nhà: `python train.py --data data/train.txt --resume --device cuda --batch-size 4 --grad-accum 4` (encode 1 lần ~1.5-2h rồi cache) |

## 📏 Luật bất biến (in ra dán tường)

1. **≤ 12,000 steps/session** — commit chỉ lưu output khi **chạy xong**, chết giữa chừng = mất
2. **Save & Run All đúng MỘT lần** — bấm 2 lần = 2 session đốt quota gấp đôi
3. 🆘 **DRAFT session = chưa lưu gì cả.** Log draft lặp lại y hệt từng mốc giây là lỗi RENDER của viewer (browser reconnect), KHÔNG phải restart — nhưng checkpoint chỉ thật sự an toàn khi đã COMMIT xong hoặc tải qua dl folder. Vừa train vừa muốn ngủ → BẤM COMMIT, đừng tiếc steps trong draft.
4. Sau khi push dataset version mới → **nhớ Update Input Version ▾** trong notebook
5. Val loss **nhảy 1 lần rồi giảm tiếp là noise** — nhìn trend 3-4 lần eval liên tiếp
6. Mọi file nặng luôn nằm **D:** (`D:\tmp`, `D:\kaggle`, `D:\jin-kaggle`) — kiểm tra `pip cache dir` = `d:\pip-cache`

---

## 🗺️ Bản đồ đường xa (JINAI 2026 → 2056)

```
Hôm nay:   base 10k→22k (Kaggle) + SFT grounded (nhà) → web chat biết tra mạng
Tuần này:  resume 22k→46k→...→100k (Kaggle reset 30h/tuần = ~2 phiên/tuần)
           → mỗi +10k pretrain steps: re-SFT nhanh 3-4k steps + bench
Tháng tới: thử preset medium (360M) khi quota cho phép — cùng pipeline, to hơn
Val loss cần chạm:  ~5.0 (đọc hiểu cơ bản) → ~4.3 (trôi chảy) → ~3.8 (khá "thông minh")
```

*Chuẩn bị ở nhà đã xong 100% (zip mới 0.7MB + checkpoint đặt cạnh). Việc duy nhất của bạn: BƯỚC 1 → BƯỚC 2 → 5 cell → bấm nút → ngủ. Chúc JINAI ngủ ngon cùng bạn đêm nay!*
