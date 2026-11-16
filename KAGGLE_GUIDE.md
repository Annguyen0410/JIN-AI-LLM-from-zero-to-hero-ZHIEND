# Hướng dẫn Kaggle — dùng workflow v4 hiện tại

**Bắt đầu tại [hướng dẫn Kaggle v4](docs/KAGGLE_V4_VI.md) và [notebook sẵn có](docs/JIN_KAGGLE_V4.ipynb).** Workflow mới đóng gói gọn trên D, dùng hash/validation riêng, không patch source, không đoán glob, không ép thay cấu hình Windows toàn máy. Phần bên dưới là hướng dẫn lịch sử: các quota/tốc độ/đường dẫn và lời hứa chất lượng chưa được xác minh cho run hiện tại; không thực hiện lệnh dọn/reset hoặc sửa pagefile theo phần cũ.

---

# 🖥️ HƯỚNG DẪN KAGGLE GPU — phiên bản lịch sử

> **Mục tiêu:** Dùng máy tính miễn phí của Google (Kaggle) — mạnh gấp 5 lần con
> RTX 2060 nhà bạn — để train Jin. **Tốn 0 đồng.**
>
> Cách đọc: làm **đúng thứ tự** từ trên xuống. Mỗi bước có:
> - 🔵 phần giải thích "tại sao" (đọc để hiểu, không cần làm gì)
> - ⬛ khối lệnh/lời dẫn để bạn **copy-paste hoặc bấm theo**
> - ✅ dấu tick để bạn tự đánh dấu đã xong

---

## Trước khi bắt đầu — 3 khái niệm cần hiểu

| Khái niệm | Ví dụ đời thường |
|---|---|
| **Kaggle** | Giống YouTube, nhưng thay vì up video thì up "sổ tay code" (notebook) để máy chủ Google chạy giúp bạn |
| **Dataset trên Kaggle** | Một "túi đồ" bạn gửi trước lên máy chủ: chứa code + dữ liệu train |
| **Notebook** | Trang giấy điện tử, bạn viết từng ô lệnh (cell), bấm chạy từng ô. Có gắn GPU miễn phí |

**Luồng tổng thể (nhìn 1 cái là hiểu):**

```
NHÀ BẠN                          MÁY CHỦ KAGGLE (miễn phí)
─────────                        ─────────────────────────
1. Nén code + corpus  ──upload──▶  Túi đồ (dataset)
2. Tạo notebook + chọn GPU
3. Bấm chạy                    ──▶  Train hàng giờ (không cần giữ máy mở*)
4. Tải checkpoint về           ◀──  File .pt
5. Chép vào D:\...\checkpoints\
   → về nhà tiếp Bước 8, 9

* khi dùng chế độ "Save & Run All" (Phần 7)
```

---

# PHẦN 0 — Chặn không cho ổ C đầy (làm TRƯỚC, 5 phút)

> **Tại sao:** Ổ C của bạn chỉ còn trống ~13GB. Các công cụ (kaggle CLI, file tạm
> Windows) mặc định vứt rác vào ổ C. Mình **dổi nhà rác sang ổ D** (D còn trống 657GB).

### 0.1 — Tạo 2 thư mục đích trên D

⬛ Mở **PowerShell**, dán lệnh này (1 lần thôi):

```powershell
New-Item -ItemType Directory -Force "D:\pip-cache" | Out-Null
New-Item -ItemType Directory -Force "D:\tmp" | Out-Null
New-Item -ItemType Directory -Force "D:\kaggle" | Out-Null
New-Item -ItemType Directory -Force "D:\jin-kaggle" | Out-Null
echo "Đã tạo xong 4 thư mục trên ổ D"
```

✅ Thấy 4 dòng "Đã tạo xong..." hoặc không báo lỗi.

### 0.2 — Ra lệnh cho pip và file tạm nằm ở D **vĩnh viễn**

⬛ Dán tiếp (1 lần thôi — lệnh này ghi vào cấu hình Windows):

```powershell
# pip cache -> D (nếu chưa làm ở Bước 1 HUONG_DAN_TRAIN.md)
pip config set global.cache-dir "D:\pip-cache"

# File tạm người dùng -> D (đặt vào PowerShell Profile để tự áp dụng mỗi lần mở)
$line1 = '$env:TEMP="D:\tmp"; $env:TMP="D:\tmp"'
if (-not (Test-Path $PROFILE)) { New-Item -ItemType File -Path $PROFILE -Force | Out-Null }
Add-Content -Path $PROFILE -Value $line1
echo "Xong. Đóng PowerShell và mở lại để có hiệu lực."
```

⬛ **Đóng PowerShell, mở lại**, rồi gõ `echo $env:TEMP` → phải in ra `D:\tmp` ✅

### 0.3 — Lệnh "giữ C sạch" dùng mỗi phiên (dán trước khi chạy lệnh kaggle)

```powershell
$env:TEMP="D:\tmp"; $env:TMP="D:\tmp"; $env:KAGGLE_CONFIG_DIR="D:\kaggle"
```

> 🔵 `$env:KAGGLE_CONFIG_DIR` = chỗ kaggle CLI để "chìa khóa" (token API) —
> đưa về D luôn cho gọn.

### 0.4 — (Tùy chọn, khuyến khích) Dời pagefile sang D — giải phóng ~11GB trên C

> 🔵 pagefile.sys là "RAM ảo" của Windows, chiếm 10.8GB trên C của bạn.
> Dời sang D vừa đỡ ổ C, vừa giúp các lệnh build dataset bớt MemoryError.

1. Bấm `Win` + gõ **"Advanced system settings"** → Enter
2. Tab **Advanced** → nút **Settings** (trong nhóm Performance)
3. Tab **Advanced** → nút **Change** (Virtual memory)
4. **Bỏ tick** "Automatically manage..."
5. Chọn ổ **C:** → chọn **No paging file** → **Set**
6. Chọn ổ **D:** → chọn **System managed size** → **Set**
7. **OK** hết → khởi động lại máy
8. Sau khởi động lại: mở PowerShell gõ `Get-ChildItem C:\pagefile.sys -Force` → không thấy = thành công ✅

---

# PHẦN 1 — Chọn "nhánh" train (đọc 2 phút, chọn 1)

> 🔵 Cùng 1 corpus (1.28 tỷ token). Model lớn hơn = thông minh hơn nhưng lâu hơn.

| | **Nhánh A: `small` (147M)** | **Nhánh B: `medium` (360M)** |
|---|---|---|
| Tương đương | GPT-2 small | SmolLM-class (2024) |
| Tốc độ trên Kaggle T4 | ~10.000-15.000 token/s | ~2.500-3.500 token/s |
| 100.000 steps (~410M tokens) | **~8-11 giờ = 1 buổi** | ~32-45 giờ = **3-4 buổi** |
| VRAM cần | ~5-7GB (T4 16GB dư sức) | ~10-14GB (T4 16GB vừa đủ) |
| Khuyến nghị | **Chọn nếu lần đầu** — thấy kết quả nhanh | Chọn khi đã quen quy trình |

> 💡 **Mẹo của tôi:** Làm Nhánh A trước để học quy trình 1 buổi. Sau đó quay lại
> Nhánh B "ngon" hơn. Quy trình 2 nhánh y hệt nhau, chỉ khác 1 dòng lệnh.

---

# PHẦN 2 — Đóng gói đồ lên đường (15 phút, ở nhà)

### 2.1 — Tạo thư mục đóng gói

⬛ PowerShell (nhớ dán Phần 0.3 trước!):

```powershell
cd "D:\Code folder\Jin New"
Remove-Item -Recurse -Force "D:\jin-kaggle\pack" -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force "D:\jin-kaggle\pack\data\seed" | Out-Null
New-Item -ItemType Directory -Force "D:\jin-kaggle\pack\data\eval"  | Out-Null
```

### 2.2 — Chép đúng những thứ cần

⬛ Dán:

```powershell
# 1) Toàn bộ code .py + file cấu hình
Copy-Item *.py, bpe.json, requirements.txt, sources.json, hf_datasets.json "D:\jin-kaggle\pack\"

# 2) Eval + seed (nhẹ, cần cho bench + instruct)
Copy-Item data\eval\*   "D:\jin-kaggle\pack\data\eval\"
Copy-Item data\seed\conversations.jsonl, data\seed\fluency_pack.jsonl, data\seed\knowledge_pack.jsonl "D:\jin-kaggle\pack\data\seed\"

# 3) Seed HF (cho instruct tuning sau; nặng ~vài trăm MB)
Copy-Item data\seed\hf "D:\jin-kaggle\pack\data\seed\hf" -Recurse

echo "Đã chép xong"
```

> ⬛ **KHÔNG chép** (cố tình!): `.venv`, `cache`, `data\raw`, `__pycache__`, `.git`,
> `checkpoints`, `data\train.txt` — các cái này hoặc Kaggle không cần, hoặc đưa riêng.

### 2.3 — Nén code thành 1 file zip

⬛ Dán:

```powershell
Compress-Archive -Path "D:\jin-kaggle\pack\*" -DestinationPath "D:\jin-kaggle\jin-code.zip" -Force
# Kiểm tra dung lượng (mong đợi vài trăm MB - 1GB vì seed/hf)
"{0:N0} MB" -f ((Get-Item "D:\jin-kaggle\jin-code.zip").Length/1MB)
```

✅ File zip được tạo, in ra số MB.

### 2.4 — Copy corpus vào vị trí upload (file trần, KHÔNG nén)

> 🔵 Corpus 5GB nén zip chỉ nhỏ hơn chút mà mất công giải nén trên máy chủ.
> Để **file trần** ở tầng gốc — kaggle CLI upload trực tiếp, không ngốn TEMP ổ C.

```powershell
Copy-Item "D:\Code folder\Jin New\data\train.txt" "D:\jin-kaggle\train.txt"
```

### 2.5 — Tạo "giấy khai sinh" cho dataset

⬛ Tạo file `D:\jin-kaggle\dataset-metadata.json` với nội dung:

```json
{
  "title": "jin-train-data",
  "id": "annguyen0410/jin-train-data",
  "licenses": [ { "name": "CC0-1.0" } ]
}
```

> ⚠️ Thay `TEN_BAN` = tên tài khoản Kaggle của bạn (chữ thường, không dấu cách) —
> sẽ có sau Phần 3.4. Làm Phần 3 trước, quay lại điền chỗ này sau nếu chưa biết tên.

✅ Có file `dataset-metadata.json` đã sửa tên.

---

# PHẦN 3 — Đăng ký Kaggle + chìa khóa API (15 phút)

### 3.1 — Tạo tài khoản

1. Mở trình duyệt → vào **https://www.kaggle.com**
2. Bấm **Sign Up** → đăng ký bằng email Google/GitHub (miễn phí)

### 3.2 — Xác minh số điện thoại (BẮT BUỘC để mở GPU!)

> 🔵 Không xác minh SĐT = không thấy nút chọn GPU. SĐT chỉ dùng để chống spam,
> không ai gọi cho bạn.

1. Vào **kaggle.com** → góc phải trên bấm **ảnh đại diện** → **Settings**
2. Kéo xuống mục **Phone Verification** → bấm **Verify Phone number**
3. Nhập số ĐT VN (+84...), nhận SMS, nhập mã
4. ✅ Mục chuyển thành "Verified"

### 3.3 — Lấy chìa khóa API (kaggle.json)

1. Vẫn trong **Settings** → kéo tới mục **API** → bấm **Create New Token**
2. Trình duyệt tự tải file **`kaggle.json`** xuống (nhỏ xíu, ~70 bytes)

### 3.4 — Đặt chìa khóa vào ổ D

⬛ PowerShell (nhớ Phần 0.3 trước — `KAGGLE_CONFIG_DIR=D:\kaggle`):

```powershell
# Sửa chỗ "Downloads\kaggle.json" nếu trình duyệt để chỗ khác
Move-Item "$env:USERPROFILE\Downloads\kaggle.json" "D:\kaggle\kaggle.json" -Force

# Cài kaggle CLI
pip install kaggle

# Kiểm tra chìa khóa hoạt động — sẽ in ra thông tin tài khoản
kaggle config view
```

> 🔵 Xem **tên tài khoản** (username): mở `D:\kaggle\kaggle.json` bằng Notepad —
> dòng `"username": "abcxyz"` → đây chính là `TEN_BAN` ở Phần 2.5!

✅ `kaggle config view` không báo lỗi.

### 3.5 — Quay lại điền tên ở Phần 2.5 nếu chưa làm

---

# PHẦN 4 — Gửi "túi đồ" lên Kaggle (30-60 phút, tùy mạng)

> 🔵 Upload 5GB. Mạng 50Mbps ≈ 15 phút; 20Mbps ≈ 35 phút. Làm 1 lần duy nhất —
> các lần sau chỉ upload thêm checkpoint (nhỏ hơn nhiều).

### 4.1 — Tạo dataset

⬛ PowerShell (nhớ Phần 0.3!):

```powershell
cd D:\jin-kaggle
kaggle datasets create -p D:\jin-kaggle
```

- Nếu thành công: thấy `Your dataset title "jin-train-data"...` / `successfully created`
- **Đang chạy:** `%` tăng dần — đừng tắt cửa sổ!

⚠️ **Nếu báo lỗi tên trùng:** đổi `title` + `id` trong `dataset-metadata.json`
thành `jin-train-data-2` rồi chạy lại.

### 4.2 — Kiểm tra trên web + CLI (QUAN TRỌNG — đừng bỏ qua!)

1. Vào **kaggle.com/datasets** (bấm ảnh đại diện → Datasets)
2. ✅ Thấy **jin-train-data** với **ĐỦ 2 file**: `jin-code.zip` (~600MB) và `train.txt` (~5GB), tổng ~5.7GB
3. ⬛ Kiểm tra bằng CLI (chắc chắn hơn nhìn web):

```powershell
kaggle datasets files annguyen0410/jin-train-data
```

   ✅ Danh sách in ra phải có **CẢ HAI** file. Nếu thiếu file nào → chạy lại
   `kaggle datasets version -p D:\jin-kaggle -m "bo sung file"` (lệnh này chỉ
   upload phần thiếu, không gửi lại 5GB).

> ⚠️ **Bẫy lớn nhất Phần 4:** tạo dataset bằng kéo-thả qua **web** dễ bị sót file
> (upload 1 trong 2 file xong tưởng là xong). Luôn xác minh bằng lệnh CLI ở bước 3.
>
> 🔵 URL dataset của bạn có dạng `kaggle.com/datasets/TEN_BAN/jin-train-data`
> — phần `/TEN_BAN/jin-train-data` gọi là **slug**, dùng ở Phần 6.

---

# PHẦN 5 — Tạo Notebook + gắn GPU (10 phút)

### 5.1 — Tạo notebook mới

1. Vào **kaggle.com/code** → nút **New Notebook**
2. Đặt tên: bấm vào chữ "Untitled" trên cùng → đổi thành `jin-train`

### 5.2 — Thêm "túi đồ" vào notebook

1. Thanh bên phải → **Input** (biểu tượng 📥) → nút **+ Add Input**
2. Gõ tìm: `jin-train-data` → click chọn dataset của bạn → **Add**
3. ✅ Dataset hiện trong danh sách Input, đường dẫn mount là `/kaggle/input/...`
   (tên có thể là `jin-train-data` hoặc lồng sâu hơn — Cell 1 bản mới tự tìm,
   không cần quan tâm)

> ⚠️ **Khi dataset có version mới sau này** (bổ sung file, thêm checkpoint):
> notebook **KHÔNG tự thấy** — phải vào Input → click vào tên dataset → dropdown
> **Version ▾** chọn bản mới nhất → Save → cho restart session. Hoặc gỡ (X) →
> Add Input lại. Kiểm tra bằng cell: `!find /kaggle/input -name "*.zip"`.

### 5.2b — (Phương án B) Nếu version update cứ không ăn: dataset zip riêng

Dataset lớn (5.7GB) dễ bị notebook ghim version cũ. Cách né hẳn: tách **code zip**
ra dataset riêng nhỏ, upload 5 phút, mount thêm — Cell 1 tự `find` ra zip ở dataset mới:

⬛ Ở nhà (PowerShell, cửa sổ đã `kaggle auth login`):

```powershell
New-Item -ItemType Directory -Force D:\jin-kaggle-code | Out-Null
Copy-Item D:\jin-kaggle\jin-code.zip D:\jin-kaggle-code\
@'
{
  "title": "jin-train-code",
  "id": "annguyen0410/jin-train-code",
  "licenses": [ { "name": "CC0-1.0" } ]
}
'@ | Set-Content D:\jin-kaggle-code\dataset-metadata.json -Encoding UTF8
kaggle datasets create -p D:\jin-kaggle-code
```

⬛ Trên notebook: **+ Add Input** → tìm `jin-train-code` → Add → chạy Cell 1
bình thường (Cell 1 tự tìm zip trong mọi dataset đang mount).

### 5.3 — Gắn GPU

1. Vẫn thanh phải → **Session options** (biểu tượng ⚙️)
2. **Accelerator** → chọn **GPU T4 x2**
3. **Persistence** → bật **Files only** (giữ file giữa các lần chạy trong session)

> 🔵 Script của mình chỉ dùng **1 GPU** trong 2 GPU — không sao, T4 1 cái
> vẫn mạnh hơn RTX 2060 nhà bạn ~5 lần.

### 5.4 — Bật Internet (để pip install nếu cần)

- Session options → **Internet** → **On**
- ✅ Nếu nó đòi verify lại SĐT → làm theo link nó đưa (đã làm ở 3.2 thì thường OK)

---

# PHẦN 6 — Viết các ô lệnh (cell) trong notebook

> 🔵 Notebook mới có sẵn 1 cell trống. Thêm cell: bấm **+ Code** ở giữa màn hình.
> Chạy cell: bấm nút **▶** bên trái cell, hoặc `Shift+Enter`.
> ⬛ = copy nguyên khối vào cell rồi chạy.

### Cell 1 — Bung đồ ra nơi làm việc

⬛ **Bước 1.0 — Xác định dataset của bạn đang ở dạng nào** (chạy cell này):

```python
!find /kaggle/input -name "jin-code.zip" -o -name "bpe.json" | head -5
```

- **Dạng A** — thấy đường dẫn kết thúc `jin-code.zip`: zip còn nguyên → dùng Cell 1A
- **Dạng B** — thấy đường dẫn kiểu `.../jin-code/bpe.json` (KHÔNG có .zip): web Kaggle
  đã **tự bung zip** thành thư mục khi kéo-thả upload → dùng Cell 1B (bỏ unzip luôn!)

⬛ **Cell 1A — bản zip** (dạng A):

```python
!mkdir -p /kaggle/working/jin/data
!unzip -q -o "$(find /kaggle/input -name jin-code.zip | head -1)" -d /kaggle/working/jin
!cp "$(find /kaggle/input -name train.txt | head -1)" /kaggle/working/jin/data/train.txt
%cd /kaggle/working/jin
!ls
```

⬛ **Cell 1B — bản thư mục đã bung sẵn** (dạng B — phổ biến hơn):

```python
!cp -r "$(dirname "$(find /kaggle/input -name bpe.json | head -1)")"/* /kaggle/working/jin/
!cp "$(find /kaggle/input -name train.txt | head -1)" /kaggle/working/jin/data/train.txt
%cd /kaggle/working/jin
!ls
```

> ⚠️ **KHÔNG dùng `rm -rf /kaggle/working/jin`!** Lệnh đó sẽ XÓA SẠCH
> checkpoint + token cache đã tiết kiệm từ session trước. Copy đè lên là đủ —
> file trong dataset không chứa `checkpoints/` hay `cache/` nên chúng sống sót.

✅ Cả 2 bản: `ls` phải in ra `agent.py  app.py  bpe.json  config.py  data  train.py ...`

> 🔵 **Tại sao phải copy?** Thư mục `/kaggle/input` là **CHỈ ĐỌC** (không ghi được
> gì vào). Train phải ghi checkpoint → mình chép toàn bộ sang `/kaggle/working`
> (vùng có thể ghi).
>
> 🔵 **Nếu find không thấy gì** → notebook đang mount **version cũ** của dataset:
> Input → click tên dataset → dropdown **Version ▾** → chọn bản mới nhất → Save →
> cho restart session → chạy lại Bước 1.0.

### Cell 2 — Khám sức khỏe GPU

⬛ Dán, chạy:

```python
import torch
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA OK:", torch.cuda.is_available())
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), "GB")
```

✅ In ra `GPU: Tesla T4`, `CUDA OK: True`, `VRAM: 15.3 GB` — nếu OK → qua Cell 3.

> ⬛ Kaggle có sẵn torch mới — **KHÔNG cần pip install torch** (đỡ 5 phút).
> Nhánh A pretrain không cần pyarrow (chỉ instruct build mới cần).

---

### 🅰️ NHÁNH A — Cell 3 (small 147M)

⬛ Dán, chạy:

```python
!python train.py --device cuda --optimizer muon --batch-size 8 --grad-accum 2 --run-iters 20000
```

> 🔵 `--batch-size 8 --grad-accum 2` = 8×2×256 = **4096 tokens/step** — GPU "no
> bụng" (T4 chạy ~8× nhanh hơn về token so với 1×2). 20k steps ≈ 4-5h — an toàn
> trong khung 12h, và 82M tokens seen ≈ gấp 25× corpus cũ.
>
> ⚠️ **Nếu log in ra `batch 1x2`** → dataset chứa code cũ chưa nhận cờ này (bug đã
> sửa ở nhà) → chạy **Cell PATCH** bên dưới rồi chạy lại Cell 3.

### Cell PATCH — (chỉ chạy nếu log ra `batch 1x2`)

```python
p = "/kaggle/working/jin/train.py"
src = open(p).read()
if "config.GRAD_ACCUM = args.grad_accum" not in src:
    anchor = '    print(f"[Train] Mode: pretrain'
    fix = ("    # Honor --batch-size / --grad-accum in pretrain\n"
           "    config.BATCH_SIZE = args.batch_size\n"
           "    config.GRAD_ACCUM = args.grad_accum\n")
    open(p, "w").write(src.replace(anchor, fix + anchor))
    print("Patched!")
else:
    print("Already patched")
```
Sau patch chạy lại Cell 3 — log phải in `batch 8x2`.

### Cell DL — lấy checkpoint ra khi Files panel lag (mẹo thực chiến)

```python
# (1) Interrupt cell train trước (an toàn — checkpoint đã lưu trên đĩa)
# (2) Dời train.txt 5GB khỏi cây file (resume không cần nó — token cache dùng rồi)
!mkdir -p /kaggle/temp/keep
!mv /kaggle/working/jin/data/train.txt /kaggle/temp/keep/ 2>/dev/null || true
# (3) Thư mục tải chỉ 1 file → mở trong Files panel không lag
!mkdir -p /kaggle/working/dl
!cp /kaggle/working/jin/checkpoints/jingpt_latest.pt /kaggle/working/dl/
!ls -lh /kaggle/working/dl/
```
Files 📁 (phải) → expand `dl` → tải file về. Trình duyệt lưu vào
`C:\Users\nguye\Downloads` → chép ngay sang D:
```powershell
Move-Item "$env:USERPROFILE\Downloads\jingpt_latest.pt" "D:\jin-kaggle\" -Force
```
> 🔵 Session sau muốn resume: cell copy train.txt sẽ không còn nguồn (đã mv đi) —
> nhưng **không sao**: cache token còn nguyên → train vào thẳng Cell RESUME.

### Cell RESUME — train tiếp từ checkpoint (session sau)

```python
!python train.py --device cuda --optimizer muon --batch-size 8 --grad-accum 2 --resume --run-iters 18000
```
> 🔵 `--resume` đọc `checkpoints/jingpt_latest.pt`; `--run-iters` là số step
> **cộng thêm** (resume từ 9000 + 18000 → target 27000).
>
> ⚠️ **Nếu Cell DL đã mv `train.txt` đi:** bản code cũ sẽ crash
> `FileNotFoundError: data/train.txt` khi tính fingerprint cache. 2 cách:
> ```python
> # Cách 1 — trả file về chỗ cũ (mv giữ nguyên mtime → fingerprint giữ nguyên → cache HIT):
> !mv /kaggle/temp/keep/train.txt /kaggle/working/jin/data/train.txt
> # Cách 2 — trỏ thẳng vào cache có sẵn (bản code mới hỗ trợ env này):
> import os
> os.environ["JINGPT_TOKEN_CACHE"] = "/kaggle/working/jin/cache/tokens_<HASH>stream.u16"
> ```
> (Xem đúng tên file cache: `!ls /kaggle/working/jin/cache/`)

- 20-40 phút đầu (session mới): `[Encode] ...` tiến trình encode chạy 1 lần rồi
  cache ra `cache\*.u16` — session sau sẽ cache HIT, bỏ qua sau vài giây
- Sau đó: `step 1000 | ...` — **để nó chạy!**

### 🅱️ NHÁNH B — Cell 3 (medium 360M) — thay cho Cell 3A

⬛ Dán, chạy (30k steps ≈ 17-20h → **chia 2-3 buổi**, xem Phần 8.5):

```python
import os
os.environ["JINGPT_SCALE"] = "medium"   # chọn preset 360M
!python train.py --device cuda --optimizer muon --batch-size 4 --grad-accum 2 --run-iters 12000
```

- Sẽ thấy dòng `[Model] ~360M params` và `[Model] Gradient checkpointing enabled`
- Nếu cuối còn thấy `CUDA out of memory`: chạy lại với `--low-vram` thêm vào cuối lệnh

### Cell 4 — (CẢ HAI NHÁNH) Dọn cache trước khi lưu output

⬛ Khi train xong (hoặc trước khi bấm Save ở Phần 7), thêm cell này chạy 1 lần:

```python
!rm -f /kaggle/working/jin/cache/*.u16
!ls -lh /kaggle/working/jin/checkpoints/
```

✅ In ra thấy `jingpt_best.pt`, `jingpt_latest.pt` với cỡ **hàng trăm MB - 1.6GB**.

> 🔵 Xóa cache stream (2.5GB) để output gọn — session sau encode lại 1 lần cũng nhanh.

---

# PHẦN 7 — Cách chạy cho ĐÚNG (rất quan trọng!)

> 🔵 Notebook Kaggle có 2 chế độ:
> - **Interactive** (bấm từng cell): bạn phải **giữ tab mở**, tắt máy/tắt tab = mất
> - **Save & Run All (Commit)**: bấm 1 lần, nó chạy **toàn bộ các cell từ trên xuống**
>   trên máy chủ. Bạn **tắt máy về ngủ thoải mái** — máy chủ vẫn chạy.

### Quy trình khuyên dùng:

1. Làm xong Cell 1 → 2 → 3 (bấm ▶ từng cell để kiểm tra Cell 1, 2 chạy OK)
2. **Kiểm tra log Cell 3 in `batch 8x2`** — nếu `1x2` → chạy Cell PATCH rồi chạy lại
3. Khi tới Cell 3 (train): **KHÔNG cần chờ** — bấm nút **Save Version** (phải trên cùng)
   → chọn **Save & Run All (Commit)** → **Save**
4. Máy chủ khởi động lại session và chạy lại toàn bộ từ Cell 1 → 4 tự động
   (encode chạy lại 1 lần ~30 phút — bình thường; session sau sẽ cache HIT)
5. Đóng laptop, đi ngủ 😴 — mở kaggle.com/code trên điện thoại vẫn xem được tiến độ
6. Sáng mở lại: **kaggle.com/code** → notebook của bạn → thấy **"Run successfully"**
7. ⏰ **Tính giờ trước khi ngủ:** tổng thời gian các cell (encode + train) phải
   **< 11h** — session bị kill ở 12h và output không lưu. Nếu cần nhiều hơn →
   chia buổi bằng Cell RESUME (Phần 8.5).
8. 🚨 **CHỈ bấm Save & Run All MỘT lần!** Bấm 2 lần = 2 version chạy song song =
   đốt quota GPU gấp đôi. Nếu lỡ có 2 version đang chạy:
   giữ bản **mới hơn** (Version số lớn), vào bản cũ → bấm **⏹ Stop**.
   (Run bị stop thì output của nó không lưu — nhưng bản trùng lặp thì không tiếc.
   Bản giữ lại chạy tới target sẽ tự lưu Output.)

> ⚠️ Giới hạn session GPU: **12 tiếng liên tục**. Nhánh A (~8-11h) vừa trong 1 session.
> Nhánh B (~32-45h) phải chia 3-4 buổi — xem Phần 8.5.

---

# PHẦN 8 — Lấy thành quả về nhà

### 8.1 — Tải checkpoint (cách 1: trình duyệt)

1. Mở notebook → tab **Output** (thanh phải, hoặc cuối trang notebook)
2. Vào `jin/checkpoints/` → bấm **⬇** cạnh `jingpt_best.pt`
3. File ~1.6GB tải về thư mục Downloads

### 8.2 — Tải checkpoint (cách 2: CLI — dùng khi trình duyệt tải hỏng giữa đường)

⬛ Ở nhà, PowerShell (nhớ Phần 0.3):

```powershell
# TEN_BAN/tên-notebook xem ở URL trang notebook của bạn
kaggle kernels output TEN_BAN/jin-train -p "D:\jin-kaggle\out"
# File sẽ nằm trong D:\jin-kaggle\out\jin\checkpoints\
```

### 8.3 — Chép về chỗ dự án

⬛ PowerShell:

```powershell
# Sửa đường dẫn nguồn nếu khác
Copy-Item "D:\jin-kaggle\out\jin\checkpoints\*.pt" "D:\Code folder\Jin New\checkpoints\" -Force
Copy-Item "D:\jin-kaggle\out\jin\checkpoints\*.pt" "D:\Code folder\Jin New\checkpoints\" -Force
```

> ⚠️ **Quan trọng:** checkpoint Kaggle dùng **BPE mới** (bpe.json upload cùng).
> Ở nhà bpe.json của bạn cũng đã là bản mới → khớp nhau. Nếu sau này bạn
> train BPE khác thì phải tải cả `bpe.json` trên Kaggle về cùng lúc.

### 8.4 — Tiếp tục ở nhà

Checkpoint mới là **pretrain**. Về nhà chạy tiếp **Bước 8 + 9** trong
`HUONG_DAN_TRAIN.md` (build chat corpus + instruct tuning + bench) —
các bước này nhẹ, RTX 2060 nhà làm được (~2-4 tiếng).

### 8.5 — Nhánh B (medium): vòng lặp nhiều buổi

Session 12h không đủ → mỗi buổi:

1. Trước khi session cũ hết giờ: đảm bảo đã **Save & Run All** xong buổi đó
   (checkpoint mới nằm ở Output của notebook)
2. Upload checkpoint lên dataset để buổi sau mount vào:
   ```powershell
   # Tải checkpoint của buổi đó về (8.2), rồi:
   Copy-Item "D:\jin-kaggle\out\jin\checkpoints\jingpt_latest.pt" "D:\jin-kaggle\jingpt_latest.pt"
   kaggle datasets version -p D:\jin-kaggle -m "checkpoint buổi 2"
   ```
3. Sáng sau: mở notebook → Cell 1 thêm 2 dòng **sau dòng unzip**:
   ```python
   !mkdir -p /kaggle/working/jin/checkpoints
   !cp /kaggle/input/jin-train-data/jingpt_latest.pt /kaggle/working/jin/checkpoints/
   ```
   và Cell 3 thêm chữ `--resume` vào cuối lệnh train
4. Bấm **Save & Run All** — nó load checkpoint buổi trước và train tiếp

---

# PHẦN 9 — Bẫy thường gặp (đọc kỹ, tiết kiệm giờ ×100)

| 🚨 Bẫy | 🔧 Cách xử lý |
|---|---|
| `kaggle: command not found` | Chưa `pip install kaggle` trong venv đúng — kích hoạt `.venv` rồi cài lại |
| `401 Unauthorized` khi kaggle CLI | `kaggle.json` sai chỗ. Đảm bảo đã `$env:KAGGLE_CONFIG_DIR="D:\kaggle"` trong cùng cửa sổ đang chạy lệnh |
| Upload 413/429 hay đứt giữa đường | Chạy lại y nguyên lệnh — CLI tự resume. Nếu hoài: thử giờ thấp điểm (khuya VN) |
| Notebook không thấy GPU | Session options → Accelerator: chọn lại T4. Mới verify SĐT thì chờ ~5 phút |
| Cell 1 báo `No such file /kaggle/input/...` | Dataset mount ở path khác (`/kaggle/input/datasets/<user>/<slug>/`) — Cell 1 bản mới tự tìm bằng `find`; nếu `find` cũng trống → notebook đang ghim **version cũ** của dataset → Input → dataset → bấm **↻ Update** (xem ghi chú 5.2) |
| `unzip: cannot find ... jin-code.zip` nhưng web thấy đủ file | 2 khả năng: (1) notebook ghim version cũ → Input → click dataset → **Version ▾** → bản mới → Save → restart (xem 5.2); (2) web Kaggle **tự bung zip thành thư mục** khi kéo-thả upload → chạy Bước 1.0 rồi dùng **Cell 1B** (copy thư mục, bỏ unzip) |
| `unzip: cannot find or open , .zip` (path rỗng) | `find` không thấy zip nào → dataset chứa thư mục bung sẵn, không phải zip → chạy Bước 1.0 → Cell 1B |
| `cp: cannot create .../data/train.txt` | Thiếu dòng tạo thư mục `data` trước khi cp — Cell 1 bản mới đã có sẵn `mkdir -p .../data` |
| Dataset thiếu file trên web | Ở nhà chạy `kaggle datasets files <user>/<slug>` xem thiếu gì → đặt đủ file vào thư mục upload → `kaggle datasets version -p D:\jin-kaggle -m "bo sung"` (chỉ gửi phần thiếu) |
| Train ngắt giữa chừng `CUDA out of memory` | Nhánh B: thêm `--low-vram`. Cả 2: giảm `--grad-accum` xuống 8 |
| Session chết giữa giờ (mất điện Kaggle) | Không mất gì nếu checkpoint đã lưu (mỗi 1000 step). Mở lại notebook, chạy Cell 8.5-bước-2+3 để resume |
| `Daily quota exceeded` | Hết 30h GPU tuần rồi — đợi reset (xem đếm giờ ở thanh phải Session options) |
| Chạy 2 notebook cùng lúc | **Đừng** — vừa đốt quota 2×, vừa dễ bị kill session |
| Ổ C lại đầy | Mở PowerShell: `echo $env:TEMP` (phải là `D:\tmp`), `pip cache dir` (phải là `d:\pip-cache`) |

---

# PHỤ LỤC — Thứ tự cell chuẩn cho 1 buổi RESUME (dọn notebook loạn)

> Notebook bị nhiều cell rối? Xóa bớt cell cũ (di chuột lên cell → biểu tượng 🗑️),
> chỉ giữ đúng 5 cell theo thứ tự dưới đây cho các buổi resume tiếp theo.

**CELL 1 — Sức khỏe GPU (10 giây)**
```python
import torch
print("GPU:", torch.cuda.get_device_name(0))
```

**CELL 2 — Trả train.txt về chỗ cũ** (nếu buổi trước đã mv đi; mv giữ nguyên
mtime → fingerprint giữ nguyên → cache HIT, không encode lại)
```python
!mv /kaggle/temp/keep/train.txt /kaggle/working/jin/data/train.txt
!ls -lh /kaggle/working/jin/data/train.txt
```

**CELL 3 — Kiểm tra code đã patch** (phải in ra `1`; nếu `0` → chạy Cell PATCH
trong Phần 6 rồi quay lại)
```python
!grep -c "config.GRAD_ACCUM = args.grad_accum" /kaggle/working/jin/train.py
```

**CELL 4 — TRAIN resume** (log mong đợi: `[Cache] Hit -> ...` →
`[Resume] Continuing from iteration N` → `batch 8x2` → `step ... | loss ...`)
```python
!python train.py --device cuda --optimizer muon --batch-size 8 --grad-accum 2 --resume --run-iters 18000
```

**CELL 5 — Lấy checkpoint về trước khi tắt máy** (chạy sau khi Interrupt Cell 4):
```python
!mkdir -p /kaggle/working/dl
!cp /kaggle/working/jin/checkpoints/*.pt /kaggle/working/dl/
!ls -lh /kaggle/working/dl/
```
→ Files 📁 (phải) → thư mục `dl` → tải từng file → trình duyệt lưu vào
`C:\Users\nguye\Downloads` → chép sang D:
```powershell
Move-Item "$env:USERPROFILE\Downloads\*.pt" "D:\jin-kaggle\" -Force
```

> 🔵 Buổi sau lặp lại đúng Cell 1→5 (đổi `--run-iters` tùy ý). Khi đã đạt
> (~25-30k steps): tải checkpoint về nhà → Bước 8-9 trong `HUONG_DAN_TRAIN.md`.

---

# ✅ CHECKLIST TỔNG (in ra, gạch từng cái)

- [ ] Phần 0: 4 thư mục D + TEMP/profile + (pagefile)
- [ ] Phần 1: Chọn nhánh A hoặc B
- [ ] Phần 2: pack\ → jin-code.zip + train.txt + metadata (điền TEN_BAN)
- [ ] Phần 3: Tài khoản + verify SĐT + kaggle.json vào D:\kaggle + pip install kaggle
- [ ] Phần 4: `kaggle datasets create` xong, thấy 2 file trên web
- [ ] Phần 5: Notebook + Add Input + GPU T4 x2 + Internet On
- [ ] Phần 6: Cell 1 (unzip OK), Cell 2 (GPU OK), Cell 3 (train), Cell 4 (dọn cache)
- [ ] Phần 7: Save Version → Save & Run All → đi ngủ
- [ ] Phần 8: Tải .pt về → chép vào checkpoints\ → chạy Bước 8+9 ở HUONG_DAN_TRAIN.md
- [ ] Phần 9: Không dính bẫy nào 😎

---

*Số liệu tham khảo: Kaggle T4 ≈ 5× RTX 2060 (FP16); quota 30h GPU/tuần, session ≤ 12h;
dataset ≤ 100GB; notebook output ≤ 20GB. Mọi lệnh đã dùng tên thật của dự án
(`jingpt_*`, `train.py`, `bpe.json`) — copy là chạy.*
