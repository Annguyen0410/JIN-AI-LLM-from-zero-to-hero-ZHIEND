# Train JIN trên Kaggle — bộ dữ liệu v4, không cần GPU máy nhà

**Cách khuyến nghị cho lần đầu:** tiếp tục instruction-tune model chat hiện có với bộ v4 đã lọc. Máy nhà chỉ nén/upload/download; training chạy trên GPU Kaggle.

Notebook đã chuẩn bị: [JIN_KAGGLE_V4.ipynb](JIN_KAGGLE_V4.ipynb). Đây là notebook để bạn tải/import vào Kaggle, **chưa chạy thực tế trên Kaggle**. Đã kiểm tra cú pháp và các helper bằng tests, chưa đo thời gian hay chất lượng học trên GPU Kaggle. [Báo cáo tests](../bench_results/kaggle_workflow_tests.json) ghi rõ bài nào bỏ qua: máy nhà gặp lỗi cấp phát bộ nhớ khi chạy bộ đầy đủ, nên regression cuối chạy CPU và bỏ riêng test dựng preset model lớn; không gọi lượt lỗi đó là pass.

## 0. Bạn đang có notebook cũ như ảnh: đọc chỗ này trước

Trong ảnh có `jin-train-data`, `jin-tokencache`, `jin-ckpt`; các lệnh copy `train.txt`, `jingpt_latest.pt`, `.u16` là workflow **continued pretrain cũ**, không phải SFT v4. Không xóa Input/notebook cũ, không trộn cells mới vào cells cũ. Tạo notebook mới riêng để thử v4.

| Lệnh trong ảnh | Nó làm gì | Nên làm gì |
|---|---|---|
| `!ls -lh ... 2>&1` | Xem danh sách/kích thước file, đưa lỗi ra cùng output | Có thể chạy: không train/ghi đè. Missing working folder trong session mới chưa chứng minh Input mất |
| `!find /kaggle/input -name 'bpe.json' ...` | Tìm tên file trong Input | Dùng để nhìn path. Nếu chữ đang ở ô Markdown thì không thực thi; thêm ô **Code** |
| `!mkdir -p ...` | Tạo thư mục working | Hợp lý, nhưng chưa chọn/xác minh data/checkpoint |
| `!cp ... $(find ... head -1)` | Đoán file đầu tiên rồi copy | Không khuyến nghị: có thể lấy sai version, ghi đè working checkpoint hoặc copy code cũ |
| `%cd /kaggle/working/jin` | Đổi thư mục làm việc trong notebook | Lệnh Kaggle/Jupyter, không dán vào Windows CMD |
| `assert sz > 1_000_000_000` | Chặn checkpoint quá nhỏ theo heuristic | Không chứng minh đúng model, đúng tokenizer hay file không hỏng |
| `torch.load(...)` | Nạp checkpoint để xem step | Dùng `weights_only=True`, chỉ checkpoint đáng tin, không kết luận từ dung lượng |
| `!grep -c 'Honor --batch-size' ...` | Đếm comment trong source | Không chứng minh version/behavior code đúng, bỏ check này ở workflow mới |
| `glob(...*.u16)` | Liệt kê token cache | Không chọn cache đầu tiên để reuse; phải có manifest/tokenizer lineage phù hợp |

Trong ảnh `Draft Session off` chỉ cho biết draft session chưa chạy tại lúc chụp, không cho biết checkpoint có/mất. Nút **Run All** chạy mọi cell, có thể gồm train phía dưới chưa thấy trong ảnh. Lần đầu bấm nút tam giác **từng ô**.

Muốn kiểm kê notebook cũ mà chưa train: bấm **+ Code**, dán và chạy ô này:

```python
from pathlib import Path
names = {'bpe.json', 'train.py', 'train.txt', 'jingpt_latest.pt',
         'upload_manifest.json', 'jin-code.zip', 'jin-data.zip'}
count = 0
for p in sorted(Path('/kaggle/input').rglob('*')):
    if p.is_file() and (p.name in names or p.name.endswith(('.u16', '.manifest.json'))):
        print(p, 'bytes=', p.stat().st_size)
        count += 1
        if count >= 80:
            print('Dừng hiển thị sau 80 file; không xóa/đổi file nào.')
            break
if not count:
    print('Chưa thấy file cần dùng; mở Input để xem cấu trúc thực tế.')
```

Ô này chỉ đọc danh sách, không tải/nạp model/huấn luyện. Gửi output nếu muốn tiếp run cũ: cần xác định chính xác code, parent checkpoint, tokenizer và token cache trước, không đoán theo tên Dataset.

**Phân biệt nơi chạy:** `D:\\...` và lệnh `.venv\\Scripts\\python.exe` là máy nhà/PowerShell; `/kaggle/...`, `!ls`, `%cd` là notebook Kaggle. `!` gọi shell Linux; không phải dấu phải thêm vào mọi đoạn Python.

## 1. Chuẩn bị gói upload trên máy nhà — tất cả trên D

Mở PowerShell trong `D:\Code folder\Jin New`, chạy:

```powershell
cd "D:\Code folder\Jin New"
.\.venv\Scripts\python.exe scripts/python_d.py scripts/pack_kaggle.py --output-dir kaggle_upload/v4_chat --checkpoint checkpoints/jingpt_chat.pt
```

Lệnh này **không train, không upload**, không cần nạp model/GPU. Nó kiểm tra hash rồi tạo:

- `kaggle_upload/v4_chat/jin-code.zip`: source Python + scripts + BPE + config.
- `kaggle_upload/v4_chat/jin-data.zip`: prepared_v4 + seed nhỏ + eval.
- `kaggle_upload/v4_chat/upload_manifest.json`: hash code/data và checkpoint chính xác.

Checkpoint **không bị copy nhân đôi**: lấy thêm file gốc `checkpoints/jingpt_chat.pt` để upload riêng. Tổng cần upload là 3 file trên + checkpoint. Không upload `.venv`, cache, corpus cũ 5,31 GB, raw dumps, logs riêng tư, API keys hoặc toàn project.

Nếu folder `v4_chat` đã tồn tại, chọn tên mới `v4_chat_02`; packer từ chối ghi đè. Bộ v4 là batch thử khoảng 38,24 MB pretrain và 18k hội thoại cả train/val, **chưa phải bộ đủ lớn để train model từ đầu**. Sau khi quy trình chạy được mới nâng lượng data.

File trên D; không sửa TEMP toàn Windows. [Chính sách lưu trữ](STORAGE_D_VI.md).

## 2. Upload thành PRIVATE Dataset

1. Vào [Kaggle Datasets](https://www.kaggle.com/datasets) → **New Dataset / Create Dataset**.
2. Tên dễ nhớ: **jin-v4-training**.
3. Chọn **Private**. Xem quyền chia sẻ dữ liệu trước khi publish; đừng đổi hỗn hợp giấy phép thành MIT vì tiện.
4. Thả **hai ZIP + upload_manifest.json + jingpt_chat.pt**, rồi Create.
5. Chờ upload/process xong. Giữ checkpoint nguyên tên và nội dung.

**Kaggle có thể tự giải nén ZIP** ([datasets docs](https://www.kaggle.com/docs/datasets)). Notebook hỗ trợ cả ZIP còn nguyên lẫn các file đã giải nén (kể cả khi giải nén vào thư mục con `jin-code/`, `jin-data/`): ô 1 tự dò, không lấy file đầu tiên bằng glob.

Có thể tách checkpoint thành Dataset riêng để lần sau chỉ cập nhật code/data nhỏ: khi đó sửa `CHECKPOINT_INPUT` sang path Dataset checkpoint đã attach.

## 3. Tạo notebook + bật GPU

1. Vào [Kaggle Code](https://www.kaggle.com/code) → **New Notebook**.
2. Trong File/menu notebook chọn **Import/Upload Notebook**, chọn `docs/JIN_KAGGLE_V4.ipynb` trên D. Tên/menu có thể thay đổi theo giao diện Kaggle.
3. **Add Input / Add Data** → chọn PRIVATE Dataset vừa tạo (đúng version).
4. **Settings → Accelerator → GPU** (T4/P100 tùy đang được cấp).
5. Bật **Internet** nếu phải cài package thiếu; notebook không tải lại corpus.
6. Nếu GPU bị khóa, thực hiện account/phone verification theo yêu cầu Kaggle và xem GPU quota còn lại trong giao diện.

Không chọn TPU: trainer là PyTorch CUDA, chưa hỗ trợ TPU. **T4 x2 không tự cộng VRAM thành 32 GB**; trainer hiện dùng GPU 0, chưa có DDP. Bắt đầu batch 1/block 256; chỉ tăng sau khi đo VRAM và thời gian.

Kaggle có giới hạn session/GPU quota/output. Xem con số **đang hiển thị trên tài khoản** và [notebook docs](https://www.kaggle.com/docs/notebooks), đừng dựa vào tốc độ hoặc quota trong hướng dẫn lịch sử. Không hứa train xong trong một đêm.

## 4. Chạy từng ô lần đầu — 20 steps smoke

### Ô 1: để mặc định — ô tự dò input

**Copy file path** trong panel Input trả về đường dẫn của **FILE** (ví dụ `/kaggle/input/datasets/<owner>/<slug>/upload_manifest.json`), không phải thư mục dataset. Ô 1 không cần path đó: nó tự tìm `upload_manifest.json` và parent checkpoint (đối chiếu đúng sha256 đã pin trong manifest) ở bất kỳ đâu trong `/kaggle/input`, kể cả khi Kaggle tự giải nén vào thư mục con. Cứ để:

```python
INPUT = Path('/kaggle/input')
DATA_INPUT = None
CHECKPOINT_INPUT = None
EXTRACTED_ROOTS = {}
MODE = 'instruct'
RUN_LONG = False
STEPS = 500
```

> **Notebook đang pin cho run đêm general_v1 (2026-10-01):** `MODE='pretrain'`, `RUN_LONG=True`, `ALLOW_LONG=True`, `STEPS=50.000` — ô 4 vẫn chạy smoke 20 bước trước, rồi ô 5 chạy dài tiếp trong cùng một version. Muốn chỉ smoke thì đặt `RUN_LONG=False` (khối code trên là giá trị mặc định chung khi chưa pin).

Chỉ điền tay khi ô 1 dừng vì báo lỗi:

```python
DATA_INPUT = Path('/kaggle/input/datasets/<owner>/jin-v4-training')     # THƯ MỤC chứa upload_manifest.json
CHECKPOINT_INPUT = Path('/kaggle/input/<dataset-checkpoint>/jingpt_chat.pt')  # file parent đã pin
EXTRACTED_ROOTS = {'jin-code.zip': Path('/kaggle/input/.../jin-code')}  # chỉ khi giải nén vào thư mục con
```

Các assert hay gặp:

- `Cần đúng 1 upload_manifest.json...`: có nhiều bản trong Input — bỏ dataset thừa hoặc điền `DATA_INPUT`.
- `Không thấy jingpt_chat.pt đúng sha...`: chưa attach dataset chứa parent checkpoint đúng version — thêm vào Input, không đoán theo tên file.
- `Sai parent checkpoint...`: file khác hash đã pin khi đóng gói — dùng đúng file, không bỏ assert.

Ô 1 kiểm tra hash parent, code/data, dựng `/kaggle/working/jin`. Nếu workspace đã tồn tại thì dừng, **không chép đè**; restart session để chạy lại từ đầu hoặc chọn workspace mới trong notebook. Nếu hash sai, sửa nguồn/version/file, không bỏ assert.

### Ô 2: GPU/dependencies

Phải in tên GPU và `torch.cuda.is_available()` qua assert thành công. Nếu GPU không có thì notebook dừng, không train CPU âm thầm.

Kaggle thường có PyTorch CUDA. **Không chạy pip install torch mới đè lên môi trường có sẵn**. Notebook chỉ cài numpy/flask nếu thiếu; parquet đã chuyển đổi nên không cần pyarrow cho đợt này.

### Ô 3: dữ liệu/tokenizer

Notebook xác minh dữ liệu rồi tái pin các đường dẫn mixture từ Windows sang Kaggle và chạy verification. Kiểm tra tokenizer trong parent trùng `bpe.json`; **không rebuild BPE khi tiếp weights cũ**.

Dữ liệu trong `/kaggle/input` là read-only. Code, cache, checkpoint phải nằm trong `/kaggle/working/jin`. Chính sách ép D chỉ áp dụng trên Windows; Kaggle Linux không có ổ D.

### Ô 4–6: smoke, optional long run, output index

- Ô 4 chạy preflight rồi **20 optimizer steps** với batch 1 × grad accumulation 4, block 256, LR 1e-5 cho SFT.
- Dùng prepared `sft_train` và `sft_val` riêng; seeds nhỏ thêm một lần, không nhân 40 lần; grounded ngoài bộ bị tắt.
- Output mới: `jin/checkpoints/kaggle_v4_instruct.pt`, bản best tương ứng, manifests và run records.
- SFT lần đầu warm-start weights, **optimizer mới và step của run mới**, không gọi là resume optimizer của chat cũ.
- `RUN_LONG=False`: ô 5 bỏ qua run dài.
- Ô cuối tạo `jin/output_index.json` chứa kích thước/hash checkpoint. Không copy thêm nhiều file GB vào một folder dl.

Thấy `SMOKE PASS` mới đi tiếp. Nếu OOM: restart session, giảm block về 128 hoặc grad accumulation theo nhu cầu (giảm grad accumulation không giảm VRAM microbatch), không đổi architecture/depth giữa resume. Dataset tokenization/packing là việc một lần trong mỗi tiến trình, có thể mất vài phút trước dòng train.

## 5. Run dài hơn và lưu phiên

Sau smoke thành công, đổi ở ô 1:

```python
RUN_LONG = True
STEPS = 500   # nâng theo bảng dưới, sau khi đọc it/s ở dòng cuối ô 4
```

Cách chọn `STEPS` cho commit qua đêm (bảng là **số học từ it/s đo ở ô 4**, không phải cam kết tốc độ): chạy ô 4 xong, đọc dòng cuối dạng `[step 20/20] loss ... | X.X it/s | ...`, chọn mức chừa biên để tổng ≤ ~7 giờ (giới hạn session GPU ~9 giờ, có overhead setup/tokenize):

| it/s đo được | ~giây/bước | STEPS | Thời gian train ước tính |
|---|---|---|---|
| ≥ 0.20 | ≤ 5 | 5000 (trần) | ≤ ~6,9 giờ |
| 0.167 | 6 | 4000 | ~6,7 giờ |
| 0.125 | 8 | 3000 | ~6,7 giờ |
| 0.10 | 10 | 2500 | ~6,9 giờ |
| 0.067 | 15 | 1500 | ~6,3 giờ |
| 0.05 | 20 | 1200 | ~6,7 giờ |

- Trần 5000 có phanh ở ô 5; muốn vượt thì bật `ALLOW_LONG = True` ở ô 1 (ô 5 sẽ in ước lượng số vòng trước khi chạy).
- Số bước/vòng đo thật (prepared_v4, block 256, grad-accum 4, batch 1): **34.563 block → 1 vòng ≈ 8.640 bước** (Kaggle 17.440 hội thoại ⇒ ≈8,4–8,7k). 5.000 bước ≈ 0,6 vòng; 1–3 vòng = 8,5k–26k bước (≈1–3 giờ ở 2,5 it/s); quá 3 vòng là lặp data — chỉ chạy khi hiểu rõ.
- Commit chạy lại notebook từ đầu: smoke 20 bước rồi tiếp `STEPS` bước từ smoke checkpoint (tổng `20 + STEPS` bước).
- Commit bị timeout có thể mất output: chốt `STEPS` theo bảng và giữ `checkpoint-interval 100`.

**Đo thật trên T4 (model 147,74M, batch 1×4, block 256):** đầu run ~2,4–2,8 it/s nhưng **giảm dần theo phiên**; lượt 60.020 bước hoàn tất trong khoảng **8 giờ 42 phút 40 giây** (đồng hồ notebook, tốc độ tích lũy cuối run ~1,9 it/s). Vì vậy các dòng phía trên chỉ đúng cho đoạn đầu; muốn cam kết qua đêm phải chừa biên lớn hơn và theo dõi log thật. Số đo chi tiết + hash đã tải: [bench_results/kaggle_v4_first_run.json](../bench_results/kaggle_v4_first_run.json).

Với **general_v1** (continued pretrain, cùng recipe 1×4×256 = 1.024 token/bước): corpus train ~508 MB ≈ **~122,5M token** (đo sample 256 KB/ngành, 0,21–0,39 tok/B — ước lượng) ⇒ 1 vòng ≈ **~119.600 bước**; 50.000 bước ≈ **0,42 vòng**. Công thức chọn bước qua đêm: `STEPS ≈ it/s đo được × 3600 × số giờ × 0,9` (bớt 10% biên + overhead setup). Ví dụ 1,914 it/s → 50.000 bước ≈ **7,3 giờ**; còn trần cũ 5.000 chỉ ~43 phút — quá ngắn cho một đêm, nên cần `ALLOW_LONG=True`.

Sau đó **Save Version → Save & Run All / Run All** theo UI.

- Đừng chạy ô 1 lại vào workspace cũ để mong nó ghi đè.
- Checkpoint lưu định kỳ mỗi 100 steps, tại eval và cuối run. Run 20 vẫn save cuối.
- Theo dõi log và thời gian thật; 500 chỉ là đợt kiểm tra học, không bảo đảm đủ tốt.
- Chọn mức steps phù hợp thời gian session còn lại. Lưu định kỳ vào working **chưa đồng nghĩa chắc chắn được Kaggle persist** nếu job bị timeout/crash; save version/output trước giới hạn.
- Không chỉ chạy interactive rồi đóng tab và hy vọng output tự còn. Dùng saved version, xác minh status thành công và tab **Output** có files.

Phần này không gọi API model bên ngoài hay pretrained weights mới. Model vẫn là JIN hiện có.

## 6. Tải checkpoint về ổ D và không thay model phục vụ ngay

Vào version đã chạy → **Output**, tìm:

- `jin/checkpoints/kaggle_v4_instruct.pt` (latest, có optimizer để train tiếp).
- `jin/checkpoints/kaggle_v4_instruct_best.pt` nếu đã có best.
- Các `.manifest.json`, `jin/output_index.json`, `jin/runs/*.json`.

Trình duyệt: bật **Ask where to save each file** và chọn một thư mục mới trên D, ví dụ `D:\Code folder\Jin New\checkpoints\kaggle_v4_run01`. Browser mặc định Downloads trên C không được launcher Python kiểm soát.

**Chưa chép đè `jingpt_chat.pt`.** Giữ candidate riêng, đo loss/benchmark trước khi chọn phục vụ. Checkpoint training nặng vì optimizer; khi chỉ cần chat có thể export inference ra tên mới sau, nhưng phải giữ latest training để resume.

## 7. Phiên Kaggle tiếp theo

Không giả định `/kaggle/working` còn sau restart. Tạo PRIVATE Dataset checkpoint mới hoặc attach Output của version trước thành Input. Code/data Dataset giữ nguyên.

Tại workspace mới đã setup và verify như ô 1–3, để tiếp SFT (không chạy lại ô smoke init từ parent cũ):

```python
# SỬA path chính xác của latest checkpoint + manifest trong output đã lưu.
previous = Path('/kaggle/input/YOUR-PREVIOUS-OUTPUT/jin/checkpoints/kaggle_v4_instruct.pt')
previous_manifest = Path(str(previous) + '.manifest.json')
info = json.loads(previous_manifest.read_text())
assert sha(previous) == info['output_artifact']['sha256']

from utils import load_checkpoint_file
from bpe_tokenizer import BPETokenizer
from lineage import object_hash
c = load_checkpoint_file(str(previous), map_location='cpu')
t = BPETokenizer(); t.load(str(ROOT / 'bpe.json'))
assert object_hash(c['tokenizer_state']) == object_hash(t.to_state_dict())
assert c.get('optimizer_state_dict'), 'Cần latest training checkpoint, không inference-only weights'
assert all('betas' in g and g.get('mode', 'adamw') == 'adamw'
           for g in c['optimizer_state_dict']['param_groups']), 'Run SFT này dùng AdamW'
del c

next_output = ROOT / 'checkpoints/kaggle_v4_instruct_session02.pt'
assert not next_output.exists()
next_output.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(previous, next_output)

command = [sys.executable, str(ROOT / 'scripts/python_d.py'), 'train.py',
    '--mode', 'instruct', '--resume', '--checkpoint', str(next_output),
    '--hf-seed-dir', str(PREPARED / 'sft_train'), '--sft-val-dir', str(PREPARED / 'sft_val'),
    '--grounded-dir', '', '--max-convs', '0', '--duplication', '1',
    '--device', 'cuda', '--optimizer', 'adamw', '--batch-size', '1', '--grad-accum', '4',
    '--block-size', '256', '--lr', '1e-5', '--steps', '500',
    '--eval-interval', '100', '--eval-iters', '5', '--checkpoint-interval', '100']
subprocess.run(command, check=True)
```

Step count mới bằng step cũ + 500. Giữ cùng dataset/tokenizer/optimizer/batch recipe để tiếp dễ kiểm chứng. Scheduler của trainer tính lại theo target run mới, không hứa tái hiện bit-for-bit cả quá trình nhiều phiên.

## 8. Nếu muốn continued PRETRAIN thay vì SFT

Hai việc khác nhau:
- **SFT**: dạy trả lời/hội thoại, mặc định notebook.
- **Continued pretrain**: học tiếp văn bản/kiến thức, nên bắt đầu từ checkpoint pretrain và sau đó SFT.

Đóng gói khác trên D:

```powershell
.\.venv\Scripts\python.exe scripts/python_d.py scripts/pack_kaggle.py --output-dir kaggle_upload/v4_pretrain --checkpoint checkpoints/jingpt_latest.pt
```

Upload gói/parent đúng, sửa notebook `CHECKPOINT_INPUT` thành `jingpt_latest.pt` và `MODE='pretrain'`. Notebook nhận dạng AdamW/Muon của parent, chạy mixture đã relocation và resume từ bản copy tên mới. Parent phải có optimizer state hỗ trợ; không dùng inference-only artifact hoặc ép optimizer khác rồi giả như resume liên tục.

**Trạng thái đóng gói (2026-10-01):** gói pretrain cho bộ general_v1 đã có tại `kaggle_upload/pretrain_general_v1/` (jin-code.zip + jin-data.zip + upload_manifest.json); parent pin là `checkpoints/kaggle_v4_60020/kaggle_v4_instruct.pt` (step 60.020, AdamW, `tokenizer_state` khớp `bpe.json`, SHA-256 `fe9fb2b9…`). Trên Kaggle: tạo PRIVATE Dataset mới (ví dụ `jin-pretrain-general-v1`), thả **2 ZIP + upload_manifest.json + đúng file parent đó**; import notebook — ô 1 đã pin sẵn `MODE='pretrain'`, `RUN_LONG=True`, `ALLOW_LONG=True`, `STEPS=50.000` (~7,3 giờ ở 1,914 it/s, ~0,42 vòng corpus); muốn chỉ smoke 20 bước trước thì đặt `RUN_LONG=False`. Đây là "tiếp từ bản mới nhất của v4", chưa qua benchmark năng lực nên chưa phải kết luận best mọi tác vụ. Ví dụ `jingpt_latest.pt` phía trên là mẫu cũ: phải kiểm tra `tokenizer_state` khớp `bpe.json` trước khi dùng (BPE mới có thể không khớp checkpoint rất cũ).

Validation mới khác validation lịch sử; `best_loss` cũ không so sánh trực tiếp, đừng tự coi checkpoint tên best là tốt hơn nếu chưa đánh giá trên cùng bộ. Bộ v4 chỉ là batch nhỏ; sau kiểm thử workflow hãy mở rộng corpus đã lọc/cân bằng. **Không tăng depth/model lên 360M giữa resume 147M**.

## 9. Lỗi thường gặp

| Lỗi | Xử lý |
|---|---|
| Missing upload_manifest / checkpoint | Ô 1 tự dò; nếu dừng: attach đúng dataset/version rồi điền tay DATA_INPUT/CHECKPOINT_INPUT (path FILE của panel Input không dùng làm thư mục) |
| ZIP không tồn tại | Kaggle auto-unzip; ô 1 tự dò cả thư mục con; chỉ điền EXTRACTED_ROOTS khi bị dừng |
| Hash mismatch | Sai file/version hoặc upload chưa hoàn tất; không bỏ hash checks |
| Mixture trỏ D:/... | Chạy ô relocation, không patch train.py hoặc bỏ manifest check |
| GPU unavailable | Settings GPU + kiểm tra quota/account verification |
| Tokenizer khác parent | Upload BPE đúng parent, không rebuild tokenizer |
| Workspace already exists | Dùng session/workspace mới, không copy đè dữ liệu/run |
| CUDA out of memory | Bắt đầu batch 1, block 128/256, GPU 0 đủ VRAM; chưa hỗ trợ dùng cộng VRAM T4 x2 |
| Download rơi về C | Chọn đường dẫn tải của browser trên D |
| Muốn dataset public | Review license/PII/provenance trước, không public tự động |

Hướng dẫn này thay thế các snippet glob/patch/PID/tốc độ trong runbook lịch sử. Không chạy upload, commit notebook hay train từ máy nhà thay người dùng.
