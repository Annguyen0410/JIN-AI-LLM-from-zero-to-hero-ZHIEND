# Model Lab: checkpoint trống + vì sao không dùng `best`/`latest`

Ngày 16/09/2026 — sửa 2 hiểu nhầm thật khi mở panel **Model Lab**.

## 1. Model Lab trống: server cũ đang chạy code cũ

`/api/status` của process đang giữ port 5000 chỉ trả **14 key**:

```
auth_required, best_val_loss, checkpoint_file, checkpoint_step, context_window,
device, kv_cache, model_loaded, model_params, quant, scale, tokenizer_kind, vocab_size,
web_search_available
```

Thiếu `checkpoints`, `checkpoint_default`, `defaults`, `uptime_s`, `vram_used_mb` →
`renderLab()` đọc `d.checkpoints || []` ra rỗng nên `<select id="labCheckpoint">` không có
option nào. Nguyên nhân không phải JSON sai mà là **process Python cũ** (PID 23308) đã import
`app.py` trước khi endpoint được thêm; sửa file trên đĩa không ảnh hưởng tới tiến trình đang chạy.

Cách kiểm tra nhanh (không cần mở UI):

```powershell
(Invoke-RestMethod http://127.0.0.1:5000/api/status).PSObject.Properties.Name -contains "checkpoints"
```

`False` = server đang chạy bản cũ → `Ctrl+C` ở cửa sổ đó rồi `python app.py` lại.
Nếu process bị bỏ quên: `netstat -ano | findstr :5000`, rồi `Stop-Process -Id <pid>`.

## 2. Vì sao phục vụ `jingpt_chat.pt` (step 4000) chứ không phải `best` / `latest`

Số thật đọc trực tiếp từ header từng file (`torch.load(..., mmap=True)`, 0.03s/file):

| File | Vai trò | Step | Val loss | Dung lượng | Ghi chú |
|---|---|---|---|---|---|
| `jingpt_chat.pt` | **chat** | 4.000 | **2.9957** | 1773 MB | model đang phục vụ |
| `jingpt_chat_46k_sft4k.pt` | chat | 4.000 | 2.9957 | 1773 MB | bản golden, y hệt |
| `jingpt_chat_pre46k.pt` | chat | 4.000 | 3.0594 | 1773 MB | trước khi SFT lại |
| `jingpt_chat_10kbase.pt` | chat | 9.200 | 2.1522 | 1773 MB | base 10k, vocab khác đợt |
| `jingpt_chat_{1600,2000,2400}.pt` | chat | 1.6k–2.4k | 3.4–3.7 | 1773 MB | snapshot giữa đợt |
| `jingpt_chat_seed_v1.pt` | chat | 2.400 | 0.033 | 1773 MB | model char-level cũ, val không cùng thang đo |
| `jingpt_latest.pt` | pretrain | 10.000 | 6.4572 | 1247 MB | **pretrain**, không biết chat template |
| `jingpt_best.pt` | pretrain | 2.000 | 7.1048 | 1247 MB | **pretrain**, đời đầu |
| `jingpt.pt` | legacy | 30.000 | — | 58.6 MB | char-level |

Ba lý do `best`/`latest` **không** được chọn:

1. **Chúng là pretrain**, chưa bao giờ thấy chat template → trả lời kiểu nối tiếp văn bản
   (`train.py --mode instruct` chỉ ghi ra `jingpt_chat.pt`).
2. **Val loss của hai nhóm không so được** (2.99 vs 7.10) vì khác dung lượng file (1773 MB vs
   1247 MB ⇒ khác vocab) và khác đợt dữ liệu. Cùng lý do đó, `jingpt_chat_10kbase.pt` có val
   2.15 “đẹp” hơn mà không chắc tốt hơn — val chỉ so được **trong cùng một lần train**.
3. Model base 46.000 step thật đang nằm **ngoài** `checkpoints/`, ở `D:\jin-ckpt\jingpt_latest.pt`;
   file `checkpoints/jingpt_latest.pt` chỉ là bản 10k cũ.

## 3. Đã sửa gì

`config.py`

* `checkpoint_role(name)` → `chat` / `pretrain-best` / `pretrain-latest` / `legacy`.
* `chat_snapshot_paths()` → các `jingpt_chat_*.pt`, mới nhất trước.
* `resolve_checkpoint() -> (path, why)`: `JINGPT_CHECKPOINT` → `jingpt_chat.pt` →
  snapshot chat mới nhất → `jingpt_best.pt`. Trước đây chỉ có 2 bậc và nhảy thẳng về pretrain.
* `get_checkpoint_path()` giữ nguyên chữ ký (mọi entry point cũ không phải sửa).

`app.py`

* `_probe_checkpoint()` đọc `iteration` / `val_loss` / `best_loss` bằng `mmap=True`,
  cache theo `(mtime, size)` → poll 5s không đọc lại đĩa.
* `_checkpoint_files(probe=False)` thêm `role`, `default`, `mtime_epoch`; sắp xếp
  đang chạy → tự chọn → chat → còn lại (mới nhất trước).
* `GET /api/model/checkpoints` (có step/val, tách khỏi `/api/status` vì lần đầu probe tốn ~15s
  với các file legacy không zip; lần sau 0.06s nhờ cache).
* `/api/status` thêm `checkpoint_default` (`{path, name, why}`) để UI giải thích được lựa chọn tự động.

Frontend (`static/index.html`, `static/script.js`)

* Dòng hint **“Tự chọn khi khởi động: jingpt_chat.pt (checkpoint chat mới nhất).”**
* `<option>` giờ là `jingpt_chat.pt · chat · step 4.000 · val 2.9957 · 1773 MB · 16/09 19:34 · ← đang chạy`.
* Chọn file `jingpt_best/latest.pt` → hiện cảnh báo “không biết format chat”.
* Model Lab có thêm dòng **Loại file** trong bảng trạng thái.
* Probe chỉ chạy khi mở panel (`toggle`), không phải mỗi 5s.
* Nhãn đổi theo ngôn ngữ VI/EN (`ckptCache` + `paintCheckpointPicker` khi `refreshDynamic`).
* Lựa chọn tay không bị poll ghi đè (giữ `prev` nếu file còn trong danh sách).

## 4. Kiểm chứng

* `tests/test_checkpoint_picker.py`: 15 test mới — phân loại role, thứ tự resolve (kể cả
  env override trỏ sai bị bỏ qua), probe + cache (patch `torch.load` để chắc không đọc lại),
  file hỏng không làm vỡ picker, và hợp đồng JSON của `/api/model/checkpoints` + `/api/status`.
* Toàn bộ: **202 test OK**.
* Server thật (`python app.py`, port 5000, PID mới): `/api/status` → `step 4000 · cuda ·
  jingpt_chat.pt`, 11 checkpoint; `/api/model/checkpoints` lần đầu 15.8s, lần sau 0.064s.
* Preview: picker 11 option đúng nhãn, đổi ngôn ngữ đổi nhãn, chọn `jingpt_best.pt` hiện cảnh báo,
  console 0 lỗi.
