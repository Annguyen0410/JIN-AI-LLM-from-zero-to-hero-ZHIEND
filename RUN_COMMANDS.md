# Lệnh Chạy JIN AI (v3) — Step-by-step: Train từ 0 → Launch

> **Current verified commands (2026-09-30):** [upgrade report](docs/UPGRADE_2026.md).
> Use `python scripts/check_project.py` for tests and
> `python scripts/run_training.py --config training_run.json` for hash-pinned
> preflight. Self-learning is NOT operational; historical full-pipeline claims
> below are not fresh verification. `--block-size` now controls both training
> modes; SFT validation is conversation-disjoint. Keep candidate checkpoints
> separate from served weights.

File này là lộ trình đầy đủ: từ cài đặt → chuẩn bị dữ liệu → train (pretrain → SFT → DPO) → benchmark → launch (chat / web / agent / tự học). Mọi lệnh đã kiểm tra với code hiện tại; nếu đang ở chế độ `cmd` dùng `python`, PowerShell cũng dùng `python`.

> Mẹo nhanh: muốn **auto chạy nguyên pipeline** thì nhảy thẳng tới **mục 7 (Full auto)**.

---

## 1. Vào thư mục dự án

```powershell
cd "d:\Code folder\Jin New"
```

## 2. Tạo môi trường ảo

```powershell
python -m venv .venv
```

## 3. Kích hoạt môi trường ảo

### PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### Command Prompt (cmd)

```cmd
.\.venv\Scripts\activate.bat
```

## 4. Cài thư viện

```powershell
pip install -r requirements.txt
```

> **CUDA (muốn train trên GPU NVIDIA):** cài bản PyTorch có CUDA phù hợp driver:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
> ```
> **Pyarrow (tùy chọn):** chỉ cần khi bật dataset dạng parquet trong `hf_datasets.json` (Cosmopedia, Wikipedia parquet, OpenHermes…).

---

## 5. Quick start — chạy với checkpoint đã có

```powershell
python chat.py                                      # chat trong terminal
python app.py                                       # web UI → mở http://localhost:5000
python generate.py --prompt "The future of AI is"   # sinh 1 đoạn văn
python agent.py "What is sqrt(144) * 3?" --rigor high   # agentic: calculator + tự kiểm tra
python bench.py --checkpoint checkpoints/jingpt_chat.pt # đo toàn bộ chỉ số
```

Nếu chưa có checkpoint, các lệnh trên báo rõ "chưa có model" — đi tiếp mục 6.

---

## 6. Train model từ đầu (từng bước)

### 6.1 Tải dữ liệu thô (public, có giấy phép)

```powershell
python fetch_datasets.py                # tải các nguồn enabled trong sources.json
python fetch_datasets.py --list         # danh sách nguồn (x = enabled)
python fetch_datasets.py --only gutenberg-alice      # chỉ tải 1 nguồn
python fetch_datasets.py --catalog hf_datasets.json  # tải dataset HF (cần pyarrow cho parquet)
```

Hoặc tự đặt file text của bạn vào `data\raw`.

### 6.2 Build corpus + làm sạch + dedupe/chống rò rỉ

```powershell
python build_dataset.py --report        # gộp data\raw -> data\train.txt (format <doc path=...>)
python data_quality.py --input data\train.txt --output data\train_clean.txt --report out.json
```

`data_quality.py` làm 4 việc: **dedupe MinHash-LSH** (exact + near-dup), **quality filter** kiểu Gopher/C4, **phát hiện ngôn ngữ** vi/en (`--lang-filter vi`), và **contamination check** chống rò rỉ benchmark (`data/eval/probes.jsonl`). Tắt từng phần bằng `--no-dedupe` / `--no-contamination`.

```powershell
# nếu dùng bản đã làm sạch, trỏ train vào nó:
Copy-Item data\train_clean.txt data\train.txt
```

### 6.3 Train BPE tokenizer

```powershell
python bpe_tokenizer.py --data data\train.txt --out bpe.json --vocab-size 16384
```

> Token đặc biệt mở rộng (`<|think|>`, `<|tool|>`, ...) chỉ có hiệu lực khi **rebuild** tokenizer lần tới (ID cũ không dịch chuyển — checkpoint cũ vẫn load được).

### 6.4 Pretrain (next-token trên corpus)

```powershell
python train.py --run-iters 100000 --device cuda          # GPU
python train.py --run-iters 20000 --device cpu            # CPU (chậm, dùng preset tiny)
$env:JINGPT_SCALE="tiny"; python train.py --run-iters 5000    # PowerShell: ép preset nhỏ test nhanh
set JINGPT_SCALE=tiny && python train.py --run-iters 5000      # cmd: tương đương
```

Checkpoint lưu: `checkpoints\jingpt_latest.pt` (mới nhất) + `jingpt_best.pt` (val loss thấp nhất).

### 6.5 SFT — dạy chat (instruct)

```powershell
python build_instruct_dataset.py --output data\chat_train.txt   # xây corpus chat (seed + synth + literary)
python train.py --mode instruct --fast --steps 2000 --device cuda
python train.py --mode instruct --steps 5000 --device cuda --batch-size 1 --grad-accum 32   # chống OOM 6GB
```

- `--fast` = profile VRAM thấp (block 256, duplication 8) — test nhanh trên RTX 2060.
- Mặc định instruct warm-start từ `jingpt_best.pt`; dùng `--no-pretrained-init` để train từ đầu, `--resume` để tiếp tục đúng optimizer/scaler.
- `--optimizer muon` thay AdamW (nhanh hơn ~1.1–1.4× khi đã tune, tiết kiệm 1 buffer momentum).

### 6.6 Dữ liệu suy luận + tự cải thiện (rejection sampling → SFT/DPO)

```powershell
# a) Sinh dữ liệu CoT CÓ KIỂM CHỨNG (toán/logic/code chạy unit test — đáp án không thể sai)
python build_reasoning_data.py --n-math 2000 --n-logic 800 --n-code 1200 --out data\reasoning.jsonl

# b) Rejection sampling: sinh k đáp án, verifier tự động giữ đáp án ĐÚNG nhất
python self_distill.py --checkpoint checkpoints\jingpt_chat.pt --prompts data\reasoning.jsonl --k 8 --out data\distill.jsonl --pref-out data\pref.jsonl

# c) SFT lại trên đáp án tốt: bỏ distill.jsonl vào data\seed\ (trainer tự gom MỌI *.jsonl trong đó)
copy data\distill.jsonl data\seed\distill.jsonl
python train.py --mode instruct --fast --steps 500 --device cuda

# d) Preference tuning: DPO (cần reference, --beta) hoặc ORPO (không cần reference, tiết kiệm VRAM)
python train_pref.py --data data\pref.jsonl --method orpo --checkpoint checkpoints\jingpt_chat.pt --steps 200 --device cuda
```

> Lưu ý: `train.py --mode instruct` **không đọc `--data`** — nó gom toàn bộ `data\seed\*.jsonl` (kể cả file bạn tự thêm, `distill.jsonl` ghi đúng chat format `{"messages": [...]}`). Muốn bỏ dữ liệu cũ chỉ cần xoá file khỏi `data\seed\`. Cặp `(chosen, rejected)` trong `pref.jsonl` là **best vs worst** trong số các đáp án đã verify — DPO/ORPO không bao giờ học từ text chưa kiểm chứng.

### 6.7 Curriculum mixing (tuỳ chọn, chất lượng cao)

```powershell
python train.py --mixture data\mixture.json --steps 2000 --device cuda
```

`data\mixture.json` khai báo tỉ lệ domain theo giai đoạn (văn xuôi/code/math/vi…); trainer sample theo weight và log tỉ lệ thực tế đã thấy.

---

## 7. Full auto — pipeline một lệnh

| Mode | Làm gì |
|---|---|
| `chat` | build instruct data → BPE → SFT |
| `raw` | fetch → build corpus → (pretrain nếu `--train`) |
| `bootstrap` | **data → BPE → pretrain → SFT → DPO/ORPO → bench** (full v3) |
| `evolve` | bench baseline → vòng tò mò → gate report |

```powershell
python pipeline.py raw --train --resume                        # fetch + build + pretrain
python pipeline.py chat --resume --skip-bpe                    # toàn bộ pipeline chat
python pipeline.py bootstrap --steps 500 --pref-steps 200 --device cuda   # full v3 từ 0
python pipeline.py bootstrap --steps 300 --pref-steps 0        # bỏ DPO
python pipeline.py evolve --cycles 3 --dry-run                 # kế hoạch tò mò (không train)
```

---

## 8. Resume train từ checkpoint

```powershell
python train.py --resume
python train.py --resume --checkpoint checkpoints\jingpt_latest.pt
python train.py --mode instruct --resume --steps 1000
```

`--resume` khôi phục đúng optimizer (AdamW hoặc Muon), scaler AMP và RNG state — không phải train lại.

---

## 9. Benchmark & đánh giá (đo trước, tối ưu sau)

```powershell
python bench.py --checkpoint checkpoints\jingpt_chat.pt        # 5 nhóm: quality/probes/fluency/speed
python bench.py --checkpoint checkpoints\jingpt_chat.pt --no-train-bench
python bench.py --train-bench --checkpoint checkpoints\jingpt_chat.pt --train-out t.json   # riêng train tok/s
python bench.py --merge-train bench.json t.json                # gộp kết quả train
python bench.py --compare bench_results\baseline_v25.json bench_results\<mới>.json   # diff + forgetting_score
```

- Output: `bench_results\<timestamp>_<checkpoint>.json` + bảng markdown in ra terminal; mọi lần chạy được nối vào `bench_results\history.json` cho Growth Dashboard.
- `python bench.py --init-eval` (chạy 1 lần): trích `data\eval\heldout.txt` ra khỏi `train.txt` — tập held-out khoá, không bao giờ train lên nó.

---

## 10. Chat trong terminal

```powershell
python chat.py
python chat.py --checkpoint checkpoints\jingpt_best.pt --temperature 0.8
python chat.py --quant int8            # weight-only quant int8 (tiết kiệm VRAM)
python chat.py --static-kv --int8-kv   # KV cache pre-allocated + int8 (decode nhanh hơn, ít VRAM)
```

Gõ `quit` để thoát, `reset` để xoá lịch sử.

---

## 11. Agentic mode — Jin biết dùng công cụ

```powershell
python agent.py "What is sqrt(144) * 3?"                 # dùng calculator, trả lời kèm kết quả
python agent.py "Latest news about AI?" --no-search      # tắt web theo chính sách
python agent.py "Who is the president of France?" --rigor high   # high = bắt buộc citation + resample kiểm tra
python agent.py "Solve 2+2*2" --max-tool-calls 4
```

Tool có sẵn: `calculator` (parser an toàn, không eval), `python_exec` (sandbox subprocess: chặn mạng/ghi file/import ngoài whitelist), `web_search`, `fetch_page`, `memory_search`, `memory_write`, `now`. `--rigor` chọn mức tự kiểm tra: `fast` / `normal` / `high`.

---

## 12. Web app + Growth Dashboard

```powershell
python app.py                                    # http://localhost:5000 (chỉ loopback)
python app.py --static-kv --int8-kv              # KV cache nhanh + tiết kiệm VRAM
python app.py --quant int8                       # quant weight cho inference
python app.py --no-memory                        # tắt bộ nhớ dài hạn (hành vi v2.5)
```

### Admin API (bảng điều khiển sinh trưởng)

Chạy local mặc định (`python app.py`, bind 127.0.0.1) thì admin **mở luôn** —
model bar ở topbar đổi checkpoint được ngay, không cần token.

Khi nào cần token: muốn bật token mode, hoặc mở server ra LAN. Token thống nhất
cho máy này là `Zerosd Logiol` (nhập đúng chuỗi đó vào ô *Admin token* trong
Model Lab — UI sẽ tự gắn `Authorization` cho mọi API):

```powershell
$env:JINGPT_TOKEN = "Zerosd Logiol"   # PowerShell
set JINGPT_TOKEN=Zerosd Logiol        # cmd
python app.py
```

Bind ra LAN mà không có token thì admin **đóng** (403) — cố ý.

```powershell
curl -s -H "Authorization: Bearer mysecret" http://127.0.0.1:5000/api/learn/status
curl -s -X POST -H "Authorization: Bearer mysecret" -H "Content-Type: application/json" `
     -d '{"topic":"quantum computing","dry_run":true}' http://127.0.0.1:5000/api/learn/trigger
curl -s -X POST -H "Authorization: Bearer mysecret" -H "Content-Type: application/json" `
     -d '{"path":"checkpoints/jingpt_chat.pt","changelog":{"learned":"x"}}' http://127.0.0.1:5000/api/checkpoints/promote
```

Bấm nút **📈 Growth** trên sidebar để xem: val loss / probe / tok/s theo thời gian, memory stats, knowledge frontier sống, registry checkpoint (golden bất khả xâm phạm + N gần nhất). Muốn mở admin cho máy khác trên LAN (không khuyến khích): `$env:JINGPT_ALLOW_REMOTE_ADMIN="1"`.

### Memory API (RAG có citation)

```powershell
curl -X POST -H "Content-Type: application/json" -d '{"text":"Paris is the capital of France","source_url":"https://example.com","confidence":0.95}' http://127.0.0.1:5000/api/memory/add
curl -X POST -H "Content-Type: application/json" -d '{"query":"capital of France","k":5}' http://127.0.0.1:5000/api/memory/search
curl http://127.0.0.1:5000/api/memory/stats
curl -X POST -H "Content-Type: application/json" -d '{"id":"..."}' http://127.0.0.1:5000/api/memory/forget
```

---

## 13. Tự học liên tục — Curiosity Engine

```powershell
python learn_daemon.py --cycles 2 --dry-run --topic "quantum computing"   # lên kế hoạch, không train
python learn_daemon.py --cycles 5 --topics data\curiosity\topics.txt     # chạy thật (cần train provider)
```

Vòng 8 bước: phát hiện lỗ hổng → đặt câu hỏi + quiz (viết TRƯỚC khi học) → harvest (robots.txt + allowlist + SSRF guard) → **kiểm chứng chéo ≥2 nguồn độc lập** → self-edit (SEAL) → học có replay → **eval gate** (quiz ≥80%, forgetting ≤1%, val loss ≤+1%) → ledger + UCB. Nguồn không verify được → chỉ vào memory, **không bao giờ vào weight**. Budget & phanh: `JINGPT_CURIOSITY_MAX_REQUESTS`, `JINGPT_CURIOSITY_MAX_GB_DAY`, `JINGPT_CURIOSITY_MAX_CYCLES`.

---

## 14. Kiểm tra / test

```powershell
python -m unittest discover -s tests -v      # toàn bộ suite (CPU, ~1 phút)
python -m unittest tests.test_tool_sandbox -v    # riêng sandbox tool
python config.py --describe --depth 20            # dự đoán params/VRAM/tokens theo depth
```

---

## 15. Nếu web app báo chưa có model

```powershell
python train.py --mode instruct --fast --steps 2000 --device cuda   # hoặc pipeline bootstrap (mục 7)
python app.py
```

---

## 16. Biến môi trường hay dùng

| Biến | Giá trị | Ý nghĩa |
|---|---|---|
| `JINGPT_SCALE` | `tiny` `small` `medium` `large` `xl-moe` | Preset model (mặc định `small`) |
| `JINGPT_DEVICE` | `cuda` `mps` `cpu` `auto` | Ép device |
| `JINGPT_HOST` | `0.0.0.0` | Mở web ra LAN (chỉ khi thật sự cần) |
| `JINGPT_PORT` | số | Cổng web (mặc định 5000) |
| `JINGPT_TOKEN` | chuỗi | Bật Bearer token + admin API (máy này: `Zerosd Logiol`) |
| `JINGPT_ALLOW_REMOTE_ADMIN` | `1` | Cho admin API truy cập ngoài loopback |
| `JINGPT_QUANT` | `int8` `int4` | Weight-only quant mặc định |
| `JINGPT_STATIC_KV` / `JINGPT_KV_INT8` | `1` | KV cache pre-allocated / int8 |
| `JINGPT_COMPILE` | `1` | Bật torch.compile (tự fallback nếu lỗi) |
| `JINGPT_PROMPT_LOOKUP_K` | số | Độ sâu speculative decoding (0 = tắt) |
| `JINGPT_NO_MEMORY` | `1` | Tắt bộ nhớ dài hạn |
| `JINGPT_CURIOSITY_*` | số | Budget tự học (request/GB/cycles) |

---

## 17. Tắt môi trường ảo

```powershell
deactivate
```

> Lộ trình khuyến nghị mỗi lần nâng cấp model: **đo baseline (mục 9) → train dữ liệu mới (mục 6) → đo lại + so sánh (mục 9) → chỉ merge khi có số liệu tốt hơn**. Không có số liệu = không merge.