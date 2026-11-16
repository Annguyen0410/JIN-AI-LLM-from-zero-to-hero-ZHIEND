# JIN AI — Mô hình ngôn ngữ từ con số 0

**Training chia rõ 3 loại:** mở [trung tâm Pretrain / SFT / Preference](docs/training/README.md) để chọn đúng dữ liệu, checkpoint, test và lệnh tiếp tục. [Kế hoạch dữ liệu tổng quát 70/30](docs/training/DATA_GENERALIST.md) tải từ nguồn công bố, trần mới 20 GiB; xem báo cáo tiến độ, không coi kế hoạch là dữ liệu đã sẵn sàng train. V4 60k là SFT, hash checkpoint đã khớp log; chưa có benchmark năng lực v4 mới.

> **Verified status — 2026-09-30:** See [upgrade report](docs/UPGRADE_2026.md)
> for current evidence, maturity and limitations. Base model remains 13/165
> probes (no new learned-capability claim). Retrieved chat now defaults to
> cited evidence quotations or abstention; `grounded:false` opts into unverified
> model synthesis. `/api/agent` supports verified calculator + bounded tools;
> Python execution is disabled in agents. Self-learning is **not operational**.
> Training now uses conversation-disjoint SFT validation and hashed manifests;
> the old validation metrics are not directly comparable. The default mixture
> remains a template; a separate verified dataset is now available at
> [prepared_v4](data/prepared_v4/manifest.json). No pretrained model or new dependency was added.

**Kho dữ liệu mới:** Đã bổ sung 20.000 hội thoại/code thô, mẫu FineWeb tiếng Việt/giáo dục và chuẩn bị bộ riêng gồm **38,24 MB pretrain + 17.086 SFT train / 914 val**. Xem [hướng dẫn dữ liệu tiếng Việt](docs/DATA_TRAINING_VI.md) để mở rộng từ kho nhiều GB, tải có ngân sách và kiểm tra giấy phép. Corpus/checkpoint cũ không bị thay thế; chưa chạy train dài.

**Train trên Kaggle:** xem [hướng dẫn từng bước cho bộ v4](docs/KAGGLE_V4_VI.md) và [notebook để import](docs/JIN_KAGGLE_V4.ipynb). Máy nhà chỉ đóng gói/upload; bắt đầu bằng smoke-test 20 steps trên GPU Kaggle, checkpoint đầu ra tên riêng.

**Lưu trữ trên ổ D:** Data/checkpoint/cache nằm trong project; file tạm và cache thư viện cũng được cấu hình về project. Dùng [launcher Python trên D](scripts/python_d.py) và xem [hướng dẫn lưu trữ](docs/STORAGE_D_VI.md) để pip/compiler không quay về Temp/cache trên C. Chỉ đổi môi trường tiến trình, không cấu hình toàn Windows.

**JIN AI** là một mô hình ngôn ngữ kiểu GPT (decoder-only transformer) được xây dựng **hoàn toàn từ đầu bằng PyTorch** — không dùng pretrained weights, không gọi API ngoài, không phụ thuộc HuggingFace models. Toàn bộ hệ thống chỉ dùng **PyTorch + thư viện chuẩn của Python**.

Dự án bao gồm cả một pipeline trọn vẹn: thu thập dữ liệu → làm sạch → xây BPE tokenizer → huấn luyện (pretrain + instruct tuning) → kiểm tra chất lượng → chat trong terminal → giao diện web có streaming và tìm kiếm web.

---

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Yêu cầu hệ thống & cài đặt](#2-yêu-cầu-hệ-thống--cài-đặt)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Cấu hình trung tâm — `config.py`](#4-cấu-hình-trung-tâm--configpy)
5. [Kiến trúc mô hình — `model.py`](#5-kiến-trúc-mô-hình--modelpy)
6. [BPE Tokenizer — `bpe_tokenizer.py`](#6-bpe-tokenizer--bpe_tokenizerpy)
7. [Định dạng hội thoại — `chat_template.py`](#7-định-dạng-hội-thoại--chat_templatepy)
8. [Dataset chat có loss masking — `chat_dataset.py`](#8-dataset-chat-có-loss-masking--chat_datasetpy)
9. [Pipeline dữ liệu](#9-pipeline-dữ-liệu)
10. [Huấn luyện — `train.py`](#10-huấn-luyện--trainpy)
11. [Pipeline tự động — `pipeline.py`](#11-pipeline-tự-động--pipelinepy)
12. [Sinh văn bản — `generate.py`](#12-sinh-văn-bản--generatepy)
13. [Chat terminal — `chat.py`](#13-chat-terminal--chatpy)
14. [Web App — `app.py`](#14-web-app--apppy)
15. [Tìm kiếm web — `web_search.py`](#15-tìm-kiếm-web--web_searchpy)
16. [Kiểm tra độ trôi chảy — `fluency_check.py`](#16-kiểm-tra-độ-trôi-chảy--fluency_checkpy)
17. [Tiện ích — `utils.py`](#17-tiện-ích--utilspy)
18. [Biến môi trường](#18-biến-môi-trường)
19. [Quy trình huấn luyện từ A→Z](#19-quy-trình-huấn-luyện-từ-az)
20. [Câu hỏi thường gặp & khắc phục lỗi](#20-câu-hỏi-thường-gặp--khắc-phục-lỗi)
21. [Ý tưởng nâng cấp](#21-ý-tưởng-nâng-cấp)
22. [Benchmark & đánh giá — `bench.py`](#22-benchmark--đánh-giá--benchpy)

---

## 1. Tổng quan kiến trúc

```
                    ┌────────────────────────────────────────────────────┐
                    │                 DỮ LIỆU THÔ                        │
                    │  sources.json (Gutenberg, Wikipedia...)           │
                    │  hf_datasets.json (TinyStories, Dolly, Alpaca...) │
                    │  data/raw/*  (file local của bạn)                 │
                    └──────────────────────┬─────────────────────────────┘
                                           │  fetch_datasets.py
                    ┌──────────────────────▼─────────────────────────────┐
                    │           XÂY CORPUS                                │
                    │  build_dataset.py        → data/train.txt          │
                    │  build_instruct_dataset.py → data/chat_train.txt   │
                    │  (seed + HF + synth + literary echo)               │
                    └──────────────────────┬─────────────────────────────┘
                                           │  bpe_tokenizer.py
                    ┌──────────────────────▼─────────────────────────────┐
                    │        TOKENIZER (bpe.json, vocab 16.384)          │
                    └──────────────────────┬─────────────────────────────┘
                                           │
                    ┌──────────────────────▼─────────────────────────────┐
                    │          HUẤN LUYỆN (train.py)                     │
                    │  pretrain  → checkpoints/jingpt_best.pt            │
                    │  instruct  → checkpoints/jingpt_chat.pt            │
                    └──────────────────────┬─────────────────────────────┘
                                           │
              ┌────────────────────────────┼────────────────────────────┐
              ▼                            ▼                            ▼
      generate.py / chat.py         app.py (web UI)            fluency_check.py
      (terminal)                    (SSE streaming +          (chẩn đoán chất lượng)
                                     web search)
```

**Luồng chat hoàn chỉnh chỉ với 1 lệnh:**

```bash
python pipeline.py chat
```

Lệnh này tự động chạy: `build_instruct_dataset.py` → `bpe_tokenizer.py` → `train.py --mode instruct`.

---

## 2. Yêu cầu hệ thống & cài đặt

### Phần cứng

| Cấu hình | VRAM / RAM | Ghi chú |
|---|---|---|
| **Tiny (~35M)** | chạy được trên CPU | Laptop bình thường, dùng để smoke-test |
| **Small (~143M)** | ~6GB GPU hoặc CPU (chậm) | Checkpoint hiện tại, chạy tốt trên RTX 2060 |
| **Medium (~360M)** | ≥ 11GB GPU | Mục tiêu SLM khuyến nghị |
| **Large (~1.0B)** | GPU thật sự | Cần GPU mạnh |

### Cài đặt (Windows — cmd / PowerShell / Git Bash)

```powershell
# 1. Vào thư mục dự án
cd "d:\Code folder\Jin New"

# 2. Tạo môi trường ảo
python -m venv .venv

# 3. Kích hoạt (chọn 1 trong 2)
.\\.venv\\Scripts\\Activate.ps1     # PowerShell
.\\.venv\\Scripts\\activate.bat     # Command Prompt

# 4. Cài thư viện
pip install -r requirements.txt
```

`requirements.txt` chỉ cần 3 gói: `flask>=3.0.0`, `numpy>=2.0`, `torch>=2.0.0`.

> **Lưu ý CUDA:** nếu muốn train trên GPU NVIDIA, hãy cài bản PyTorch có CUDA:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
> ```
> (chọn wheel `cu1xx` phù hợp với driver của bạn tại https://pytorch.org/get-started/locally/)

> **Gói tùy chọn:** `pyarrow>=15.0.0` — chỉ cần khi bạn bật các dataset dạng parquet trong `hf_datasets.json` (Cosmopedia, Wikipedia parquet, OpenHermes…). `web_search.py` **không** cần gói nào ngoài stdlib.

### Quick start — chạy ngay

```bash
# 1. Chạy web UI (mở http://localhost:5000) — dùng checkpoint sẵn có
python app.py

# 2. Hoặc chat trong terminal
python chat.py

# 3. Đo baseline / kiểm tra checkpoint
python bench.py --checkpoint checkpoints/jingpt_chat.pt

# 4. Train từ đầu (xem mục 10 & RUN_COMMANDS.md để đầy đủ)
python pipeline.py bootstrap --steps 500 --pref-steps 200 --device cuda
```

Khi chưa có checkpoint nào, `app.py`/`chat.py` báo rõ "chưa có model" và chỉ dẫn
train. Xem **RUN_COMMANDS.md** cho lộ trình từng bước (cài đặt → train → generate
→ chat → web app).

---

## 3. Cấu trúc thư mục

```text
Jin New/
│
├── config.py                  # Toàn bộ hyperparameter, đường dẫn, cấu hình device
├── model.py                   # GPT decoder-only transformer (2 biến thể kiến trúc)
├── train.py                   # Trainer thống nhất: pretrain + instruct
├── pipeline.py                # Điều phối end-to-end (chat / raw)
├── fetch_datasets.py          # Tải dữ liệu: sources.json + hf_datasets.json
├── build_dataset.py           # Xây corpus pretrain từ file local (data/raw)
├── build_instruct_dataset.py  # Xây corpus chat (seed + HF + synth + literary)
├── chat_dataset.py            # Dataset chat dạng packed, có loss masking
├── chat_template.py           # Định dạng ChatML-style cho hội thoại
├── bpe_tokenizer.py           # Byte-level BPE tokenizer (tự viết, không thư viện ngoài)
├── tokenizer.py               # Char tokenizer cũ (chỉ để load checkpoint legacy)
├── chat.py                    # Chat REPL trong terminal
├── app.py                     # Flask web app (SSE streaming + web search)
├── generate.py                # Sinh văn bản từ prompt đơn
├── fluency_check.py           # Chẩn đoán "model đã sẵn sàng chat chưa?"
├── web_search.py              # Tìm kiếm DuckDuckGo + fetch trang + robots.txt + Wikipedia/arXiv API (chỉ stdlib)
├── clean_corpus.py            # Hàm làm sạch văn bản cho corpus
├── optim_muon.py              # Muon optimizer tự viết (Newton-Schulz, 1 buffer momentum) (v3 Phase 2)
├── quantize.py                # Weight-only quant int8 per-channel / int4 group-128 (v3 Phase 2)
├── utils.py                   # Checkpoint I/O, tạo model, cache token, device setup
├── opencode.jsonc             # Cấu hình ARIS / opencode (custom commands)
├── bench.py                   # Benchmark 1-lệnh (v3 Phase 0)
├── data_quality.py            # Dedupe MinHash-LSH, quality filter, language, contamination (v3 Phase 3)
├── build_reasoning_data.py    # Dữ liệu CoT toán/logic/code có kiểm chứng Python (v3 Phase 3)
├── self_distill.py            # Rejection-sampling self-distillation k=8 (v3 Phase 3)
├── train_pref.py              # Preference tuning: DPO + ORPO tự viết (v3 Phase 3)
├── memory.py                  # Bộ nhớ dài hạn 3 tầng + consolidation/decay (v3 Phase 4)
├── retriever.py               # Embedding Jin + VectorIndex + BM25 + RRF + rerank (v3 Phase 4)
├── curiosity_engine.py        # Vòng 8 bước: gap → query+quiz → harvest → verify → self-edit → train → gate → ledger (v3 Phase 5)
├── knowledge_ledger.py        # Sổ append-only mọi vòng học + UCB bandit (v3 Phase 5)
├── lora.py                    # LoRA tự viết cho cập nhật nhẹ, merge/unmerge rollback (v3 Phase 5)
├── learn_daemon.py            # Daemon học nền: budget + phanh an toàn + --dry-run (v3 Phase 5)
├── tools.py                   # Registry tool schema JSON: calculator an toàn, python_exec sandbox, memory (v3 Phase 6)
├── agent.py                   # ReAct loop: <|tool|>{...} → thực thi → tiếp tục + self-check verifier (v3 Phase 6)
├── tests/                     # Unit test (unittest stdlib, CPU-safe)
├── scripts/
│   └── demo_xl_moe.py         # Demo preset xl-moe + expert offload
├── bench_results/             # Kết quả benchmark (JSON + baseline_v25.json)
│
├── data/eval/                 # Held-out set + probes (benchmark)
│   ├── heldout.txt
│   └── probes.jsonl
│
├── sources.json               # Danh mục nguồn raw text (Gutenberg, Wikipedia...)
├── hf_datasets.json           # Danh mục dataset HuggingFace (SLM-grade)
│
├── bpe.json                   # BPE vocabulary đã train (16384 tokens)
├── vocab.json                 # Char vocab cũ (legacy)
│
├── data/
│   ├── train.txt              # Corpus pretrain (raw text, có thẻ <doc>)
│   ├── chat_train.txt         # Corpus chat định dạng ChatML
│   ├── raw/                   # Text thô đã tải về / bạn tự bỏ vào
│   ├── curated/snapshots/     # Snapshot & backup của train.txt
│   └── seed/
│       ├── conversations.jsonl    # Hội thoại seed viết tay (JSONL)
│       └── hf/                    # Conversation shard từ HF datasets
│
├── cache/                     # Token cache (tensor .pt đã mã hoá)
├── checkpoints/
│   ├── jingpt_latest.pt       # Checkpoint mới nhất (pretrain/instruct)
│   ├── jingpt_best.pt         # Checkpoint val-loss thấp nhất (pretrain)
│   ├── jingpt_chat.pt         # Checkpoint instruct tuning tốt nhất
│   └── jingpt.pt              # Checkpoint legacy (char-level, cũ)
│
└── static/
    ├── index.html             # Giao diện chat web
    ├── style.css              # Style (dark theme)
    └── script.js              # Logic frontend (SSE, settings, search)
```

---

## 4. Cấu hình trung tâm — `config.py`

Đây là "bộ não" cấu hình: mọi script trong dự án đều import `config`.

### 4.1 Đường dẫn quan trọng

| Hằng số | Đường dẫn | Dùng cho |
|---|---|---|
| `DATA_PATH` | `data/train.txt` | Corpus pretrain |
| `CHAT_DATA_PATH` | `data/chat_train.txt` | Corpus instruct |
| `SEED_CONVERSATIONS_PATH` | `data/seed/conversations.jsonl` | Hội thoại seed |
| `BPE_PATH` | `bpe.json` | BPE vocab |
| `VOCAB_PATH` | `vocab.json` | Char vocab (legacy) |
| `CHECKPOINT_DIR` | `checkpoints/` | Thư mục checkpoint |
| `LATEST_CHECKPOINT_PATH` | `checkpoints/jingpt_latest.pt` | Mới nhất |
| `BEST_CHECKPOINT_PATH` | `checkpoints/jingpt_best.pt` | Val loss tốt nhất |
| `CHAT_CHECKPOINT_PATH` | `checkpoints/jingpt_chat.pt` | Sau instruct tuning |
| `LEGACY_CHECKPOINT_PATH` | `checkpoints/jingpt.pt` | Checkpoint cũ |
| `JINGPT_CHECKPOINT` (env) | *(không có)* | Ghim checkpoint cụ thể, bỏ qua mọi suy đoán |

Hàm `resolve_checkpoint()` → `(path, vì_sao)` chọn checkpoint **tại thời điểm gọi** (không phải lúc import), theo thứ tự:

1. `JINGPT_CHECKPOINT` — chỉ định tay, thắng tất cả;
2. checkpoint **biết chat** mới nhất (theo mtime) ở *bất kỳ đâu* dưới `checkpoints/`: `jingpt_chat.pt`, bản sao `jingpt_chat_*.pt`, hay bản instruct vừa train trên Kaggle (`kaggle_v4_60020/kaggle_v4_instruct.pt`). Copy file mới vào là app dùng ngay ở lần khởi động sau, không phải sửa code;
3. `jingpt_best.pt` — phương án cuối, **chỉ pretrain** (trả lời kiểu nối tiếp văn bản, không biết format chat).

`iter_checkpoint_paths()` quét `checkpoints/` **đệ quy** (bỏ thư mục ẩn và `__pycache__`, giới hạn `CHECKPOINT_WALK_DEPTH`), nên checkpoint nằm trong thư mục con của một run Kaggle vẫn hiện đủ trong Model Lab lẫn model bar ở topbar — mỗi dòng kèm thư mục, step, val loss, dung lượng, mtime và dấu `← đang chạy`.

Đừng nhầm tên: `jingpt_best.pt` / `jingpt_latest.pt` nghe như “bản tốt nhất / mới nhất” nhưng chúng là mốc **pretrain** (val loss không so được với `jingpt_chat.pt` vì khác vocab/đợt dữ liệu). `checkpoint_role()` phân loại file để Model Lab ghi rõ `chat` / `instruct` / `pretrain best` / `pretrain latest` / `pretrain` / `legacy`; chỉ `chat` và `instruct` mới được coi là “biết chat” (`config.CHAT_ROLES`), phần còn lại chỉ nối tiếp văn bản.

Muốn chạy đúng một bản cụ thể:

```powershell
$env:JINGPT_CHECKPOINT="checkpoints/jingpt_chat_46k_sft4k.pt"; python app.py
python app.py --checkpoint checkpoints/jingpt_chat_46k_sft4k.pt   # tương đương
```

### 4.2 Các preset model (theo `MODEL_SCALE`)

| Preset | Số tham số | `embed_dim` | Layers | Heads | KV heads | `block_size` | Dùng khi nào |
|---|---|---|---|---|---|---|---|
| `tiny` | ~35M | 384 | 6 | 6 | 6 | 512 | Laptop / smoke test |
| `small` | ~143M | 992 | 12 | 16 | 8 (GQA) | 1024 | **Checkpoint hiện tại** |
| `medium` | ~360M | 1280 | 20 | 20 | 4 (GQA) | 2048 | SLM mục tiêu (GQA + grad-checkpointing) |
| `large` | ~1.0B | 2048 | 24 | 32 | 8 (GQA) | 2048 | Cần GPU mạnh |
| `xl-moe` | ~0.78B total / ~0.31B active | 1280 | 12 | 20 | 4 (GQA) | 4096 | MoE 8 expert top-2 ở layer chẵn, expert offload → chạy được 6GB |

Chọn scale bằng biến môi trường `JINGPT_SCALE` (mặc định `small`). Mọi preset v3 có GQA + QK-norm; `xl-moe` thêm sliding window (1024, 3 local : 1 global) và MoE.

**Auto-scale một núm:** `python config.py --describe --depth N` in bảng dự đoán params/active/VRAM/tokens cho mọi preset, kèm config suy ra từ depth N (dùng trong `train.py --depth N`).

### 4.3 Hyperparameter huấn luyện mặc định

- `BATCH_SIZE=1`, `GRAD_ACCUM=2`, `TRAIN_BLOCK_SIZE=256`
- `RUN_ITERS=100_000`, `LR=2e-4`, `MIN_LR_RATIO=0.1`, `WARMUP_ITERS=200`
- `WEIGHT_DECAY=0.1`, `BETA1=0.9`, `BETA2=0.95`, `MAX_GRAD_NORM=1.0`
- `EVAL_INTERVAL=1000`, `EVAL_ITERS=5`, `CHECKPOINT_INTERVAL=1000`
- `TRAIN_SPLIT=0.9` (90% train / 10% val), `SEED=1337`
- `AMP_DTYPE="float16"` (tự động chuyển sang bfloat16 nếu cấu hình)
- `GRADIENT_CHECKPOINTING=False` (tắt trên CPU vì giảm throughput; bật trên GPU để tiết kiệm VRAM)

### 4.4 Tham số sinh văn bản mặc định

`TEMPERATURE=0.65`, `TOP_K=40`, `TOP_P=0.9`, `REPETITION_PENALTY=1.12`, `MAX_NEW_TOKENS=256`.

### 4.4b Chống lặp (anti-echo)

`NO_REPEAT_NGRAM_SIZE=3`, `STOP_ON_REPEAT=True`.

Penalty token-đơn không chặn được echo nguyên cụm: mọi token của
"a small language model" đều hợp lệ, chỉ *n-gram* là không. Ba lớp xử lý
(xem `model.repeated_tail_period`, `reply_quality.trim_repeated_tail`):

1. **Ban n-gram** — token nào hoàn tất một n-gram đã có trong chuỗi thì bị đặt
   `-inf` (không bao giờ ban token dừng, nếu không model không kết thúc được lượt).
2. **Tự dừng khi lặp** — đuôi sinh ra mà lặp lại một cụm 3+ lần thì dừng decode,
   `stop_reason="repeat"` trả về cho client.
3. **Cắt đuôi văn bản** — bản cuối (`done.reply` trong SSE, `response` trong
   `/api/chat`) đã cắt phần lặp đã stream đi rồi.

### 4.5 Bảo mật web server

- `SERVER_HOST=127.0.0.1` (chỉ localhost — ghi đè bằng `JINGPT_HOST=0.0.0.0` khi chủ động muốn expose LAN)
- `SERVER_PORT=5000` (`JINGPT_PORT`)
- `MAX_REQUEST_BYTES=32KB`, `MAX_USER_MESSAGE_CHARS=4000`
- Rate limit: `RATE_LIMIT_RPS=4`, `RATE_LIMIT_BURST=12` (token bucket theo IP)
- `REQUIRE_LOOPBACK_TOKEN=False`, `SERVER_AUTH_TOKEN=""` (đặt `JINGPT_TOKEN` để bật Bearer token; không đặt + bind loopback thì admin API mở cho loopback — xem 4.5 phần bảo mật và `config.host_is_loopback()`)

### 4.6 Chọn device

`resolve_device(request)` — thứ tự ưu tiên: `auto` (mặc định) → `cuda` nếu có → `mps` (Apple Silicon) → `cpu`.

Các giá trị hợp lệ: `auto`, `gpu` (CUDA/MPS, cảnh báo nếu không có), `cuda`, `mps`, `cpu`. Dùng `--device` trên CLI hoặc `JINGPT_DEVICE` env var.

---

## 5. Kiến trúc mô hình — `model.py`

`JinAI` là decoder-only transformer, hỗ trợ **2 biến thể kiến trúc** qua flag `modern`:

| | Classic (`modern=False`) | Modern (`modern=True`) |
|---|---|---|
| Normalization | `nn.LayerNorm` | `RMSNorm` |
| Feed-forward | Linear → GELU → Linear | **SwiGLU** (gate × up × down) |
| Positional encoding | Học được (`pos_emb` Embedding) | **RoPE** (rotary) |
| Checkpoint hiện tại | Có (checkpoint cũ) | Khuyến nghị khi train mới |

### 5.1 Các lớp thành phần

- **`RMSNorm`** — chuẩn hoá theo root-mean-square, không bias, không trừ mean.
- **`build_rope_cache` / `apply_rope`** — precompute bảng cos/sin cho rotary position embeddings, shape `(block_size, head_dim)`. RoPE cho phép model tổng quát hoá tốt hơn với chuỗi dài.
- **`SwiGLU`** — feed-forward hiện đại: `w_down(silu(w_gate(x)) * w_up(x))`, hidden size = `embed_dim × 8/3` làm tròn lên bội số 64.
- **`CausalSelfAttention`** — một `qkv_proj` duy nhất, chia 3, dùng `F.scaled_dot_product_attention` (fallback bằng masked matmul thủ công nếu torch cũ). Hỗ trợ **KV cache** để sinh nhanh, **GQA** (`n_kv_heads < n_heads` → cache nhỏ ~4x), **QK-norm** (RMSNorm q/k, chống loss spike), **sliding window** (layer local chỉ nhìn `window` token gần nhất, cache trim đúng window).
- **`MoEFeedForward`** (v3) — Mixture-of-Experts: router linear + softmax, top-k (mặc định 2/8), capacity factor, load-balancing aux loss kiểu Switch + z-loss tuỳ chọn; `--expert-offload` giữ expert trên CPU, router prefetch lên GPU (cho VRAM 6GB). Đếm tham số **total vs active** chính xác.
- **RoPE scaling** (v3) — `rope_theta` cấu hình được + NTK-aware / YaRN-lite (`--extend-context N` trong `train.py`) để mở rộng ctx 1024 → 4096 → 8192 mà không train lại từ đầu.
- **`FeedForward`** — MLP cổ điển 4× cho biến thể classic.
- **`TransformerBlock`** — pre-norm residual: `x + attn(ln1(x))` rồi `x + ffn(ln2(x))`.

### 5.2 `JinAI` — model chính

- **Weight tying**: `head.weight = tok_emb.weight` (share embedding input/output).
- **Khởi tạo**: `_init_weights` — Linear/Embedding ~ `N(0, 0.02)`, bias = 0, LayerNorm = 1/0.
- **`forward(idx, targets, kv_caches, loss_mask)`**:
  - Trả về `(logits, loss)`.
  - Nếu không có `loss_mask`: cross-entropy chuẩn trên toàn bộ token.
  - Nếu có `loss_mask`: cross-entropy **per-token** nhân mask, chia cho tổng mask — đây là cơ chế để gradient chỉ tập trung vào token của assistant (xem mục 8).
  - Nếu `targets=None`: chỉ trả logits (dùng cho generation).
- **`generate()`** — sinh autoregressive:
  - Dùng KV cache (tăng tốc ~5-10×) kèm `_trim_kv_cache` để giữ cache trong `block_size`.
  - `_apply_repetition_penalty` — penalty có dấu (nhân nếu logit âm, chia nếu dương — đúng theo HF Transformers, tránh lỗi "thưởng" token lặp).
  - `_filter_logits` — áp temperature → top-k → top-p.
  - `temperature <= 0` → greedy (argmax).
  - Dừng sớm nếu gặp token trong `stop_ids` (ví dụ `<|end|>`).
- **`stream_generate()`** — generator trả từng token một cho SSE streaming, kèm KV cache.

---

## 6. BPE Tokenizer — `bpe_tokenizer.py`

Byte-level BPE viết tay, **không dùng** tiktoken / sentencepiece / transformers.

### Đặc điểm thiết kế

1. **Byte-level**: mọi byte của UTF-8 đều biểu diễn được → encode **mọi ngôn ngữ** losslessly (tiếng Việt, emoji, code…).
2. **Pre-tokenization**: regex kiểu GPT-2 giữ token `" the"` khác với `"the"` (khoảng trắng đầu quan trọng cho sinh trôi chảy).
3. **Incremental training**: giữ index ngược `pair → {word}`, mỗi bước merge chỉ chạm vào các word bị ảnh hưởng → train 16k merges trên corpus ~80MB chỉ mất vài phút Python thuần.
4. **Special tokens được bảo vệ**: không bao giờ bị BPE tách, luôn mã hoá nguyên khối.

### Cấu trúc vocab

```text
[0 .. 255]                        → 256 byte thô (luôn có)
[256 .. 256+M-1]                  → M merges học được
[cuối - S .. cuối]                → S special tokens (ID cố định ở đuôi)
```

6 special tokens mặc định: `<|endoftext|>`, `<|user|>`, `<|assistant|>`, `<|system|>`, `<|end|>`, `<|pad|>`.

### API chính

| Phương thức | Chức năng |
|---|---|
| `train(text, vocab_size=16384)` | Học merges từ corpus |
| `encode(text, allow_special=True)` | Text → list token ID (special token → ID riêng) |
| `decode(ids)` | Token ID → text (utf-8, lỗi → `replace`) |
| `save(path)` / `load(path)` | Lưu/đọc JSON (`bpe.json`) |
| `to_state_dict()` / `load_state_dict()` | Embed vào checkpoint (để resume không cần file riêng) |
| `preview(text)` | Hiển thị token đầu tiên để gỡ lỗi |

**CLI:**
```bash
python bpe_tokenizer.py --data data/chat_train.txt --out bpe.json --vocab-size 16384
```

---

## 7. Định dạng hội thoại — `chat_template.py`

Mọi hội thoại được bọc bằng special tokens để model học cấu trúc hội thoại tường minh:

```text
<|system|>   ... system message ... <|end|>
<|user|>     ... user message ...   <|end|>
<|assistant|> ... reply ...         <|end|>
<|endoftext|>   (kết thúc hội thoại / tài liệu)
```

- `format_message(role, content)` — bọc 1 message.
- `format_conversation(messages, add_generation_prompt=False, default_system=...)`:
  - Nếu không có message system và `default_system` được set → tự chèn system prompt mặc định.
  - `add_generation_prompt=True` → kết thúc bằng thẻ `<|assistant|>\n` để model tiếp tục sinh câu trả lời.
- `format_training_document(messages)` — render thành 1 tài liệu training hoàn chỉnh kết thúc bằng `<|endoftext|>`.
- `DEFAULT_SYSTEM_PROMPT` — prompt system mặc định giới thiệu "Jin, a small but capable AI assistant built from scratch in PyTorch…"
- `STOP_TOKEN_STRINGS = (<|endoftext|>, <|end|>)` — dùng để dừng generation.

---

## 8. Dataset chat có loss masking — `chat_dataset.py`

### Tại sao cần loss masking?

Nếu tính loss trên mọi token như pretrain, model sẽ bị phạt vì không đoán được *lời người dùng* ngang bằng với việc đoán câu trả lời — lãng phí sức học vào việc bắt chước cách đặt câu hỏi. Cách chuẩn của mọi mô hình instruct hiện đại: **chỉ lan truyền gradient trên token của assistant**. Kết quả là model tập trung học "cho hội thoại này, viết câu trả lời đúng".

Với mỗi hội thoại, module tạo ra 3 tensor:

```text
input_ids: [t0, t1, ..., tN-1]
target_ids: [t1, t2, ..., tN]        # dịch chuyển next-token
loss_mask:  [m0, m1, ..., mN-1]      # 1 chỉ trên nội dung assistant
```

### Các thành phần

- **`build_masked_example(messages, tokenizer)`** — token hoá 1 hội thoại và sinh mask. Mask = 1 trên content + suffix (`<|end|>`) của assistant turn (và system nếu `train_on_system=True`); = 0 trên prefix thẻ, content user, và `<|endoftext|>`.
- **`iter_conversations(paths)`** — đọc các file JSONL `{"messages": [...]}`; lọc message rác, đảm bảo hội thoại kết thúc bằng turn assistant.
- **`PackedChatDataset`** — nối nhiều hội thoại thành một chuỗi dài rồi cắt thành block cố định (`block_size + 1` token). Giữa các hội thoại chèn `<|endoftext|>` làm ranh giới cứng để attention không tràn qua tài liệu khác. `__getitem__` trả `(x, y, m)`.
- **`build_dataset(duplication, block_size, seed)`** — hàm tiện: load BPE từ `bpe.json`, gom mọi `*.jsonl` trong `data/seed/` và các `extra_dirs`, nhân bản hội thoại lên `duplication` lần rồi pack.

Chạy thử để xem thống kê:
```bash
python chat_dataset.py --block-size 512 --duplication 30
```

---

## 9. Pipeline dữ liệu

### 9.1 Corpus pretrain — `build_dataset.py`

Đọc toàn bộ text từ `data/raw/` (hỗ trợ `.txt, .md, .rst, .json, .jsonl, .csv, .tsv, .yaml, .yml, .log, .xml`), làm sạch từng dòng, bọc trong thẻ `<doc path="...">`:

```bash
# Chạy cơ bản
py build_dataset.py --report

# Giữ dataset cũ + cộng dồn dataset mới (có backup + snapshot)
py build_dataset.py --append-existing --backup-existing --snapshot --report
```

Quy tắc:
- Bỏ file < `--min-file-bytes` (mặc định 8KB).
- Bỏ dòng quá ngắn/quá dài, dòng ít chữ cái, dòng meta Gutenberg.
- `strip_gutenberg` tự bật cho file có "gutenberg" trong đường dẫn (cắt boilerplate START/END OF PROJECT GUTENBERG).
- `strip_xml` tự bật cho file `.xml`.
- `--report` ghi `train.txt.report.json` chi tiết từng file (chars in/out, dòng giữ/bỏ).
- Cảnh báo nếu corpus < 500MB (mục tiêu khuyến nghị cho SLM).

### 9.2 Corpus chat — `build_instruct_dataset.py`

Tổng hợp 4 nguồn thành `data/chat_train.txt`:

1. **Seed conversations** (`data/seed/conversations.jsonl`) — hội thoại viết tay. Mỗi hội thoại được nhân bản `--seed-duplication` lần (mặc định 40) để nhấn mạnh.
2. **HF seed** (`data/seed/hf/*.jsonl`) — tạo bởi `fetch_datasets.py` từ các dataset hướng dẫn (Dolly, Alpaca, OASST…), có cap `--hf-max-per-dataset` (mặc định 20k) và giới hạn độ dài message để hội thoại khổng lồ không lấn át corpus.
3. **Synthetic Q&A từ corpus raw** (`--synth-samples`, mặc định 2000) — sinh hội thoại bằng template thuần:
   - *Summary* (55%): đưa 1 đoạn văn + hỏi tóm tắt → câu trả lời = 1-2 câu đầu đoạn.
   - *Continuation* (30%): đưa nửa đầu đoạn + hỏi viết tiếp → trả 3 câu tiếp theo.
   - *Definition* (15%): chọn từ dài trong văn bản + giải nghĩa theo ngữ cảnh câu chứa nó.
4. **Literary echo** (`--literary-target-mb`, mặc định 2MB) — cắt ngẫu nhiên các đoạn sách, bọc vào hội thoại giả "Read me a passage from X" để model không quên sự trôi chảy văn xuôi. **Khuyến nghị tắt khi fine-tune chat** (`--no-literary`) vì Gutenberg có thể lấn át dữ liệu chat.

```bash
python build_instruct_dataset.py                       # đầy đủ
python build_instruct_dataset.py --synth-samples 4000 --literary-target-mb 2.0
python build_instruct_dataset.py --no-synth --no-literary   # chỉ dùng seed + HF
```

### 9.3 Tải dữ liệu — `fetch_datasets.py`

Hợp nhất 2 script cũ (`fetch_corpus.py`, `fetch_hf_datasets.py`). Tự nhận diện catalog theo tên file:

**`sources.json`** — raw text công khai (tất cả sách Gutenberg public domain, Tiny Shakespeare, Wikipedia English Abstracts…). Mỗi nguồn có `type`: `text`, `gzip_text`, `wikimedia_abstract_gz`, `zip_text`. File tải về nằm trong `data/raw/<id>/`.

```bash
python fetch_datasets.py --catalog sources.json
python fetch_datasets.py --catalog sources.json --only gutenberg-alice gutenberg-moby-dick
python fetch_datasets.py --catalog sources.json --max-sources 3
```

**`hf_datasets.json`** — dataset kiểu SLM (TinyStories, Dolly 15k, Alpaca 52k, GSM8K, CodeAlpaca, OASST2, Bactrian-X tiếng Việt…). Mỗi entry khai báo `kind` (pretrain/instruction/chat/math) và `format` — bộ chuyển đổi tương ứng:

| Format | Dataset điển hình |
|---|---|
| `alpaca_json` | Alpaca, CodeAlpaca |
| `dolly_jsonl` | Dolly 15k |
| `bactrian_json` | Bactrian-X (tiếng Việt) |
| `lima_jsonl` | LIMA |
| `gsm8k_jsonl` | GSM8K |
| `metamath_json` | MetaMathQA |
| `oasst_trees_jsonl_gz` | OASST2 (cây hội thoại đa lượt, có tiếng Việt) |
| `parquet_messages` | OpenHermes (cần pyarrow) |
| `text` / `parquet` (pretrain) | TinyStories, Cosmopedia, Wikipedia |

Kết quả pretrain → `data/raw/hf-<id>/`; kết quả chat → `data/seed/hf/<id>-<shard>.jsonl` (sẵn sàng cho `build_instruct_dataset.py`).

```bash
python fetch_datasets.py --catalog hf_datasets.json --list          # liệt kê
python fetch_datasets.py --catalog hf_datasets.json --only dolly-15k --max-rows 5000
python fetch_datasets.py --limit-mb 50                              # giới hạn tải
```

> Nhiều dataset mặc định `enabled: false` (MetaMathQA 250MB, Cosmopedia, OpenHermes 1M, Wikipedia parquet) — bật `enabled: true` trong JSON khi bạn muốn dùng.

### 9.4 Làm sạch văn bản — `clean_corpus.py`

Các helper được `build_dataset.py` dùng:
- `read_text_file` — thử dần utf-8 / utf-8-sig / utf-16 / cp1252 / latin-1.
- `strip_gutenberg_boilerplate` — cắt khối START/END OF PROJECT GUTENBERG.
- `normalize_line` — unescape HTML, NFKC normalize, bỏ control chars, nén khoảng trắng.
- `is_low_quality` — bỏ dòng: quá dài, không có chữ cái, tỉ lệ chữ cái < 55%, hoặc quá ngắn và không giống hội thoại.
- `write_report` — xuất JSON thống kê.

---

## 10. Huấn luyện — `train.py`

Trainer thống nhất 2 chế độ: **pretrain** (next-token prediction trên raw text) và **instruct** (instruction tuning có loss masking).

### 10.1 Pretrain

```bash
python train.py                          # mặc định
python train.py --resume                 # tiếp tục từ checkpoint mới nhất
python train.py --resume --checkpoint checkpoints/jingpt_latest.pt
python train.py --device cpu             # ép CPU
python train.py --data data/train.txt --run-iters 50000
```

Quy trình:
1. Load text → build/load tokenizer (BPE từ `bpe.json`, hoặc restore từ checkpoint khi resume).
2. Encode text → token tensor, cache trong `cache/tokens_<hash>.pt` (hash dựa trên path + size + mtime + fingerprint tokenizer — đổi data hoặc tokenizer là tự miss cache).
3. Split 90/10 train/val.
4. AdamW + cosine LR schedule (warmup 200, min ratio 0.1) + weight decay 0.1 + grad clip 1.0.
5. AMP (float16) tự bật trên CUDA kèm GradScaler.
6. Eval mỗi `EVAL_INTERVAL` step: tính loss train/val, save latest; nếu val loss thấp hơn best → save `jingpt_best.pt`. Checkpoint lưu cả optimizer, scaler, RNG state để resume chính xác.
7. Kết thúc: load best checkpoint và sinh thử 3 prompt mẫu (`"The "`, `"Once upon a time, "`, `"I think that "`).

### 10.2 Instruct tuning

```bash
python train.py --mode instruct --steps 5000
python train.py --mode instruct --fast --steps 1000 --device cuda        # GPU 6GB
python train.py --mode instruct --batch-size 1 --grad-accum 32 --device cuda  # hết VRAM
```

Các cờ quan trọng:

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--init-from` | (tự dùng `jingpt_best.pt`) | Checkpoint pretrain để warm-start |
| `--no-pretrained-init` | off | Không load `jingpt_best.pt`, train từ khởi tạo ngẫu nhiên |
| `--duplication` | 40 | Nhân bản hội thoại seed |
| `--fast` | off | Profile tiết kiệm VRAM: block-size ≤ 256, duplication ≤ 8, eval-iters ≤ 5 |
| `--block-size` | 512 | Độ dài chuỗi huấn luyện |
| `--batch-size` / `--grad-accum` | 8 / 4 | Micro-batch & tích luỹ gradient (batch hiệu dụng = 8×4) |
| `--lr` / `--warmup` / `--min-lr-ratio` | 3e-4 / 100 / 0.1 | Lịch LR cosine |
| `--steps` | 4000 | Số step thêm vào |
| `--val-fraction` | 0.02 | Tỷ lệ validation |
| `--checkpoint-interval` | 500 | Lưu checkpoint định kỳ |
| `--resume` | off | Resume đúng optimizer/scaler từ `jingpt_chat.pt` |
| `--optimizer` | adamw | `adamw` hoặc `muon` (Muon = momentum + orthogonalization Newton-Schulz cho matrix 2D, AdamW cho embed/head/norm — 1 buffer momentum thay vì 2) |
| `--low-vram` | off | Selective gradient checkpointing (`gc_mode="layer2"`) — giảm ~1/2 activation memory |
| `--pack` | off | Sequence packing với **document mask thật** — attention không tràn qua ranh giới `<|endoftext|>` |

Cơ chế tự động:
- **`_adapt_microbatch_for_vram`** — nếu GPU < 11GB, tự giảm batch xuống (4 nếu ≥8GB, 1 nếu nhỏ hơn) và nhân grad-accum tương ứng để giữ batch hiệu dụng.
- **Compatibility check** — trước khi warm-start, kiểm tra vocab checkpoint vs tokenizer; nếu lệch (ví dụ đổi từ char sang BPE) thì bỏ qua pretrained weights với cảnh báo rõ ràng.
- Sau mỗi eval: in val loss + sinh mẫu chat nhanh `[sample] jin: ...`.

Checkpoint instruct được lưu vào `checkpoints/jingpt_chat.pt`.

---

## 11. Pipeline tự động — `pipeline.py`

Orchestrator hợp nhất 2 script cũ (`train_jin_v2.py`, `auto_pipeline.py`).

### Chế độ `chat` — toàn bộ pipeline chat

```bash
python pipeline.py chat
python pipeline.py chat --resume --skip-bpe
python pipeline.py chat --synth-samples 4000 --vocab-size 16384
```

Chạy lần lượt: `build_instruct_dataset.py` → `bpe_tokenizer.py` (nếu chưa có `bpe.json`) → `train.py --mode instruct`.

### Chế độ `raw` — fetch → build → (train)

```bash
python pipeline.py raw --train --resume          # fetch + build + pretrain
python pipeline.py raw --only gutenberg-alice --skip-build
python pipeline.py raw --skip-fetch --train      # chỉ train với dữ liệu có sẵn
```

- Fetch từng nguồn enabled trong `sources.json` (chịu lỗi từng nguồn).
- Build corpus: gộp file mới vào `train.txt` hiện có (nếu tồn tại) và **luôn snapshot** vào `data/curated/snapshots/train_auto_<timestamp>_train.txt`.
- `--train` → gọi `train.py` pretrain.

### Chế độ `bootstrap` (Phase 7) — full path v3: data → BPE → pretrain → SFT → DPO → bench

```bash
python pipeline.py bootstrap --steps 500 --pref-steps 200 --device cuda
python pipeline.py bootstrap --steps 300 --pref-steps 0          # bỏ DPO
```

Mỗi stage gọi lại CLI hiện có (cùng flag/checkpoint), nên có thể resume: stage đã
có output (corpus/BPE) tự bỏ qua; `--pref-data` chỉ định sẵn cặp preference thay vì
sinh bằng `self_distill`; kết thúc bằng `bench.py` trên checkpoint SFT cuối.

---

## 12. Sinh văn bản — `generate.py`

```bash
python generate.py --prompt "The future of AI is"
python generate.py --prompt "Hello" --temperature 0.65 --top_k 20 --top_p 0.9 --repetition-penalty 1.12
python generate.py --max_tokens 512
```

- Không có `--prompt` → bắt đầu từ token `<|endoftext|>`.
- In ra checkpoint file + step + text sinh được.
- Nhận checkpoint tùy chọn qua `--checkpoint`.
- Cờ v3 (Phase 2): `--quant int8|int4` (weight-only quant), `--static-kv` (KV pre-allocated), `--int8-kv` (KV cache int8, ít VRAM hơn).

---

## 13. Chat terminal — `chat.py`

```bash
python chat.py
```

Hỗ trợ **2 loại checkpoint**:
- **BPE chat checkpoint** (mới): enforce cấu trúc ChatML, dừng sinh khi gặp `<|end|>` / `<|endoftext|>`, tự cắt nếu model nhả thẻ role khác. Giữ lịch sử đa lượt, `--max-turns` (mặc định 12) để giới hạn prompt.
- **Char-level checkpoint** (legacy): mode tương thích ngược — chỉ tiếp tục text, không có template.

Lệnh trong chat: `quit` / `exit` để thoát, `reset` để xoá bộ nhớ hội thoại.

Cờ: `--checkpoint`, `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`, `--max-new-tokens`, `--max-turns`, `--system-prompt`, `--device`.
Cờ v3 (Phase 2): `--quant int8|int4`, `--static-kv`, `--int8-kv`.

---

## 14. Web App — `app.py`

Flask app phục vụ UI chat + API. Chạy:

```bash
python app.py
# mở http://localhost:5000
```

### API endpoints

| Endpoint | Method | Chức năng |
|---|---|---|
| `/` | GET | Trang chat UI |
| `/api/status` | GET | Trạng thái model (params, vocab, context window, checkpoint step, device, scale…) |
| `/api/chat` | POST | Chat hoàn chỉnh (JSON đồng bộ), trả `response`, `session_id`, `sources` |
| `/api/chat/stream` | POST | SSE streaming — sự kiện: `session`, `sources`, `data` (delta từng token), `done` |
| `/api/search` | POST | Chỉ tìm kiếm web, trả context + sources |
| `/api/reset` | POST | Xoá bộ nhớ session |
| `/api/kv-cache` | POST | Xoá toàn bộ KV prefix cache (giải phóng VRAM / ép prefill lại) |
| `/api/memory/search` | POST | Tra cứu top-k mẩu kiến thức (RAG) — body `query`, `k` |
| `/api/memory/add` | POST | Thêm mẩu kiến thức — body `text`, `claim`, `source_url`, `license`, `confidence`, `tier` |
| `/api/memory/stats` | GET | Thống kê bộ nhớ (số mẩu theo tier, disputed, index mode) |
| `/api/memory/forget` | POST | Xoá theo `id` hoặc `source_url` (quyền riêng tư) |
| `/api/memory/consolidate` | POST | Chạy 1 vòng gộp/decay/phát hiện mâu thuẫn |
| `/api/bench/history` | GET | Lịch sử benchmark cho dashboard (admin) |
| `/api/learn/status` | GET | Growth dashboard: frontier, ledger, memory, registry, budget (admin) |
| `/api/learn/trigger` | POST | Chạy 1 vòng tò mò nền — body `topic`, `dry_run` (admin) |
| `/api/checkpoints` | GET | Registry + danh sách checkpoint trên đĩa (admin) |
| `/api/checkpoints/promote` | POST | Đăng ký checkpoint đã qua gate — body `path`, `changelog` (admin) |
| `/api/checkpoints/rollback` | POST | Rollback về checkpoint cũ — body `path`, `reason` (admin) |

**Admin API (Phase 7)**: các endpoint `/api/bench/history`, `/api/learn/*`, `/api/checkpoints`,
`/api/model/reload` theo chính sách *máy cá nhân trước*:

- Không đặt `JINGPT_TOKEN` **và** server bind loopback (`127.0.0.1`) **và** client là
  loopback → admin **mở**: model bar đổi checkpoint được ngay, không cần mật khẩu.
  Đây là trường hợp thường ngày (`python app.py`).
- Đặt `JINGPT_TOKEN` → phải có `Authorization: Bearer <token>` (so sánh hằng thời
  gian) và chỉ loopback, trừ khi `JINGPT_ALLOW_REMOTE_ADMIN=1`.
- Bind ra LAN (`JINGPT_HOST=0.0.0.0`) mà không có token → admin **đóng** (403).

Token dùng chung cho cả chat/`reset`/memory: UI tự gắn `Authorization` cho mọi
API khi bạn nhập token ở **Model Lab** (lưu localStorage). Token thống nhất cho
máy này là `Zerosd Logiol` — chỉ cần khi bạn muốn bật token mode hoặc mở LAN:
`$env:JINGPT_TOKEN="Zerosd Logiol"; python app.py`.

Đây là "bảng điều khiển sinh trưởng" của Jin: xem frontier, lịch sử val loss/probe/tok/s,
quản lý checkpoint — mở bằng nút **Growth** (📈) trên sidebar của UI.

Body request chat gồm: `message`, `session_id` (tùy chọn), `system_prompt`, `max_tokens`, `temperature`, `top_k`, `top_p`, `repetition_penalty`, `no_repeat_ngram_size` (0 = tắt), `stop_on_repeat` (bool), `web_search` (`auto`/`on`/`off`), `search_max_results`, `search_fetch_top`, `memory` (`auto`/`on`/`off`).

Response có thêm số đo: `stop_reason` (`eos` | `repeat` | `length`) + `stop_label`, `elapsed_s`, `tok_per_s`. SSE phát thêm event `done` chứa `reply` (bản cuối đã cắt echo/role tag), `tok_per_s`, `first_token_ms`, `elapsed_s`.

### Tính năng

- **Streaming theo token** — dùng `model.stream_generate()` + KV cache; giải mã UTF-8 theo từng chunk (chờ đủ byte cho ký tự đa byte).
- **Prefix cache (v3)**: KV của lượt trước được tái dùng giữa các lượt trong cùng session — lượt 2+ chỉ prefill phần mới (TTFT giảm đáng kể). LRU budget 16 session; tắt bằng `--no-prefix-cache`.
- **Cờ khởi động v3**: `--quant int8|int4`, `--static-kv`, `--int8-kv`, `--no-prefix-cache`, `--no-memory`, `--ctx N` (mở rộng context bằng RoPE scaling lúc load — xem `config.context_override`).
- **Hot-swap checkpoint**: `POST /api/model/reload` (admin: token + loopback) đổi checkpoint đang phục vụ mà không restart; KV prefix cache của mọi session bị xoá vì thuộc weight cũ. UI dùng nó cho picker A/B trong panel **Model Lab** và cho **model bar** ở topbar (bấm chip model → danh sách checkpoint trên đĩa, đổi tại chỗ).
- **Web search tích hợp** — tự quyết định có tìm không (heuristic `looks_like_search_query`: câu có `?` hoặc chứa từ khóa who/what/latest/news/price…); kết quả được chèn thành "context block" vào prompt để model trả lời có căn cứ + liệt kê nguồn `[1]`, `[2]…`.
- **Bộ nhớ dài hạn (v4)** — RAG từ chính Jin: câu hỏi được embed bằng hidden state của model (mean-pool `ln_f`), tra cứu hybrid BM25 + vector (RRF) rồi rerank bằng logprob của Jin; câu trả lời luôn kèm citation trỏ về `source_url` đã lưu. Tắt bằng `--no-memory` (giữ đúng hành vi v2.5).
- **Session server-side** — tối đa 256 session, 16 lượt/session (trim lịch sử), GC nền xoá session stale sau 30 phút.
- **Multithread an toàn** — `_model_lock` bao quanh generation (không thread-safe), token bucket theo IP có lock riêng.

### Bảo mật (được thiết kế "không thể bị hack dễ dàng")

- Mặc định bind `127.0.0.1`; chỉ mở LAN khi chủ động set `JINGPT_HOST=0.0.0.0`.
- Token-bucket rate limit: 4 rps, burst 12, theo IP.
- Giới hạn body JSON 32KB, message ≤ 4000 ký tự.
- `JINGPT_TOKEN` → yêu cầu header `Authorization: Bearer <token>` (so sánh hằng thời gian `secrets.compare_digest`).
- Header an toàn trên mọi response: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, CSP hạn chế (chỉ same-origin + Google Fonts cụ thể).
- Trả lỗi chuẩn: 400 (message rỗng), 401 (sai token), 413 (body quá lớn), 429 (rate limit), 503 (chưa có model), 404.

### UI (static/)

- **Layout tối giản** (kiểu ChatGPT/Gemini): sidebar chỉ còn danh sách hội thoại + trạng thái model (ẩn/hiện được, nhớ qua localStorage), topbar hiện chip `checkpoint · step · device`, vùng chat canh giữa 760px, **assistant trả lời dạng chữ trần** (không bubble) còn tin của bạn là khối bo tròn — composer hình viên thuốc 22px với nút gửi tròn bên trong. Font hệ thống (bỏ webfont), chỉ còn border 1px mờ, không glow/gradient trang trí.
- **Sidebar kéo giãn tự do**: kéo vạch ngăn cách (200–460px), nháy đúp để về 260px, hoặc dùng slider *Bề rộng thanh bên*; nút `‹` ẩn/hiện sidebar. Bề rộng + trạng thái ẩn nhớ qua localStorage.
- **Cài đặt trong sheet trượt phải** (nút *Cài đặt* ở đáy sidebar, hoặc Esc để đóng), **6 mục gấp/mở được** — mặc định chỉ mở *Sampling* để sheet gọn: Sampling (**temperature, top-k, top-p, repetition penalty, max tokens** + preset **Chuẩn xác/Cân bằng/Sáng tạo**), **Chống lặp** (chặn lặp n-gram + tự dừng), **Hành vi** (stream, web search, bộ nhớ dài hạn, system prompt), **Model Lab**, **Bộ nhớ**, **Giao diện** (ngôn ngữ, bề rộng sidebar, về mặc định).
- **Guide tại chỗ**: gần như mọi nhãn/nút có **huy hiệu ⓘ** — lia chuột (hoặc focus bằng bàn phím) là hiện bong bóng giải thích tính năng đó làm gì; bong bóng do JS đặt vị trí nên luôn nằm trong màn hình, không bị sheet/sidebar cắt.
- **Đổi ngôn ngữ VI/EN** ngay trong *Giao diện* (nhớ qua localStorage): toàn bộ nhãn, tooltip, gợi ý preset, thông báo lỗi và cả tiêu đề hội thoại mới đều dịch theo.
- **Theme đỏ**: nền đen ám đỏ (`#0c0809` / surface `#1a1114`), viền ngả hồng đỏ, vạch gradient đỏ→cam dưới topbar, tin của bạn là khối đỏ mờ bo góc, nút gửi gradient đỏ→cam; chữ trắng và cam `#ff8a3d` cho cảnh báo/trạng thái online. Slider có track tối + phần đã kéo màu đỏ (không dùng look mặc định của trình duyệt). `color-scheme: dark` + `option` được tô tay nên **popup `<select>` cũng nền tối chữ sáng**, không còn cảnh nền trắng mà chữ cũng trắng.
- **Mỗi câu trả lời có dải số đo** (tok/s, độ trễ token đầu, số token, step của checkpoint) và cảnh báo vàng `⚠ model bắt đầu lặp — đã tự cắt` khi `stop_reason=repeat`.
- **Hội thoại nhiều phiên** lưu `localStorage` (tiêu đề = câu hỏi đầu, chuyển/xoá được), xuất `.md` / `.json`; refresh trang vẫn còn nguyên chat.
- **Markdown**: heading, list, blockquote, link, code block có nút copy; nút **Copy / Sinh lại / Sửa / Xoá** trên từng tin nhắn (sửa = cắt từ đó rồi gửi lại, server session được reset cho khớp).
- **Model Lab**: checkpoint đang chạy, step, val loss, thiết bị, VRAM, uptime, số request, sparkline held-out loss từ `/api/bench/history`, và picker đổi checkpoint (chỉ cần `JINGPT_TOKEN` khi server bật token mode).
- **Model bar (topbar)**: chip `file · step · device` bấm vào là xổ danh sách **mọi** checkpoint trên đĩa (quét đệ quy, mới nhất xếp trên, kèm thư mục · role · step · val · dung lượng, dấu `★ tự chọn` cho bản sẽ được nạp khi restart); bấm một dòng là hot-swap ngay, không phải mở Cài đặt. Chạy local mặc định đổi được luôn; chỉ khi server bật `JINGPT_TOKEN` (hoặc mở ra LAN) menu mới đòi token và chỉ đường vào Model Lab.
- **Bộ nhớ**: số entry (total/semantic/disputed/indexed) + ô tra cứu (`/api/memory/search`) hiện điểm và nguồn.
- **Growth Dashboard (Phase 7)** — nút 📈 ở sidebar: thẻ số liệu (memory entries, learning cycles, budget, cycle running), biểu đồ SVG val loss / probe pass rate / decode tok/s theo thời gian (từ `/api/bench/history`), danh sách **knowledge frontier** sống, registry checkpoint (golden bất khả xâm phạm + N gần nhất). Ô `JINGPT_TOKEN` lưu vào localStorage; không có token thì dashboard báo rõ "Admin API is OFF" thay vì im lặng.
- Welcome screen với 4 prompt gợi ý, nút **New chat**, nút **Stop** khi đang sinh, hiển thị model info (params, context window) từ `/api/status`.

---

## 15. Tìm kiếm web — `web_search.py`

Cho Jin khả năng trả lời về những thứ ngoài dữ liệu huấn luyện — **chỉ dùng stdlib** (urllib + html.parser), không API key.

- **Search**: endpoint HTML của DuckDuckGo, regex bóc `result__a` / `result__snippet`, unwrap link `//duckduckgo.com/l/?uddg=...`.
- **Fetch trang**: `urllib` + HTMLParser tùy biến (bỏ script/style/nav, giữ block tags), cap 2MB/trang, tự giải nén gzip, đoán charset.
- **An toàn (SSRF guard)**: chỉ nhận `http/https`; từ chối IP loopback/private/link-local (kiểm tra cả khi host là hostname — resolve qua getaddrinfo).
- **Polite**: User-Agent đầy đủ, timeout 8s, rate limit toàn cục 2 req/s.
- `looks_like_search_query(message)` — heuristic quyết định câu hỏi có cần tìm web không.
- `web_context(query)` — trả `(context_block, sources)`; block chứa title/URL/snippet (+ excerpt của 1 trang top nếu `fetch_top>0`) để model đọc và trả lời có căn cứ.

CLI test: `python web_search.py "latest GPT-5 release" --results 4 --fetch-top 1`

---

## 15b. Bộ nhớ dài hạn — `memory.py` + `retriever.py` (Phase 4)

Cho Jin **nhớ vĩnh viễn mà không cần train lại** — RAG thuần numpy/PyTorch, không faiss:

- **Embedding từ chính Jin** — mean-pool hidden state của `ln_f` (forward hook, không cần model ngoài), L2-normalize, lưu fp16.
- **`VectorIndex`** — brute-force cosine (đủ nhanh ~100k vector) tự chuyển sang **IVF-flat** (kmeans numpy, nlist≈√N, nprobe) khi vượt ngưỡng.
- **`BM25Index`** — keyword retrieval từ đầu (k1=1.5, b=0.75, IDF).
- **`HybridRetriever`** — hợp nhất BM25 + vector bằng **Reciprocal Rank Fusion**, rerank top-20 bằng logprob của chính Jin (`P(đoạn trả lời câu hỏi)`) → top-5.
- **3 tầng bộ nhớ**: `episodic` (hội thoại, TTL 30 ngày), `semantic` (kiến thức đã kiểm chứng — có `source_url`, `license`, `fetched_at`, `confidence`, `verified_by`), `procedural` (cách làm/skill).
- **Consolidation & decay** — gộp mẩu trùng (content-addressed id + cosine > 0.95), hạ confidence theo tuổi với kiến thức dễ lỗi thời, **phát hiện mâu thuẫn** (2 mẩu cùng chủ đề khác claim) → đánh dấu `needs_verification` cho Curiosity Engine (Phase 5).
- **RAG có citation** — khi chat, câu hỏi được tra cứu memory trước (cờ `memory: auto/on/off`); context chèn `[1] [2]…`, mọi nguồn trả về kèm `source_url`.

```bash
# Thêm/tra cứu thủ công qua API (khi app đang chạy):
curl -X POST http://127.0.0.1:5000/api/memory/add -H "Content-Type: application/json" \
  -d '{"text": "Paris is the capital of France", "source_url": "https://wiki.example/paris", "confidence": 0.95}'
curl -X POST http://127.0.0.1:5000/api/memory/search -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?", "k": 5}'
# Tắt hoàn toàn (hành vi v2.5):
python app.py --no-memory
```

---

## 15c. CURIOSITY ENGINE — tự học liên tục (Phase 5)

Trái tim của v3: Jin tự phát hiện mình chưa biết gì, tự tìm nguồn, tự kiểm chứng,
tự sinh dữ liệu huấn luyện, học có replay, và chỉ ghi nhớ khi vượt cổng kiểm định.

```
┌─ 1. Gap detection ─────────────────────────────────────────────┐
│  score = uncertainty × importance × novelty × learnability     │
│  (entropy + self-consistency k=5 + entity novelty vs corpus)   │
└──────────────┬─────────────────────────────────────────────────┘
               ▼
┌─ 2. Query + quiz (quiz viết TRƯỚC khi học) ────────────────────┐
└──────────────┬─────────────────────────────────────────────────┘
               ▼
┌─ 3. Harvest: DuckDuckGo + Wikipedia API + arXiv API ───────────┐
│  robots.txt + per-domain rate limit + allowlist + SSRF guard   │
└──────────────┬─────────────────────────────────────────────────┘
               ▼
┌─ 4. Kiểm chứng chéo (lá chắn chống học rác) ───────────────────┐
│  ≥2 domain độc lập → high · 1 nguồn → medium · mâu thuẫn →     │
│  disputed (không vào weight) · không verify → memory only       │
└──────────────┬─────────────────────────────────────────────────┘
               ▼
┌─ 5. SEAL self-edit: 5-20 mẫu/claim (assertion/qa/dialogue/summary) ┐
└──────────────┬──────────────────────────────────────────────────────┘
               ▼
┌─ 6. Continual learning: re-warm + re-decay + replay (5%→50%) ──────┐
│  LoRA rank-16 (merge chỉ khi qua gate; rollback 1 giây)             │
└──────────────┬──────────────────────────────────────────────────────┘
               ▼
┌─ 7. EVAL GATE — không qua thì không nhớ ───────────────────────────┐
│  quiz mới ≥80% · forgetting ≤1% · val loss ≤+1% · fluency không    │
│  tụt → promote registry.json · fail → rollback + giảm scope        │
└──────────────┬──────────────────────────────────────────────────────┘
               ▼
┌─ 8. Ledger append-only + UCB bandit (topic_kind × self_edit_form) ─┐
└─────────────────────────────────────────────────────────────────────┘
```

**An toàn**: budget request/GB/cycle, dừng sau 3 vòng fail gate liên tiếp,
`--dry-run` in kế hoạch mà không train. Không bao giờ tự ghi đè checkpoint
"golden" (registry.json giữ 3 checkpoint gần nhất + 1 golden bất khả xâm phạm).

```bash
python learn_daemon.py --cycles 5 --topics data/curiosity/topics.txt   # chạy thật
python learn_daemon.py --cycles 2 --dry-run --topic "quantum computing"  # lên kế hoạch
python pipeline.py evolve --cycles 3 --dry-run                          # bench → cycles → report
```

---

## 15d. AGENTIC — biết dùng công cụ, biết tự kiểm tra (Phase 6)

Jin có thể gọi công cụ giữa câu trả lời theo vòng **ReAct**: sinh
`<|tool|>{"name": ..., "args": {...}}<|end|>` → runtime thực thi → chèn kết quả
→ tiếp tục sinh. Giới hạn `max_tool_calls`/câu trả lời + timeout mỗi lần gọi;
mọi bước tool đều phát event qua `stream_cb` để UI hiện "Jin đang tra cứu…".

**`tools.py` — registry tool với JSON schema** (tất cả chỉ stdlib):

- `calculator` — parser **recursive-descent tự viết, không dùng `eval`**
  (`2+3*4`, `sqrt(16)`, `2**3**2`, `pi`, hàm lượng giác…; lỗi cú pháp/zero-division
  trả về chuỗi lỗi, không crash).
- `python_exec` — sandbox **subprocess riêng** (`python -I`, import whitelist,
  chặn mạng qua `socket`, chặn ghi file, timeout cứng + giới hạn output).
- `web_search` / `fetch_page` — bọc `web_search.py` (đã có SSRF guard +
  robots.txt + rate limit theo domain).
- `memory_search` / `memory_write` — đọc/ghi bộ nhớ dài hạn (Phase 4) kèm
  `source_url` + `confidence` để trích dẫn.
- `now` — thời gian hiện tại (cho câu hỏi "hôm nay…").

**`agent.py` — vòng ReAct + self-check**: model sinh tool call → thực thi →
lặp tới khi có câu trả lời cuối. Trước khi chốt, verifier tự kiểm tra theo
`--rigor`:

- `fast` — không kiểm tra.
- `normal` — **tính lại bằng calculator** các biểu thức số trong câu hỏi, đối
  chiếu với con số trong câu trả lời.
- `high` — thêm **bắt buộc citation** `[n]`/URL cho câu hỏi factual + **resample
  lần 2** và chấm self-consistency (Jaccard) — không cần model chấm ngoài.

Chạy thủ công:
```bash
python agent.py "What is the square root of 144?" --rigor high
python agent.py "Current president of France?" --no-search   # tắt web theo chính sách
```

Acceptance Phase 6 (chạy khi có GPU/checkpoint thật): ≥85% câu hỏi số học
nhiều bước đúng khi bật calculator; câu hỏi tin mới luôn có citation;
agent không bao giờ vượt `max_tool_calls`; sandbox chặn được test thoát
(`python -m unittest tests.test_tool_sandbox`).

---

## 16. Kiểm tra độ trôi chảy — `fluency_check.py`

Chẩn đoán nhanh "model đã sẵn sàng chat chưa". Chạy 8 prompt cố định (`Hi!`, `Who are you?`, `What is 2+2?`, `Write hello world in Python`…) qua chat template, chấm điểm bằng 3 heuristic thô:

1. **Real-word ratio** — tỉ lệ token giống từ thật (dài 1-20, có nguyên âm + phụ âm, không có chuỗi 3 ký tự lặp `thalllll`, từ 1-2 chữ phải nằm trong danh sách từ thông dụng).
2. **Sentence rhythm** — có dấu câu `.`/`!`/`?` theo sau bởi khoảng trắng hay không (chống ramble).
3. **Echo / collapse** — có lặp 1-2 token liên tục không (dấu hiệu model chưa đủ train hoặc over-decay).

Điểm tổng = `0.5·real_ratio + 0.25·length_score + 0.15·sentence + 0.10·echo_score`.

```bash
python fluency_check.py
python fluency_check.py --checkpoint checkpoints/jingpt_chat.pt
```

**Verdict tự động:**

| Điểm TB | Verdict | Ý nghĩa |
|---|---|---|
| ≥ 0.80 | `READY` | Sẵn sàng chat bình thường |
| ≥ 0.60 | `OK` | Tạm ổn, cần thêm instruct steps |
| ≥ 0.40 | `TRAINING` | Đang hình thành, chạy thêm `train.py --mode instruct` |
| < 0.40 | `BROKEN` | Token soup — rebuild data + train lại vài nghìn step |

---

## 17. Tiện ích — `utils.py`

- `make_tokenizer` / `tokenizer_from_state` — tạo/khôi phục tokenizer (BPE hoặc char) từ state dict trong checkpoint.
- `TextDataset` — dataset pretrain đơn giản (x, y kề nhau).
- `load_or_create_token_data` — encode + cache token tensor; cache miss khi data hoặc tokenizer thay đổi (fingerprint SHA-256).
- `split_train_val` — cắt train/val với ràng buộc block size.
- `get_batch` — lấy batch ngẫu nhiên (x, y) từ tensor.
- `load_checkpoint_file` — load checkpoint với fallback chuỗi `best → latest → legacy` khi file gốc không tồn tại (`allow_fallback=True`).
- `extract_model_config` / `create_model` — dựng model từ checkpoint (tự nhận diện config legacy nếu file là `jingpt.pt`).
- `save_checkpoint` — gói format v2: model config, state dict, optimizer, scaler, RNG state, tokenizer state, losses.
- `load_checkpoint` — khôi phục đầy đủ (optimizer/scaler/RNG/CUDA RNG).
- `build_inference_bundle` — 1 hàm chuẩn hoá để mọi script inference (chat/generate/fluency/app) load `(model, tokenizer, meta)`.
- `configure_cuda_performance` — bật TF32, cudnn benchmark, matmul khởi động nhỏ để ổn định cuBLAS trên Windows + GPU VRAM thấp.

---

## 18. Biến môi trường

| Biến | Giá trị | Mặc định | Mô tả |
|---|---|---|---|
| `JINGPT_SCALE` | `tiny` / `small` / `medium` / `large` | `small` | Chọn preset kiến trúc model |
| `JINGPT_DEVICE` | `auto` / `gpu` / `cuda` / `mps` / `cpu` | `auto` | Chọn device (có thể ghi đè bằng `--device` từng lệnh) |
| `JINGPT_HOST` | IP | `127.0.0.1` | Bind address của web app (chỉ mở khi thật sự cần) |
| `JINGPT_PORT` | số | `5000` | Port web app |
| `JINGPT_TOKEN` | chuỗi | *(trống)* | Bật Bearer token cho web API (máy này: `Zerosd Logiol`) |

Ví dụ chạy model medium:
```bash
set JINGPT_SCALE=medium && python app.py        # cmd
$env:JINGPT_SCALE='medium'; python app.py        # PowerShell
```

---

## 19. Quy trình huấn luyện từ A→Z

### Lộ trình khuyến nghị

```bash
# ① Tải raw text công khai (Gutenberg + Wikipedia abstracts)
py fetch_datasets.py --catalog sources.json

# ② (Tùy chọn) Bổ sung text local vào data/raw/ rồi build corpus pretrain
py build_dataset.py --report

# ③ Tải thêm dataset chat (Dolly, Alpaca, GSM8K, CodeAlpaca, OASST2, Bactrian-X…)
py fetch_datasets.py --catalog hf_datasets.json

# ④ Xây corpus chat
py build_instruct_dataset.py --synth-samples 4000 --no-literary

# ⑤ Train BPE trên corpus chat (lần đầu)
py bpe_tokenizer.py --data data/chat_train.txt --out bpe.json --vocab-size 16384

# ⑥ (Tùy chọn) Pretrain trước để có nền văn xuôi
py train.py --resume

# ⑦ Instruct tuning — GPU 6GB thì thêm --fast
py train.py --mode instruct --steps 5000 --device cuda --batch-size 1 --grad-accum 32

# ⑧ Kiểm tra chất lượng
py fluency_check.py

# ⑨ Dùng thử
py chat.py          # terminal
py app.py           # web UI → http://localhost:5000
```

### Hoặc tất cả bằng 1 lệnh

```bash
py pipeline.py chat
py pipeline.py raw --train --resume
```

---

## 20. Câu hỏi thường gặp & khắc phục lỗi

**"Web app báo chưa có model"** → train trước để tạo checkpoint: `python train.py`, rồi chạy lại `python app.py`.

**"CUDA out of memory"** → dùng batch nhỏ nhất: `--batch-size 1 --grad-accum 32`; thêm `--fast` để giảm block-size xuống 256. Hoặc giảm `JINGPT_SCALE=tiny`.

**"Train trên CPU quá chậm"** → chọn preset nhỏ: `JINGPT_SCALE=tiny`; giảm `--steps`; đừng bật gradient checkpointing trên CPU (đã tắt mặc định).

**"Muốn dùng `--device cuda` nhưng lỗi 'torch.cuda.is_available() is False'`** → cài driver NVIDIA + bản PyTorch có CUDA (xem mục 2). Dùng `--device gpu` nếu muốn tự fallback về CPU kèm cảnh báo.

**"Model sinh toàn token soup / lặp vô nghĩa"** → chạy `python fluency_check.py` để chẩn đoán; nếu `BROKEN`: rebuild data (`build_instruct_dataset.py`) và train thêm vài nghìn step instruct.

**"Switch từ char sang BPE checkpoint cũ không load được"** → bình thường: vocab không khớp nên pretrained weights bị bỏ qua (có log rõ `SKIPPING pretrained weights`). Train mới hoàn toàn.

**"Pipeline raw báo 'No enabled sources matched'"** → kiểm tra `sources.json`: có `enabled: true` không, hoặc `--only` có trùng `id` không.

**"BPE tốn quá nhiều thời gian"** → giảm `--vocab-size` (vd 8192) hoặc tăng `--min-pair-freq`; thuật toán incremental đã nhanh hơn naive rất nhiều.

**"Rate limit 429 khi gọi API"** → mặc định 4 req/s là đủ cho người dùng thật; nếu test tự động, chỉnh `RATE_LIMIT_RPS` trong `config.py` hoặc dùng `sleep` giữa request.

---

## 21. Ý tưởng nâng cấp

- **Scale lên `medium` (360M)** với corpus lớn hơn (bật thêm Cosmopedia / Wikipedia parquet, cần `pyarrow`).
- **SFT trên dữ liệu tiếng Việt nhiều hơn** — bật `bactrian-x-vi` (67k mẫu tiếng Việt) và `oasst2` (đã có tiếng Việt) trong `hf_datasets.json`.
- **Tinh chỉnh siêu tham số** — thử LR 1e-4~5e-4, warmup dài hơn, `GRADIENT_CHECKPOINTING=True` trên GPU để tăng block-size.
- **KV cache streaming đã có sẵn** — có thể mở rộng thêm nhiều lượt đối thoại dài hơn bằng cách tăng `MAX_TURNS_PER_SESSION`.
- **Thêm bộ chuyển đổi dataset mới** trong `fetch_datasets.py` cho các định dạng JSON/JSONL khác (đã có sẵn `converters` dict — chỉ cần thêm 1 hàm iterator).
- **Đánh giá có hệ thống** — mở rộng `fluency_check.py` thành benchmark nhiều probe hơn, hoặc so sánh val loss giữa các lần chạy.

## 22. Benchmark & đánh giá — `bench.py`

Một lệnh cho mọi số liệu của Jin (Phase 0 của v3.0). Không cần thư viện ngoài — tự chấm bằng PyTorch + numpy + stdlib.

```bash
python bench.py --checkpoint checkpoints/jingpt_chat.pt   # đo đầy đủ 5 nhóm chỉ số
python bench.py --compare A.json B.json                   # diff 2 lần chạy + forgetting_score
python bench.py --init-eval                               # trích data/eval/heldout.txt khỏi train.txt (chạy 1 lần)
python bench.py --train-bench --checkpoint X --train-out t.json   # riêng train tok/s (subprocess cách ly)
python bench.py --merge-train bench.json t.json           # gộp kết quả train vào bản full
```

Đo được:

| Nhóm | Chỉ số | Nguồn |
|---|---|---|
| **Chất lượng** | val loss + perplexity trên `data/eval/heldout.txt` (held-out cố định, khoá, không bao giờ train lên) | `bench.py` |
| **Kiến thức & năng lực** | 165 probe tự soạn, 8 domain (`data/eval/probes.jsonl`); chấm exact / numeric / keyword + logprob đáp án đúng-vs-sai (không cần model chấm ngoài) | `bench.py` |
| **Trôi chảy** | tái dùng đúng heuristic `fluency_check.py` để so sánh lịch sử | `bench.py` |
| **Tốc độ** | prefill tok/s, decode tok/s, TTFT, peak VRAM; train tok/s theo device+preset | `bench.py` |
| **Trí nhớ** | `forgetting_score` khi `--compare` checkpoint mới vs cũ trên tập probe cũ | `bench.py` |

Output: `bench_results/<timestamp>_<checkpoint>.json` + bảng markdown in ra terminal. Baseline v2.5 đã lưu tại `bench_results/baseline_v25.json` (chi tiết: `docs/v3_progress.md`).

**Tests (v3):**
```bash
python -m unittest discover -s tests -v
```
Cover: model shapes mọi preset, kv-cache equivalence (greedy có cache == không cache), GQA, MoE total/active + aux loss, sliding window cache bound, RoPE scaling, tokenizer round-trip (tiếng Việt/emoji/code), chat template masking, Muon, quantize int8/int4, prefix-reuse, data quality, DPO/ORPO — chạy trên CPU, không cần GPU.

## 23. Nâng cấp chất lượng dữ liệu — Phase 3 (v3)

Các công cụ tự viết, không dependency ngoài, giúp Jin "thông minh hơn từ dữ liệu":

| Công cụ | Chức năng | Lệnh |
|---|---|---|
| `data_quality.py` | Dedupe **MinHash+LSH** (exact + near-dup), quality filter kiểu Gopher/C4, phát hiện ngôn ngữ vi/en, **contamination check** chống rò rỉ benchmark (`data/eval/probes.jsonl`) | `python data_quality.py --input data/train.txt --output data/train_clean.txt --report out.json` |
| `build_reasoning_data.py` | Sinh dữ liệu **CoT có kiểm chứng**: toán (Fraction/Cramer), logic, code có unit test chạy sandbox — đáp án đúng được tính bằng Python nên không thể sai nhãn | `python build_reasoning_data.py --out data/reasoning.jsonl` |
| `self_distill.py` | **Rejection sampling k=8**: sinh nhiều đáp án, verifier tự động lọc (math numeric/exact, code chạy test, heuristic), giữ đáp án tốt nhất → SFT JSONL | `python self_distill.py --checkpoint checkpoints/jingpt_chat.pt --prompts data/reasoning.jsonl --out data/distill.jsonl` |
| `train_pref.py` | **DPO** (reference đóng băng, `--beta`) và **ORPO** (không cần reference, tiết kiệm VRAM) — tự viết từ đầu | `python train_pref.py --data data/pref.jsonl --method orpo --checkpoint checkpoints/jingpt_chat.pt --steps 200` |
| `train.py --mixture` | Curriculum mixing theo `data/mixture.json` (3 stage, mỗi domain một file, log tỉ lệ thực tế) | `python train.py --mixture data/mixture.json --steps 2000` |

Token mới cho suy luận/tool: `<|think|>`, `<|/think|>`, `<|tool|>` được thêm **ở đuôi vocab** (ID cũ không dịch chuyển) — chỉ có hiệu lực khi rebuild tokenizer (`train.py` tự làm khi train BPE mới). Contamination check chạy mặc định trong `build_dataset.py` / `build_instruct_dataset.py` (tắt bằng `--no-contamination-check`).

> **Ghi chú phần cứng:** trên GPU 6GB WDDM, fp32 AdamW cho preset `small` (151M) không fit bộ nhớ khả dụng (~3.9GB); `bench.py --train-bench` tự fallback sang `tiny` và ghi rõ trong kết quả. Đây là dữ liệu đo thật — động lực trực tiếp cho Phase 2 (optimizer state 8-bit / `--low-vram`).

## 24. Benchmark & kết quả thật (v3)

Mọi con số dưới đây đều từ `bench.py` (không phóng đại). Bảng **before/after** được
cập nhật khi bạn chạy lại trên GPU; baseline v2.5 đã khoá tại `bench_results/baseline_v25.json`:

| Chỉ số | Baseline v2.5 (đo thật) | v3 hiện tại (CPU/tiny) | Mục tiêu v3.0 | Trạng thái |
|---|---|---|---|---|
| Held-out loss (jingpt_chat) | **11.86** | 5.35 (jingpt_best, pretrain) | giảm | ✅ đo được |
| Probe pass rate | **10.9%** | 0.6% (best — chưa SFT) | ≥ +15% vs baseline | ⏳ chờ GPU |
| Fluency | **0.768** | 0.80 (best) | ≥ 0.85 | ⏳ chờ GPU |
| Decode tok/s (CPU, small) | **58.3** | 55.9 (best) | ≥ 2.5× (static+int8 KV) | ⏳ chờ GPU |
| TTFT lượt 2+ (prefix cache) | — | — | giảm ≥ 40% | ⏳ chờ GPU |
| Train time-to-val-loss | — | — | ≥ 1.3× (Muon) | ⏳ chờ GPU |

Cách đo lại đầy đủ (mỗi lần thay checkpoint quan trọng):
```bash
python bench.py --checkpoint checkpoints/jingpt_chat.pt   # toàn bộ 5 nhóm
python bench.py --compare bench_results/baseline_v25.json bench_results/<mới>.json
```
Mọi lần chạy đều được nối vào `bench_results/history.json` — Growth Dashboard vẽ
val loss / probe / tok/s theo thời gian từ chính file này.

## 25. An toàn & đạo đức crawl (v3)

Jin tự học từ internet nên mọi đường thu hoạch đều có phanh — thiết kế theo
nguyên tắc "tò mò vô hạn đi kèm trí nhớ có kiểm soát":

| Lớp | Cơ chế | Nơi |
|---|---|---|
| **Tôn trọng robots.txt** | `urllib.robotparser`, cache theo domain, fail-open (không robots → cho phép đọc, vẫn chặn disallow-all) | `web_search.py` |
| **Tần suất** | ≤ 1 req/2s/domain + 2 req/s toàn cục, User-Agent khai rõ bot | `web_search.py` |
| **SSRF guard** | chỉ http/https public, chặn loopback/private/link-local (IP literal + resolve hostname) | `web_search.py` |
| **Allowlist/denylist domain** | `CURIOSITY_DOMAIN_ALLOWLIST` / `DENYLIST` trong `config.py` (wiki/edu/gov/doc ưu tiên) | `config.py` |
| **Provenance bắt buộc** | mọi mẩu kiến thức lưu `source_url` + `license` + `fetched_at`; claim phải có ≥2 domain độc lập mới đạt confidence=high | `memory.py`, `curiosity_engine.py` |
| **Chống học rác** | nguồn không verify được → chỉ vào memory dạng `unverified`, **tuyệt đối không vào weight**; nguồn mâu thuẫn → `disputed` + `needs_verification` | `curiosity_engine.py` |
| **Budget & phanh** | `JINGPT_CURIOSITY_MAX_REQUESTS`, `MAX_GB_DAY`, `MAX_CYCLES`; dừng sau 3 vòng fail gate liên tiếp; `--dry-run` in kế hoạch không train | `learn_daemon.py` |
| **Eval gate trước khi nhớ** | quiz chủ đề mới ≥80%, forgetting ≤1%, val loss ≤+1%, fluency không tụt >0.02 — fail → rollback + giảm scope; registry giữ 3 gần nhất + golden bất khả xâm phạm | `curiosity_engine.py` |
| **Quyền riêng tư** | `/api/memory/forget` xoá theo id/nguồn; admin API mặc định tắt + loopback-only | `app.py` |
| **Không crawl dữ liệu cá nhân** | chỉ nguồn công khai, không paywall/login; không tự ý thu hoạch dữ liệu người dùng | policy |

Vòng tự học đầy đủ (8 bước, sơ đồ ASCII ở mục 15c): gap detection → query + quiz
(viết trước khi học) → harvest → **kiểm chứng chéo ≥2 nguồn** → SEAL self-edit →
continual learning có replay (re-warm + re-decay, 5% weak shift / 25–50% strong) →
eval gate → ledger + UCB meta-learning.

---

*Generated with Codebuff 🤖 — JIN AI: build, train, and chat with your own small language model.*