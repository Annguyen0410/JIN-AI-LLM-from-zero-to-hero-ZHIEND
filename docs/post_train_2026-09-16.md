# Hậu-train 16/09/2026 — SFT base 46k → step 4,000

> Đợt SFT tại nhà (RTX 2060 6GB) nối từ base pretrain **46k** của Kaggle.
> Mọi số dưới đây là số đo thật, log gốc ở `train_log.txt`, bench ở `bench_results/`.
> Mục đích của file: lần sau biết đã đạt tới đâu và cái gì còn nợ.

---

## 1. Đợt train

| Mục | Giá trị |
|---|---|
| Base warm-start | `D:\jin-ckpt\jingpt_latest.pt` (46k) |
| Dữ liệu | 67,838 hội thoại (746 seed ×8 + 59,248 HF + 2,622 grounded), 13 file |
| Packing | 89,249 block × 256 token; trainable tokens **12,763,154 / 22,936,993 (55.6%)** |
| Cấu hình | `--mode instruct --fast --batch-size 1 --grad-accum 8`, block 256, dup 8 |
| Bước | 0 → 4,000, eval mỗi 200, checkpoint mỗi 500 |
| Thời gian | ~80 phút (0.8 it/s) |
| **Val cuối** | **2.9957** (best) tại step 4,000 — đẹp hơn best cũ 3.0815 (step 3,400) |

Val loss theo mốc: 200→4.7608 · 1000→3.9890 · 2000→3.5874 · 3000→3.1594 · 3400→3.0815 · 3800→3.0361 · **4000→2.9957**

Checkpoint ghi ra: `iteration = 4000`, `val_loss = 2.9957`, `best_loss = 2.9957`
(đã verify bằng cách load trực tiếp metadata, không tin log).

> ⚠️ Cơ chế trong `train.py`: save "best" chỉ xảy ra khi `val < best`, **còn** cứ 500 step là ghi đè vô điều kiện.
> Lần này val cuối đẹp nhất nên file = step 4,000. Nhưng nếu một lần nào val cuối tệ hơn, file sẽ giữ bản interval
> (VD 3,500) chứ không phải bản best — luôn load metadata để kiểm tra trước khi khoe thành tích.

## 2. Bản lưu

| File | Nội dung |
|---|---|
| `checkpoints/jingpt_chat.pt` | bản đang dùng (step 4,000) |
| `checkpoints/jingpt_chat_46k_sft4k.pt` | **golden** copy của đợt này (rollback) |
| `checkpoints/jingpt_chat_pre46k.pt` | bản chat trước (đối chứng A/B) |
| `D:\jin-ckpt\jingpt_latest.pt` | base 46k (không đụng vào) |

## 3. Fluency A/B (8 probe cố định)

| | bản 46k+SFT4k | bản trước (`pre46k`) |
|---|---|---|
| Điểm trung bình | **0.79 – OK** | 0.77 – OK |
| "Hi!" | ⚠️ *"Good morning! I'm small, very small. I'm small, even if I was small…"* (echo 21.7%) | ✅ *"Hello! I'm Jin. What's next?"* |
| "Who are you?" | 0.89 | 0.78 |
| "What is 2 + 2?" | ❌ *"2 + 3 = 4."* (0.00) | ❌ *"2 + 2 = 3."* |
| "capital of Japan?" | ❌ *"The capital of France is France."* | ❌ *"The capital of China is Australia."* |
| "Write hello world in Python." | 0.97 (có block code) | 0.79 |

Kết luận: bản mới **tốt hơn về tổng thể** nhưng có thiên hướng "nói về bản thân" khi mở đầu câu chào
(`I'm small, very small…`) — dấu hiệu echo nhẹ ở greeting, chỉ xuất hiện ở 1/8 probe.

## 4. Benchmark (`bench_results/20260916_194140_jingpt_chat.json`)

| Chỉ số | Lần này | 15/09 | Δ |
|---|---|---|---|
| held-out loss | **5.6904** | 5.718 | −0.5% |
| held-out ppl | **296.02** | 304.31 | −2.7% |
| probe pass rate | **0.079** | 0.067 | +17.9% |
| fluency (bench) | 0.76 | 0.768 | −1.0% (nhiễu) |
| decode tok/s | **39.6** | 27.9 | +41.9% |
| TTFT (ms) | **27.8** | 47.3 | −41.2% |
| peak VRAM | 0.643 GB | 0.643 GB | — |

`forgetting_score = 0.0179`. Domain **tụt**: `vietnamese` 0.045 → **0.000**, `instruction_following` 0.20 → 0.15,
`long_context` 0.048 → 0.000. Domain **lên**: `code` 0.111 → 0.222, `math` 0.083 → 0.125, `factual` 0 → 0.045,
`reasoning` 0.05 → 0.10.

Chưa vượt được `baseline_v25` probe (0.109) — nhưng file đó là checkpoint char-level legacy, không cùng hệ, đừng lấy làm mốc.

## 5. Smoke test

| Hạng mục | Kết quả |
|---|---|
| `app.py` + `/api/status` | ✅ checkpoint `jingpt_chat.pt`, **step 4000**, device **cuda** |
| `/api/chat` với web search ON | ✅ `web_search_used: true`, 3 nguồn thật (RTX 5060 tracker $459.99, Newegg…), câu trả lời của model vẫn sai số (kỳ vọng với 148M) |
| `/api/chat/stream` | ✅ 1 event `session` + 10 `data` delta + `done` |
| `generate.py` | ✅ mặc định đúng `jingpt_chat.pt` step 4000 |
| `agent.py --rigor high` | ❌ **không trả lời gì** — chỉ in self-check *"factual/recent claim without a citation"*. Test cả bản `pre46k`: **giống hệt** ⇒ không phải regression, mà là model chưa học format ReAct `<|tool|>{...}` |
| Test suite | ✅ `Ran 159 tests ... OK` |

## 6. Bug đã sửa trong lần này

1. **`config.CHECKPOINT_PATH` bị ghim vào `BEST_CHECKPOINT_PATH`** (checkpoint *pretrain*) trong khi `get_checkpoint_path()`
   đã tồn tại và README ghi rõ "ưu tiên `jingpt_chat.pt`". Hậu quả: `python app.py`, `python chat.py`, `generate.py`, `agent.py`
   đều phục vụ **model pretrain** (thực tế `/api/status` báo step 2000 của `jingpt_best.pt`).
   → Sửa bằng `__getattr__` cấp module (PEP 562), resolve tại thời điểm truy cập.
2. **`set_device(None)` là no-op + mặc định `DEVICE = "cpu"`** ⇒ mọi script không truyền `--device` chạy trên **CPU**
   dù GPU đang rảnh (đúng như comment trong `utils.py` thừa nhận; `bench.py` đã phải tự workaround bằng `or "auto"`).
   → `None` giờ = *auto* (theo đúng tài liệu §4.6), thêm `--device` cho `app.py` và `generate.py`, gọi `configure_cuda_performance`.

Test lại: 159 test OK, `/api/status` báo `device: cuda`, `checkpoint_step: 4000`.

## 7. Phán quyết & việc còn nợ

**Giữ bản này** làm golden (`jingpt_chat_46k_sft4k.pt`) — probes +17.9%, heldout tốt hơn, decode nhanh +42%.

Nợ cần trả:
- **Tiếng Việt về 0%** và `instruction_following`/`long_context` tụt → lần SFT sau tăng tỉ trọng dữ liệu VI + ngữ cảnh dài.
- Echo ở greeting → cân nhắc dừng SFT sớm hơn (~2,000–2,500 step) hoặc giảm `--lr` cuối; đồng thời rà lại dữ liệu self-referential ("a small language model…").
- **Agent mode chưa chạy** → cần dữ liệu ReAct (`<|tool|>{"name": ...}<|end|>`) thì 148M mới có cơ hội học format.
- `checkpoints/` đang 14GB, ổ C: chỉ còn 17GB → giữ tối đa 2–3 bản gần nhất + golden, phần còn lại để trên D:.
