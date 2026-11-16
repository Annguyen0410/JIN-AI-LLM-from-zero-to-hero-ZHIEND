# JIN AI v3.0 — "Infinite Curiosity" — Nhật ký tiến độ

Mọi con số trong file này đều đo thật từ `bench.py`, không ước lượng.
Checkpoint baseline v2.5: `checkpoints/jingpt_chat.pt` (151.49M params, BPE 8.256, ctx 1024, iter 5000).
Môi trường đo: RTX 2060 6GB (Turing sm75, WDDM), torch 2.5.1+cu121, Windows.

---

## PHASE 0 — Đo lường & hạ tầng đánh giá (DONE)

### Đã làm

| Hạng mục | File | Nội dung |
|---|---|---|
| Eval set held-out | `data/eval/heldout.txt` | ~1.1MB text trích từ cuối `data/train.txt` (1.094.816 chars), cắt đúng ranh giới `<doc>`, đã snapshot corpus gốc tại `data/curated/snapshots/train_pre_phase0_20260907_145749.txt` và **xoá khỏi train.txt** → không bao giờ train lên nó. |
| Probe set | `data/eval/probes.jsonl` | **165 câu** tự soạn, 8 domain (factual 22, math 24, code 18, reasoning 20, vietnamese 22, instruction_following 20, safety 18, long_context 21). Mỗi câu có reference answer, distractor sai, cơ chế chấm tự động (exact / numeric / keyword / forbidden). |
| Công cụ đo | `bench.py` | Quality (val loss + ppl held-out), probes (pass rate theo domain + logprob đúng-vs-sai), fluency (tái dùng heuristic `fluency_check.py` nguyên vẹn), speed (prefill tok/s, decode tok/s, TTFT, peak VRAM), train tok/s (subprocess cách ly), `--compare` kèm forgetting_score, `--init-eval`, output `bench_results/<ts>_<ckpt>.json` + bảng markdown. |
| Baseline | `bench_results/baseline_v25.json` | Số liệu v2.5 đầy đủ (bảng bên dưới). |

### Số liệu baseline v2.5 (jingpt_chat.pt — acceptance)

| Metric | Giá trị |
|---|---|
| params | 151.5 M |
| held-out loss | 11.8638 |
| held-out ppl | 142,034 |
| probe pass rate (165 câu) | 10.9% |
| probe mean logprob gap | −1.088 |
| fluency avg score | 0.768 (verdict OK) |
| prefill tok/s | 8,637 |
| decode tok/s | 58.3 |
| TTFT (ms) | 15.8 |
| peak VRAM inference (GB) | 0.649 |
| train tok/s | 963.8 (tiny preset — xem ghi chú) |
| train peak VRAM (GB) | 0.297 |

Probe pass rate theo domain (baseline v2.5): factual 18.2% · math 25.0% · code 11.1% · reasoning 5.0% · vietnamese 9.1% · instruction_following 15.0% · safety 0.0% · long_context 0.0%.
Logprob pass rate (đáp án đúng khả dĩ hơn đáp án sai): factual 68.2% · math 37.5% · code 16.7% · reasoning 10.0% · vietnamese 36.4% · instruction_following 30.0% · safety 83.3% · long_context 61.9%.

### So sánh pretrain → SFT (minh hoạ `--compare`)

`bench.py --compare bench_results/20260907_154554_jingpt_best.json bench_results/baseline_v25.json`
(jingpt_best.pt = pretrain step 1000, jingpt_chat.pt = instruct)

| Metric | best (pretrain) | chat (SFT) | diff |
|---|---|---|---|
| held-out loss | 5.35 | 11.86 | +6.52 (SFT làm mất trôi chảy văn xuôi — dự kiến) |
| probe pass rate | 0.6% | 10.9% | +10.3pp (SFT giúp trả lời câu hỏi) |
| fluency avg | 0.800 | 0.768 | −0.032 |
| decode tok/s | 55.9 | 58.3 | +4.3% |

**forgetting_score = 0.0000** trên tập probe cũ (mọi domain đều không giảm).

### Lỗi/giới hạn phần cứng phát hiện (đo thật, đáng giá cho các phase sau)

1. **WDDM "khoá" bộ nhớ GPU**: RTX 2060 6GB ở chế độ WDDM chỉ dành ~3.1–3.9GB cho compute (desktop chiếm phần còn lại). Hệ quả đo được:
   - **fp32 AdamW 151M KHÔNG fit**: cần model 605MB + grads 605MB + 2 moment 1.2GB + temp foreach ~605MB ≈ 3.3GB > trần. `train.py --mode instruct` chế độ thật trên GPU host này sẽ OOM — **đây là lý do trực tiếp cho Phase 2 (optimizer state 8-bit/fp16) và --low-vram**.
   - Load checkpoint 1.8GB (chứa optimizer state) vào RAM + khởi tạo CUDA liên tiếp có thể làm `cudaMalloc` fail dù báo còn 3–4GB "free" (báo cáo sai) hoặc segfault driver — đã xử lý bằng `torch.load(mmap=True)`.
   - `configure_cuda_performance` (cudnn.benchmark + warmup matmul) dễ segfault khi đi sau load checkpoint lớn → `bench.py` chỉ bật TF32, bỏ phần không ổn định.
2. **Train throughput đo được** trên tiny preset (35M→13.8M với vocab 8.256): 963.8 tok/s, peak 0.297GB. Same-arch 151M không đo được vì lý do (1).

### Bug pre-existing sửa được trong Phase 0 (để giữ đường chạy)

- `train.py --mode instruct` crash ngay lập tức: `train_instruct` dùng `args.seed` nhưng `parse_args` không định nghĩa `--seed` → thêm `--seed` (mặc định 1337). Lệnh acceptance `python train.py --mode instruct --steps 20` đã chạy lại được (smoke 20 steps OK).
- `utils.build_inference_bundle` load cả checkpoint (kèm optimizer state ~1.2GB) lên GPU → OOM trên 6GB; chuyển sang load CPU + mmap, chỉ đưa weight lên device. Lợi cho mọi script inference (app/chat/generate/fluency).
- `utils._torch_load` mmap mặc định (fallback an toàn) → giảm RAM, tránh wedge CUDA.

### Kỹ thuật thử nhưng KHÔNG hiệu quả / không dùng

- Chạy train benchmark ngay trong tiến trình chính: gây segfault/OOM wedge → chuyển sang subprocess cách ly (mỗi attempt một tiến trình sạch).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: không hỗ trợ trên Windows (warning từ PyTorch).

### Rủi ro còn lại / nợ kỹ thuật

- Driver WDDM host này flaky: vài lần chạy CUDA liên tiếp (đặc biệt sau OOM) có thể segfault — nên chạy `bench.py --train-bench` tách riêng nếu full run lỗi; merge bằng `--merge-train`.
- Held-out được trích từ `train.txt` hiện tại; checkpoint cũ có thể đã "thấy" phần text này trong lúc train → con số val loss với checkpoint cũ là ước lượng thận trọng; với model train TỪ BÂY GIỜ trở đi thì held-out hoàn toàn sạch.
- Probe safety chấm bằng keyword refusal — model có thể từ chối theo cách khác (false negative); đã ghi lại, Phase 3 sẽ cải thiện.
- Chưa có long-context probe > 1024 token (ctx hiện tại); Phase 1 sẽ thêm khi có RoPE scaling.

---

## PHASE 1 — Kiến trúc v3: to hơn mà vẫn chạy được (CODE DONE — chờ verify GPU)

> Trạng thái: code + unit test (CPU) xong và pass 22/22. Các bước verify GPU
> (A/B train, xl-moe load+sinh, extend-context) do người dùng tự chạy vì host
> WDDM này crash khi agent lặp lại nhiều vòng CUDA — lệnh cụ thể ở cuối phase.

### Đã làm (thay đổi code)

| File | Nội dung |
|---|---|
| `model.py` | Viết lại có cấu trúc, backward-compat 100%: **GQA** (`n_kv_heads`, qkv_proj vẫn 3·embed khi MHA), **QK-norm** (RMSNorm q/k), **RoPE** `rope_theta` cấu hình được + **NTK-aware / YaRN-lite** scaling (`rope_scaling`), **sliding window** local/global (3:1 qua `local_layer_every`, mask cắt theo window, KV cache trim đúng window + đếm `pos` tuyệt đối), **MoE** (`MoEFeedForward`: 8 expert, top-2, capacity factor, aux loss Switch + z-loss, router không noise ở inference, `expert_offload` CPU prefetch theo router), đếm `num_parameters` (total) vs `active_parameters` (active), `max_positions` cho ctx mở rộng. |
| `config.py` | Preset `medium` 360M (GQA 20H/4KV, grad-checkpointing), `xl-moe` (12L, embed 1280, 20H/4KV, block 4096, sliding 1024 3:1, MoE layer chẵn, offload mặc định → **~0.78B total / ~0.31B active** đo thật), CLI `python config.py --describe [--depth N]` (bảng params/active/VRAM/tokens + auto-scale một núm), `derive_config_from_depth`. |
| `train.py` | `--depth N` (auto-scale, kèm lr/warmup suy ra, qk_norm=True), `--extend-context N --rope-scale-type ntk|yarn --rope-scale-factor F` (pretrain, block dài + RoPE scaling). |
| `utils.py` | `create_model(..., override_config=...)` để train.py chèn config từ --depth/--extend-context. |
| `tests/` | unittest stdlib (CPU): `test_model_shapes.py` (forward+generate mọi preset, kv-cache equivalence, GQA shape, MoE total/active + aux/z loss, sliding window cache bound, RoPE scaling > block, backward-compat v2.5), `test_tokenizer.py` (round-trip vi/emoji/code, special token), `test_chat_template.py` (masking chỉ token assistant). |
| `scripts/demo_xl_moe.py` | Load xl-moe + offload, sinh text, in total/active + peak VRAM. |

### Kết quả đo được (CPU, không phải acceptance cuối)

- Unit test: **22/22 pass** (`python -m unittest discover -s tests`).
- Checkpoint v2.5 (`jingpt_chat.pt`) load + generate nguyên vẹn trên code mới: 151.49M, MHA (16KV), qk_norm=False, ctx 1024 — không migrate cần thiết.
- `config.py --describe`: tiny 0.02B · small 0.15B · medium 0.37B · large 1.10B · **xl-moe 0.78B total / 0.31B active** (VRAM fp32-AdamW ước tính: small 2.4GB, medium 5.8GB, xl-moe 12.6GB).
- Sửa 2 bug phát hiện trong lúc làm: preset medium ban đầu khai `n_kv_heads=8` với 20 heads (không chia hết — vô hiệu GQA) → đổi 4; `active_parameters` đếm gate MoE 2 lần → sửa.

### Việc cần chạy (GPU — người dùng tự verify, theo lệnh bên dưới)

1. **A/B kiến trúc cũ vs mới cùng token** (tiny, fit 6GB):
   ```bash
   python train.py --mode instruct --fast --steps 300 --device cuda --no-pretrained-init --checkpoint checkpoints/_ab_old.pt
   JINGPT_SCALE=tiny python train.py --mode instruct --fast --steps 300 --device cuda --no-pretrained-init --depth 6 --checkpoint checkpoints/_ab_new.pt
   python bench.py --checkpoint checkpoints/_ab_old.pt --no-train-bench
   python bench.py --checkpoint checkpoints/_ab_new.pt --no-train-bench
   python bench.py --compare bench_results/*_ab_old.json bench_results/*_ab_new.json
   ```
   Tiêu chí: probe pass rate của `_ab_new` ≥ `_ab_old` (hoặc val loss ≤) ở cùng 300 step.
2. **xl-moe trên 6GB:** `python scripts/demo_xl_moe.py --device cuda` → in total ~780M / active ~310M, peak VRAM < 5GB.
3. **Extend-context:** `JINGPT_SCALE=tiny python train.py --extend-context 2048 --rope-scale-type ntk --rope-scale-factor 4 --steps 60 --device cuda --no-pretrained-init --checkpoint checkpoints/_ctx.pt` rồi sinh với prompt dài ~1500 token để test long-context probe.
4. **medium load+sinh:** `python scripts/demo_xl_moe.py` chỉ là xl-moe; để kiểm medium: `JINGPT_SCALE=medium python -c "..."` (hoặc chạy `python generate.py` với checkpoint medium khi có).

### Ghi chú trung thực / nợ kỹ thuật

- `xl-moe` theo đúng shape spec (12L, embed 1280, 8 exp top-2) cho **0.78B total / 0.31B active** — active khớp mục tiêu ~300M; total ~0.8B (spec ước lượng 1.2B, số đo thật thấp hơn vì vocab 8.256 + tied embedding).
- `expert_offload` hoạt động khi model được khởi tạo đúng device (đường đi của `build_inference_bundle`); nếu `model.to(device)` sau khi khởi tạo CPU thì expert bị kéo lên device — ghi chú đã thêm trong model.py.
- MuP (tuỳ chọn trong spec) chưa làm — ROI thấp so với các mục khác; sẽ cân nhắc nếu cần transfer LR giữa các cỡ model.
- Train thật preset small/medium trên host này vẫn bị chặn bởi fp32 AdamW (Phase 0 đã đo) → Phase 2 (optimizer state 8-bit) là cổng để medium train được.### Việc tiếp theo

Phase 2 — Nhanh hơn: Muon optimizer (optim_muon.py), torch.compile fallback, prefix cache, int8 KV cache, prompt-lookup decoding; ưu tiên cao: optimizer state 8-bit để mở khoá train small/medium trên 6GB.

---

## PHASE 2 — Nhanh hơn (train + inference) (CODE DONE — chờ verify GPU)

> Trạng thái: code + unit test (CPU) xong, **35/35 pass**. Phần verify GPU
> (A/B Muon vs AdamW, decode tok/s ×2.5, TTFT lượt 2+) do người dùng tự chạy
> bằng các lệnh ở cuối phase — host WDDM này crash khi agent lặp CUDA.

### Đã làm (thay đổi code)

| File | Nội dung |
|---|---|
| `optim_muon.py` (mới) | Muon tự viết: Newton-Schulz 5 bước (hệ số Moonshot), momentum 1 buffer, weight decay decoupled, scale `sqrt(max(1, fan_out/fan_in))`, nesterov; AdamW cho embed/head/norm/bias. `train.py --optimizer muon`. Tiết kiệm ~1 buffer momentum cho mọi matrix 2D. |
| `model.py` | **Static KV cache** (pre-allocated, bỏ `cat` mỗi step) + **int8 KV cache** (per-token scale, dequant `buf/127*scale`); **prompt-lookup speculative decoding** (`prompt_lookup_k`, verify batch + rewind cache khi reject — đúng cả partial accept); **document mask** (`doc_mask` trong forward, cắt attention tại ranh giới tài liệu); **prefix reuse** (`kv_caches` + `skip_prefill_tokens` trong `generate`/`stream_generate`); **selective grad checkpointing** (`gc_mode="layer2"`, `--low-vram`); mask attention theo **absolute position** (sửa bug window + trimmed cache). |
| `quantize.py` (mới) | Weight-only quant tự viết: int8 per-channel / int4 group-128 (tự co group nếu in_features không chia hết), lazy dequant fp16, `--quant int8|int4` cho chat/generate/app. |
| `utils.py` | `build_inference_bundle(..., quant=...)` áp quant sau load; `get_batch_with_doc_mask` (doc_mask 1 tại đầu block + sau `<|endoftext|>`); load checkpoint state optimizer linh hoạt (Muon↔AdamW state không tương thích → warning rõ ràng, không im lặng bỏ weight). |
| `train.py` | `--optimizer muon`, `--pack` (sequence packing với doc mask thật), `--low-vram` → `gc_mode="layer2"`. |
| `app.py` | `--quant/--static-kv/--int8-kv/--no-prefix-cache`; **prefix cache per-session** (turn 2+ chỉ prefill phần mới — TTFT giảm), LRU budget 16 session; endpoint `/api/kv-cache`; status báo kv_cache/quant. |
| `chat.py`, `generate.py` | Cờ `--quant/--static-kv/--int8-kv`; `generate_with_stop` chuyển sang incremental cache đúng (prefill 1 lần, sau đó chỉ feed token mới). |
| `tests/` | +`test_optim_muon.py` (Muon step/loss, partition, state_dict; quant int8/int4 round-trip), +`TestPhase2KV` (static≡dynamic, int8≡fp32, prompt-lookup≡greedy, prefix-reuse≡full-prefill, multi-turn reuse, doc-mask). |

### Bug nghiêm trọng phát hiện & sửa (đáng ghi)

1. **int8 KV dequant thiếu `/127`**: sai số logits ~0.134 → sau fix 0.0005 (gấp ~260×). Test greedy equivalence bắt được.
2. **Sliding-window + prefill vào cache đã trim**: query dài hơn keys → SDPA shape mismatch + rò rỉ future token; sửa bằng mask absolute-position cho mọi forward nhiều token.
3. **quantize int4 group-128 crash** khi `in_features % 128 != 0` (992 của small preset) → tự co group đến ước số đúng.
4. **chat.py manual loop double-write KV** (feed cả context mỗi step vào cache) → đổi sang prefill-once + incremental.

### Kết quả đo được (CPU, không phải acceptance cuối)

- Unit test: **35/35 pass** (`python -m unittest discover -s tests`).
- Checkpoint v2.5 vẫn load + generate nguyên vẹn (chat.py, generate.py, app.py smoke đều OK).
- int8 KV: logits diff 0.0005 (sau fix), greedy output bằng fp32 — test pass.
- int4 weight quant: logit drift ~0.03 trên toy net (bảng `test_optim_muon` pass).

### Việc cần chạy (GPU — người dùng tự verify)

```bash
# 1. A/B Muon vs AdamW cùng seed/data (tiny, fit 6GB)
JINGPT_SCALE=tiny python train.py --mode instruct --fast --steps 300 --device cuda --no-pretrained-init --optimizer adamw --checkpoint checkpoints/_muon_adamw.pt
JINGPT_SCALE=tiny python train.py --mode instruct --fast --steps 300 --device cuda --no-pretrained-init --optimizer muon   --checkpoint checkpoints/_muon_muon.pt
python bench.py --compare bench_results/*_muon_adamw.json bench_results/*_muon_muon.json   # muon: time-to-val-loss <= adamw

# 2. Decode speed: baseline vs static+int8 KV (target decode >= 2.5x so với Phase 0: 58.3 tok/s)
python bench.py --checkpoint checkpoints/jingpt_chat.pt --no-train-bench
python bench.py --checkpoint checkpoints/jingpt_chat.pt --no-train-bench --static-kv --int8-kv
python bench.py --compare bench_results/*.json bench_results/*_static.json

# 3. TTFT lượt 2+ (prefix cache): chạy app rồi 2 lượt chat cùng session
python app.py --static-kv
#   POST /api/chat x2 cùng session_id; đo thời gian lượt 2 (mong đợi giảm >=40%)

# 4. Quantized inference (PP L delta)
python bench.py --checkpoint checkpoints/jingpt_chat.pt --no-train-bench --quant int8

# 5. Packing có doc mask
python train.py --mode instruct --fast --steps 20 --device cpu --no-pretrained-init --pack --checkpoint checkpoints/_pack.pt
```

### Ghi chú trung thực / nợ kỹ thuật

- Prompt-lookup chỉ có lãi khi prompt có n-gram lặp (code/JSON/chat); trên văn xuôi ngẫu nhiên draft thường rỗng → chi phí ~0. Đo tốc độ thật bằng bench ở bước (2).
- Prefix cache giữ KV của session trong RAM/VRAM (LRU 16 session) — `--no-prefix-cache` để tắt nếu VRAM chật.
- torch.compile chưa bật mặc định (Windows + Triton hay lỗi — đã có fallback try/except trong train.py từ v2.5); test riêng trên máy người dùng nếu muốn.
- Optimizer state 8-bit chưa làm (mục trong spec) — Muon đã tiết kiệm 1 buffer; nếu medium vẫn OOM thì Phase 2.5 sẽ bổ sung block-wise quantization.
- Dataloader memmap `.npy` chưa làm (corpus hiện tại ~76MB vẫn nằm trong RAM); làm khi corpus vượt RAM (Phase 3).

### Việc tiếp theo

Phase 3 — Thông minh hơn: data_quality.py (dedupe MinHash-LSH, filter, contamination check), mixture curriculum, CoT data, tool-use format, rejection sampling self-distill, DPO/ORPO tự viết. Điểm probe mục tiêu +15% vs baseline v2.5.

---

## PHASE 3 — Thông minh hơn: dữ liệu & huấn luyện (CODE DONE — chờ verify GPU)

> Trạng thái: code + unit test (CPU) xong, **56/56 pass**. Phần train thật
> (SFT trên dữ liệu mới, DPO/ORPO, tự distill) do người dùng chạy trên GPU.

### Đã làm (thay đổi code)

| File | Nội dung |
|---|---|
| `data_quality.py` (mới) | **Dedupe MinHash+LSH** tự viết (shingle 5-gram, 128 hash, 16 band, ngưỡng 0.65): exact + near-dup mức document; **quality filter** kiểu Gopher/C4 rút gọn (độ dài, tỉ lệ chữ cái, dòng lặp, stopword, ký tự đặc biệt, boilerplate, dòng dài, OCR lỗi); **language detection** n-gram 3-char tự train cho vi/en (kèm seed mặc định); **contamination check** so data/eval/probes.jsonl (n-gram 8 từ, tự co cho prompt ngắn) — loại doc rò rỉ benchmark. CLI + report JSON. |
| `build_dataset.py`, `build_instruct_dataset.py` | Gọi `check_contamination` mặc định sau khi build (cờ `--no-contamination-check` để tắt). |
| `data/mixture.json` + `utils.MixtureSampler` | Curriculum mixing: 3 stage theo tiến độ train (foundation 60% → reasoning 30% → anneal 10%), mỗi domain một file nguồn, `train.py --mixture data/mixture.json` sample theo weight + log tỉ lệ thực tế đã thấy. |
| `bpe_tokenizer.py` | `default_special_tokens(extended=True)` thêm `<|think|>`, `<|/think|>`, `<|tool|>` **ở đuôi vocab** — ID cũ không dịch chuyển, checkpoint cũ load nguyên vẹn (chỉ dùng khi rebuild tokenizer, đúng kế hoạch Phase 3). |
| `chat_template.py` | Role `tool` (`<|tool|>`), `format_tool_trace` (query → call → result → answer có citation), `format_cot_document` (`<|think|>...<|/think|>`). |
| `build_reasoning_data.py` (mới) | Dữ liệu suy luận **có kiểm chứng bằng Python**: toán GSM8K-style + đại số + hệ phương trình (Fraction/Gramer), logic puzzle, code có unit test chạy sandbox — đáp án đúng được tính tại thời điểm sinh, không thể sai nhãn. Output JSONL. |
| `self_distill.py` (mới) | **Rejection sampling**: k=8 candidate từ checkpoint hiện tại, verifier tự động (math numeric/exact, code chạy test sandbox, generic heuristic), giữ candidate tốt nhất đạt chuẩn → SFT JSONL. |
| `train_pref.py` (mới) | **DPO** (mặc định, reference model đóng băng, beta=0.1) + **ORPO** (`--method orpo`, không cần reference — tiết kiệm VRAM); AdamW, cosine LR, AMP, save checkpoint. |
| `config.py` | Thêm `EVAL_DIR`, `PROBES_PATH`, `MIXTURE_PATH`. |

### Kết quả đo được (CPU, số thật)

- Unit test: **56/56 pass** (35 Phase 0-2 + 21 Phase 3).
- Checkpoint v2.5 vẫn load nguyên vẹn (vocab 8.256, 6 special token — token mở rộng chỉ xuất hiện khi rebuild tokenizer).
- **Data quality trên 3MB slice `data/chat_train.txt`** (đo thật): 6.067 conversation → quality filter loại 21 (repeated_lines), dedupe loại **5.436** (5.302 exact + 134 near-dup — exact trùng vì seed duplication ×40 trong build_instruct_dataset), **contamination loại 6 doc** (rò rỉ n-gram probes.jsonl). Còn 604 doc sạch.
- `build_reasoning_data.py`: 100 mẫu test sinh đúng (50 math, 20 logic, 30 code), **100% verified**, code snippet pass đủ unit test.
- `train_pref.py` smoke (tiny CPU): ORPO loss 2.34 → step 4, DPO loss 0.35 → lưu checkpoint OK.
- `self_distill.py` smoke: verifier hoạt động đúng — model ngẫu nhiên sinh 8 candidate, tất cả bị loại (math_mismatch/numeric) vì không đúng đáp án (hành vi trung thực, không giữ rác).

### Lưu ý trung thực / nợ kỹ thuật

- `MAX_DOC_CHARS=200k` trong quality filter sẽ loại các `<doc>` rất lớn trong `data/train.txt` (9/11 doc của 8MB slice bị loại) — nếu corpus pretrain có doc >200k chars, tăng ngưỡng. Đây là knobs cần chỉnh theo corpus thật.
- Dedupe pure-Python/numpy chậm trên toàn corpus lớn (~1 phút/MB đo được trên 3MB chat) — đủ dùng cho corpus hiện tại; nếu corpus lên GB thì cần batching (ghi chú cho Phase sau).
- `<|think|>` / `<|tool|>` chỉ có hiệu lực khi **rebuild tokenizer** (train BPE mới); checkpoint cũ không có embedding cho 3 id này — đúng thiết kế, cần lưu ý khi tiếp tục train từ ckpt cũ với dữ liệu CoT/tool.
- `self_distill` generic verifier là heuristic (độ dài + repetition) — yếu hơn verifier chuyên biệt; Phase 3.7 RLVR (reward = đáp án đúng) sẽ mạnh hơn cho toán/code.
- DPO cần reference model đóng băng — trên 6GB, chạy reference bằng no_grad (đã làm) và có thể int8 (chưa tích hợp; ghi chú).

### Việc cần chạy (GPU — người dùng tự verify)

```bash
# 1. Sinh dữ liệu reasoning + tự distill (chạy CPU/GPU đều được)
python build_reasoning_data.py --n-math 2000 --n-logic 800 --n-code 1200 --out data/reasoning.jsonl
python self_distill.py --checkpoint checkpoints/jingpt_chat.pt --prompts data/reasoning.jsonl --k 8 --out data/distill.jsonl

# 2. SFT trên dữ liệu mới (chuyển distill JSONL thành chat corpus qua format_tool_trace/format_cot_document rồi train instruct)
python train.py --mode instruct --fast --steps 500 --device cuda --checkpoint checkpoints/jingpt_chat.pt

# 3. DPO (cần cặp chosen/rejected):
python train_pref.py --data data/pref.jsonl --method dpo --checkpoint checkpoints/jingpt_chat.pt --steps 200 --device cuda
#    ORPO (không cần reference):
python train_pref.py --data data/pref.jsonl --method orpo --checkpoint checkpoints/jingpt_chat.pt --steps 200 --device cuda

# 4. Chạy data quality trên toàn corpus + report:
python data_quality.py --input data/train.txt --output data/train_clean.txt --report data/train_clean.report.json
python data_quality.py --input data/chat_train.txt --output data/chat_train_clean.txt --report data/chat_train_clean.report.json

# 5. Curriculum pretrain:
python train.py --mixture data/mixture.json --steps 2000 --device cuda
```

### Bug hunt Phase 2-3 (tìm & sửa trước khi sang Phase 4)

Theo yêu cầu "find bug first → optimize → re-check → Phase 4", rà soát + stress-test toàn bộ code Phase 2-3. Các bug **thật** tìm được (kèm test hồi quy mới trong `tests/test_model_shapes.py`, tổng suite tăng 56 → **60 test**):

1. **Prompt-lookup KHÔNG BAO GIỜ chạy** (bug lớn nhất, im lặng): `if suffix in prompt_ids[:-n]` dùng list `in` (so sánh phần tử) — không bao giờ khớp sublist đa token → mọi feature speedup chết ngầm. Sửa: thêm `_rfind_sublist` (tìm lần xuất hiện cuối, đúng thứ tự token). Test mới `test_prompt_lookup_drafts_actually_fire` xác nhận draft sinh ra được.
2. **Draft-room tính thiếu `cur_input.size(1)`**: batch forward ghi `cur_input + drafts` token, nhưng `max_draft` chỉ trừ `start_len` — cache window có thể slide giữa batch (không unwind được) → hỏng KV cache âm thầm. Sửa: trừ cả `cur_input.size(1)`.
3. **Off-by-`cur_len` trong verify draft**: `logits[:, d_i]` sai khi `cur_input` nhiều token (prefill lần đầu / prefix-reuse tail dài) — draft ở vị trí `cur_len - 1 + d_i`. Sửa cả vị trí verify lẫn vị trí sample sau reject.
4. **Rewind cache thiếu `cur_input.size(1)`**: `keep_len = 1 + n_accept` xoá nhầm prompt token đã ghi cùng batch (cache len=2 nhưng pos=7 → garbage, output lệch ở token 11). Sửa: `keep_len = cur_input.size(1) + n_accept`.
5. **Draft shape 3D**: `target.unsqueeze(0)` trên tensor đã là `(1,1)` → `torch.cat` crash khi draft bắt đầu chạy. Sửa: cat trực tiếp.
6. **All-accepted + hết budget sinh thừa 1 token** (29 vs 28): full-accept branch bị bỏ qua khi `remaining == 0`, rơi xuống partial branch sample thêm 1. Sửa: `if remaining <= 0: break` trong nhánh full-accept.
7. **`stream_generate` thiếu guard empty-tail**: `skip_prefill_tokens >= idx.size(1)` → `logits[:, -1]` IndexError (đường `/api/chat/stream` lượt 2+). Sửa: cùng guard như `generate`.
8. **Muon nesterov double-mutate momentum**: `m_nes = m.mul_(beta1).add_(grad)` ghi đè `m` lần 2 → momentum không còn là EMA sạch (gradient hiện tại vào 2 lần). Sửa theo đúng Keller-Jordan single-buffer: `m.mul_(beta1).add_(grad)` rồi lookahead `grad.add(m, alpha=beta1)` không chạm lại `m`.
9. **Contamination rewrite phá format corpus** (đã sửa ở đầu phase): `build_dataset.py` mất `</doc>`, `build_instruct_dataset.py` mất `<|endoftext|>` — cả hai đều phá ranh giới tài liệu mà `chat_dataset.py` cần. Sửa: split giữ nguyên delimiter.

Không phải bug (đã xác minh bằng stress-test 12 seed × static/dynamic, cache content so sánh từng tensor): KV cache static/dynamic/int8 nhất quán, prefix reuse tương đương full prefill, `_rfind_sublist` đúng 5 trường hợp biên.

---

## PHASE 4 — Bộ nhớ dài hạn: nhớ vĩnh viễn mà không cần train lại (CODE DONE — chờ verify GPU)

> Trạng thái: code + unit test (CPU) xong. Các bước verify số liệu lớn
> (1000 mẩu → ≥80% trả lời đúng, retrieval <100ms/100k vector) do người
> dùng chạy trên GPU/máy thật.

### Đã làm (thay đổi code)

| File | Nội dung |
|---|---|
| `memory.py` (mới) | **3 tầng bộ nhớ** (episodic TTL 30 ngày / semantic có provenance đầy đủ / procedural). Schema mỗi mẩu: `id` content-addressed (sha1 text), `claim`, `source_url`, `license`, `fetched_at`, `confidence`, `embedding_id`, `verified_by`, `contradicts[]`, `needs_verification`. Append-only JSONL + compaction; **consolidation** (gộp near-dup cosine>0.95 cùng claim, decay confidence theo tuổi cho kiến thức có TTL, **phát hiện mâu thuẫn** cùng chủ đề khác claim → `needs_verification` cho Curiosity Engine). Dedupe ngay tại `add` (id content-addressed → cùng text chỉ 1 mẩu). |
| `retriever.py` (mới) | **Embedding từ chính Jin** (mean-pool hidden state `ln_f` qua forward hook, không model ngoài, L2-normalize); **`VectorIndex`** brute-force cosine → **IVF-flat tự viết** (kmeans numpy, nlist≈√N, nprobe) khi >100k; **`BM25Index`** (k1=1.5, b=0.75); **`HybridRetriever`** = RRF (BM25 + vector) → rerank top-20 bằng logprob của Jin → top-5. |
| `app.py` | RAG có citation: `_maybe_use_memory` (cờ `memory: auto/on/off`, auto chỉ khi câu ≥3 từ và store không rỗng), context chèn `[1] [2]` + nguồn trả về kèm `kind: memory`, `memory_id`, `confidence`; **5 API mới**: `/api/memory/search|add|stats|forget|consolidate`; `--no-memory` (hành vi v2.5); index tăng dần theo yêu cầu (chỉ embed mẩu chưa có). |
| `config.py` | `MEMORY_DIR/PATH/INDEX_PATH`, `MEMORY_ENABLED` (`JINGPT_NO_MEMORY=1`), `MEMORY_POOL_LAYER`. |
| `tests/test_memory.py` (mới) | **10 test**: CRUD + persist reload + delete theo source_url, TTL compaction, consolidation gộp near-dup, phát hiện mâu thuẫn, content-addressed id, VectorIndex brute-force + IVF, BM25, RRF fusion. |

### Kết quả đo được (CPU + smoke, số thật)

- Unit test memory: **10/10 pass**. Tổng suite: **60 + 10 = 70 test** (chạy lại ở cuối phase).
- Smoke app thật (CPU): add 2 mẩu → `/api/memory/stats` = 2 semantic, index brute-force dim 992 (đúng embed_dim small preset); search "What is the capital of France?" → mẩu Paris **đứng #1** (rerank logprob -6.07 vs -7.24); chat `memory:on` → response kèm 2 sources (kind memory + url), `web_search_used=true`; forget theo id → store còn 1; consolidate chạy không lỗi; **`--no-memory` → cả stats/add đều 503** (đúng thiết kế, server không crash).
- Latency note: embedding query tốn 1 forward của model 151M trên CPU (~vài trăm ms) — bỏ qua khi store rỗng (early-out đã thêm).

### Bug tìm được trong chính Phase 4 (sửa ngay)

1. **Content-addressed id trùng text** → 2 mẩu cùng id + cùng embedding slot: consolidation đánh dấu CẢ HAI deleted. Sửa: `add()` dedupe ngay (update mẩu cũ, không append), merge-phase bỏ qua `ej["id"] == ei["id"]`.
2. **Merge phase nuốt mâu thuẫn**: consolidation merge (0.95) chạy trước contradiction pass (0.85) → cặp cùng chủ đề khác claim bị gộp mất, `disputed` không bao giờ > 0. Sửa: merge chỉ khi **cùng claim**; khác claim = contradiction, để cho pass sau.
3. **`v not in (None, [], False)`** trong update path: `0.0 == False` → confidence=0.0 bị bỏ qua. Sửa: so sánh `is not None`.
4. **`memory_forget` index-sync sai**: xoá theo `source_url` đưa `set()` rỗng vào `vector.remove` (no-op) và **không bao giờ sync BM25** → mẩu bị forget vẫn search được. Sửa: tính `affected` ids trước, xoá cả vector lẫn bm25.
5. **`_maybe_use_memory` embed query mọi request** kể cả khi store rỗng → early-out khi `not MEMORY_STORE.all()`.

### Rủi ro / nợ kỹ thuật

- Embedding bằng hidden state của Jin hiện tại (chưa train contrastive head) — chất lượng retrieval vừa đủ; Phase sau có thể train `contrastive projection head` (in-batch negatives) nếu cần top-k chính xác hơn.
- Rerank logprob của Jin trên 151M model yếu — với model mạnh hơn sẽ phân biệt tốt hơn.
- IVF nlist/nprobe chưa tinh chỉnh theo N thật (mặc định 64/8) — đo lại khi store >100k.
- Index nằm trong RAM process app; restart server → rebuild index theo yêu cầu (lazy) — chưa persist `.npy` index riêng (đã có `MEMORY_INDEX_PATH` trong config để dành).

---

## PHASE 5 — CURIOSITY ENGINE: trí tò mò vô hạn (CODE DONE — chờ verify GPU/network)

> Trạng thái: code + unit test (CPU) xong, **23 test mới pass**. Các bước chạy
> thật (harvest mạng, train continual, gate trên checkpoint thật) do người dùng
> chạy vì host này không ổn định khi agent lặp CUDA/mạng.

### Đã làm (thay đổi code)

| File | Nội dung |
|---|---|
| `curiosity_engine.py` (mới) | Vòng **8 bước** đầy đủ. B1 gap detection: `score = uncertainty × importance × novelty × learnability` (entropy token + self-consistency disagreement k-sample + entity novelty vs corpus), mọi thành phần log. B2 query đa góc + **quiz viết TRƯỚC** khi học (tiêu chí thành công, không tự lừa). B3 harvest qua fetch injectable + domain allowlist/denylist + min-text filter. B4 **verification**: ≥2 domain độc lập → high; 1 → medium; khác claim → disputed (không vào weight); đối chiếu `needs_verification` từ memory → conflict = disputed. B5 **SEAL self-edit**: 4 form (assertion/qa/dialogue/summary), 4-20 mẫu/claim. B7 **eval gate thuần** (`eval_gate`: quiz ≥80%, forgetting ≤1%, val loss ≤+1%, fluency không tụt >0.02) + `promote_checkpoint` registry.json (3 gần nhất + golden bất khả xâm phạm). B8 `run_cycle` ghi ledger đầy đủ. |
| `knowledge_ledger.py` (mới) | Sổ append-only mọi vòng học (topic, queries, quiz, sources, claims verified/rejected, form, replay/lr/steps, score/forgetting/val/fluency delta, decision + reason). **UCB bandit** trên (topic_kind × self_edit_form): thưởng = score_delta, rollback = 0, arm < min_samples được cộng exploration bonus. |
| `lora.py` (mới) | LoRA tự viết: `LoRALinear` (A/B rank-r, init A~N(0,1/r), B=0), wrap qkv/o/gate/up/down, **merge/unmerge fold delta vào base weight có thể hoàn tác 1 giây**, `trainable_parameters` chỉ A/B. |
| `learn_daemon.py` (mới) | Chạy nền có **budget** (request/GB/cycle từ config) + **phanh an toàn** (dừng sau N vòng fail gate liên tiếp) + `--dry-run`. Meta-learning thật: pick arm UCB **cùng kind** với topic hiện tại → truyền `self_edit_form` vào cycle. |
| `web_search.py` | Mở rộng cho harvest: **robots.txt** (`urllib.robotparser`, cache per-domain, fail-open), **per-domain rate limit** (≤1 req/2s/domain), **Wikipedia API** (search + extract), **arXiv API** (Atom XML), `harvest()` facade gộp DDG+Wiki+arXiv, mỗi nguồn fail độc lập không chết chung. |
| `config.py` | Khối `CURIOSITY_*`: paths, budgets (`MAX_REQUESTS/MAX_GB_DAY/MAX_CYCLES`), `MAX_CONSECUTIVE_FAILS=3`, `SAMPLES_PER_CLAIM=8`, `VERIFY_MIN_SOURCES=2`, `REPLAY_RATIO` 0.05/0.3, `LORA_RANK=16`, domain allowlist/denylist, **gate thresholds** (`GATE_QUIZ_PASS_RATE=0.80`…). |
| `pipeline.py` | Mode mới **`evolve`**: bench baseline → curiosity cycles → báo cáo merged/rolled_back → ledger path. |
| `tests/test_curiosity_gate.py` (mới) | **23 test**: eval gate 5 trường hợp (pass + 4 kiểu fail), verification (2 nguồn→high, 1→medium, spam 1 nguồn không verify), gap ranking (mystery > known, entity novelty), harvest filter + allowlist, ledger persist + UCB ưu tiên arm thắng, registry golden + bounded history, **engine rollback** (rác → memory_only + train không bao giờ chạy; gate fail → rolled_back + brake tăng), self-edit 4 form, **LoRA merge≡adapter-on + unmerge khôi phục chính xác**. |

### Kết quả đo được (CPU, số thật)

- **23/23 test Phase 5 pass**; tổng suite **91/91 pass** (60 + 10 memory + 21… tăng từ 70).
- Logic check thuần: gap detection xếp `quantum computing` (score 0.25) > `Paris` (đã biết); UCB pick `(factual, qa)` sau khi arm qa thắng 0.2 vs 0.01; gate pass đủ điều kiện; Budget chặn request quá hạn + GB quá hạn đúng.
- Kiểm chứng bằng fake (không mạng/GPU): 2 domain độc lập → high; 1 nguồn spam → **memory_only**, `train` KHÔNG được gọi (lá chắn chống học rác hoạt động); 3 vòng fail gate → brake.

### Bug tìm được trong chính Phase 5 (sửa ngay)

1. **UCB không được nối vào vòng học** (meta-learning hở): `run_cycle` hardcode `form="qa"`, daemon đọc `best_arm()` nhưng không truyền đi. Sửa: `run_cycle(self_edit_form=...)` + daemon chọn arm **cùng kind** rồi truyền form.
2. **LoRA merge sai phép tính**: `B @ A.t()` là (rank,out)@(rank,in) → crash. Delta đúng là `B.t() @ A.t()` (out,in). Test mới `test_merge_then_unmerge_restores_weights` xác nhận merge ≡ adapter-on và unmerge khôi phục base chính xác.
3. **Test yếu của tôi**: 2 test dùng text < 200 chars bị harvest filter chặn (→ "no harvestable pages" thay vì mục tiêu), 1 test assert ngược chiều (cho rằng merged ≠ before — thực ra merge PHẢI bằng before). Sửa test cho đúng spec.
4. **Quiz template sai ngữ pháp** số nhiều: "Why does black holes matter?" → "Why is {topic} important?".
5. **`_allowed_domain` import sai module** trong test → sửa sang `curiosity_engine`.

### Rủi ro / nợ kỹ thuật

- **Train provider chưa nối** vào model thật: `learn_daemon.make_engine` raise hướng dẫn rõ ràng (train.py --mode curiosity) thay vì train mù — chạy thật cần implement continual-pretrain có replay (re-warm + re-decay) hoặc LoRA merge gate.
- Gate thật cần `bench.py` trên checkpoint ứng viên (quiz riêng chủ đề mới + probes cũ + fluency) — chưa tự động hoá thành hàm in-process (test dùng fake metric).
- `verify_claims` là sentence-overlap heuristic (token overlap ≥0.6) — đủ chặn rác cơ bản, nhưng claim trùng nghĩa khác từ có thể không gộp; cải thiện bằng embedding cosine khi có contrastive head.
- Harvest thật phụ thuộc mạng + robots.txt fail-open; SSRF guard giữ nguyên.

### Việc tiếp theo

Phase 6 — Agentic: `tools.py` (registry schema JSON: calculator an toàn, python_exec sandbox subprocess no-network, memory_search/write, now), `agent.py` ReAct loop (sinh `<|tool|>` → runtime thực thi → tiếp tục, giới hạn vòng + timeout + stream UI), self-check verifier (`--rigor`), SFT trên tool-trace. Acceptance: ≥85% số học nhiều bước đúng khi bật calculator; sandbox chặn test thoát.

---

## PHASE 6 — Agentic: biết dùng công cụ, biết tự kiểm tra (CODE DONE — chờ verify GPU)

> Trạng thái: code + test CPU xong (131/131 pass). Không chạy project/server/GPU theo yêu cầu — bạn chạy lệnh acceptance ở cuối.

### Đã làm

| File | Nội dung |
|---|---|
| `tools.py` (mới) | Registry tool schema JSON: **calculator** recursive-descent **không eval** (độ ưu tiên chuẩn: `-2**2 == -4`, `2**3**2 == 512`, right-assoc, `_MAX_CALC_DEPTH`, error string thay vì crash); **python_exec sandbox** subprocess `-I` + import whitelist + chặn `socket`/`os.system` + chặn ghi file + timeout cứng + cap output; `web_search`/`fetch_page` (SSRF guard sẵn có); `memory_search`/`memory_write` (keyword-overlap + kèm source/confidence); `now`. |
| `agent.py` (mới) | **ReAct loop**: model sinh `<|tool|>{json}<|end|>` → `execute_tool` → chèn `<tool result>` → lặp; parse JSON bằng **scanner cân bằng ngoặc nhọn** (bỏ qua string literal) — regex non-greedy cũ vỡ với args lồng nhau; `max_tool_calls` budget → hết hạn trả lời thật "ran out of tool calls" kèm tool output; `stream_cb` phát event tool_call/tool_result/final; **self-check verifier** `--rigor fast/normal/high` (math re-check bằng calculator, citation bắt buộc, resample + Jaccard self-consistency); `generate_fn` injectable cho test không cần GPU; CLI `python agent.py "..."`. |
| `tests/test_tool_sandbox.py` (mới) | **35 test**: calculator đúng + an toàn (chặn `__import__('os')`, `1/0`, `2**(`); sandbox chặn `import os/subprocess/sys/socket`, chặn `socket.connect`, chặn ghi file, timeout giết `while True`, `1/0` báo lỗi không crash; parse nested args; max_tool_calls enforced; self-check math mismatch/citation/consistency; memory tools trên MemoryStore thật. |
| `model.py` (sửa) | **Fix bug rope table overflow** (chi tiết dưới) + thêm 3 regression test. |

### Kết quả đo được (CPU, số thật)

- **35/35 test Phase 6 pass**; tổng suite **131/131 pass** (tăng từ 91).
- Smoke thật (CPU, model 0.1M + fake tokenizer): prefill đúng `max_positions`, sinh vượt **3× ctx** (prompt 4 → 388 token) không crash cả `generate` lẫn `stream_generate`; agent loop chạy qua real `model.generate`.

### Bug tìm được trong chính Phase 6 (sửa ngay)

1. **BUG SẢN XUẤT (model.py, tìm nhờ test agent)**: sinh vượt `max_positions` tổng token bị crash. Nguyên nhân: `pos` (absolute) tăng không giới hạn sau khi `_trim_kv_cache` cắt cache còn `max_positions-1`, nhưng bảng `rope_cos/sin` chỉ precompute đúng `max_positions` dòng → `rope_cos[offset:offset+seq]` rỗng → q/k co về **0 token** → SDPA crash `Expected [4,127] got [4,128]`. Ảnh hưởng: bất kỳ prompt+sinh ≥ ctx (vd prompt 600 + sinh 425 trên block 1024). Sửa: **clamp rope lookup** `rope_off = min(offset, table - seq_len)` (chuẩn Llama.cpp/Mistral sliding-window cap) — vị trí chính xác tới hết bảng, sau đó alias mềm thay vì crash; q/k dùng chung clamp nên cặp nhất quán. 3 regression test mới.
2. **`parse_tool_call` regex non-greedy vỡ với args dict lồng nhau**: `\{.*?\}` dừng ở `}` đầu tiên → JSON cụt → mọi tool call có `args: {...}` đều fail. Sửa bằng scanner đếm ngoặc nhọn cân bằng + bỏ qua string literal.
3. **`python_exec` vỡ lúc khởi tạo**: wrapper dùng `%`-format 2 placeholder (`%r` whitelist + `% top` trong lỗi import) nhưng chỉ truyền 1 arg → `TypeError: not enough arguments for format string`. Sửa: inject whitelist bằng `str.replace` thay vì `%`.
4. **`-2**2` sai kết quả (4 thay vì -4)**: grammar cũ cho unary minus độ ưu tiên CAO hơn `**`. Sửa theo đúng grammar Python: `factor := ('+'|'-') factor | power`, `power := atom ('**' factor)?` → `-2**2 == -4`, `2**-2 == 0.25`, `(-2)**2 == 4`.
5. **`agent.py build_bundle` crash khi chạy CLI**: import `quantize_weights` (không tồn tại — tên thật `apply_weight_quantization`), gọi `load_checkpoint(str)` với signature sai (đối số đầu là model), gọi `build_inference_bundle(model)` sai tham số. Sửa: dùng thẳng `build_inference_bundle(checkpoint, device, quant=...)` — nó đã lo checkpoint load + quantization.
6. **Test của tôi viết sai**: `test_low_self_consistency_flagged` feed cùng text cho cả answer lẫn resample (counter bắt đầu 0) → Jaccard 1.0, không bao giờ flagged; `max_positions` không phải tham số của JIN AI → bỏ kwarg.

### Rủi ro / nợ kỹ thuật

- **Rope cap làm mất tín hiệu vị trí ở đuôi** sau khi `pos` vượt bảng (mọi token mới alias về góc cuối) — đây là hành vi chuẩn của sliding-window cap; với global layer toàn cảnh, chất lượng dài hạn có thể giảm nhẹ ở đoạn vượt ctx (không crash). Nếu cần chính xác tuyệt đối: precompute bảng lớn hơn hoặc regenerate bảng theo window — ghi là debt.
- Sandbox `python_exec` không có rlimit POSIX trên Windows (chỉ subprocess timeout) — đủ cho mục đích, note cho Linux thì thêm `resource.setrlimit`.
- `memory_search` tool là keyword-overlap (model-free) — chưa dùng HybridRetriever logprob rerank (cần model+tokenizer trong ctx).
- **SFT trên tool-trace chưa chạy** (Phase 3.4 có `format_tool_trace` — cần sinh trace thật bằng agent rồi SFT; acceptance ≥85% số học nhiều bước cần model thật đã học format `<|tool|>`).
- Chưa tích hợp `/api/agent/chat` + UI stream tool steps (để dành cho Phase 7 vận hành).

### Việc tiếp theo

Phase 7 — Vận hành & giao diện "sống": tab Growth Dashboard (static/) + API mới `/api/learn/trigger`, `/api/learn/status`, `/api/checkpoints`, tất cả sau `JINGPT_TOKEN` + không mở ngoài 127.0.0.1 trừ khi env tường minh; config v3 khối CURIOSITY/MEMORY/MOE/QUANT/SPEED (đã có CURIOSITY); README v3 hoàn chỉnh; `pipeline.py evolve` + `v3-bootstrap`; đóng các acceptance còn thiếu (SFT tool-trace, DPO, RLVR, xl-moe trên 6GB).

---

## PHASE 7 — Vận hành & giao diện "sống" (CODE DONE — chờ verify GPU/server)

> Trạng thái: code + test CPU xong (148/148 pass). Không chạy project/server/GPU theo yêu cầu — bạn chạy verify ở cuối.

### Đã làm

| File | Nội dung |
|---|---|
| `app.py` | **Admin API gated**: `_admin_ok()` — mặc định **TẮT** (403 khi chưa set `JINGPT_TOKEN`) + chỉ loopback trừ khi `JINGPT_ALLOW_REMOTE_ADMIN=1`. Endpoint mới: `/api/bench/history`, `/api/learn/status` (frontier + ledger + memory + registry + budget), `/api/learn/trigger` (1 vòng tò mò chạy nền, `dry_run=True` mặc định an toàn, lock chống chạy song song), `/api/checkpoints` GET + `/promote` + `/rollback` (ghi action=rollback vào registry, không tự ghi đè golden). |
| `bench.py` | `append_history()` — mỗi lần chạy nối bản ghi rút gọn (loss, probe, fluency, tok/s, VRAM) vào `bench_results/history.json` (chịu file hỏng). |
| `static/` | **Growth Dashboard**: nút 📈 sidebar ↔ panel `#growthPanel`; thẻ số liệu (memory, cycles, budget, cycle running), **biểu đồ SVG tự vẽ** (held-out loss / probe pass rate / decode tok/s theo thời gian, không thư viện ngoài), frontier sống, registry checkpoint; ô `JINGPT_TOKEN` lưu localStorage, báo rõ "Admin API is OFF" khi chưa có token. |
| `config.py` | Khối v3 **MOE / QUANT / SPEED** (env `JINGPT_MOE_*`, `JINGPT_QUANT_GROUP`, `JINGPT_COMPILE`, `JINGPT_PROMPT_LOOKUP_K`, `JINGPT_GC_MODE`) + `ALLOW_REMOTE_ADMIN` — giữ pattern env-override. |
| `pipeline.py` | Mode **`bootstrap`**: data → BPE → pretrain → SFT → DPO/ORPO → bench; mỗi stage tái dùng CLI hiện có (resume/skip tự nhiên), cặp preference tự sinh bằng `self_distill` nếu chưa có. |
| `tests/test_phase7.py` (mới) | **17 test**: config blocks, bench history (tạo + chịu corrupt), admin auth 6 case (tắt mặc định / thiếu token / token đúng loopback / chặn non-loopback / mở remote tường minh / bench history gated), promote + rollback + 404, trigger validation + chạy nền qua fake `run_daemon` (không mạng). |
| README | Mục 24 **"Benchmark & kết quả thật"** (bảng before/after từ số đo thật + mục tiêu chờ GPU), mục 25 **"An toàn & đạo đức crawl"** (9 lớp phanh), cập nhật bảng API + mục 11 (bootstrap). |

### Kết quả đo được (CPU, số thật)

- **17/17 test Phase 7 pass**; tổng suite **148/148 pass** (tăng từ 131).
- Baseline v2.5 khoá tại `bench_results/baseline_v25.json` (đo lại: loss 11.86, probe 10.9%, fluency 0.768, decode 58.3 tok/s, TTFT 15.8ms) — số thật để đối chiếu before/after trong README mục 24.

### Bug tìm được trong chính Phase 7 (sửa ngay)

1. **`COMPILE_MODEL` dùng trước khi định nghĩa** trong config.py (khối SPEED đặt trước khối Training → NameError khi import): sửa thành env-only `SPEED_COMPILE` + ghi chú train.py vẫn đọc `COMPILE_MODEL`.
2. (Phòng ngừa) `/api/learn/trigger` không được chạy song song → `_LEARN_LOCK` + `_LEARN_RUNNING`; worker bọc try/except không bao giờ kéo sập server.

### Rủi ro / nợ kỹ thuật

- `/api/learn/trigger` mặc định `dry_run=True` — **train provider thật chưa nối** (nợ Phase 5): endpoint chạy đủ vòng harvest → verify → self-edit → ledger nhưng không đụng weight cho tới khi `train.py --mode curiosity` xong.
- Dashboard biểu đồ SVG tối giản (không lib) — đủ theo dõi, chưa zoom/interact.
- UI chưa verify bằng browser thật (không chạy server theo yêu cầu) — cần mở http://localhost:5000 bấm 📈.
- Acceptance tổng còn chờ GPU: decode ≥2.5×, TTFT lượt 2+ ≥40%, probe ≥+15%, SFT tool-trace + DPO + RLVR + xl-moe trên 6GB, 5 vòng curiosity thật, 1000 mẩu memory → ≥80%.

### Bạn chạy để hoàn tất acceptance Phase 7

```bash
JINGPT_TOKEN=mysecret python app.py          # bật admin API
# mở http://127.0.0.1:5000 → nút 📈 Growth Dashboard
curl -s -H "Authorization: Bearer mysecret" http://127.0.0.1:5000/api/learn/status
curl -s -X POST -H "Authorization: Bearer mysecret" -H "Content-Type: application/json" \
     -d '{"topic": "quantum computing", "dry_run": true}' http://127.0.0.1:5000/api/learn/trigger
curl -s -X POST -H "Authorization: Bearer mysecret" -H "Content-Type: application/json" \
     -d '{"path": "checkpoints/jingpt_chat.pt", "changelog": {"learned": "x"}}' http://127.0.0.1:5000/api/checkpoints/promote
python pipeline.py bootstrap --steps 500 --pref-steps 200 --device cuda   # full path v3
```

### Việc tiếp theo

Toàn bộ 8 phase đã có code + test CPU. Còn lại là **đóng acceptance GPU/server thật**: SFT tool-trace (agent sinh trace → SFT), DPO/ORPO chạy thật trên checkpoint, RLVR toán/code, xl-moe inference 6GB, 5 vòng curiosity thật (cần train provider), 1000 mẩu memory → ≥80% RAG, và đo lại bảng before/after trong README mục 24 để thay "⏳ chờ GPU" bằng số thật.

---

## BUG HUNT + TỐI ƯU CẤU TRÚC + README (round cuối)

> Theo yêu cầu "fix bug, tối ưu structure, README đầy đủ + hướng dẫn chạy" — tất cả verify CPU (149/149 test), không chạy GPU/server.

### Bug thật tìm & sửa

1. **`pipeline.py` không import được** (ImportError `fetch_sources` — không tồn tại; tên đúng `fetch_source`) → toàn bộ pipeline module chết từ Phase 0. Sửa import. **Đây là bug lớn nhất round này** (tìm bằng import-sweep 31 module).
2. **`pipeline.py bootstrap` stage 5 crash**: gọi `self_distill.py` thiếu `--prompts` (required) → argparse fail ngay. Sửa: truyền `--prompts data/reasoning.jsonl` + tự sinh reasoning data nếu chưa có.
3. **`pipeline.py bootstrap` stage 5 sai ngữ nghĩa**: `self_distill --out` ghi SFT chat-format nhưng `train_pref --data` cần cặp `prompt/chosen/rejected` → DPO không bao giờ chạy được qua pipeline. Sửa: thêm `--pref-out` vào `self_distill.py` (chosen = best verified, rejected = worst verified — đúng spec Phase 3.6 "best vs worst") + nối vào bootstrap.
4. **`self_distill` không nhận dữ liệu reasoning**: `build_reasoning_data.py` ghi `answer` nhưng verifier math đọc `reference` → record bị verify rớt hết. Sửa: `load_prompts` tự fallback `reference ← answer`.
5. **`frontier.json` không bao giờ được ghi** (dashboard đọc file chết) → thêm `append_frontier()` + gọi sau mỗi cycle trong `run_daemon` + test.
6. **RUN_COMMANDS.md chứa lệnh của file đã xoá**: `fetch_corpus.py`, `auto_pipeline.py`, `train_chat.py`, `train.py --fetch-data --prepare-data` → cập nhật sang lệnh hiện tại (`fetch_datasets.py`, `pipeline.py raw/chat/bootstrap`, `train.py --mode instruct`).

### Tối ưu cấu trúc

- **Untrack 22 file `__pycache__/*.pyc`** khỏi git (build artifact đã nằm trong .gitignore nhưng bị commit từ trước) — `git rm --cached`, file vẫn còn trên đĩa.
- Import-sweep toàn bộ 31 module chính: tất cả import sạch.
- CLI verify: `pipeline.py/agent.py/self_distill.py/app.py/config.py --describe` đều parse đúng.
- Smoke acceptance thật (CPU): `train.py --mode instruct --fast --steps 5` → data load → pack (2.548 blocks, 29.6% trainable) → train → eval → save checkpoint → exit 0; `chat.py` chạy được.

### README / hướng dẫn chạy

- README mục 3 (file tree): bổ sung `optim_muon.py`, `quantize.py`, làm rõ `web_search.py`.
- README mục 2: thêm **Quick start** (app.py → chat.py → bench → bootstrap) ngay sau phần cài đặt.
- README mục 11: thêm chế độ `bootstrap` (data → BPE → pretrain → SFT → DPO → bench).
- RUN_COMMANDS.md: sửa toàn bộ lệnh cũ hỏng (5.1/5.3/5.4/6.1).

### Số liệu

- **149/149 test pass** (tăng từ 148 — thêm test frontier).
- Nợ kỹ thuật còn lại (không sửa để tránh rủi ro): `app.py` ~1.150 dòng nên tách module khi có test E2E; `checkpoints/*.pt` + `data/train.txt` vẫn được track trong git dù nằm trong .gitignore (đề xuất `git rm --cached` nếu muốn repo gọn — KHÔNG tự làm vì là dữ liệu người dùng); `chat.py`/`agent.py`/`app.py` mỗi nơi một `_stop_token_ids` (trùng lặp nhỏ, giữ nguyên để tránh churn).
