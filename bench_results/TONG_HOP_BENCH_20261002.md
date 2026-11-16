# Tổng hợp bench 2026-10-02 — 4 checkpoint, cùng một thước đo

## Đã chạy gì
- Script: `bench_results/run_logs/bench_4ckpts_20261002.sh`
- Log đầy đủ: `bench_results/run_logs/20261002_4ckpts.log`
- Lệnh mỗi checkpoint: `bench.py --checkpoint <ck> --device cuda --no-train-bench`
- Baseline: `20260930_171852_jingpt_chat.json` (chạy lại hôm nay cho ra **y hệt từng số** → protocol ổn định)

## Bảng kết quả (165 probes, heldout local, greedy)

| Checkpoint | iter | heldout loss | ppl | probe pass | fluency | VI pass |
|---|---|---|---|---|---|---|
| chat đang serving (`jingpt_chat.pt`) | 4000 | 5.6904 | 296.02 | 0.079 | 0.770 OK | 0/22 |
| pretrain_best (`kaggle_v4_pretrain_best.pt`) | 107900 | **5.3887** | **218.92** | 0.030 | 0.204 BROKEN | 1/22 |
| v4 instruct_best (`kaggle_v4_instruct_best.pt`) | 57400 | 5.8285 | 339.83 | **0.145** | 0.664 OK | 0/22 |
| v4 instruct (`kaggle_v4_instruct.pt`) | 60020 | 5.8331 | 341.41 | 0.133 | 0.730 OK | 0/22 |

## So sánh chính (`bench.py --compare`)
- **chat → v4 instruct_best**: probe pass 0.079 → 0.145 (**+83.5%**); thắng gần như mọi domain
  (factual +0.046, math +0.042, code +0.111, reasoning +0.150, instruction +0.150, long_context +0.048);
  **forgetting_score = 0.0000** (không mất gì so với bản cũ); fluency OK (0.77 → 0.664);
  heldout loss +2.4% xấu hơn (đo trên corpus cũ — xem mục "đọc số cho đúng").
- **pretrain_best → v4 instruct_best**: SFT biến base thành chat — fluency BROKEN → OK,
  probes 0.03 → 0.145 (×4.8), mất 1 câu VI (0.045 → 0).
- **v4 instruct_best vs instruct 60020**: gần như song sinh; bản 57400 nhỉnh hơn nhẹ (0.145 vs 0.133).

Các file compare: `bench_results/compare_20261002_*.txt`

## Đọc số cho đúng
- Cột "heldout loss" đo trên corpus cũ (`data/eval/heldout.txt`) → dùng để trả lời "đổi model thì mất gì so với bản đang chạy", **không** dùng để xếp hạng chất lượng corpus general_v1 của v4.
- Val loss trên Kaggle (1.4165 / 1.5842 / 1.5870...) đo trên val set khác → không so ngang cột heldout local.
- **Tiếng Việt: 0/22 pass ở mọi model; lp_pass_rate = 0.591 giống hệt nhau ở cả 4 model + baseline** → nghi vấn chỉ số VI của probe không phân biệt được model; cần soi lại cách tính trước khi kết luận "model dở tiếng Việt".

## Kết luận
1. v4 pretraining là tiến bộ thật: heldout 5.6904 → 5.3887 (ppl 296 → 219) trên cùng bộ heldout.
2. SFT là bước "mở khoá" năng lực: probes 0.03 → 0.145, fluency BROKEN → OK.
3. Ứng viên thay model serving: **v4 instruct_best (iter 57,400)** — hơn bản đang chạy +83.5% probes, forgetting 0, tốc độ tương đương. Không ghi đè `checkpoints/jingpt_chat.pt`.
4. Trước khi promote: chat thử vài câu bằng tay để kiểm tra cảm giác thực tế, vì probe test vẫn chỉ là proxy.
