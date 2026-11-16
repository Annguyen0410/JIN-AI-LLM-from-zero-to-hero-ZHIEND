# 02 — SFT / instruction tuning / chat

[Quay về mục lục](README.md) · [Dữ liệu tổng quát](DATA_GENERALIST.md)

**Học gì:** câu trả lời đúng, hội thoại, làm theo yêu cầu, trình bày lời giải; có thể dạy kiến thức. **Chat đời trước và v4 60k đều thuộc loại này.** Không cần merge snapshots để gom kiến thức.

**Dữ liệu:** JSONL UTF-8:

```json
{"messages":[{"role":"user","content":"Câu hỏi có nguồn/đã kiểm tra"},{"role":"assistant","content":"Đáp án đúng đã kiểm tra"}],"provenance":{"dataset_id":"nguon-thuc","license":"xem dieu khoan"}}
```

Ví dụ là schema, không phải bộ train tạo bằng AI. Tách cả hội thoại theo train/val trước packing; cùng bài gốc/diễn đạt lại cần giữ cùng split khi có family ID. Không train vào eval/probes hoặc đổ logs riêng tư vào seed.

## Tình trạng checkpoint v4

- Parent đã nạp là `jingpt_chat.pt`, warm-start weights, optimizer mới.
- Smoke 20 → resume thêm 60k → latest step 60.020; best validation 1,5842 theo log.
- Hai checkpoint đã tải, byte/SHA256 khớp log. Best step/benchmark rộng chưa được xác minh ở công việc sắp xếp này.
- Giữ nguyên [thư mục v4](../../checkpoints/kaggle_v4_60020), không di chuyển GB hoặc promote tự động.

## Lượt tiếp theo

1. Chọn parent sau benchmark chat cũ/best/latest, không mặc định best thắng mọi tác vụ.
2. Lọc/trộn dataset mới (người/AI có provenance), thêm replay tác vụ cũ, verify train/val.
3. Tạo output mới và warm-start cho run/recipe mới. Resume chỉ khi thực sự tiếp cùng trạng thái đã xác nhận.
4. Smoke → run có ngân sách → benchmark → promote khi vượt tiêu chí.

Ví dụ lệnh **trên Kaggle sau setup**, không chạy ở đây; thay đường dẫn theo bản đã chọn:

```bash
python train.py --mode instruct \
  --init-from checkpoints/PARENT.pt \
  --checkpoint checkpoints/sft/general_v1/smoke.pt \
  --hf-seed-dir data/prepared_general_v1/sft_train \
  --sft-val-dir data/prepared_general_v1/sft_val \
  --grounded-dir '' --max-convs 0 --duplication 1 \
  --device cuda --optimizer adamw --batch-size 1 --grad-accum 4 \
  --block-size 256 --lr 1e-5 --warmup 20 --steps 20 \
  --eval-interval 20 --eval-iters 20 --checkpoint-interval 20
```

Output phải mới; direct trainer không có guard output như launcher. Ưu tiên [launcher preflight](../../scripts/run_training.py)/[notebook](../JIN_KAGGLE_V4.ipynb) để pin parent/tokenizer. Packer hỗ trợ `--prepared` + `--checkpoint`, chọn upload directory mới. Seed local vẫn thêm vào trainer; log count/packed blocks là số thật, không giả định bằng count prepared.

**SFT không tự nạp cả corpus raw mới tải.** Pretrain là một đợt riêng, nếu benchmark cho thấy cần bổ sung nền.

## Test / đo

- `check_project.py --group sft --group shared --skip-preset-shapes`: template, masking, dataset/validation và hạ tầng.
- Validation assistant-token loss trên split đó; best của `eval-iters=5` chỉ nhìn một phần nhỏ.
- `bench.py --checkpoint ... --device cuda --no-train-bench`: cùng heldout/probes và generation settings; xem từng câu VI/toán/code.
- Fluency heuristic thiên về EN, không đủ để nói model “thông minh”. Tools/calculator trong app cũng không phải năng lực weights tự trả lời.
