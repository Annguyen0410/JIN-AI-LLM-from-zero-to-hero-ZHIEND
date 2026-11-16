# Trung tâm training JIN — đọc từ đây

**Ba loại train, một checkpoint dùng chính.** Không cần chạy cả ba ở mỗi đợt. V4 là phiên bản dữ liệu/run SFT, không phải thuật toán thứ tư. Tất cả đường dẫn tương đối bên dưới tính từ project root trên D.

| Mục | Mục đích | Trainer | Hướng dẫn |
|---|---|---|---|
| **01 — Pretrain** | Học ngôn ngữ/kiến thức/mẫu văn bản và code | `train.py --mode pretrain` | [01_PRETRAIN.md](01_PRETRAIN.md) |
| **02 — SFT / Chat** | Học trả lời, giải bài, làm theo yêu cầu; cũng học kiến thức | `train.py --mode instruct` | [02_SFT.md](02_SFT.md) |
| **03 — Preference** | Ưu tiên đáp án tốt hơn trong các cặp so sánh | `train_pref.py` (thử nghiệm DPO/ORPO) | [03_PREFERENCE.md](03_PREFERENCE.md) |
| **Dữ liệu tổng quát** | Nguồn người/AI, tải có ngân sách, chất lượng và trộn | không tự train | [DATA_GENERALIST.md](DATA_GENERALIST.md) |
| **Checkpoint hiện có** | Parent, best/latest, export phục vụ | không merge mù | [bản đồ checkpoint](../TRAINING_MAP_VI.md) |

## Phân biệt ba nghĩa của “test”

1. **Test phần mềm:** mã có chạy đúng không; không cho biết weights thông minh hơn.
2. **Validation trong training:** theo dõi loss trên split chưa học; chỉ so loss cùng mục tiêu/dataset/recipe. Best SFT không phải best mọi tác vụ.
3. **Benchmark năng lực:** cùng bộ eval/probes và setting, so parent/candidate sau train. `bench.py` đo tác vụ model, còn `scripts/system_eval.py` đo hạ tầng/tools, không thay thế nhau.

## Test phần mềm đã chia nhóm

Không di chuyển test cũ, không sửa đường dẫn import. [check_project.py](../../scripts/check_project.py) hỗ trợ `--group` lặp lại; mặc định vẫn chạy **all**. Một test nằm đúng một nhóm; các test data/tokenizer/model/Kaggle dùng chung ở `shared` vì phục vụ nhiều loại train.

```powershell
cd 'D:\Code folder\Jin New'
# CPU + không để test GPU làm ảnh hưởng máy đang chặt RAM.
$env:JINGPT_DEVICE = 'cpu'
$env:CUDA_VISIBLE_DEVICES = ''
.\.venv\Scripts\python.exe scripts/check_project.py --list-groups
.\.venv\Scripts\python.exe scripts/check_project.py --group pretrain --skip-preset-shapes --report bench_results/training/pretrain_tests.json
.\.venv\Scripts\python.exe scripts/check_project.py --group sft --skip-preset-shapes --report bench_results/training/sft_tests.json
.\.venv\Scripts\python.exe scripts/check_project.py --group preference --skip-preset-shapes --report bench_results/training/preference_tests.json
.\.venv\Scripts\python.exe scripts/check_project.py --group shared --skip-preset-shapes --report bench_results/training/shared_tests.json
# Full regression vẫn cần trước thay đổi quan trọng:
.\.venv\Scripts\python.exe scripts/check_project.py --skip-preset-shapes --report bench_results/training/all_tests.json
```

Test class thuộc phase3 cũ được chia theo chức năng, không gán cả file hỗn hợp vào một loại. Report ghi rõ nhóm, số test và skip. Mốc kiểm tra nhóm riêng: **Pretrain 17 / SFT 19 / Preference 3**; nhóm `shared` chứa hạ tầng dùng chung và có thể tăng theo test mới. [Báo cáo nhóm trainer](../../bench_results/training/trainer_groups_tests.json), [regression toàn dự án](../../bench_results/training/all_tests.json). Không có static typechecker cấu hình; compile cú pháp + unittest không được gọi là static typecheck. Chọn nhóm vẫn import suite hiện có để discovery lỗi không bị bỏ qua; không bảo đảm hoàn toàn không import các module ngoài nhóm.

**Trước chat/benchmark/training CUDA trong PowerShell đó:**

```powershell
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
$env:JINGPT_DEVICE = 'cuda'
```

`--skip-preset-shapes` chỉ bỏ test dựng preset nặng; CUDA smoke sẽ skip khi CUDA bị ẩn, không phải GPU đã được xác minh.

## Đặt tên/giữ file từ giờ

Không di chuyển các GB checkpoint hiện có; dùng tên/nhánh mới cho run tương lai:

```text
checkpoints/                 # Giữ checkpoint cũ nguyên chỗ
  kaggle_v4_60020/           # SFT v4 đã tải, không phải pretrain/preference
  pretrain/<run>/           # Run mới, khi chủ động tạo
  sft/<run>/
  preference/<run>/
bench_results/training/     # Test phần mềm theo nhóm
bench_results/              # Giữ benchmark lịch sử; xem checkpoint trong nội dung
runs/                       # Manifest gốc do trainer ghi; không đổi chỗ
```

Tên file chỉ gợi ý; **phase phải đọc từ `training_config.mode`/lệnh/manifest nếu có**. Cũ thiếu provenance = chưa xác minh, không xếp theo tên như bằng chứng. Hiện v4 đã xác nhận hash hai checkpoint; chưa có benchmark năng lực v4 mới trong phiên này.

## Thứ tự tiếp theo

1. Benchmark chat cũ/v4 best/v4 latest cùng setting, từng file một.
2. Hoàn tất tải, kiểm tra nguồn → lọc/giữ cả tài liệu → split/dedupe/contamination → pin manifest.
3. Chọn Pretrain hoặc SFT dựa lỗi thực tế. Không bấm train ngay với parquet mới tải.
4. Smoke 20 updates, đo VRAM/tốc độ, chọn ngân sách có biên session; không dùng projection 20 step làm cam kết cả đêm.
5. Benchmark đối chiếu rồi mới promote một bản phục vụ. Preference để sau khi có nhãn cặp chất lượng.

Mục tiêu là model nhỏ đa dụng tiếng Việt/EN/kiến thức/toán/code. Khoảng 148M tham số và 20 GiB tải **không bảo đảm đạt năng lực GPT-3**; phải đặt mục tiêu và đo từng tác vụ.
