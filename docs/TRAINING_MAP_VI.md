# JIN — tải bản 60k, thống nhất model và học tiếp

**Hướng dẫn đã chia theo loại:** [01 Pretrain](training/01_PRETRAIN.md), [02 SFT/Chat](training/02_SFT.md), [03 Preference](training/03_PREFERENCE.md). [Mục lục test/training](training/README.md) và [dữ liệu tổng quát 70/30](training/DATA_GENERALIST.md).

## 1. Việc làm ngay: tải đúng Output về D

Theo log Kaggle người dùng cung cấp, lượt SFT v4 đã hoàn tất ở **step 60.020** (20 smoke + 60.000 bước tiếp tục). Validation cuối **1,5870**, tốt nhất **1,5842**; đồng hồ notebook cuối khoảng **8 giờ 42 phút 40 giây**. Hai checkpoint đã được tải về D và **đối chiếu SHA-256 khớp log** (2026-10-01); best = **step 57.400** theo manifest của `_best.pt`. Chưa có benchmark năng lực độc lập.

1. Mở **version vừa chạy** của notebook Kaggle, kiểm tra trạng thái hoàn tất.
2. Mở tab **Output**, không lấy checkpoint parent ở tab Input.
3. Mở `jin/checkpoints/`, tải các file dưới đây. Nếu có menu ba chấm cạnh file, chọn Download; tên nút có thể thay đổi theo giao diện.
4. Bật tùy chọn trình duyệt hỏi nơi lưu mỗi file; lưu vào **thư mục mới trên D**:

```text
D:\Code folder\Jin New\checkpoints\kaggle_v4_60020\
```

Giữ nguyên tên file, không chép đè [jingpt_chat.pt](../checkpoints/jingpt_chat.pt). Không bấm **Save & Run All** để tải output: thao tác đó có thể chạy lại toàn bộ training. Nếu chưa thấy Output được lưu, xác minh trạng thái/version trong giao diện trước khi restart session hoặc chạy lại.

| File trong Output | Tải để làm gì? |
|---|---|
| `jin/checkpoints/kaggle_v4_instruct_best.pt` | Ứng viên đánh giá/chat; tốt nhất theo validation SFT của run này, chưa chắc tốt nhất trên mọi tác vụ |
| `jin/checkpoints/kaggle_v4_instruct.pt` | Bản cuối step 60.020; giữ để tiếp training cùng run/recipe |
| `jin/checkpoints/kaggle_v4_instruct_best.pt.manifest.json` | Hash, metadata và parent/run của bản best |
| `jin/checkpoints/kaggle_v4_instruct.pt.manifest.json` | Hash, metadata và parent/run của bản cuối |
| `jin/output_index.json` | Danh sách kích thước/hash hai checkpoint |

Năm file trên là bộ tối thiểu cần giữ (các manifest nếu có trong Output). Khi tải riêng, có thể đặt cả năm cùng thư mục đích. Giữ thêm nếu có: `jin/runs/*.json`, `jin/checkpoints/*.launch.json`, `jin/smoke_run.json`, `jin/continued_run.json`, `jin/bpe.json`, notebook và log. Các bản ghi này giúp lần ngược nguồn gốc run. Không cần tải cache token hay cả corpus để chat.

Hai checkpoint mỗi file **1.774.134.089 byte**; tổng khoảng **3,55 GB** (3,30 GiB), chưa tính file phụ. Kích thước không đủ chứng minh file đúng: phải đối chiếu hash.

| Checkpoint | SHA-256 từ log |
|---|---|
| `kaggle_v4_instruct.pt` | `fe9fb2b9554fab0a398e71642f1633f463094244db2490cc708afcc0efb060b7` |
| `kaggle_v4_instruct_best.pt` | `1e91a9283ec835bf6049afbf95269ce95d07c2073e3cb47a7ebb3c3dcd9b1e99` |

**Trạng thái: đã tải xong và đối chiếu PASS (2026-10-01)** — `checkpoints/kaggle_v4_60020/` hiện có 2 `.pt` + 2 `.manifest.json` + `output_index.json`; cả hai file đúng 1.774.134.089 byte. Có thể chạy lại đoạn kiểm tra dưới bất cứ lúc nào.

Sau khi tải riêng hai checkpoint vào thư mục trên, mở **PowerShell** và chạy đoạn này. Chỉ đọc/hash file, không nạp model vào RAM/GPU và không sửa checkpoint:

```powershell
$dir = 'D:\Code folder\Jin New\checkpoints\kaggle_v4_60020'
$expected = @{
    'kaggle_v4_instruct.pt' = 'fe9fb2b9554fab0a398e71642f1633f463094244db2490cc708afcc0efb060b7'
    'kaggle_v4_instruct_best.pt' = '1e91a9283ec835bf6049afbf95269ce95d07c2073e3cb47a7ebb3c3dcd9b1e99'
}
foreach ($name in $expected.Keys) {
    $path = Join-Path $dir $name
    $file = Get-Item -LiteralPath $path -ErrorAction Stop
    if ($file.Length -ne 1774134089) { throw "Sai kich thuoc: $name" }
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256 -ErrorAction Stop).Hash
    if ($hash -ne $expected[$name]) { throw "Sai SHA256: $name" }
    Write-Host "PASS: $name"
}
```

Nếu tải nguyên ZIP, giải nén **trên D** và dùng đường dẫn thực tế tới hai checkpoint; không giả định cấu trúc ZIP phẳng. [Công cụ tải streaming](../scripts/fetch_kernel_output.py) có thể dùng khi biết đúng `owner/slug` notebook; pattern mặc định của công cụ dành cho tên checkpoint cũ, không dùng nguyên mặc định để tải v4.

## 2. Ba lượt train không phải ba mảnh cần ghép

Nếu lịch sử thực sự là nối tiếp weights:

```text
pretrain A → SFT/chat B khởi tạo từ A → SFT v4 C khởi tạo từ B
```

thì **C đã bắt đầu từ weights của B, còn B bắt đầu từ A**. Khi chat chỉ nạp **một checkpoint C**; không cần nạp cả A/B/C hay cộng trọng số. Việc học tiếp vẫn có thể làm quên khả năng cũ, nên kế thừa weights không đồng nghĩa giữ mọi kiến thức một cách hoàn hảo.

**Đã xác nhận cho v4 qua setup/log:** notebook nạp weights của `jingpt_chat.pt` (parent iteration 4000), tạo optimizer mới cho smoke, sau đó resume checkpoint smoke để chạy 60k. Reset optimizer/step không xóa weights đã học. Cạnh `jingpt_chat.pt → v4` có bằng chứng; nguồn gốc của các checkpoint đời trước chưa được xác minh đầy đủ.

Không khẳng định các file `jingpt_chat_1600/2000/2400`, `jingpt_chat_46k_sft4k`, `jingpt_latest` là cùng một dòng chỉ vì tên, ngày hoặc dung lượng. Cần manifest/run records, parent hash, tokenizer và model config; nếu bản cũ không có provenance thì phải ghi **ancestry unknown**, không tự dựng lịch sử.

Nếu ba lượt là **ba nhánh độc lập**, không có thao tác ghép bảo đảm lấy toàn bộ ưu điểm:

- Mặc định: chọn nhánh tốt nhất qua benchmark, rồi học tiếp bằng dữ liệu mới + dữ liệu cũ đã lọc để hạn chế quên.
- Nếu các nhánh có khả năng bổ sung nhau: có thể thử distillation từ các câu trả lời được kiểm chứng vào một model học trò; cần đánh giá, không dùng câu trả lời sai làm nhãn.
- Weight merging chỉ là thí nghiệm sau khi kiểm tra tensor layout/kiến trúc, ánh xạ tokenizer, nguồn khởi tạo và độ tương thích. Cùng số tham số không đủ; lấy trung bình có thể làm model kém đi. Luôn giữ bản gốc và đánh giá bản ghép riêng.

**Không merge best + latest:** đây là hai snapshot của cùng lượt v4, không phải hai bộ kiến thức rời nhau.

## 3. Một model để dùng, checkpoint training để phát triển

Quy trình chọn bản phục vụ:

1. Hash đúng → đọc metadata/manifest từng bản. Best step chưa được biết từ đoạn log cuối.
2. So sánh parent chat, v4 best và v4 latest trên cùng eval/probes, tokenizer, context và thiết lập generation. Chạy từng checkpoint một để giảm RAM/VRAM.
3. Xem cả điểm tự động lẫn câu trả lời tiếng Việt, kiến thức, toán/code và làm theo yêu cầu. Validation v4 dùng `eval-iters=5`; danh hiệu best không thay cho benchmark rộng hơn.
4. Chọn **một bản vượt tiêu chí chất lượng/không thoái lui quá mức**, không mặc định bản cuối hoặc bản best là người thắng.
5. Có thể export bản thắng thành inference artifact tên mới; vẫn giữ checkpoint training nguyên vẹn để phát triển/rollback.

[bench.py](../bench.py) hỗ trợ `--checkpoint`, `--device cuda`, `--no-train-bench` và `--compare A.json B.json`. Dùng bộ eval hiện có; **không chạy `--init-eval` chỉ để thử model** vì thao tác đó sửa corpus. [fluency_check.py](../fluency_check.py) chỉ là heuristic thiên về tiếng Anh, không dùng verdict của nó để khẳng định model thông minh hoặc giỏi tiếng Việt.

Ví dụ **chỉ sau khi best được kiểm tra/chọn**, chạy trong PowerShell ở project root:

```powershell
cd 'D:\Code folder\Jin New'
.\.venv\Scripts\python.exe lineage.py --export-inference 'checkpoints/kaggle_v4_60020/kaggle_v4_instruct_best.pt' 'checkpoints/jin_v4_serving.pt'
.\.venv\Scripts\python.exe chat.py --checkpoint 'checkpoints/jin_v4_serving.pt' --device cuda
```

Nếu latest thắng, đổi nguồn export sang latest. [lineage.py](../lineage.py) từ chối ghi đè destination đã tồn tại; dùng tên version mới khi nâng cấp. Export giữ weights/config/tokenizer, bỏ optimizer để file nhẹ hơn; **không làm model thông minh hơn**, không phải phép merge. [chat.py](../chat.py) có thể nạp thẳng training checkpoint nếu chưa export.

Để chọn cùng một bản cho app và các entry point đọc config, đặt biến môi trường **trong PowerShell hiện tại** trước khi khởi động chương trình:

```powershell
$env:JINGPT_CHECKPOINT = 'D:\Code folder\Jin New\checkpoints\jin_v4_serving.pt'
```

[config.py](../config.py) hỗ trợ override này. Không đổi default bằng cách chép đè model cũ; luôn kiểm tra log `Loaded` cho đúng đường dẫn. Inference artifact dùng để chat hoặc warm-start weights; không dùng như full optimizer resume checkpoint.

## 4. Học tiếp để tăng khả năng, không chỉ tăng step

Sau benchmark mới chọn đợt tiếp theo:

- **Thiếu kiến thức/ngôn ngữ:** thử continued pretraining trên corpus sạch, đa dạng, cân bằng; có thể khởi đầu từ weights của nhánh đã chọn. Nó có thể ảnh hưởng khả năng chat, nên đo lại và SFT phục hồi nếu cần. Notebook pretrain hiện tại yêu cầu optimizer tương thích để resume; không đổi `MODE` rồi bỏ qua checks.
- **Biết nhưng trả lời sai dạng/tiếng Việt yếu:** mở rộng SFT có câu trả lời đúng và ví dụ làm theo yêu cầu. SFT cũng có thể dạy kiến thức, không chỉ style. Ưu tiên lỗi được benchmark/chat thực tế chỉ ra.
- **Quên tác vụ cũ:** thêm dữ liệu replay đã lọc từ các tác vụ cũ; tách train/validation/eval và kiểm tra overlap. Chọn tỷ lệ qua thử nghiệm, không cố định một tỷ lệ là tối ưu.
- **Cần kiến thức mới cập nhật thường xuyên:** retrieval/RAG có thể bổ sung thông tin có nguồn mà không phải train lại cho từng thay đổi; đây là khả năng của hệ thống, không phải bằng chứng weights đã học kiến thức đó.

Mỗi đợt: pin parent hash + tokenizer/config + data/version → smoke ngắn → train có ngân sách → benchmark đối chiếu → promote hoặc giữ bản đang dùng. Không cần lặp đủ ba phase ở mọi lần, không có số vòng/step bảo đảm thông minh hơn. Chưa nên chạy thêm 60k trên cùng bộ chỉ vì lượt trước xong.

**Warm start:** nạp weights, optimizer mới, bước của run mới. **Resume:** phục hồi weights + optimizer/scaler và bước; kiểm tra trạng thái tương thích. Cùng phase không tự bảo đảm resume hợp lệ; đổi mục tiêu/dataset/recipe cần xem lại optimizer, scheduler và validation. Trainer hiện tính lại lịch LR theo target của run mới, không hứa tái hiện bit-for-bit qua nhiều phiên.

Số epoch cần số **packed train blocks thực tế trên Kaggle**, batch và accumulation; không suy ra chính xác từ số hội thoại. Số đo local trước đây: 34.563 blocks / (batch 1 × accumulation 4) ≈ 8.641 updates mỗi lượt (bỏ qua rounding). Đó không phải phép đo chính xác cho dữ liệu Kaggle vì seeds khác nhau. Khoảng 7 lượt cho 60k chỉ là ước lượng theo cấu hình local; nhiều lượt là rủi ro cần đo, không phải chứng minh overfit. Validation giảm cũng chưa đủ chứng minh tổng quát hóa.

**Bước kế tiếp cụ thể:** tải năm file ở mục 1 về D, gửi thư mục tải cùng tên ba checkpoint/lượt mà bạn muốn thống nhất. Khi có file mới xác minh hash, xem ancestry và đánh giá trước khi chọn một bản phục vụ.
