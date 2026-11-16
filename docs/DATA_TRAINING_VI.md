# Kho dữ liệu huấn luyện JIN — hướng dẫn sử dụng và mở rộng

**Ổ đĩa:** mọi dữ liệu/cache mặc định ở project trên D. Dùng `python scripts/python_d.py <script> ...` để cả cache thư viện và file tạm của tiến trình đều nhận môi trường trước imports. Xem [hướng dẫn lưu trữ D](STORAGE_D_VI.md); các đầu ra lớn sang ổ khác bị từ chối trên Windows.

## 1. Kết quả đã làm thật

Không phải database SQL: model cần **corpus văn bản** để pretrain và **hội thoại JSONL** để instruction-tune (SFT).

- Corpus cũ `data/train.txt`: **5.305.760.075 bytes (~5,31 GB)**, không bị ghi đè.
- Kho HF đã chuyển đổi: khoảng **5,42 GB văn bản** và **2,27 GB hội thoại**; đây chưa phải dung lượng duy nhất sau dedupe.
- Inventory cộng cả dữ liệu chuyển đổi, sách và bản cache: **14.860.603.909 bytes**. Không gọi con số này là 14,86 GB dữ liệu huấn luyện độc lập.
- Hai parquet mới tải đủ: **363.186.828 bytes (~363 MB)**; đã xuất **10.000 Smol-SmolTalk + 10.000 OpenCodeInstruct** hội thoại.
- Web samples mới: **100 tài liệu FineWeb2 tiếng Việt + 300 FineWeb-Edu tiếng Anh**. Đây là mẫu nhỏ, **chưa tải toàn bộ web corpus**.
- Bộ được khuyến nghị: [data/prepared_v4](../data/prepared_v4/manifest.json), **38.238.712 bytes pretrain kể cả validation**, **17.086 hội thoại SFT train + 914 validation**.
- [Verification](../data/prepared_v4/verification.json): 46.143 hash nội dung pretrain train / 1.245 validation; giao tập hash bằng 0. Chưa chứng minh các bản diễn đạt lại hoặc họ bài toán tương tự không rò rỉ.
- Chưa chạy huấn luyện dài, chưa thay weights phục vụ, chưa có tuyên bố model thông minh hơn.

Báo cáo: [inventory](../bench_results/data_inventory.json), [download plan](../bench_results/data_download_plan.json), [test results](../bench_results/data_pipeline_tests.json).

**Không dùng các bản thử `prepared_v1`, `prepared_v2`, `prepared_v3`:** v2 bị ngắt giữa chừng; v1/v3 đã được thay thế bởi v4 có xử lý ranh giới tài liệu tốt hơn. Giữ nguyên chúng để kiểm tra, không tự xóa.

## 2. Nguồn dữ liệu và công dụng

| Nhóm | Dataset | Công dụng / lưu ý |
|---|---|---|
| Tiếng Việt | Wikipedia VI, 4 shard đã có (~1,52 GB text) | Bách khoa, đọc parquet gốc để giữ ranh giới bài viết |
| Tiếng Việt web | [FineWeb2 — vie_Latn](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | Mới có 100 mẫu; cần lọc nhiễu/PII, không xem mọi bài web là sự thật |
| Giáo dục EN | Cosmopedia v2 shard đã có (~1,42 GB text) | Nội dung tổng hợp; không đảm bảo mọi đáp án chính xác |
| Giáo dục EN | [SmolLM FineWeb-Edu dedup](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | Mới có 300 mẫu, nguồn lớn để mở rộng |
| Kiến thức EN | Simple Wikipedia (~274,5 MB text) | Văn bản dễ đọc, kiến thức tổng quát |
| Hội thoại | [Smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) | Đã tải 1 parquet, xuất 10k hội thoại; ưu tiên hội thoại ngắn cho model nhỏ |
| Hội thoại | Dolly 15k, OpenAssistant OASST2 | Đã có; OASST đa ngôn ngữ, chưa đo/cân bằng tỷ lệ tiếng Việt |
| Toán | GSM8K train, MetaMathQA | Đã có; chỉ lấy TRAIN; bài toán diễn đạt lại có thể liên quan nhau |
| Code | [OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) | Đã tải 1 parquet, xuất 10k cặp; lọc `average_test_score >= 1` của upstream, **không chạy code tải về** |
| Trôi chảy EN | TinyStories v2 (~2,21 GB text) | Chỉ 2% trọng số đề xuất, tránh để truyện lấn át kiến thức/VI |
| VI SFT nghiên cứu | Bactrian-X VI đã có (~67,6 MB JSONL) | NC: **không đưa vào bộ mặc định**, chưa bảo đảm câu trả lời đúng |

Catalog [hf_datasets.json](../hf_datasets.json) có 20 entries. Bốn nguồn mới mặc định `enabled:false`: bật bằng `--only` để không bất ngờ tải hàng GB.

### Giấy phép không đồng nghĩa “chắc chắn dùng thương mại được”

- FineWeb2 / SmolLM corpus: ODC-BY-1.0, giữ attribution và kiểm tra quyền nội dung gốc.
- Smol-SmolTalk: card Apache-2.0, vẫn phải kiểm tra constituent datasets.
- OpenCodeInstruct: CC-BY-4.0, giữ attribution.
- GSM8K / MetaMathQA: MIT theo catalog; cần kiểm tra upstream cụ thể khi phát hành.
- Wikipedia / Dolly: có nghĩa vụ attribution/share-alike; xác minh revision/điều khoản trước phân phối.
- TinyStories: CDLA-Sharing-1.0.
- Alpaca / CodeAlpaca / Bactrian-X: **non-commercial**, chỉ thêm khi chủ động chọn profile nghiên cứu tương ứng.
- OpenHermes: giấy phép tùy từng nguồn; **loại khỏi bộ mặc định**.

Đây là bộ có chính sách loại NC/giấy phép hỗn hợp chưa rõ, **không phải chứng nhận pháp lý thương mại**.

## 3. Bộ trộn đã chuẩn bị

[Mixture](../data/prepared_v4/mixture.json) lấy mẫu theo nhóm, không theo kích thước dump:

| Nhóm | Trọng số khởi đầu | Train bytes |
|---|---:|---:|
| Vietnamese | 35% | 7.070.055 |
| Educational EN | 30% | 5.565.083 |
| Knowledge EN | 15% | 8.233.157 |
| Code | 10% | 4.542.734 |
| Math | 8% | 3.609.819 |
| Stories | 2% | 8.220.969 |

Đây là **recipe khởi đầu**, chưa được tối ưu bằng thử nghiệm. Math/code pretrain dùng lại cặp SFT TRAIN; phần val tương ứng chỉ đưa vào validation. Hai giai đoạn không tạo ra hai kho nội dung độc lập.

Pipeline có:
- Giữ whole document từ parquet/EOT; bỏ record quá 100k ký tự thay vì chia train/val theo đoạn.
- Dedupe nguyên bản sau chuẩn hóa văn bản; SFT dedupe theo hội thoại.
- Chặn ngram từ probes và heldout cố định. Heuristic không chứng minh sạch toàn bộ benchmark trên thế giới.
- Kiểm tra encoding, độ dài, nội dung ít thông tin, lặp dòng, một số instruction-injection; SFT còn lọc email/khóa bí mật dễ nhận biết.
- **Chưa có PII detector toàn diện**, language balancing hay semantic/family dedupe.
- Pretrain là prefix có ngân sách từng nguồn; SFT reservoir có seed trong tối đa 30.000 dòng mỗi dataset, **không phải mẫu ngẫu nhiên toàn corpus**.
- Manifest `building` khi bắt đầu, chỉ `ready` khi xong; MixtureSampler từ chối manifest chưa sẵn sàng, file sai hash và cache override dùng chung cho nhiều domain.
- SHA256 đầu ra, nguồn/giấy phép và revision nếu có. Viewer samples là trạng thái hiện tại, không pin được revision; các nguồn legacy chưa rõ revision.
- Không tự lấy logs riêng tư, không thực thi lời giải/code tải về.

## 4. Kiểm tra và mở rộng từ kho đã có

Chạy tại root sau khi activate `.venv` (Git Bash cũng có thể dùng `./.venv/Scripts/python.exe` thay `python`). Máy hiện tại đã có pyarrow cho parquet.

```bash
python data_pipeline.py --inventory
python data_pipeline.py --verify --output-dir data/prepared_v4

# Batch lớn hơn, KHÔNG ghi đè v4. Có thể mất nhiều phút.
python data_pipeline.py --prepare --output-dir data/prepared_128mb --mb-per-domain 128 --sft-per-source 10000
python data_pipeline.py --verify --output-dir data/prepared_128mb
```

`--mb-per-domain` là cap MB nhị phân, không đảm bảo nhóm nào cũng có đủ lượng đó. Web samples nhỏ nên chưa có nhiều dữ liệu web mới dù tăng cap. SFT cap gộp các shard cùng dataset, không ghi đè shard trước.

Nếu chủ động chấp nhận NC cho mục đích phù hợp, tạo **thư mục khác**, không thay bộ mặc định:

```bash
python data_pipeline.py --prepare --output-dir data/prepared_research_nc --mb-per-domain 128 --sft-per-source 10000 --allow-noncommercial
python data_pipeline.py --verify --output-dir data/prepared_research_nc
```

Lệnh này thêm Bactrian-X VI, Alpaca, CodeAlpaca theo chính sách NC; **không** biến chúng thành nguồn được phép dùng thương mại. Bộ mặc định hiện chưa có nguồn VI SFT lớn, chuyên biệt.

## 5. Tải thêm thật nhiều nhưng có giới hạn

Xem kế hoạch trước; `--plan` chỉ lấy metadata, không tải payload:

```bash
python fetch_datasets.py --catalog hf_datasets.json --only fineweb2-vi fineweb-edu-dedup smol-smoltalk opencodeinstruct --max-shards 1 --plan
```

Theo metadata đã kiểm tra, shard đầu FineWeb2 VI khoảng **4,84 GB**, FineWeb-Edu **2,40 GB**. Vì vậy `--max-output-mb 128` **không** khiến parquet tải về chỉ còn 128 MB: vẫn cần tải đủ shard, rồi mới chuyển đổi một phần.

### Web: lấy mẫu nhỏ trước

```bash
# 1k train rows mỗi nguồn, ngân sách network 128 MiB cho cả lệnh.
python fetch_datasets.py --catalog hf_datasets.json --only fineweb2-vi fineweb-edu-dedup --sample-rows 1000 --total-download-mb 128 --max-output-mb 32
```

Viewer có thể trả 429/502; nếu vẫn lỗi thì không coi là tải thành công. File `.part` cũ được giữ để xem xét. Đừng xóa file không rõ chủ sở hữu; chọn tên/batch mới nếu cần. Samples hiện có không bị ghi đè.

### Khi chấp nhận tải khoảng 7,24 GB parquet web

```bash
python fetch_datasets.py --catalog hf_datasets.json --only fineweb2-vi fineweb-edu-dedup --max-shards 1 --total-download-mb 8000 --max-output-mb 256
```

Ngân sách 8.000 MiB không phải lượng bắt buộc tải. Cần thêm chỗ cho text, prepared data và token cache. Sau khi tải, tạo một phiên bản prepared mới và verify như mục 4. Không chạy lệnh này tự động trong phiên vừa rồi.

### Tăng hội thoại/code

```bash
# Đã có cache shard đầu: xuất nhiều hàng hơn sang seed-dir mới, không ghi đè JSONL cũ.
python fetch_datasets.py --catalog hf_datasets.json --only smol-smoltalk opencodeinstruct --max-shards 1 --max-rows 30000 --seed-dir data/seed/hf_expanded --total-download-mb 512
```

`data_pipeline.py` tự đọc `data/seed/hf`, **không tự đọc hf_expanded**. Có thể dùng thư mục mở rộng trực tiếp bằng `train.py --hf-seed-dir`, nhưng phải tạo/filter split riêng trước; không dùng chung file train/test. Muốn pipeline tự nhận nhiều shard, tải các shard mới vào thư mục HF mặc định; output đã tồn tại sẽ được giữ/bỏ qua, shard mới được gộp theo dataset. Quá trình prepare hiện giới hạn scan 30k dòng/dataset nên cần nâng ngân sách scan và thêm sampling toàn nguồn trước khi sử dụng hàng triệu hội thoại.

Tải fullfiles pin revision được resolve tại thời điểm chạy và lưu SHA256. Ngân sách download vượt giới hạn sẽ **báo lỗi**, không xuất JSON/parquet cụt giả như hoàn chỉnh. Catalog legacy vẫn có nguồn enabled; **luôn chỉ định `--only`**, đừng tải cả catalog vô tình.

## 6. Đưa bộ dữ liệu vào trainer

Không cần dựng lại `data/train.txt` hay chạy lại BPE để dùng prepared mixture. **Khi tiếp tục weights cũ phải giữ tokenizer tương thích**; xây tokenizer mới là một nhánh huấn luyện khác.

- Pretrain: `--mixture data/prepared_v4/mixture.json` tự dùng validation của mixture, không dùng tail corpus 5,31 GB cũ.
- Mixture v4 lưu đường dẫn tuyệt đối của máy hiện tại. Khi chuyển lên Kaggle/máy khác, prepare lại tại workspace đích hoặc tái pin cả mixture/manifest; không chỉ sửa đường dẫn rồi bỏ qua hash checks.
- Instruct: `--hf-seed-dir data/prepared_v4/sft_train --sft-val-dir data/prepared_v4/sft_val` dùng split riêng đã kiểm tra, không random-split lại.
- `--max-convs 0` đọc toàn bộ HF prepared. Seed cục bộ vẫn được thêm; chọn `--duplication 1` ban đầu để không nhân seed quá mạnh. `--grounded-dir ""` tắt dữ liệu grounded ngoài bộ này.
- Seed hoặc nguồn train khác trùng tập SFT val sẽ bị trainer từ chối. Trainer có thể lọc thêm nên count thực tế phải đọc log/manifests, không giả định bằng đúng 17.086.
- SFT: chỉ khởi tạo từ checkpoint đã chọn/đo và lưu tên mới bằng `--checkpoint`; tránh mặc định ghi đè model chat đang phục vụ.
- Pretrain mới từ đầu và continued pretrain là hai việc khác nhau: hiện pretrain tiếp weights dùng `--resume` với bản checkpoint ở đường dẫn run riêng. Không chạy scratch rồi gọi là “train tiếp”.

Ví dụ **chỉ để người dùng chủ động chạy**, chưa được chạy trong phiên này:

```bash
# SFT trên checkpoint pretrain đã chọn; output phải là đường dẫn mới.
python train.py --mode instruct --init-from checkpoints/jingpt_best.pt --checkpoint checkpoints/jingpt_data_v4.pt --hf-seed-dir data/prepared_v4/sft_train --sft-val-dir data/prepared_v4/sft_val --grounded-dir "" --max-convs 0 --duplication 1 --device cuda --low-vram --batch-size 1 --grad-accum 4 --block-size 256 --lr 1e-5 --steps 200
```

200 steps chỉ là đợt kiểm tra ban đầu, không lời hứa đủ học. RTX2060 6GB có thể train chậm/lỗi CUDA tùy runtime; ưu tiên smoke-test nhỏ, không tự hạ sang CPU cho run dài. Đo frozen benchmark trước/sau, heldout loss, tiếng Việt, toán và code; không lấy train loss làm bằng chứng thông minh.

## 7. Tests

```bash
python scripts/check_project.py --report bench_results/data_pipeline_tests.json
```

Đợt cuối: **257 tests pass**, 62 file Python compile thành công. Không có static typechecker cấu hình; không tuyên bố đã static-typecheck. Preset model nặng được test hiện có bỏ qua nội bộ để tránh hết RAM. Tests mới bao gồm ngân sách/truncation download, giữ partial, pin train shards, parquet quality filter, JSON gzip streaming, giữ ranh giới EOT, gộp nhiều SFT shard, provenance/hash, chặn preparation chưa xong, và trainer validation SFT riêng.
