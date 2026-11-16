# Dữ liệu tổng quát JIN — ngân sách 20 GiB, mục tiêu nguồn 70/30

[Quay về mục lục](README.md) · [Kho cũ và pipeline](../DATA_TRAINING_VI.md)

## Quyết định của người dùng

- Ưu tiên tiếng Việt, kiến thức rộng, toán/logic và Python, hướng model nhỏ đa dụng.
- **Không tạo thêm bộ dữ liệu bằng agent/AI trong phiên này.** Công cụ tạo template thử vừa thêm đã được bỏ; không chạy tạo corpus bằng nó.
- Tải dataset đã công bố trên mạng, mới tối đa **20 GiB** (21.474.836.480 byte).
- Mục tiêu **70% nguồn người có provenance + 30% nguồn AI có kiểm tra**. Đây là recipe thử nghiệm, không tỷ lệ đã chứng minh tối ưu.

## Tách nguồn và độ tin cậy

| Nguồn | Nhóm gốc | Vai trò | Giới hạn |
|---|---|---|---|
| Wikipedia VI/EN/Simple, snapshot 2023-11 | human_documented | Kiến thức/ngôn ngữ | Cộng đồng biên tập; không bảo đảm 100% từng đoạn do người viết hoặc đúng |
| Dolly 15k, OASST2 | human_documented | Hội thoại/làm theo yêu cầu | Review nhãn, nguồn và quyền sử dụng; OASST không bảo đảm mọi turn đúng |
| GSM8K train | human_documented | Toán có lời giải | Không đưa test split vào train; kiểm chứng nhãn, phân family |
| Cosmopedia v2 | synthetic_documented | Văn bản giáo dục tổng hợp | **Chưa fact-check từng thông tin ở local**, giữ nhóm riêng, không gọi là 30% đã chính xác |
| OpenCodeInstruct | synthetic_documented | Code/SFT | Chỉ lọc upstream full test score; không thực thi code tải về |
| Smol-SmolTalk | synthetic_or_mixed | Hội thoại | Phải review thành phần trước tính vào 70/30 |
| Web crawl/FineWeb | mixed_unknown | Mở rộng văn bản | Nguồn trang/copyright/PII/AI authorship khó xác minh; không tính là human chắc chắn |

Nguồn có uy tín và hash đúng là bằng chứng provenance/integrity, **không phải kiểm chứng sự thật**. Để phần AI đạt yêu cầu chính xác cần kiểm tra toán bằng reference, code bằng test an toàn, kiến thức đối chiếu nguồn đáng tin + người review/sampling. Không chạy untrusted code tải về bằng exec; chưa có cơ chế fact-check toàn corpus.

Upstream: [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia), [SmolLM corpus/Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus). Catalog giấy phép là ghi chú, không tư vấn/chứng nhận pháp lý. Wikipedia có attribution/share-alike; Cosmopedia/ODC-BY cần attribution và review quyền nội dung.

## Đợt tải thực tế

[Catalog riêng](download_catalog.json) + [downloader](../../scripts/download_training_data.py) không dùng mặc định enabled của catalog legacy. Nó resolve SHA revision, chọn shard **TRAIN**, kiểm kích thước/hash upstream, stream xuống D, không nạp payload vào RAM và không thay corpus/model.

Kế hoạch đã resolve:

- 41 shard Wikipedia EN 2023-11 + 5 shard Cosmopedia v2 **mới**.
- **Đã tải đủ 46/46 shard**, tổng **17.513.308.811 byte (~17,51 GB / 16,31 GiB)**, dưới 20 GiB. 41 Wikipedia = 11.630.929.031 byte; 5 Cosmopedia = 5.882.379.780 byte. Không còn partial và không lỗi tải tại lần kiểm tra hoàn tất. Từng shard khớp size + SHA256 upstream, chưa fact-check từng tài liệu.
- Complete shards không lấp đầy trần một cách chính xác; phần payload thực tế khoảng 66,4/33,6 theo bytes, không phải 70/30 training tokens.
- Không tải lại Cosmopedia shard đầu đã có; Wikipedia VI/Simple và dữ liệu SFT người tận dụng từ cache cũ.
- Ngân sách acquisition: 14 GiB human, 6 GiB synthetic. **Lượng thực tải không chính xác 70/30** do shard sizes/cached files, càng không phải tỷ lệ token training.

Đích:

```text
D:\Code folder\Jin New\data\downloads\general_v1\
  human_documented/wikipedia-en-20231101/*.parquet
  synthetic_documented/cosmopedia-v2-sample/*.parquet
  download_report.json
```

[Báo cáo tiến độ](../../data/downloads/general_v1/download_report.json) là nguồn trạng thái thật. Có thể job bị ngắt/timeout; file `.part` không được dùng cho train, không tự xóa partial để tải lại ngoài ngân sách. Lượt tải đang tiến hành/partial **không được gọi là toàn bộ dataset đã có**.

Kiểm tra tiến độ, không mạng:

```powershell
.\.venv\Scripts\python.exe scripts/python_d.py scripts/download_training_data.py --output-dir data/downloads/general_v1 --status
```

Khi không có downloader khác chạy và báo cáo không ghi lỗi/hash sai, tiếp **saved plan**, gồm file dở, theo batch có thời gian dừng:

```powershell
.\.venv\Scripts\python.exe scripts/python_d.py scripts/download_training_data.py --output-dir data/downloads/general_v1 --execute --resume --resume-partial --max-files 5 --max-seconds 240
```

Resume rehash file đã hoàn tất, không tải lại. `--resume-partial` yêu cầu SHA256 pin và HTTP Range chính xác để nối từ byte đã có, kiểm hash cuối trước publish; nếu server bỏ qua Range, hash sai hoặc phản hồi thiếu thì dừng và giữ partial. `--max-seconds` dừng có kiểm soát, file đang tải vẫn là `.part`. Lock ngăn hai downloader ghi cùng batch; nếu tiến trình bị kill để lại lock, phải xác minh tiến trình cũ đã dừng trước khi xử lý lock, không xóa mù.

Nếu đã có destination chưa ghi completed hoặc failure thì script dừng, không tự ghi đè/retry/reset ngân sách. Download report ghi checksum/file source, chưa kiểm tra nội dung parquet hay filter toàn bộ. `status=downloading` là trạng thái lần ghi cuối, **không chứng minh tiến trình vẫn sống**.

Muốn hoàn tất tất cả file còn lại trên máy bạn, khi không có downloader khác chạy:

```powershell
.\.venv\Scripts\python.exe scripts/python_d.py scripts/download_training_data.py --output-dir data/downloads/general_v1 --execute --resume --resume-partial
```

Lệnh có thể mất nhiều giờ, giữ PowerShell mở; không có tiến trình tải nền do agent để lại. File lỗi/mạng đứt cần review trước retry; không tự tải lại toàn bộ.

## Vì sao chưa bấm train ngay?

Downloader này **chỉ acquisition**. [data_pipeline.py](../../data_pipeline.py) vẫn đọc kho cũ theo mặc định, và giờ có `--download-report` để nhập **chỉ các parquet completed** trong batch đã hoàn tất, kiểm hash/size/plan trước khi đọc. Không copy GB thủ công hoặc đọc `.part`. Giữ origin/source/revision/giấy phép trong provenance. Bộ prepared mới (`data/prepared_general_v1`) đã được tạo + verify (mục trên); gói Kaggle đã đóng tại `kaggle_upload/pretrain_general_v1` nhưng **chưa upload, chưa train**.

Lệnh tạo **pilot từ dataset có sẵn, không tạo đáp án AI và không train**:

```powershell
.\.venv\Scripts\python.exe scripts/python_d.py data_pipeline.py --prepare --download-report data/downloads/general_v1/download_report.json --output-dir data/prepared_general_v1 --mb-per-domain 128 --sft-per-source 10000
.\.venv\Scripts\python.exe scripts/python_d.py data_pipeline.py --verify --output-dir data/prepared_general_v1
```

Lệnh có thể mất nhiều phút vì rehash 17,51 GB và đọc/lọc parquet. **Hai lệnh này đã chạy xong cho `data/prepared_general_v1`** (2026-10-01): manifest `status: ready`, verify pass — 519.660.893 byte pretrain (6 domain), 54.091 hội thoại SFT train / 2.901 val, `exact_split_overlap: 0`, không có `verification.json` cũ trước đó. Domain cap là cap, không bảo đảm giữ đủ mọi MB hoặc phân bố ngẫu nhiên toàn corpus. Hiện domain mixture của pipeline **chưa thực thi mục tiêu origin 70/30**; manifest ghi `origin_mixture_status: Not enforced`. Muốn gọi bộ train là 70/30 cần bước origin-aware mixing/review ở mục dưới, không chỉ dùng ngay pilot như đã đạt tỷ lệ.

Trạng thái hiện tại của pilot: prepare + verify đều pass; chưa train trên bộ này. Bước nhập/prepare tiếp phải:

1. Xác minh mọi shard, giữ revision/source ID/giấy phép và origin; nếu dùng cache cũ thì kiểm hash/revision hoặc đánh dấu legacy unknown.
2. Đọc whole-document parquet, giữ article ID/URL, lọc encoding/nhiễu/PII/repetition, tránh benchmark contamination.
3. Dedupe exact + near/family nơi có thể, chia cùng article/bài gốc vào một split. Không để đoạn cùng bài sang val.
4. Tách domain **và** origin; `human_vi`, `human_knowledge`, `human_math`, `synthetic_edu`, `synthetic_code`... Nguồn unknown tách riêng, không lén tính vào human.
5. Tạo mixture có tổng trọng số domain human=.70, synthetic=.30 **khi có đủ nguồn được review**; cân VI/EN trong mỗi nhóm. Không phân theo dung lượng dump. Đếm token/realized batches để báo tỷ lệ đạt thật. SFT/preference phải tính riêng, không chỉ dùng mixture pretrain.
6. Giữ eval/probes cố định, validation đủ đại diện, lưu manifest/hash/status ready rồi verify. Không dùng synthetic chưa review làm tập đo truth độc lập.
7. Batch pilot nhỏ trước khi prepare toàn kho nhiều GB; đo tokenization/RAM/disk/VRAM, không ước tính token từ bytes như sự thật.

## Mở rộng tổng quát, không chỉ Wikipedia EN

Đợt tải này mở rộng kiến thức EN; tận dụng VI đã có, không hứa nâng VI bằng riêng dữ liệu EN. Các bước sau có thể thêm sách/tài liệu giáo dục/nguồn code có giấy phép phù hợp, cặp SFT người, bài toán có reference. Không để truyện/nhãn AI lấn át toàn bộ kiến thức/VI. Tỷ lệ và số bước chỉnh theo benchmark, không đặt “càng nhiều GB càng thông minh”.

[01 Pretrain](01_PRETRAIN.md) dùng corpus/mixture, [02 SFT](02_SFT.md) dùng hội thoại, [03 Preference](03_PREFERENCE.md) dùng cặp lựa chọn. Các nguồn có thể liên quan nhau nên split cần thống nhất xuyên phase, không cho validation của phase này trở thành train ở phase khác rồi vẫn gọi held-out.
