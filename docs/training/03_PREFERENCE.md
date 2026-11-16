# 03 — Preference tuning: DPO / ORPO (thử nghiệm)

[Quay về mục lục](README.md)

**Học gì:** với cùng prompt, ưu tiên `chosen` hơn `rejected`. Không thay thế pretrain/SFT khi model thiếu nền kiến thức. Chưa có kết quả preference run được đo cho JIN trong công việc này.

**Dữ liệu:** cặp có nhãn đáng tin, người review/tiêu chí kiểm chứng. Không chỉ lấy đáp án AI tự tin làm chosen; không mặc định nguồn uy tín = mọi nhãn đúng.

```json
{"prompt":"Câu hỏi","chosen":"Đáp án đã kiểm chứng tốt hơn","rejected":"Đáp án kém hơn đã gán nhãn","provenance":{"source":"nguon","review":"cach danh gia"}}
```

## Mức sẵn sàng của code

[train_pref.py](../../train_pref.py) đã có thử nghiệm DPO/ORPO nhưng **không gọi là pipeline production/chuẩn đã được xác minh**:

- Logprob đang lấy trung bình **toàn sequence**, không chỉ assistant response; cần review so với objective mong muốn.
- Không có validation preference riêng, không early stopping và chưa tích hợp launcher pin/lineage đầy đủ như pretrain/SFT.
- DPO nạp thêm reference model, tốn bộ nhớ hơn; ORPO không nạp reference nhưng không bảo đảm fit RTX2060.
- `--out` có thể ghi đè; dùng đường dẫn mới và guard trước.
- Unit test loss sign/template không chứng minh training thực tế cải thiện chất lượng.

Do đó **ưu tiên benchmark + pretrain/SFT tiếp trước**. Muốn chạy preference nghiêm túc cần harden objective/masking, validation split, checksum/reference compatibility, lineage và test smoke trước.

Lệnh tham khảo **không thực thi tự động**, chỉ dùng sau review trên môi trường phù hợp:

```bash
python train_pref.py --method orpo --data data/preference/VERSION/train.jsonl \
  --checkpoint checkpoints/PARENT_SFT.pt --out checkpoints/preference/VERSION/smoke.pt \
  --device cuda --batch-size 1 --grad-accum 4 --max-length 256 \
  --lr 5e-6 --warmup 20 --steps 20
```

Không dùng `data/prepared_v4/sft_train` làm preference: schema khác. Không tạo rejected sai bừa hàng loạt và gọi là preference người.

## Test / đo

- `check_project.py --group preference --group shared --skip-preset-shapes`.
- Cần đánh giá held-out pair preference (response-only logprob), correctness, verbosity/style và benchmark khả năng cũ. Hiện chưa có CLI pair-eval hoàn chỉnh.
- Train loss preference giảm không bảo đảm đáp án thật chính xác hơn; chỉ promote nếu benchmark/chất lượng đáp án tốt hơn và không thoái lui quá mức.
