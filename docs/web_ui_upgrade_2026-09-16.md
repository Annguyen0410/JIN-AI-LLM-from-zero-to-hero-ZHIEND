# Nâng cấp web + chống echo — 2026-09-16

Bốn gói việc đã làm: **A** chất lượng câu trả lời, **B** chat app thật,
**C** model lab, **D** long-context + memory. Tất cả đã đo trên checkpoint
`checkpoints/jingpt_chat.pt` (step 4,000, val 2.9957) chạy CUDA.

## Vấn đề gốc: model lặp nguyên cụm

Log SFT kết thúc ở step 4,000:

```
"Hello! I'm Jin, a small language model — a small language model — a small language model..."
```

`REPETITION_PENALTY` chỉ nhìn từng token, mà mọi token trong câu đó đều hợp lệ —
chỉ *n-gram* là sai. Nên penalty không chặn được, model chạy hết 256 token.

## Ba lớp chống echo (pack A)

| Lớp | Ở đâu | Làm gì |
|---|---|---|
| Ban n-gram | `model.JinAI._apply_no_repeat_ngram` | token hoàn tất n-gram đã có → `-inf`. Không bao giờ ban token dừng. |
| Tự dừng khi lặp | `model.repeated_tail_period` dùng trong `generate` + `stream_generate` | đuôi lặp một cụm 3+ lần (cụm dài hơn thì ít lần hơn) → dừng, `stop_reason="repeat"` |
| Cắt đuôi văn bản | `reply_quality.trim_repeated_tail` | bản cuối (`done.reply`, `response`) đã cắt phần đã stream đi |

Mọi penalty đi qua một chỗ duy nhất (`JinAI._steer`) nên đường KV-cache,
đường speculative decoding và đường stream không lệch nhau.

### Kết quả đo (cùng checkpoint, cùng prompt)

| Prompt | Trước | Sau |
|---|---|---|
| `Hi!` | *"I'm small, very small. I'm small, even if I was small…"* (chạy tới hạn token) | *"I'm Jin, I'm Jin. How can I help you?"* — **15 token, `stop_reason=eos`** |
| `Hi!` | — | 14.6 → 38.3 tok/s, 0.9–1.0 s/lượt |
| SSE stream | delta có thể lẫn role tag, không có số đo | 21 delta + event `done`: `first_token_ms 131`, `28.6 tok/s`, `stop_reason=eos`, `reply` sạch |

Model giờ **tự kết thúc lượt** thay vì đốt hết ngân sách token vào echo.

## Pack B — chat app

- Markdown thật: heading/list/blockquote/link, `<pre>` có nút copy.
- Nút **Copy · Sinh lại · Sửa · Xoá** trên từng tin nhắn; Sửa cắt hội thoại từ
  điểm đó rồi gửi lại (reset session server cho khớp với những gì đang thấy).
- **Hội thoại lưu `localStorage`** (`jin.conversations`): tiêu đề = câu hỏi đầu,
  chuyển/xoá được, refresh vẫn còn chat; xuất `.md` / `.json`.
- Nhãn tiếng Việt cho toàn bộ chrome UI (prompt gửi cho model vẫn tiếng Anh vì
  model train tiếng Anh).

## Pack C — Model Lab

`/api/status` giờ trả: `checkpoints` (name/size/mtime/current), `defaults`
(thông số sinh), `max_positions`, `vram_used_mb/total`, `gpu_name`, `uptime_s`,
`requests_served`.

`POST /api/model/reload` (admin: `JINGPT_TOKEN` + loopback) hot-swap checkpoint
— dùng để A/B `jingpt_chat.pt` với bản golden `jingpt_chat_46k_sft4k.pt` ngay
trên UI. Xoá toàn bộ KV prefix cache vì cache đó thuộc weight cũ.

## Pack D — long-context + memory

- `--ctx N` trên `app.py` / `chat.py` / `bench.py`: mở rộng cửa sổ bằng RoPE
  scaling lúc load (`config.context_override`, NTK mặc định). Đây là "dùng được"
  chứ **chưa phải "đã học"** — phải đo:

  ```powershell
  python bench.py --checkpoint checkpoints\jingpt_chat.pt                # nền
  python bench.py --checkpoint checkpoints\jingpt_chat.pt --ctx 2048     # so sánh
  ```
  Nếu probe `long_context` không tăng thì trả về 1024 và để dành `--extend-context`
  trong `train.py` (fine-tune trên block dài).
- Panel **Bộ nhớ**: số entry + ô tra cứu hybrid (`/api/memory/search`).

## Sửa kèm 2 bug UI

- `.stop-btn` và `.message-actions` là `display:flex` nên **đè luôn thuộc tính
  `hidden`** → nút Stop hiện thường trực, hàng nút hành động hiện khi hover dù
  chưa sinh xong. Thêm `.stop-btn[hidden], .message-actions[hidden] { display:none }`
  (đúng pattern `.growth-panel[hidden]` đã có).
- Sidebar nhiều panel: `overflow-y` chuyển từ từng `.sidebar-section` (mỗi cái
  `flex:1`) lên chính `.sidebar`, nếu không các panel chia đôi chiều cao.

## Kiểm thử

`python -m unittest discover -s tests` → **187 OK** (thêm 22 test mới):
`tests/test_reply_quality.py` (n-gram ban, loop breaker, cắt đuôi, property
"không 3-gram nào lặp lại"), `tests/test_context_override.py`,
`tests/test_model_reload_api.py` (gates admin + xoá prefix cache).

## Việc còn lại

1. Chạy 2 lệnh bench ở trên để quyết định `--ctx 2048` có bật mặc định không.
2. Đổi checkpoint A/B thử trong panel Model Lab (cần `JINGPT_TOKEN`).
3. Model vẫn thua ở *chất lượng* (câu trả lời ngắn, lạc) — đó là chuyện token
   chứ không phải UI: xem bảng token/param trong `docs/post_train_2026-09-16.md`.
