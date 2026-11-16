/**
 * JinAI — web frontend.
 *
 * Talks to /api/chat (synchronous) or /api/chat/stream (server-sent events).
 * Conversation memory lives on the server, indexed by session_id which we
 * persist in localStorage so a refresh keeps the same chat.
 */

(() => {
"use strict";

// ── DOM ───────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const chatMessages   = $("chatMessages");
const chatInput      = $("chatInput");
const sendBtn        = $("sendBtn");
const stopBtn        = $("stopBtn");
const newChatBtn     = $("newChatBtn");
const modelInfo      = $("modelInfo");
const modelMeta      = $("modelMeta");
const footerStatus   = $("footerStatus");

const tempSlider     = $("temperature");
const tempValue      = $("tempValue");
const tokensSlider   = $("maxTokens");
const tokensValue    = $("tokensValue");
const topkSlider     = $("topK");
const topkValue      = $("topkValue");
const toppSlider     = $("topP");
const toppValue      = $("toppValue");
const repSlider      = $("repPenalty");
const repValue       = $("repValue");
const streamToggle   = $("streamToggle");
const searchMode     = $("searchMode");
const systemPrompt   = $("systemPrompt");

const menuBtn        = $("menuBtn");
const sidebar        = $("sidebar");
const appShell       = $("app");
const topModel       = $("topModel");
// Model bar: the topbar chip doubles as a quick checkpoint switcher.
const modelBarBtn    = $("modelBarBtn");
const modelBarMenu   = $("modelBarMenu");
const modelBarList   = $("modelBarList");
const modelBarHint   = $("modelBarHint");
const statusDot      = $("statusDot");
const settingsSheet  = $("settingsSheet");
const scrim          = $("scrim");
const collapseBtn    = $("collapseBtn");

// Anti-echo + behaviour controls
const presetRow      = $("presetRow");
const presetHint     = $("presetHint");
const ngramSelect    = $("ngramSelect");
const stopRepeat     = $("stopRepeatToggle");
const memoryMode     = $("memoryMode");

// Conversation store
const convoListEl    = $("convoList");

// Model Lab + memory panels
const labGrid        = $("labGrid");
const labCheckpoint  = $("labCheckpoint");
const labAuto        = $("labAuto");
const labSection     = $("labSection");
const labToken       = $("labToken");
const labHint        = $("labHint");
const labChart       = $("labChart");
const memStatsEl     = $("memStats");
const memResults     = $("memResults");

// ── State ─────────────────────────────────────────────────────────────
const STATE = {
    sessionId:   localStorage.getItem("jin.sessionId") || null,
    isGenerating: false,
    abort:       null,        // AbortController for the in-flight stream
    history:     [],          // local mirror for "regenerate"
    convoId:     localStorage.getItem("jin.activeConvo") || null,
    status:      null,        // last /api/status payload
};

// Sampling presets. Tuned for a ~147M chat model: it needs a *higher*
// repetition penalty and n-gram ban than a big model, and long answers decay
// into echo, so "precise" also caps the budget.
const PRESETS = {
    precise:  { temperature: 0.50, topK: 20, topP: 0.85, rep: 1.25, ngram: 3, maxTokens: 192 },
    balanced: { temperature: 0.70, topK: 40, topP: 0.90, rep: 1.20, ngram: 3, maxTokens: 256 },
    creative: { temperature: 1.00, topK: 80, topP: 0.95, rep: 1.12, ngram: 2, maxTokens: 384 },
};
// ── i18n (VI / EN) ────────────────────────────────────────────────────
// Mọi chuỗi của UI đi qua t(); đổi ngôn ngữ = dịch lại các node mang
// data-i18n / data-i18n-ph / data-tip-key rồi render lại phần động.
const I18N = {
    vi: {
        "sidebar.conversations": "Hội thoại",
        "sidebar.export_md": "Xuất .md",
        "sidebar.export_json": "Xuất .json",
        "sidebar.settings": "Cài đặt",
        "sidebar.growth": "Growth",
        "status.loading": "Đang tải…",
        "status.no_model": "Model chưa load",
        "status.train_first": "Train model trước đã.",
        "status.offline": "Server offline",
        "status.chip": "{file} · step {step} · {device}",
        "status.no_server": "Mất kết nối tới server",

        "composer.placeholder": "Nhắn cho JinAI…",
        "composer.hint": "JinAI chạy hoàn toàn trên máy bạn — model nhỏ nên có thể sai.",

        "welcome.sub": "Small LM tự train từ số 0. Hỏi gì cũng được — model nhỏ nên trả lời có thể sai.",
        "welcome.c1.title": "Chào hỏi", "welcome.c1.desc": "Để Jin tự giới thiệu",
        "welcome.c2.title": "Giải thích khái niệm", "welcome.c2.desc": "Hỏi cho dễ hiểu",
        "welcome.c3.title": "Viết code", "welcome.c3.desc": "Xin một đoạn snippet",
        "welcome.c4.title": "Kể chuyện", "welcome.c4.desc": "Cho model sáng tạo",

        "msg.you": "Bạn", "msg.jin": "JinAI",
        "msg.copy": "Copy", "msg.copied": "Đã copy", "msg.regen": "Sinh lại",
        "msg.edit": "Sửa", "msg.del": "Xóa", "msg.stopped": "(đã dừng)",
        "msg.empty": "(không có nội dung)", "msg.error": "Lỗi",
        "msg.web_sources": "Nguồn web",

        "metrics.tok_s": "{v} tok/s", "metrics.first": "token đầu {v} ms",
        "metrics.tokens": "{v} token", "metrics.step": "step {v}",
        "metrics.web": "web search",
        "metrics.warn_repeat": "⚠ model bắt đầu lặp — đã tự cắt",
        "metrics.stop_eos": "kết thúc bằng token dừng",
        "metrics.stop_repeat": "cắt vì lặp", "metrics.stop_length": "hết ngân sách token",
        "metrics.stop_aborted": "bạn bấm dừng",

        "convo.empty": "Chưa có hội thoại nào.",
        "convo.new": "Hội thoại mới",
        "convo.nothing": "Chưa có gì để xuất — hãy chat trước.",

        "preset.precise": "Chuẩn xác", "preset.balanced": "Cân bằng", "preset.creative": "Sáng tạo",
        "preset.note.precise": "Bám sát câu hỏi, ít bịa. Dùng cho tra cứu / hỏi đáp.",
        "preset.note.balanced": "Mặc định: đủ tự nhiên mà vẫn không lặp.",
        "preset.note.creative": "Kể chuyện, viết văn. Dễ lạc đề hơn — model 147M mà.",
        "preset.note.custom": "Đang dùng thông số tự chỉnh.",

        "row.preset": "Preset", "row.rep": "Repetition penalty", "row.maxtokens": "Max tokens",
        "row.ngram": "Chặn lặp cụm từ", "row.stoprepeat": "Tự dừng khi lặp",
        "row.stream": "Stream token", "row.websearch": "Web search",
        "row.memory": "Bộ nhớ dài hạn", "row.sysprompt": "System prompt",
        "row.sysprompt_ph": "(Tùy chọn) Jin là ai / cư xử thế nào",
        "row.checkpoint": "Checkpoint đang phục vụ", "row.labtoken": "Admin token",
        "row.lang": "Ngôn ngữ", "row.sidebarw": "Bề rộng thanh bên",
        "opt.auto": "Tự quyết", "opt.on": "Luôn bật", "opt.off": "Tắt", "opt.off2": "Tắt",
        "opt.ngram2": "2 token", "opt.ngram3": "3 token", "opt.ngram4": "4 token",

        "sheet.title": "Cài đặt", "sheet.sampling": "Sampling", "sheet.antirepeat": "Chống lặp",
        "sheet.behavior": "Hành vi", "sheet.lab": "Model Lab", "sheet.memory": "Bộ nhớ",
        "sheet.appearance": "Giao diện",

        "btn.reload": "Đổi checkpoint", "btn.search": "Tìm", "btn.close": "Đóng",
        "btn.resetlayout": "Về mặc định",
        "mem.placeholder": "Tìm trong bộ nhớ…",

        "tip.newchat": "Mở cuộc trò chuyện mới (vẫn giữ lại hội thoại cũ trong danh sách).",
        "tip.msgcopy": "Copy nội dung tin nhắn này vào clipboard.",
        "tip.msgregen": "Sinh lại câu trả lời cho đúng câu hỏi phía trên.",
        "tip.msgedit": "Sửa tin nhắn này rồi gửi lại — mọi tin phía sau nó bị cắt bỏ.",
        "tip.msgdel": "Xoá tin nhắn này khỏi hội thoại (chỉ nằm trên máy bạn).",
        "tip.collapse": "Ẩn thanh bên để vùng chat rộng hơn. Bấm lại để hiện.",
        "tip.settings": "Mở bảng cài đặt: sampling, chống lặp, hành vi, model, bộ nhớ.",
        "tip.growth": "Growth Dashboard (cần JINGPT_TOKEN): frontier, lịch sử bench, registry checkpoint.",
        "tip.convolist": "Hội thoại lưu trong localStorage của trình duyệt — refresh vẫn còn. Bấm để mở, ✕ để xoá.",
        "tip.exportmd": "Tải hội thoại đang mở ra file Markdown (kèm số đo từng câu).",
        "tip.exportjson": "Tải hội thoại đang mở ra JSON đầy đủ (text + metadata).",
        "tip.close": "Đóng bảng cài đặt (Esc cũng được).",
        "tip.resize": "Kéo ngang để chỉnh bề rộng thanh bên (200–460px). Nháy đúp để về 260px.",
        "tip.topchip": "Checkpoint đang phục vụ, bước train và thiết bị — cập nhật mỗi 5 giây.",
        "tip.modelbar": "Bấm để đổi model đang phục vụ (không cần mở Cài đặt).",
        "tip.stop": "Dừng sinh giữa dòng; phần đã sinh vẫn được giữ lại.",
        "tip.send": "Gửi (Enter). Shift+Enter để xuống dòng.",
        "tip.presets": "Preset = một bộ thông số sampling đã chỉnh sẵn cho model 147M.",
        "tip.temperature": "Độ ngẫu nhiên. Thấp (0.4–0.6) = bám sát câu hỏi; cao (0.9+) = sáng tạo nhưng dễ lạc.",
        "tip.topk": "Chỉ lấy mẫu từ K token có xác suất cao nhất. 0 = tắt giới hạn này.",
        "tip.topp": "Lấy mẫu trong nhóm token chiếm P% xác suất. 0.9 là mức cân bằng.",
        "tip.rep": "Phạt token đã xuất hiện. Model nhỏ cần 1.15–1.25; trên 1.3 dễ lạc đề.",
        "tip.maxtokens": "Số token tối đa cho một câu trả lời. Model dễ lặp thì để 192–256 là đủ.",
        "tip.ngram": "Cấm lặp lại nguyên cụm n token đã có. Đây là thứ thật sự chặn echo.",
        "tip.stoprepeat": "Tự dừng khi đuôi câu trả lời lặp đi lặp lại một cụm.",
        "tip.stream": "Hiện chữ ngay khi model sinh (thấy token đầu sau ~0.1–0.3s).",
        "tip.websearch": "Cho Jin tra web trước khi trả lời. Auto = chỉ tra khi câu hỏi có vẻ cần dữ kiện.",
        "tip.memory": "Dùng kho kiến thức dài hạn đã lưu trên máy (RAG) để trả lời kèm nguồn.",
        "tip.sysprompt": "Câu lệnh nền luôn đứng trước hội thoại — dùng để định hình tính cách / cách trả lời.",
        "tip.checkpoint": "Chọn file .pt trên đĩa để đổi model đang phục vụ (A/B). Nhãn cho biết file là chat (biết format hội thoại) hay pretrain (chỉ nối tiếp văn bản).",
        "tip.labtoken": "JINGPT_TOKEN của server. Cần để gọi các API admin (đổi checkpoint, bench, growth).",
        "tip.reload": "Nạp lại weight từ checkpoint đã chọn — không cần restart server.",
        "tip.memsearch": "Tra thử kho bộ nhớ xem Jin đang nhớ gì cho một truy vấn.",
        "tip.lang": "Đổi ngôn ngữ giao diện. Lựa chọn được nhớ cho lần sau.",
        "tip.resetlayout": "Trả bề rộng thanh bên và trạng thái ẩn/hiện về mặc định.",

        "lab.checkpoint": "Checkpoint", "lab.role": "Loại file",
        "lab.step": "Step", "lab.valloss": "Val loss (best)",
        "lab.device": "Thiết bị", "lab.vram": "VRAM", "lab.ctx": "Ctx / tối đa",
        "lab.uptime": "Đã chạy", "lab.requests": "Requests",
        "lab.none": "Chưa có checkpoint nào trên đĩa.",
        "lab.current": "← đang chạy",
        "lab.need_token": "Cần JINGPT_TOKEN (admin) để đổi checkpoint.",
        "lab.loading": "Đang load lại weight… (vài giây)",
        "lab.done": "Đã đổi sang {file} (step {step}, {sec}s).",
        "lab.failed": "Lỗi: {err}",
        "lab.call_failed": "Không gọi được /api/model/reload: {err}",
        "lab.chart_token": "Nhập token để xem lịch sử bench.",
        "lab.chart_admin": "Bench history cần admin token.",
        "lab.chart_err": "Không đọc được bench history.",
        "lab.auto": "Tự chọn khi khởi động: {file} ({why}).",
        "lab.why.chat": "checkpoint chat mới nhất",
        "lab.why.chat-snapshot": "bản sao chat mới nhất",
        "lab.why.instruct": "instruct mới nhất trên đĩa",
        "lab.why.pretrain-fallback": "chưa có checkpoint chat nên tạm dùng pretrain",
        "lab.why.JINGPT_CHECKPOINT": "theo biến môi trường JINGPT_CHECKPOINT",
        "lab.role.chat": "chat",
        "lab.role.instruct": "instruct",
        "lab.role.pretrain-best": "pretrain best",
        "lab.role.pretrain-latest": "pretrain latest",
        "lab.role.pretrain-run": "pretrain",
        "lab.role.legacy": "legacy",
        "lab.opt_step": "step {step}", "lab.opt_val": "val {val}",
        "lab.default_mark": "★ tự chọn",
        "lab.no_step": "chưa có step",
        "lab.pretrain_note": "File này không phải bản chat/instruct — nó chưa từng học format hội thoại, chọn vào sẽ trả lời kiểu nối tiếp văn bản.",
        "tip.lab": "Mở Model Lab: thông số model, chọn checkpoint, biểu đồ bench.",

        "modelbar.title": "Model đang phục vụ",
        "modelbar.note": "bấm để đổi · mới nhất xếp trên",
        "modelbar.open_lab": "Model Lab",
        "modelbar.current": "đang chạy",
        "modelbar.empty": "Chưa thấy file .pt nào trong checkpoints/.",
        "modelbar.no_admin": "Cần JINGPT_TOKEN (nhập ở Model Lab) để đổi model.",
        "modelbar.admin_off": "Admin đang tắt: server mở ra LAN mà không có JINGPT_TOKEN — bind 127.0.0.1 hoặc đặt token.",
        "auth.needed": "Server đang bật JINGPT_TOKEN — nhập token ở Model Lab (Cài đặt) rồi thử lại.",
        "modelbar.pretrain_tag": "pretrain",
        "modelbar.pretrain_live": "⚠ Đang phục vụ {file} — bản pretrain không biết format chat (chỉ nối tiếp văn bản) nên trả lời dễ sai/lạc. Chọn một dòng instruct/chat ở trên.",
        "modelbar.confirm_pretrain": "⚠ {file} là checkpoint pretrain — nó chưa từng học format hội thoại nên trả lời sẽ lạc. Bấm lại vào đúng dòng đó để đổi.",
        "modelbar.switching": "Đang nạp weights… (vài giây)",
        "modelbar.switched": "Đã đổi sang {file} (step {step}, {sec}s).",
        "modelbar.failed": "Không đổi được: {err}",

        "mem.total": "Tổng", "mem.semantic": "Semantic", "mem.disputed": "Disputed",
        "mem.indexed": "Đã index", "mem.disabled": "Bộ nhớ đang tắt.",
        "mem.read_err": "Không đọc được bộ nhớ.",
        "mem.searching": "Đang tìm…", "mem.no_hits": "Không có kết quả.", "mem.error": "Lỗi: {err}",

        "export.you": "Bạn", "export.jin": "JinAI",
        "export.note": "Xuất từ JinAI — {time}",
    },
    en: {
        "sidebar.conversations": "Chats",
        "sidebar.export_md": "Export .md",
        "sidebar.export_json": "Export .json",
        "sidebar.settings": "Settings",
        "sidebar.growth": "Growth",
        "status.loading": "Loading…",
        "status.no_model": "Model not loaded",
        "status.train_first": "Train a model first.",
        "status.offline": "Server offline",
        "status.chip": "{file} · step {step} · {device}",
        "status.no_server": "Lost connection to the server",

        "composer.placeholder": "Message JinAI…",
        "composer.hint": "JinAI runs entirely on your machine — it is a small model, so it can be wrong.",

        "welcome.sub": "A small LM trained from scratch. Ask anything — it is small, so answers can be wrong.",
        "welcome.c1.title": "Say hello", "welcome.c1.desc": "Have Jin introduce itself",
        "welcome.c2.title": "Explain a concept", "welcome.c2.desc": "Ask for a simple explanation",
        "welcome.c3.title": "Write code", "welcome.c3.desc": "Ask for a snippet",
        "welcome.c4.title": "Tell a story", "welcome.c4.desc": "Let the model be creative",

        "msg.you": "You", "msg.jin": "JinAI",
        "msg.copy": "Copy", "msg.copied": "Copied", "msg.regen": "Regenerate",
        "msg.edit": "Edit", "msg.del": "Delete", "msg.stopped": "(stopped)",
        "msg.empty": "(empty response)", "msg.error": "Error",
        "msg.web_sources": "Web sources",

        "metrics.tok_s": "{v} tok/s", "metrics.first": "first token {v} ms",
        "metrics.tokens": "{v} tokens", "metrics.step": "step {v}",
        "metrics.web": "web search",
        "metrics.warn_repeat": "⚠ the model started looping — cut short",
        "metrics.stop_eos": "ended on the stop token",
        "metrics.stop_repeat": "cut: it started repeating itself", "metrics.stop_length": "hit the token budget",
        "metrics.stop_aborted": "you stopped it",

        "convo.empty": "No chats yet.",
        "convo.new": "New chat",
        "convo.nothing": "Nothing to export yet — chat first.",

        "preset.precise": "Precise", "preset.balanced": "Balanced", "preset.creative": "Creative",
        "preset.note.precise": "Stays close to the question. Best for lookups and Q&A.",
        "preset.note.balanced": "Default: natural enough, still no looping.",
        "preset.note.creative": "Storytelling and prose. Wanders more — it is a 147M model.",
        "preset.note.custom": "Using hand-tuned settings.",

        "row.preset": "Preset", "row.rep": "Repetition penalty", "row.maxtokens": "Max tokens",
        "row.ngram": "Block repeated n-grams", "row.stoprepeat": "Stop when looping",
        "row.stream": "Stream tokens", "row.websearch": "Web search",
        "row.memory": "Long-term memory", "row.sysprompt": "System prompt",
        "row.sysprompt_ph": "(Optional) who Jin is / how to behave",
        "row.checkpoint": "Served checkpoint", "row.labtoken": "Admin token",
        "row.lang": "Language", "row.sidebarw": "Sidebar width",
        "opt.auto": "Auto", "opt.on": "Always on", "opt.off": "Off", "opt.off2": "Off",
        "opt.ngram2": "2 tokens", "opt.ngram3": "3 tokens", "opt.ngram4": "4 tokens",

        "sheet.title": "Settings", "sheet.sampling": "Sampling", "sheet.antirepeat": "Anti-repeat",
        "sheet.behavior": "Behaviour", "sheet.lab": "Model Lab", "sheet.memory": "Memory",
        "sheet.appearance": "Appearance",

        "btn.reload": "Switch checkpoint", "btn.search": "Search", "btn.close": "Close",
        "btn.resetlayout": "Reset layout",
        "mem.placeholder": "Search memory…",

        "tip.newchat": "Start a new chat (the old one stays in the list).",
        "tip.msgcopy": "Copy this message to the clipboard.",
        "tip.msgregen": "Generate this answer again from the question above.",
        "tip.msgedit": "Edit this message and resend — everything after it is dropped.",
        "tip.msgdel": "Delete this message from the chat (it only lives on your machine).",
        "tip.collapse": "Hide the sidebar for a wider chat area. Click again to show it.",
        "tip.settings": "Open settings: sampling, anti-repeat, behaviour, model, memory.",
        "tip.growth": "Growth Dashboard (needs JINGPT_TOKEN): frontier, bench history, checkpoint registry.",
        "tip.convolist": "Chats live in browser localStorage — they survive a refresh. Click to open, ✕ to delete.",
        "tip.exportmd": "Download the open chat as Markdown (including per-reply metrics).",
        "tip.exportjson": "Download the open chat as full JSON (text + metadata).",
        "tip.close": "Close settings (Esc works too).",
        "tip.resize": "Drag sideways to resize the sidebar (200–460px). Double-click to reset to 260px.",
        "tip.topchip": "Served checkpoint, training step and device — refreshed every 5 seconds.",
        "tip.modelbar": "Click to switch the served model (no need to open Settings).",
        "tip.stop": "Stop mid-generation; what was already generated is kept.",
        "tip.send": "Send (Enter). Shift+Enter for a new line.",
        "tip.presets": "A preset is a ready-made sampling profile tuned for this 147M model.",
        "tip.temperature": "Randomness. Low (0.4–0.6) sticks to the question; high (0.9+) is creative but wanders.",
        "tip.topk": "Sample only from the K most likely tokens. 0 disables this limit.",
        "tip.topp": "Sample from the smallest set of tokens covering P% of the probability. 0.9 is balanced.",
        "tip.rep": "Penalise repeated tokens. Small models want 1.15–1.25; above 1.3 they drift.",
        "tip.maxtokens": "Maximum tokens per reply. A loop-prone model does fine with 192–256.",
        "tip.ngram": "Forbid repeating an n-token phrase that already appeared — this is what actually kills echo.",
        "tip.stoprepeat": "Stop generation when the tail of the reply repeats a chunk over and over.",
        "tip.stream": "Show text as the model produces it (first token in ~0.1–0.3s).",
        "tip.websearch": "Let Jin search the web first. Auto = search only when the question looks like it needs facts.",
        "tip.memory": "Use the local long-term knowledge store (RAG) and answer with sources.",
        "tip.sysprompt": "A standing instruction placed before the conversation — shapes persona and style.",
        "tip.checkpoint": "Pick a .pt file on disk to swap the served model (A/B). The label says whether the file is chat (knows the conversation format) or pretrain (raw continuation only).",
        "tip.labtoken": "The server's JINGPT_TOKEN. Required for admin APIs (checkpoint swap, bench, growth).",
        "tip.reload": "Reload weights from the selected checkpoint — no server restart.",
        "tip.memsearch": "Test the memory store to see what Jin remembers for a query.",
        "tip.lang": "Switch the interface language. The choice is remembered.",
        "tip.resetlayout": "Reset sidebar width and collapsed state to defaults.",

        "lab.checkpoint": "Checkpoint", "lab.role": "File role",
        "lab.step": "Step", "lab.valloss": "Val loss (best)",
        "lab.device": "Device", "lab.vram": "VRAM", "lab.ctx": "Ctx / max",
        "lab.uptime": "Uptime", "lab.requests": "Requests",
        "lab.none": "No checkpoints on disk yet.",
        "lab.current": "← serving",
        "lab.need_token": "An admin JINGPT_TOKEN is required to switch checkpoints.",
        "lab.loading": "Reloading weights… (a few seconds)",
        "lab.done": "Switched to {file} (step {step}, {sec}s).",
        "lab.failed": "Error: {err}",
        "lab.call_failed": "Could not call /api/model/reload: {err}",
        "lab.chart_token": "Enter a token to see bench history.",
        "lab.chart_admin": "Bench history needs an admin token.",
        "lab.chart_err": "Could not read bench history.",
        "lab.auto": "Auto-picked on startup: {file} ({why}).",
        "lab.why.chat": "newest chat checkpoint",
        "lab.why.chat-snapshot": "newest chat snapshot",
        "lab.why.instruct": "newest instruct tune on disk",
        "lab.why.pretrain-fallback": "no chat checkpoint yet, falling back to pretrain",
        "lab.why.JINGPT_CHECKPOINT": "set by the JINGPT_CHECKPOINT env var",
        "lab.role.chat": "chat",
        "lab.role.instruct": "instruct",
        "lab.role.pretrain-best": "pretrain best",
        "lab.role.pretrain-latest": "pretrain latest",
        "lab.role.pretrain-run": "pretrain",
        "lab.role.legacy": "legacy",
        "lab.opt_step": "step {step}", "lab.opt_val": "val {val}",
        "lab.default_mark": "★ auto",
        "lab.no_step": "no step in header",
        "lab.pretrain_note": "This file is not a chat/instruct tune — it never saw the conversation format, so answers come back as raw text continuation.",
        "tip.lab": "Open Model Lab: model facts, checkpoint picker, bench chart.",

        "modelbar.title": "Served model",
        "modelbar.note": "click to switch · newest first",
        "modelbar.open_lab": "Model Lab",
        "modelbar.current": "serving",
        "modelbar.empty": "No .pt files under checkpoints/ yet.",
        "modelbar.no_admin": "An admin JINGPT_TOKEN (enter it in Model Lab) is required to switch models.",
        "modelbar.admin_off": "Admin is off: the server is exposed on the LAN without JINGPT_TOKEN — bind 127.0.0.1 or set a token.",
        "auth.needed": "This server requires JINGPT_TOKEN — enter it in Model Lab (Settings) and retry.",
        "modelbar.pretrain_tag": "pretrain",
        "modelbar.pretrain_live": "⚠ Serving {file} — a pretrain checkpoint that never saw the chat format (raw text continuation), so answers drift. Pick an instruct/chat row above.",
        "modelbar.confirm_pretrain": "⚠ {file} is a pretrain checkpoint — it never learned the conversation format, so answers will drift. Click the same row again to switch.",
        "modelbar.switching": "Reloading weights… (a few seconds)",
        "modelbar.switched": "Switched to {file} (step {step}, {sec}s).",
        "modelbar.failed": "Could not switch: {err}",

        "mem.total": "Total", "mem.semantic": "Semantic", "mem.disputed": "Disputed",
        "mem.indexed": "Indexed", "mem.disabled": "Memory is disabled.",
        "mem.read_err": "Could not read memory.",
        "mem.searching": "Searching…", "mem.no_hits": "No results.", "mem.error": "Error: {err}",

        "export.you": "You", "export.jin": "JinAI",
        "export.note": "Exported from JinAI — {time}",
    },
};

const DEFAULT_LANG = "vi";
let LANG = "vi";
try { LANG = localStorage.getItem("jin.lang") || DEFAULT_LANG; } catch { /* ignore */ }
if (!I18N[LANG]) LANG = DEFAULT_LANG;

/** Dịch một khoá, thay {placeholder} nếu có. */
function t(key, vars) {
    const table = I18N[LANG] || I18N[DEFAULT_LANG];
    let text = table[key] ?? I18N[DEFAULT_LANG][key] ?? key;
    if (vars) {
        for (const [name, value] of Object.entries(vars)) {
            text = text.split(`{${name}}`).join(String(value));
        }
    }
    return text;
}

const PRESET_NOTE_KEY = {
    precise: "preset.note.precise",
    balanced: "preset.note.balanced",
    creative: "preset.note.creative",
};

const SLIDERS = [
    [tempSlider, tempValue, (v) => v.toFixed(2)],
    [tokensSlider, tokensValue, (v) => String(parseInt(v, 10))],
    [topkSlider, topkValue, (v) => String(parseInt(v, 10))],
    [toppSlider, toppValue, (v) => v.toFixed(2)],
    [repSlider, repValue, (v) => v.toFixed(2)],
];

// ── Bootstrap ─────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    refreshSliderLabels();
    paintSliders();
    wireTooltips();
    document.addEventListener("input", (event) => {
        if (event.target instanceof HTMLInputElement && event.target.type === "range") {
            paintSliders();
        }
    });
    wireEvents();
    wireExtraEvents();
    autoResize();
    // Panel/bề rộng đổi (preview kéo, cửa sổ resize) → đo lại chiều cao ô nhập.
    window.addEventListener("resize", autoResize);
    // applyLanguage() dịch markup tĩnh rồi render lại phần động
    // (danh sách hội thoại, hội thoại đang mở, Model Lab, bộ nhớ).
    applyLanguage();
    checkStatus();
    setInterval(checkStatus, 5000);
});

// ── i18n: áp dụng + render lại phần động ────────────────────────────────
function applyLanguage() {
    document.documentElement.lang = LANG;
    document.querySelectorAll("[data-i18n]").forEach((el) => {
        el.textContent = t(el.dataset.i18n);
    });
    document.querySelectorAll("[data-i18n-ph]").forEach((el) => {
        el.placeholder = t(el.dataset.i18nPh);
    });
    document.querySelectorAll("[data-tip-key]").forEach((el) => {
        el.dataset.tip = t(el.dataset.tipKey);
        if (el.hasAttribute("title")) el.title = t(el.dataset.tipKey);
    });
    const langSelect = $("langSelect");
    if (langSelect) langSelect.value = LANG;
    refreshDynamic();
}

function setLanguage(lang) {
    if (!I18N[lang]) return;
    LANG = lang;
    try { localStorage.setItem("jin.lang", LANG); } catch { /* ignore */ }
    applyLanguage();
}

function refreshDynamic() {
    renderConvoList();
    const convo = loadConvos().find((c) => c.id === STATE.convoId);
    if (convo && convo.messages.length) {
        openConvo(convo.id);
    } else if (!chatMessages.querySelector(".message")) {
        chatMessages.innerHTML = "";
        chatMessages.appendChild(buildWelcome());
    }
    if (STATE.status) renderLab(STATE.status);
    if (ckptCache.length) paintCheckpointPicker(ckptCache);   // re-label in the new language
    if (!modelBarMenu.hidden) renderModelBar(ckptCache);
    refreshPresetHint();
    loadMemoryStats();
}

// ── Kéo giãn thanh bên ─────────────────────────────────────────────────
const SIDEBAR_MIN = 200;
const SIDEBAR_MAX = 460;
const SIDEBAR_DEFAULT = 260;

function setSidebarWidth(px, persist = true) {
    const width = Math.max(SIDEBAR_MIN, Math.min(SIDEBAR_MAX, Math.round(px)));
    document.documentElement.style.setProperty("--sidebar-w", width + "px");
    const slider = $("sidebarWidth");
    const label = $("sidebarWidthValue");
    if (slider) slider.value = String(width);
    if (label) label.textContent = width + "px";
    paintSliders();
    if (persist) {
        try { localStorage.setItem("jin.sidebarWidth", String(width)); } catch { /* ignore */ }
    }
    return width;
}

function initResize() {
    let saved = 0;
    try { saved = parseInt(localStorage.getItem("jin.sidebarWidth") || "0", 10); } catch { /* ignore */ }
    setSidebarWidth(saved || SIDEBAR_DEFAULT, false);

    const handle = $("sidebarResizer");
    if (!handle) return;

    const onMove = (event) => {
        const x = event.touches ? event.touches[0].clientX : event.clientX;
        setSidebarWidth(x - sidebar.getBoundingClientRect().left);
    };
    const stop = () => {
        handle.classList.remove("dragging");
        document.body.classList.remove("resizing");
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", stop);
    };
    handle.addEventListener("pointerdown", (event) => {
        if (window.innerWidth <= 768) return;   // mobile: sidebar phủ, không kéo
        event.preventDefault();
        handle.classList.add("dragging");
        document.body.classList.add("resizing");
        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", stop);
    });
    handle.addEventListener("dblclick", () => setSidebarWidth(SIDEBAR_DEFAULT));

    const slider = $("sidebarWidth");
    if (slider) slider.addEventListener("input", () => setSidebarWidth(parseInt(slider.value, 10)));
    $("resetLayoutBtn").addEventListener("click", () => {
        setSidebarWidth(SIDEBAR_DEFAULT);
        setCollapsed(false);
    });
}

function wireEvents() {
    sendBtn.addEventListener("click", () => send());
    stopBtn.addEventListener("click", () => abortStream());
    $("growthBtn").addEventListener("click", growthOpen);
    $("growthClose").addEventListener("click", growthClose);
    $("growthRefresh").addEventListener("click", loadGrowth);
    growthToken.addEventListener("keydown", (e) => {
        if (e.key === "Enter") loadGrowth();
    });

    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
            e.preventDefault();
            send();
        }
    });
    chatInput.addEventListener("input", () => {
        sendBtn.disabled = !chatInput.value.trim() || STATE.isGenerating;
        autoResize();
    });

    newChatBtn.addEventListener("click", () => resetChat());

    SLIDERS.forEach(([slider, label, fmt]) => {
        slider.addEventListener("input", () => {
            label.textContent = fmt(parseFloat(slider.value));
        });
    });

    // Suggestion cards (delegated so it survives clearChat re-render).
    chatMessages.addEventListener("click", (e) => {
        const card = e.target.closest(".suggestion-card");
        if (!card) return;
        chatInput.value = card.dataset.prompt || "";
        sendBtn.disabled = !chatInput.value.trim();
        chatInput.focus();
        autoResize();
    });

    if (menuBtn) menuBtn.addEventListener("click", toggleSidebar);
    document.addEventListener("click", (e) => {
        if (window.innerWidth <= 768 &&
            sidebar.classList.contains("open") &&
            !sidebar.contains(e.target) &&
            e.target !== menuBtn) {
            sidebar.classList.remove("open");
            removeOverlay();
        }
    });
}

function refreshSliderLabels() {
    SLIDERS.forEach(([slider, label, fmt]) => {
        label.textContent = fmt(parseFloat(slider.value));
    });
}

// ── Tooltip ⓘ ─────────────────────────────────────────────────────────
// Một bong bóng dùng chung, gắn vào <body> nên không bị overflow của
// sheet/sidebar cắt, và luôn được kéo vào trong khung nhìn.
let tipBubble = null;
let tipOwner = null;

function ensureTipBubble() {
    if (!tipBubble) {
        tipBubble = document.createElement("div");
        tipBubble.className = "tip-bubble";
        tipBubble.setAttribute("role", "tooltip");
        document.body.appendChild(tipBubble);
    }
    return tipBubble;
}

function showTip(owner) {
    const text = owner.dataset.tip;
    if (!text) return;
    if (tipOwner === owner && tipBubble && tipBubble.classList.contains("show")) return;
    tipOwner = owner;
    const bubble = ensureTipBubble();
    bubble.textContent = text;
    bubble.classList.add("show");
    placeTip(owner, bubble);
}

function placeTip(owner, bubble) {
    const margin = 8;
    const r = owner.getBoundingClientRect();
    const b = bubble.getBoundingClientRect();
    let left = r.left + r.width / 2 - b.width / 2;
    left = Math.max(margin, Math.min(window.innerWidth - b.width - margin, left));
    let top = r.top - b.height - 10;
    if (top < margin) {
        top = Math.min(window.innerHeight - b.height - margin, r.bottom + 10);
    }
    bubble.style.left = Math.round(left) + "px";
    bubble.style.top = Math.round(top) + "px";
}

function hideTip() {
    tipOwner = null;
    if (tipBubble) tipBubble.classList.remove("show");
}

function wireTooltips() {
    const ownerOf = (target) => (
        target && target.closest ? target.closest("[data-tip]") : null
    );
    document.addEventListener("pointerover", (event) => {
        const owner = ownerOf(event.target);
        if (owner) showTip(owner);
    });
    document.addEventListener("pointerout", (event) => {
        const owner = ownerOf(event.target);
        // Bỏ qua khi con trỏ chỉ di chuyển giữa các con của cùng một phần tử.
        if (owner && owner === tipOwner && !owner.contains(event.relatedTarget)) hideTip();
    });
    document.addEventListener("focusin", (event) => {
        const owner = ownerOf(event.target);
        if (owner) showTip(owner);
    });
    document.addEventListener("focusout", hideTip);
    window.addEventListener("scroll", hideTip, true);
    window.addEventListener("resize", hideTip);
}

/** Tô phần đã kéo của mọi slider bằng gradient đỏ (CSS dùng var --fill). */
function paintSliders() {
    document.querySelectorAll('input[type="range"].slider').forEach((el) => {
        const min = parseFloat(el.min) || 0;
        const max = parseFloat(el.max) || 100;
        const value = parseFloat(el.value);
        const pct = max > min ? ((value - min) / (max - min)) * 100 : 0;
        el.style.setProperty("--fill", Math.max(0, Math.min(100, pct)).toFixed(1) + "%");
    });
}

function autoResize() {
    const apply = () => {
        chatInput.style.height = "auto";
        chatInput.style.height = Math.min(chatInput.scrollHeight, 180) + "px";
    };
    apply();
    // Lần đo đầu có thể rơi vào lúc panel chưa có bề rộng cuối (placeholder
    // wrap thành nhiều dòng giả) — đo lại một nhịp sau khi layout ổn định.
    requestAnimationFrame(apply);
}

// ── Send ──────────────────────────────────────────────────────────────
async function send(text) {
    const message = (text ?? chatInput.value).trim();
    if (!message || STATE.isGenerating) return;

    hideWelcome();
    addMessage("user", message);
    STATE.history.push({ role: "user", content: message });
    pushTurn("user", message);

    chatInput.value = "";
    autoResize();
    setBusy(true);

    const requestBody = {
        message,
        session_id: STATE.sessionId,
        temperature: parseFloat(tempSlider.value),
        max_tokens: parseInt(tokensSlider.value, 10),
        top_k: parseInt(topkSlider.value, 10),
        top_p: parseFloat(toppSlider.value),
        repetition_penalty: parseFloat(repSlider.value),
        no_repeat_ngram_size: parseInt(ngramSelect.value, 10) || 0,
        stop_on_repeat: stopRepeat.checked,
        web_search: searchMode.value,
        memory: memoryMode.value,
    };
    const sysp = systemPrompt.value.trim();
    if (sysp) requestBody.system_prompt = sysp;

    const useStream = streamToggle.checked;
    const placeholder = addMessage("assistant", "", { typing: true });
    const textNode = placeholder.querySelector(".message-text");

    try {
        const result = useStream
            ? await streamReply(requestBody, placeholder, textNode)
            : await syncReply(requestBody, placeholder, textNode);
        const replyText = result.text || t("msg.empty");
        finalizeMessage(placeholder, textNode, replyText);
        attachMetrics(placeholder, result.meta);
        pushTurn("assistant", replyText, result.meta);
    } catch (err) {
        if (err.name === "AbortError") {
            const stopped = textNode.dataset.raw || t("msg.stopped");
            finalizeMessage(placeholder, textNode, stopped, true);
            pushTurn("assistant", stopped, { stop_reason: "aborted", stop_label: t("metrics.stop_aborted") });
        } else {
            finalizeMessage(placeholder, textNode, `${t("msg.error")}: ${err.message}`, true, true);
        }
    } finally {
        setBusy(false);
        chatInput.focus();
    }
}

async function syncReply(body, placeholder, textNode) {
    const response = await fetch("/api/chat", {
        method: "POST",
        headers: apiHeaders(),
        body: JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(authError(response.status) || data.error || `HTTP ${response.status}`);
    }
    if (data.session_id) saveSession(data.session_id);
    if (data.sources && data.sources.length) {
        attachSources(placeholder, data.sources);
    }
    return {
        text: data.response || t("msg.empty"),
        meta: {
            tok_per_s: data.tok_per_s,
            elapsed_s: data.elapsed_s,
            tokens_generated: data.tokens_generated,
            stop_reason: data.stop_reason,
            stop_label: data.stop_label,
            checkpoint_step: data.checkpoint_step,
            web_search_used: data.web_search_used,
        },
    };
}

async function streamReply(body, placeholder, textNode) {
    const ctrl = new AbortController();
    STATE.abort = ctrl;
    stopBtn.hidden = false;

    const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: apiHeaders({ "Accept": "text/event-stream" }),
        body: JSON.stringify(body),
        signal: ctrl.signal,
    });
    if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(authError(response.status) || data.error || `HTTP ${response.status}`);
    }

    const reader  = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let   buf     = "";
    let   acc     = "";
    let   doneData = null;
    let   clientFirstMs = null;
    const startedAt = performance.now();

    placeholder.querySelector(".typing-indicator")?.remove();

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        const events = buf.split("\n\n");
        buf = events.pop() || "";
        for (const evt of events) {
            const parsed = parseSSE(evt);
            if (!parsed) continue;
            if (parsed.event === "session" && parsed.data?.session_id) {
                saveSession(parsed.data.session_id);
            } else if (parsed.event === "sources" && parsed.data?.sources?.length) {
                attachSources(placeholder, parsed.data.sources);
            } else if (parsed.event === "error") {
                throw new Error(parsed.data?.error || "stream error");
            } else if (parsed.event === "done") {
                doneData = parsed.data || {};
            } else if (parsed.event === "message") {
                if (parsed.data?.delta) {
                    if (clientFirstMs === null) clientFirstMs = performance.now() - startedAt;
                    acc += parsed.data.delta;
                    textNode.dataset.raw = acc;
                    textNode.innerHTML   = renderMarkdown(acc);
                    scrollToBottom();
                }
            }
        }
    }
    // The server's final text is authoritative: the deltas can carry a
    // trailing role tag or an echo tail that only the last pass can strip.
    const finalText = (doneData && doneData.reply) || acc || t("msg.empty");
    return {
        text: finalText,
        meta: Object.assign({}, doneData || {}, {
            first_token_ms: doneData?.first_token_ms ?? clientFirstMs,
        }),
    };
}

function parseSSE(block) {
    let event = "message";
    const lines = block.split("\n");
    let dataLines = [];
    for (const line of lines) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    if (!dataLines.length) return null;
    let data;
    try { data = JSON.parse(dataLines.join("\n")); }
    catch { data = { raw: dataLines.join("\n") }; }
    return { event, data };
}

function abortStream() {
    if (STATE.abort) STATE.abort.abort();
}

// ── Messages ──────────────────────────────────────────────────────────
function addMessage(role, text, opts = {}) {
    const el = document.createElement("div");
    el.className = `message ${role}`;
    const avatar = role === "user" ? t("msg.you") : "J";
    const name   = role === "user" ? t("msg.you") : t("msg.jin");
    // Nút hành động của từng tin nhắn: mỗi nút kèm huy hiệu ⓘ + bong bóng giải thích.
    const copyBtn = `<button class="msg-btn has-i" data-action="copy" aria-label="${t("msg.copy")}" data-tip="${t("tip.msgcopy")}">⧉ ${t("msg.copy")}</button>`;
    const delBtn  = `<button class="msg-btn has-i" data-action="del" aria-label="${t("msg.del")}" data-tip="${t("tip.msgdel")}">✕ ${t("msg.del")}</button>`;
    const actions = role === "assistant"
        ? `${copyBtn}
           <button class="msg-btn has-i" data-action="regen" aria-label="${t("msg.regen")}" data-tip="${t("tip.msgregen")}">↻ ${t("msg.regen")}</button>
           ${delBtn}`
        : `${copyBtn}
           <button class="msg-btn has-i" data-action="edit" aria-label="${t("msg.edit")}" data-tip="${t("tip.msgedit")}">✎ ${t("msg.edit")}</button>
           ${delBtn}`;
    el.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-role">${name}</div>
            <div class="message-text"></div>
            <div class="message-actions" hidden>${actions}</div>
        </div>
    `;
    if (opts.typing) {
        el.querySelector(".message-text").innerHTML = `
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>`;
    } else if (text) {
        const node = el.querySelector(".message-text");
        node.dataset.raw = text;
        node.innerHTML = renderMarkdown(text);
        decorateCodeBlocks(node);
        // Tin đã hoàn tất (mở lại hội thoại từ localStorage) -> hiện hàng nút.
        // Nếu không làm ở đây, finalizeMessage() mới là chỗ bỏ "hidden", tức là
        // mọi tin khôi phục từ localStorage sẽ vĩnh viễn mất Copy/Sinh lại/Sửa/Xoá.
        const actions = el.querySelector(".message-actions");
        if (actions) actions.hidden = false;
    }
    chatMessages.appendChild(el);
    scrollToBottom();
    wireMessageActions(el);
    return el;
}

function finalizeMessage(messageEl, textNode, text, errored = false, isFatal = false) {
    messageEl.querySelector(".typing-indicator")?.remove();
    textNode.dataset.raw = text;
    textNode.innerHTML = renderMarkdown(text);
    decorateCodeBlocks(textNode);
    if (errored) textNode.classList.add("error-text");
    if (!errored && !isFatal) {
        STATE.history.push({ role: "assistant", content: text });
    }
    const actions = messageEl.querySelector(".message-actions");
    if (actions) actions.hidden = false;
    scrollToBottom();
}

function attachSources(messageEl, sources) {
    let panel = messageEl.querySelector(".sources-panel");
    if (!panel) {
        panel = document.createElement("div");
        panel.className = "sources-panel";
        const header = document.createElement("div");
        header.className = "sources-header";
        header.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.35-4.35"/></svg> ${t("msg.web_sources")}`;
        panel.appendChild(header);
        const list = document.createElement("div");
        list.className = "sources-list";
        panel.appendChild(list);
        messageEl.querySelector(".message-content").appendChild(panel);
    }
    const list = panel.querySelector(".sources-list");
    list.innerHTML = "";
    sources.forEach((src, i) => {
        const a = document.createElement("a");
        a.className = "source-chip";
        a.href = src.url;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        let host = "";
        try { host = new URL(src.url).hostname.replace(/^www\./, ""); } catch { /* ignore */ }
        a.innerHTML = `
            <span class="source-num">${i + 1}</span>
            <span class="source-text">
                <span class="source-title">${escapeHtml(src.title || host || src.url)}</span>
                <span class="source-host">${escapeHtml(host)}</span>
            </span>`;
        list.appendChild(a);
    });
}

function wireMessageActions(el) {
    el.addEventListener("click", async (e) => {
        const btn = e.target.closest(".msg-btn");
        if (!btn) return;
        const node = el.querySelector(".message-text");
        const text = node.dataset.raw || node.textContent || "";
        if (btn.dataset.action === "copy") {
            try {
                await navigator.clipboard.writeText(text);
                btn.textContent = `✓ ${t("msg.copied")}`;
                setTimeout(() => (btn.textContent = `⧉ ${t("msg.copy")}`), 1200);
            } catch {/* ignore */}
        } else if (btn.dataset.action === "regen") {
            // Drop the last assistant turn AND its user turn, then replay it.
            const lastUser = STATE.history.filter((m) => m.role === "user").pop();
            const prev = el.previousElementSibling;
            el.remove();
            if (prev && prev.classList.contains("user")) prev.remove();
            rebuildFromDom();
            await resetServerSession();
            if (lastUser) send(lastUser.content);
        } else if (btn.dataset.action === "edit") {
            // Rewrite a user turn: truncate everything after it, load it back
            // into the input. The server session is reset because its prompt
            // history no longer matches what the user sees.
            const raw = node.dataset.raw || node.textContent || "";
            let cursor = el.nextElementSibling;
            while (cursor) {
                const next = cursor.nextElementSibling;
                cursor.remove();
                cursor = next;
            }
            el.remove();
            rebuildFromDom();
            await resetServerSession();
            chatInput.value = raw;
            sendBtn.disabled = false;
            autoResize();
            chatInput.focus();
        } else if (btn.dataset.action === "del") {
            const next = el.nextElementSibling;
            el.remove();
            if (next && next.classList.contains("message")) next.remove();
            rebuildFromDom();
            await resetServerSession();
        }
    });
}

// ── Markdown (very small, safe) ───────────────────────────────────────
function escapeHtml(s) {
    return s
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function renderMarkdown(src) {
    if (!src) return "";
    const parts = [];
    let i = 0;
    const text = src;
    while (i < text.length) {
        const fence = text.indexOf("```", i);
        if (fence === -1) {
            parts.push(renderBlocks(text.slice(i)));
            break;
        }
        if (fence > i) parts.push(renderBlocks(text.slice(i, fence)));
        const close = text.indexOf("```", fence + 3);
        if (close === -1) {
            parts.push(`<pre><code>${escapeHtml(text.slice(fence + 3))}</code></pre>`);
            break;
        }
        const block = text.slice(fence + 3, close);
        const newline = block.indexOf("\n");
        const lang = newline >= 0 ? block.slice(0, newline).trim() : "";
        const code = newline >= 0 ? block.slice(newline + 1) : block;
        parts.push(
            `<pre data-lang="${escapeHtml(lang)}"><code>${escapeHtml(code)}</code></pre>`,
        );
        i = close + 3;
    }
    return parts.join("");
}

function renderInline(src) {
    let out = escapeHtml(src);
    // inline code `like this`
    out = out.replace(/`([^`\n]+)`/g, (_, c) => `<code>${c}</code>`);
    // links [text](https://...) — href already escaped, only http(s)/mailto
    out = out.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+|mailto:[^\s)]+)\)/g,
        (_, label, href) => `<a href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>`);
    // bold **text**
    out = out.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
    // italic *text* (avoid touching ** already handled)
    out = out.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
    return out;
}

/** Line-level markdown: headings, lists, quotes, rules.
 *  Small on purpose — this renders trust-nothing model output, so everything
 *  goes through escapeHtml first and no raw HTML ever survives. */
function renderBlocks(src) {
    const out = [];
    let list = null;
    const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
    for (const rawLine of src.split("\n")) {
        const line = rawLine.replace(/\s+$/, "");
        let m;
        if ((m = /^(#{1,4})\s+(.*)$/.exec(line))) {
            closeList();
            const level = Math.min(6, m[1].length + 2);   // h1 -> h3 inside a bubble
            out.push(`<h${level}>${renderInline(m[2])}</h${level}>`);
        } else if (/^\s*[-*+]\s+/.test(line)) {
            if (list !== "ul") { closeList(); out.push("<ul>"); list = "ul"; }
            out.push(`<li>${renderInline(line.replace(/^\s*[-*+]\s+/, ""))}</li>`);
        } else if (/^\s*\d+[.)]\s+/.test(line)) {
            if (list !== "ol") { closeList(); out.push("<ol>"); list = "ol"; }
            out.push(`<li>${renderInline(line.replace(/^\s*\d+[.)]\s+/, ""))}</li>`);
        } else if (/^\s*>\s?/.test(line)) {
            closeList();
            out.push(`<blockquote>${renderInline(line.replace(/^\s*>\s?/, ""))}</blockquote>`);
        } else if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) {
            closeList();
            out.push("<hr>");
        } else if (!line.trim()) {
            closeList();
        } else {
            closeList();
            out.push(renderInline(line));
        }
    }
    closeList();
    return out.join("\n");
}

/** Attach a copy button to every code block (pack B). */
function decorateCodeBlocks(node) {
    node.querySelectorAll("pre").forEach((pre) => {
        if (pre.querySelector(".code-copy")) return;
        const btn = document.createElement("button");
        btn.className = "code-copy";
        btn.textContent = "⧉";
        btn.title = "Copy code";
        btn.addEventListener("click", async (ev) => {
            ev.stopPropagation();
            const code = pre.querySelector("code");
            try {
                await navigator.clipboard.writeText(code ? code.textContent : "");
                btn.textContent = "✓";
                setTimeout(() => (btn.textContent = "⧉"), 1200);
            } catch { /* clipboard blocked */ }
        });
        pre.appendChild(btn);
    });
}

// ── Misc ──────────────────────────────────────────────────────────────
function hideWelcome() {
    const w = $("welcomeScreen");
    if (w) w.style.display = "none";
}

function setBusy(busy) {
    STATE.isGenerating = busy;
    sendBtn.disabled = busy || !chatInput.value.trim();
    stopBtn.hidden = !busy || !streamToggle.checked;
    if (!busy) {
        STATE.abort = null;
        stopBtn.hidden = true;
    }
}

function saveSession(id) {
    STATE.sessionId = id;
    try { localStorage.setItem("jin.sessionId", id); } catch { /* ignore */ }
}

async function resetChat() {
    if (STATE.sessionId) {
        try {
            await fetch("/api/reset", {
                method: "POST",
                headers: apiHeaders(),
                body: JSON.stringify({ session_id: STATE.sessionId }),
            });
        } catch { /* ignore */ }
    }
    STATE.sessionId = null;
    STATE.history = [];
    try { localStorage.removeItem("jin.sessionId"); } catch { /* ignore */ }
    chatMessages.innerHTML = "";
    chatMessages.appendChild(buildWelcome());
}

function buildWelcome() {
    // Giữ khớp với markup welcome trong index.html (không còn orb/emoji).
    const tpl = document.createElement("div");
    tpl.className = "welcome";
    tpl.id = "welcomeScreen";
    tpl.innerHTML = `
        <div class="welcome-badge" aria-hidden="true">J</div>
        <h1 class="welcome-title">JinAI</h1>
        <p class="welcome-sub">${esc(t("welcome.sub"))}</p>
        <div class="suggestions">
            <button class="suggestion-card" data-prompt="Hi Jin, can you introduce yourself?">
                <span class="suggestion-title">${esc(t("welcome.c1.title"))}</span>
                <span class="suggestion-desc">${esc(t("welcome.c1.desc"))}</span>
            </button>
            <button class="suggestion-card" data-prompt="Explain how a transformer language model works in simple terms.">
                <span class="suggestion-title">${esc(t("welcome.c2.title"))}</span>
                <span class="suggestion-desc">${esc(t("welcome.c2.desc"))}</span>
            </button>
            <button class="suggestion-card" data-prompt="Write a short Python function that returns the n-th Fibonacci number.">
                <span class="suggestion-title">${esc(t("welcome.c3.title"))}</span>
                <span class="suggestion-desc">${esc(t("welcome.c3.desc"))}</span>
            </button>
            <button class="suggestion-card" data-prompt="Tell me a short, original story about a curious robot.">
                <span class="suggestion-title">${esc(t("welcome.c4.title"))}</span>
                <span class="suggestion-desc">${esc(t("welcome.c4.desc"))}</span>
            </button>
        </div>`;
    return tpl;
}

async function checkStatus() {
    try {
        const r = await fetch("/api/status");
        const d = await r.json();
        STATE.status = d;
        if (d.model_loaded) {
            statusDot.className = "status-dot online";
            modelBarBtn.classList.add("online");
            modelInfo.querySelector("span").textContent = d.model_params;
            modelMeta.textContent =
                `${d.scale} · ctx ${d.context_window} · ${d.device}`;
            // Chip ở topbar: model nào, bước nào — không cần mở panel.
            const file = (d.checkpoint_file || "").split(/[\\/]/).pop() || "model";
            // Cảnh báo ngay trên chip khi bản đang chạy không phải chat/instruct:
            // model pretrain chỉ nối tiếp văn bản nên trả lời rất dễ sai.
            const served = (d.checkpoints || []).find((f) => f.current) || {};
            const drift = !!served.role && !CHAT_ROLES.has(served.role);
            modelBarBtn.classList.toggle("warn", drift);
            topModel.textContent = (drift ? "⚠ " : "") + t("status.chip", {
                file, step: d.checkpoint_step ?? "?", device: d.device,
            });
        } else {
            statusDot.className = "status-dot offline";
            modelBarBtn.classList.remove("online", "warn");
            modelInfo.querySelector("span").textContent = t("status.no_model");
            modelMeta.textContent = t("status.train_first");
        }
        renderLab(d);
    } catch {
        statusDot.className = "status-dot offline";
        modelBarBtn.classList.remove("online", "warn");
        modelInfo.querySelector("span").textContent = t("status.offline");
        topModel.textContent = t("status.no_server");
    }
}

// ── Settings sheet + thu gọn sidebar ─────────────────────────────────
function openSettings() {
    settingsSheet.hidden = false;
    scrim.hidden = false;
    requestAnimationFrame(() => {
        settingsSheet.classList.add("open");
        scrim.classList.add("open");
    });
}

function closeSettings() {
    settingsSheet.classList.remove("open");
    scrim.classList.remove("open");
    setTimeout(() => {
        settingsSheet.hidden = true;
        scrim.hidden = true;
    }, 220);
}

function setCollapsed(value) {
    appShell.classList.toggle("collapsed", value);
    try { localStorage.setItem("jin.sidebarCollapsed", value ? "1" : "0"); }
    catch { /* ignore */ }
}

// ── Model Lab (pack C) ────────────────────────────────────────────────
function renderLab(d) {
    if (!d) return;
    const vram = d.vram_used_mb != null
        ? `${Math.round(d.vram_used_mb)} / ${Math.round(d.vram_total_mb)} MB`
        : "—";
    const served = (d.checkpoints || []).find((f) => f.current);
    const rows = [
        [t("lab.checkpoint"), d.checkpoint_file ? d.checkpoint_file.split(/[\\/]/).pop() : "—"],
        [t("lab.role"), served ? t("lab.role." + served.role) : "—"],
        [t("lab.step"), d.checkpoint_step != null ? d.checkpoint_step.toLocaleString() : "—"],
        [t("lab.valloss"), d.best_val_loss != null ? d.best_val_loss.toFixed(4) : "—"],
        [t("lab.device"), d.device],
        [t("lab.vram"), vram],
        [t("lab.ctx"), `${d.context_window} / ${d.max_positions}`],
        [t("lab.uptime"), fmtUptime(d.uptime_s)],
        [t("lab.requests"), d.requests_served],
    ];
    labGrid.innerHTML = rows.map(([k, v]) =>
        `<div class="lab-cell"><span class="lab-key">${esc(k)}</span>` +
        `<span class="lab-val" title="${esc(String(v))}">${esc(String(v))}</span></div>`
    ).join("");

    const autopick = d.checkpoint_default;
    if (labAuto) {
        labAuto.textContent = autopick
            ? t("lab.auto", { file: autopick.name, why: t("lab.why." + autopick.why) })
            : "";
    }

    // Rebuild the picker only when the on-disk list changed, so a manual
    // selection is not reset by the 5s poll. Same signature for the model bar.
    const files = d.checkpoints || [];
    const signature = files.map((f) => `${f.path || f.name}:${f.mtime}`).join("|");
    if (signature && signature !== labCheckpoint.dataset.signature) {
        labCheckpoint.dataset.signature = signature;
        paintCheckpointPicker(files);          // stat-level labels, instant
        loadLabCheckpoints();                  // then upgrade with step / val
        if (!modelBarMenu.hidden) renderModelBar(files);
    }
}

/** "kaggle_v4_60020/ · kaggle_v4_instruct.pt · instruct · step 60.020 · val 1.5870 · 1774 MB · 16/09 19:34" */
function ckptOptionLabel(f) {
    const where = f.folder ? `${f.folder}/` : "";
    const bits = [`${where}${f.name}`, t("lab.role." + (f.role || "legacy"))];
    bits.push(f.step != null
        ? t("lab.opt_step", { step: f.step.toLocaleString() })
        : t("lab.no_step"));
    if (f.val_loss != null) bits.push(t("lab.opt_val", { val: f.val_loss.toFixed(4) }));
    bits.push(`${f.size_mb} MB`, f.mtime);
    if (f.current) bits.push(t("lab.current"));
    else if (f.default) bits.push(t("lab.default_mark"));
    return bits.join(" · ");
}

/** Roles that know the chat template — mirrors ``config.CHAT_ROLES``. */
const CHAT_ROLES = new Set(["chat", "instruct"]);

// Checkpoint pretrain cần bấm hai lần mới đổi. Không dùng window.confirm():
// hộp thoại của trình duyệt khoá cả trang (và treo luôn panel preview).
let armedPath = null;

/** Serving checkpoint first, then the auto-pick, then chat by step, then rest. */
function sortCheckpoints(files) {
    return files.slice().sort((a, b) =>
        (a.current ? 0 : 1) - (b.current ? 0 : 1) ||
        (a.default ? 0 : 1) - (b.default ? 0 : 1) ||
        (CHAT_ROLES.has(b.role) ? 0 : 1) - (CHAT_ROLES.has(a.role) ? 0 : 1) ||
        ((b.step ?? -1) - (a.step ?? -1)) ||
        ((b.mtime_epoch ?? 0) - (a.mtime_epoch ?? 0)));
}

let ckptCache = [];

function paintCheckpointPicker(files) {
    if (files.length) ckptCache = files;
    if (!files.length) {
        labCheckpoint.innerHTML = `<option value="">${esc(t("lab.none"))}</option>`;
        return;
    }
    const prev = labCheckpoint.value;
    labCheckpoint.innerHTML = sortCheckpoints(files).map((f) =>
        `<option value="${esc(f.path)}">${esc(ckptOptionLabel(f))}</option>`
    ).join("");
    const keep = files.find((f) => f.path === prev);
    const current = files.find((f) => f.current);
    labCheckpoint.value = keep ? prev : (current ? current.path : files[0].path);
}

/** Header probe (step / val loss) — slower, so only when the panel is open. */
async function loadLabCheckpoints() {
    try {
        const r = await fetch("/api/model/checkpoints");
        if (!r.ok) return;
        const d = await r.json();
        if (d.default && labAuto) {
            labAuto.textContent = t("lab.auto", {
                file: d.default.name, why: t("lab.why." + d.default.why),
            });
        }
        paintCheckpointPicker(d.checkpoints || []);
    } catch { /* keep the stat-level labels from /api/status */ }
}

/** Warn before picking a file that cannot chat: it answers raw continuation. */
function onCheckpointPick() {
    const entry = ckptCache.find((f) => f.path === labCheckpoint.value);
    const pretrain = entry
        ? !CHAT_ROLES.has(entry.role)
        : /pretrain|^jingpt_(best|latest)\.pt$/i.test(labCheckpoint.value || "");
    labHint.textContent = pretrain ? t("lab.pretrain_note") : "";
}

function fmtUptime(seconds) {
    if (seconds == null) return "—";
    const s = Math.round(seconds);
    if (s < 60) return `${s}s`;
    if (s < 3600) return `${Math.floor(s / 60)}m`;
    return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

/** POST /api/model/reload, then drop everything tied to the old weights.
 *
 *  Throws an Error with a readable message; ``err.code === "no_token"`` means
 *  the admin token is missing. Shared by Model Lab and the topbar model bar so
 *  both swap checkpoints the same way.
 */
async function reloadCheckpoint(path) {
    const notice = adminNotice();
    if (notice) {
        const err = new Error(notice);
        err.code = (STATE.status && STATE.status.admin_mode) === "off"
            ? "admin_off" : "no_token";
        throw err;
    }
    const r = await fetch("/api/model/reload", {
        method: "POST",
        headers: apiHeaders(),
        body: JSON.stringify({ path }),
    });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(authError(r.status) || d.error || r.status);
    // Weights changed: the old chat is a different model's memory.
    chatMessages.querySelectorAll(".message").forEach((m) => m.remove());
    STATE.history = [];
    await resetServerSession();
    chatMessages.innerHTML = "";
    chatMessages.appendChild(buildWelcome());
    return d;
}

async function labReloadCheckpoint() {
    const path = labCheckpoint.value;
    if (!path) { labHint.textContent = t("lab.none"); return; }
    labHint.textContent = t("lab.loading");
    try {
        const d = await reloadCheckpoint(path);
        labHint.textContent = t("lab.done", {
            file: d.checkpoint_file, step: d.checkpoint_step, sec: d.load_s,
        });
        checkStatus();
        loadLabCheckpoints();
        loadBenchSparkline();
    } catch (e) {
        labHint.textContent = e.code === "no_token" ? t("lab.need_token")
            : e.code === "admin_off" ? t("modelbar.admin_off")
            : t("lab.call_failed", { err: e.message });
    }
}

async function loadBenchSparkline() {
    const token = adminToken();
    if (!token) {
        labChart.innerHTML = `<div class="lab-empty">${esc(t("lab.chart_token"))}</div>`;
        return;
    }
    try {
        const r = await fetch("/api/bench/history", {
            headers: { "Authorization": "Bearer " + token },
        });
        if (!r.ok) { labChart.innerHTML = `<div class="lab-empty">${esc(t("lab.chart_admin"))}</div>`; return; }
        const d = await r.json();
        drawLineChart(labChart, d.runs || [], "heldout_loss");
    } catch {
        labChart.innerHTML = `<div class="lab-empty">${esc(t("lab.chart_err"))}</div>`;
    }
}

function adminToken() {
    const typed = (labToken.value || "").trim();
    if (typed) {
        try { localStorage.setItem("jin.adminToken", typed); } catch { /* ignore */ }
        return typed;
    }
    try { return localStorage.getItem("jin.adminToken") || ""; } catch { return ""; }
}

/** Headers for any API call: JSON + the admin token when we have one.
 *
 *  The token is only *required* when the server was started with JINGPT_TOKEN
 *  (``/api/status.admin_mode === "token"``), but sending it whenever we have it
 *  keeps chat / reset / memory working in that mode too — otherwise the browser
 *  would hit a 401 the moment the token is switched on.
 */
function apiHeaders(extra) {
    const headers = Object.assign({ "Content-Type": "application/json" }, extra || {});
    const token = adminToken();
    if (token) headers["Authorization"] = "Bearer " + token;
    return headers;
}

/** Message thân thiện cho 401/403; "" khi lỗi không phải chuyện token. */
function authError(status) {
    return (status === 401 || status === 403) ? t("auth.needed") : "";
}

/** Cảnh báo khi checkpoint đang phục vụ không biết chat (pretrain). */
function pretrainLiveNotice() {
    const files = (STATE.status && STATE.status.checkpoints) || [];
    const served = files.find((f) => f.current);
    if (!served || !served.role || CHAT_ROLES.has(served.role)) return "";
    return t("modelbar.pretrain_live", { file: served.name });
}

/** Dòng nhắc khi server chưa cho đổi model/admin — "" nghĩa là đổi được ngay. */
function adminNotice() {
    const mode = STATE.status && STATE.status.admin_mode;
    if (mode === "off") return t("modelbar.admin_off");
    if (mode === "token" && !adminToken()) return t("modelbar.no_admin");
    return "";
}

// ── Model bar (topbar): model nào đang phục vụ, đổi tại chỗ ──────────
/** Rebuild the dropdown from a checkpoint list (stat-level or probed). */
function renderModelBar(files) {
    const list = (files && files.length) ? files : ckptCache;
    if (!list.length) {
        modelBarList.innerHTML =
            `<div class="model-bar-empty">${esc(t("modelbar.empty"))}</div>`;
        return;
    }
    modelBarList.innerHTML = list.map((f) => {
        const role = f.role || "legacy";
        const risky = !CHAT_ROLES.has(role);      // pretrain: chỉ nối tiếp văn bản
        const meta = [t("lab.role." + role)];
        if (f.step != null) meta.push(t("lab.opt_step", { step: f.step.toLocaleString() }));
        if (f.val_loss != null) meta.push(t("lab.opt_val", { val: f.val_loss.toFixed(4) }));
        meta.push(`${f.size_mb} MB`);
        const dir = f.folder ? `<span class="model-item-dir">${esc(f.folder)}/</span>` : "";
        const warn = risky
            ? `<span class="model-item-warn">⚠ ${esc(t("modelbar.pretrain_tag"))}</span>` : "";
        const mark = f.current
            ? `<span class="model-item-live">${esc(t("modelbar.current"))}</span>`
            : (f.default ? `<span class="model-item-star">${esc(t("lab.default_mark"))}</span>` : "");
        return `<button class="model-item${f.current ? " current" : ""}${risky ? " warn" : ""}` +
            `${armedPath === f.path ? " armed" : ""}"` +
            ` data-path="${esc(f.path)}" role="option"` +
            ` aria-selected="${f.current ? "true" : "false"}">` +
            `<span class="model-item-top">${dir}` +
            `<span class="model-item-name">${esc(f.name)}</span>${warn}${mark}</span>` +
            `<span class="model-item-meta">${esc(meta.join(" · "))}</span></button>`;
    }).join("");
}

function setModelBarHint(text, warn) {
    modelBarHint.textContent = text || "";
    modelBarHint.classList.toggle("warn", !!warn);
}

function openModelBar() {
    modelBarMenu.hidden = false;
    armedPath = null;
    modelBarBtn.setAttribute("aria-expanded", "true");
    renderModelBar(STATE.status && STATE.status.checkpoints);
    const autopick = STATE.status && STATE.status.checkpoint_default;
    // Ưu tiên cảnh báo đúng lúc: bản pretrain đang chạy là lý do số một khiến
    // câu trả lời lạc; local-first nên chỉ nhắc token khi server thật sự đòi.
    const pretrainNotice = pretrainLiveNotice();
    setModelBarHint(pretrainNotice || adminNotice() || (autopick
        ? t("lab.auto", { file: autopick.name, why: t("lab.why." + autopick.why) })
        : ""), !!pretrainNotice);
    requestAnimationFrame(() => modelBarMenu.classList.add("open"));
    loadModelBarCheckpoints();
}

function closeModelBar() {
    if (modelBarMenu.hidden) return;
    armedPath = null;
    modelBarMenu.classList.remove("open");
    modelBarBtn.setAttribute("aria-expanded", "false");
    setTimeout(() => { modelBarMenu.hidden = true; }, 160);
}

function toggleModelBar() {
    if (modelBarMenu.hidden) openModelBar();
    else closeModelBar();
}

/** Header probe (step / val loss) — same endpoint the Model Lab picker uses. */
async function loadModelBarCheckpoints() {
    try {
        const r = await fetch("/api/model/checkpoints");
        if (!r.ok) return;
        const d = await r.json();
        if (modelBarMenu.hidden) return;      // đóng rồi thì khỏi vẽ lại
        renderModelBar(d.checkpoints || []);
    } catch { /* giữ danh sách stat-level từ /api/status */ }
}

async function switchModel(path, el) {
    if (!path) return;
    const files = (STATE.status && STATE.status.checkpoints) || ckptCache;
    const entry = files.find((f) => f.path === path);
    if (entry && entry.current) { setModelBarHint(t("modelbar.current")); return; }
    // Đổi sang pretrain là tự bắn vào chân: lần bấm đầu chỉ "lên đạn".
    // Đánh dấu trên chính nút vừa bấm — vẽ lại cả danh sách ở đây sẽ làm nút bị
    // tách khỏi DOM, và listener "bấm ra ngoài" tưởng là bấm ngoài menu.
    if (entry && !CHAT_ROLES.has(entry.role || "legacy") && armedPath !== path) {
        armedPath = path;
        modelBarList.querySelectorAll(".model-item").forEach((b) => b.classList.remove("armed"));
        if (el) el.classList.add("armed");
        setModelBarHint(t("modelbar.confirm_pretrain", { file: entry.name }), true);
        return;
    }
    armedPath = null;
    setModelBarHint(t("modelbar.switching"));
    modelBarList.querySelectorAll(".model-item").forEach((b) => { b.disabled = true; });
    try {
        const d = await reloadCheckpoint(path);
        setModelBarHint(t("modelbar.switched", {
            file: (d.checkpoint_file || "").split(/[\\/]/).pop(),
            step: d.checkpoint_step ?? "?", sec: d.load_s,
        }));
        await checkStatus();
        renderModelBar(STATE.status && STATE.status.checkpoints);
        // Sau khi đổi, đưa dòng đang chạy vào tầm mắt (danh sách có thể đang cuộn).
        const live = modelBarList.querySelector(".model-item.current");
        if (live && live.scrollIntoView) live.scrollIntoView({ block: "nearest" });
        loadLabCheckpoints();
        loadBenchSparkline();
    } catch (e) {
        setModelBarHint(e.code === "no_token" ? t("modelbar.no_admin")
            : e.code === "admin_off" ? t("modelbar.admin_off")
            : t("modelbar.failed", { err: e.message }));
    } finally {
        modelBarList.querySelectorAll(".model-item").forEach((b) => { b.disabled = false; });
    }
}

// ── Memory panel (pack D) ─────────────────────────────────────────────
async function loadMemoryStats() {
    try {
        const r = await fetch("/api/memory/stats", { headers: apiHeaders() });
        if (!r.ok) { memStatsEl.innerHTML = `<div class="lab-empty">${esc(t("mem.disabled"))}</div>`; return; }
        const d = await r.json();
        const rows = [
            [t("mem.total"), d.total ?? 0],
            [t("mem.semantic"), d.semantic ?? 0],
            [t("mem.disputed"), d.disputed ?? 0],
            [t("mem.indexed"), (d.index && d.index.size != null) ? d.index.size : "—"],
        ];
        memStatsEl.innerHTML = rows.map(([k, v]) =>
            `<div class="lab-cell"><span class="lab-key">${esc(k)}</span>` +
            `<span class="lab-val">${esc(String(v))}</span></div>`
        ).join("");
    } catch {
        memStatsEl.innerHTML = `<div class="lab-empty">${esc(t("mem.read_err"))}</div>`;
    }
}

async function searchMemory() {
    const query = $("memSearchInput").value.trim();
    if (!query) return;
    memResults.innerHTML = `<div class="lab-empty">${esc(t("mem.searching"))}</div>`;
    try {
        const r = await fetch("/api/memory/search", {
            method: "POST",
            headers: apiHeaders(),
            body: JSON.stringify({ query, k: 5 }),
        });
        const d = await r.json().catch(() => ({}));
        if (!r.ok) { memResults.innerHTML = `<div class="lab-empty">${esc(d.error || r.status)}</div>`; return; }
        const hits = d.results || [];
        memResults.innerHTML = hits.length
            ? hits.map((h) =>
                `<div class="mem-hit"><div class="mem-hit-text">${esc((h.text || "").slice(0, 220))}</div>` +
                `<div class="mem-hit-meta">score ${h.score} · ${esc(h.confidence || "?")}` +
                `${h.source_url ? ` · <a href="${esc(h.source_url)}" target="_blank" rel="noopener noreferrer">nguồn</a>` : ""}</div></div>`
              ).join("")
            : `<div class="lab-empty">${esc(t("mem.no_hits"))}</div>`;
    } catch (e) {
        memResults.innerHTML = `<div class="lab-empty">${esc(t("mem.error", { err: e.message }))}</div>`;
    }
}

// ── Sampling presets + metrics (pack A) ──────────────────────────────
function applyPreset(name) {
    const p = PRESETS[name];
    if (!p) return;
    tempSlider.value = p.temperature;
    topkSlider.value = p.topK;
    toppSlider.value = p.topP;
    repSlider.value = p.rep;
    tokensSlider.value = p.maxTokens;
    ngramSelect.value = String(p.ngram);
    refreshSliderLabels();
    presetRow.querySelectorAll(".preset-chip").forEach((chip) =>
        chip.classList.toggle("active", chip.dataset.preset === name));
    presetHint.textContent = t(PRESET_NOTE_KEY[name] || "preset.note.custom");
    try { localStorage.setItem("jin.preset", name); } catch { /* ignore */ }
}

function refreshPresetHint() {
    const name = activePresetName();
    presetHint.textContent = t(PRESET_NOTE_KEY[name] || "preset.note.custom");
}

function activePresetName() {
    const chip = presetRow.querySelector(".preset-chip.active");
    return chip ? chip.dataset.preset : null;
}

/** Metrics strip under a reply: speed, first token, why it stopped. */
function attachMetrics(messageEl, meta) {
    if (!meta) return;
    messageEl._meta = meta;
    const content = messageEl.querySelector(".message-content");
    let row = content.querySelector(".message-meta");
    if (!row) {
        row = document.createElement("div");
        row.className = "message-meta";
        content.appendChild(row);
    }
    const pills = [];
    if (meta.tok_per_s != null) pills.push(t("metrics.tok_s", { v: meta.tok_per_s }));
    if (meta.first_token_ms != null) pills.push(t("metrics.first", { v: Math.round(meta.first_token_ms) }));
    if (meta.tokens_generated != null) pills.push(t("metrics.tokens", { v: meta.tokens_generated }));
    else if (meta.tokens != null) pills.push(t("metrics.tokens", { v: meta.tokens }));
    if (meta.checkpoint_step != null) pills.push(t("metrics.step", { v: meta.checkpoint_step }));
    if (meta.web_search_used) pills.push(t("metrics.web"));
    row.innerHTML = pills.map((p) => `<span class="meta-pill">${esc(p)}</span>`).join("");
    const stopKey = { eos: "metrics.stop_eos", repeat: "metrics.stop_repeat",
                      length: "metrics.stop_length", aborted: "metrics.stop_aborted" }[meta.stop_reason];
    if (meta.stop_reason === "repeat") {
        const warn = document.createElement("span");
        warn.className = "meta-pill warn";
        warn.textContent = t("metrics.warn_repeat");
        warn.dataset.tip = stopKey ? t(stopKey) : (meta.stop_label || "");
        row.appendChild(warn);
    }
}

// ── Conversation store (pack B) ──────────────────────────────────────
const CONVO_KEY = "jin.conversations";
const MAX_CONVOS = 40;

function uid() {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}

function loadConvos() {
    try { return JSON.parse(localStorage.getItem(CONVO_KEY)) || []; } catch { return []; }
}

function saveConvos(list) {
    try { localStorage.setItem(CONVO_KEY, JSON.stringify(list.slice(0, MAX_CONVOS))); }
    catch { /* quota — keep chatting anyway */ }
}

function isDefaultConvoTitle(title) {
    return title === I18N.vi["convo.new"] || title === I18N.en["convo.new"];
}

function ensureConvo() {
    const convos = loadConvos();
    let convo = convos.find((c) => c.id === STATE.convoId);
    if (!convo) {
        convo = { id: uid(), title: t("convo.new"), sessionId: null,
                  createdAt: Date.now(), messages: [] };
        convos.unshift(convo);
        STATE.convoId = convo.id;
        saveConvos(convos);
        try { localStorage.setItem("jin.activeConvo", convo.id); } catch { /* ignore */ }
    }
    return convo;
}

function updateConvo(convo) {
    const convos = loadConvos();
    const i = convos.findIndex((c) => c.id === convo.id);
    if (i === -1) convos.unshift(convo); else convos[i] = convo;
    saveConvos(convos);
}

function pushTurn(role, text, meta) {
    const convo = ensureConvo();
    convo.messages.push({ role, text, meta: meta || null });
    if (role === "user" && isDefaultConvoTitle(convo.title)) {
        convo.title = text.slice(0, 48);
    }
    if (STATE.sessionId) convo.sessionId = STATE.sessionId;
    updateConvo(convo);
    renderConvoList();
}

function rebuildFromDom() {
    const kept = [];
    STATE.history = [];
    chatMessages.querySelectorAll(".message").forEach((el) => {
        const role = el.classList.contains("user") ? "user" : "assistant";
        const node = el.querySelector(".message-text");
        const text = node?.dataset.raw || "";
        if (!text) return;
        STATE.history.push({ role, content: text });
        kept.push({ role, text, meta: el._meta || null });
    });
    const convo = ensureConvo();
    convo.messages = kept;
    if (STATE.sessionId) convo.sessionId = STATE.sessionId;
    updateConvo(convo);
    renderConvoList();
}

async function resetServerSession() {
    if (!STATE.sessionId) return;
    try {
        await fetch("/api/reset", {
            method: "POST",
            headers: apiHeaders(),
            body: JSON.stringify({ session_id: STATE.sessionId }),
        });
    } catch { /* offline: the server will just start a fresh session */ }
}

function renderConvoList() {
    const convos = loadConvos();
    if (!convos.length) {
        convoListEl.innerHTML = `<div class="lab-empty">${esc(t("convo.empty"))}</div>`;
        return;
    }
    convoListEl.innerHTML = convos.map((c) =>
        `<div class="convo-item${c.id === STATE.convoId ? " active" : ""}" data-id="${c.id}">` +
        `<span class="convo-title" title="${esc(c.title)}">${esc(c.title)}</span>` +
        `<span class="convo-count">${c.messages.length}</span>` +
        `<button class="convo-del" data-del="${c.id}" title="${esc(t("msg.del"))}">✕</button></div>`
    ).join("");
}

function openConvo(id) {
    const convo = loadConvos().find((c) => c.id === id);
    if (!convo) return;
    STATE.convoId = id;
    STATE.sessionId = convo.sessionId || null;
    try { localStorage.setItem("jin.activeConvo", id); } catch { /* ignore */ }
    hideWelcome();
    chatMessages.querySelectorAll(".message").forEach((m) => m.remove());
    STATE.history = [];
    convo.messages.forEach((m) => {
        const el = addMessage(m.role, m.text);
        if (m.meta) attachMetrics(el, m.meta);
        STATE.history.push({ role: m.role, content: m.text });
    });
    renderConvoList();
    scrollToBottom();
}

function deleteConvo(id) {
    const convos = loadConvos().filter((c) => c.id !== id);
    saveConvos(convos);
    if (STATE.convoId === id) {
        STATE.convoId = null;
        STATE.sessionId = null;
        STATE.history = [];
        try { localStorage.removeItem("jin.activeConvo"); } catch { /* ignore */ }
        chatMessages.innerHTML = "";
        chatMessages.appendChild(buildWelcome());
    }
    renderConvoList();
}

function restoreActiveConvo() {
    const id = STATE.convoId;
    if (id && loadConvos().some((c) => c.id === id)) openConvo(id);
}

function convoToMarkdown(convo) {
    const lines = [`# ${convo.title}`, "",
        `_${t("export.note", { time: new Date(convo.createdAt).toLocaleString() })}_`, ""];
    convo.messages.forEach((m) => {
        lines.push(m.role === "user" ? `## ${t("export.you")}` : `## ${t("export.jin")}`);
        lines.push("", m.text || "", "");
        if (m.meta && m.meta.tok_per_s != null) {
            lines.push(`_${m.meta.tok_per_s} tok/s · ${m.meta.tokens_generated} token` +
                       `${m.meta.stop_label ? ` · ${m.meta.stop_label}` : ""}_`, "");
        }
    });
    return lines.join("\n");
}

function downloadFile(name, text, mime) {
    const blob = new Blob([text], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
}

function exportConvo(kind) {
    const convo = loadConvos().find((c) => c.id === STATE.convoId);
    if (!convo || !convo.messages.length) {
        footerStatus.textContent = t("convo.nothing");
        return;
    }
    const safe = (convo.title || "jin-chat").replace(/[^\p{L}\p{N} _-]/gu, "").trim().slice(0, 40) || "jin-chat";
    if (kind === "json") {
        downloadFile(`${safe}.json`, JSON.stringify(convo, null, 2), "application/json");
    } else {
        downloadFile(`${safe}.md`, convoToMarkdown(convo), "text/markdown");
    }
}

// ── Wiring for the new panels ────────────────────────────────────────
function wireExtraEvents() {
    presetRow.addEventListener("click", (e) => {
        const chip = e.target.closest(".preset-chip");
        if (chip) applyPreset(chip.dataset.preset);
    });
    $("labReload").addEventListener("click", labReloadCheckpoint);
    labCheckpoint.addEventListener("change", onCheckpointPick);
    // Header probe costs disk reads the first time — do it when it is looked at.
    labSection.addEventListener("toggle", () => {
        if (labSection.open) { loadLabCheckpoints(); loadBenchSparkline(); }
    });
    labToken.addEventListener("change", loadBenchSparkline);
    // Model bar: danh sách checkpoint trên đĩa, bấm là đổi luôn.
    modelBarBtn.addEventListener("click", (e) => { e.stopPropagation(); toggleModelBar(); });
    modelBarList.addEventListener("click", (e) => {
        const item = e.target.closest(".model-item");
        if (item && !item.disabled) switchModel(item.dataset.path, item);
    });
    $("modelBarLab").addEventListener("click", () => {
        closeModelBar();
        openSettings();
        labSection.open = true;      // mở sẵn mục Model Lab (có ô nhập token)
        labToken.focus();
    });
    document.addEventListener("click", (e) => {
        if (modelBarMenu.hidden) return;
        if (e.target.closest && e.target.closest("#modelBar")) return;
        if (e.target.isConnected === false) return;   // nút đã bị vẽ lại trong lúc xử lý click
        closeModelBar();
    });
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && !modelBarMenu.hidden) closeModelBar();
    });
    $("memSearchBtn").addEventListener("click", searchMemory);
    $("memSearchInput").addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); searchMemory(); }
    });
    $("convoExportMd").addEventListener("click", () => exportConvo("md"));
    $("convoExportJson").addEventListener("click", () => exportConvo("json"));
    convoListEl.addEventListener("click", (e) => {
        const del = e.target.closest(".convo-del");
        if (del) { deleteConvo(del.dataset.del); return; }
        const item = e.target.closest(".convo-item");
        if (item) openConvo(item.dataset.id);
    });
    // Leaving "Cân bằng" by hand un-highlights the chips.
    SLIDERS.forEach(([slider]) => slider.addEventListener("input", () => {
        const chip = presetRow.querySelector(".preset-chip.active");
        if (!chip) return;
        const p = PRESETS[chip.dataset.preset];
        if (!p) return;
        const same = parseFloat(slider.value) ===
            ({ [tempSlider.id]: p.temperature, [topkSlider.id]: p.topK,
               [toppSlider.id]: p.topP, [repSlider.id]: p.rep,
               [tokensSlider.id]: p.maxTokens }[slider.id]);
        if (!same) {
            chip.classList.remove("active");
            presetHint.textContent = t("preset.note.custom");
        }
    }));
    labToken.value = localStorage.getItem("jin.adminToken") || "";
    const savedPreset = localStorage.getItem("jin.preset");
    if (savedPreset && PRESETS[savedPreset]) applyPreset(savedPreset);
    else presetHint.textContent = t("preset.note.balanced");
    loadBenchSparkline();
    initResize();
    $("langSelect").addEventListener("change", (e) => setLanguage(e.target.value));

    // Settings sheet
    $("settingsBtn").addEventListener("click", openSettings);
    $("topSettingsBtn").addEventListener("click", openSettings);
    $("settingsClose").addEventListener("click", closeSettings);
    scrim.addEventListener("click", closeSettings);
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && !settingsSheet.hidden) closeSettings();
    });

    // Thu gọn thanh bên (nhớ lựa chọn giữa các lần mở trang).
    collapseBtn.addEventListener("click", () =>
        setCollapsed(!appShell.classList.contains("collapsed")));
    if (localStorage.getItem("jin.sidebarCollapsed") === "1") setCollapsed(true);
}

// ── Mobile ────────────────────────────────────────────────────────────
function toggleSidebar() {
    if (window.innerWidth > 768) {
        // Desktop: thanh bên trượt ra ngoài, nút menu chỉ để mở lại.
        setCollapsed(false);
        return;
    }
    sidebar.classList.toggle("open");
    if (sidebar.classList.contains("open")) addOverlay();
    else removeOverlay();
}
function addOverlay() {
    if (document.querySelector(".sidebar-overlay")) return;
    const o = document.createElement("div");
    o.className = "sidebar-overlay active";
    o.addEventListener("click", () => { sidebar.classList.remove("open"); removeOverlay(); });
    document.body.appendChild(o);
}
function removeOverlay() {
    document.querySelector(".sidebar-overlay")?.remove();
}function scrollToBottom() {
    requestAnimationFrame(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    });
}

// ── Growth Dashboard (Phase 7) ────────────────────────────────────────
// Admin-gated: /api/learn/status + /api/bench/history need JINGPT_TOKEN
// (loopback). Without a token the panel shows a clear disabled message.
const growthPanel    = $("growthPanel");
const growthToken    = $("growthToken");
const growthError    = $("growthError");
const growthCards    = $("growthCards");
const growthCharts   = $("growthCharts");
const growthFrontier = $("growthFrontier");
const growthRegistry = $("growthRegistry");

function growthOpen() {
    // Full-screen page — chat stays untouched underneath.
    growthPanel.hidden = false;
    growthToken.value = localStorage.getItem("jin.adminToken") || "";
    loadGrowth();
}
function growthClose() {
    growthPanel.hidden = true;
}
document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !growthPanel.hidden) growthClose();
});

function growthErr(msg) {
    growthError.textContent = msg;
    growthError.hidden = false;
}

async function fetchAdmin(url, opts) {
    const token = growthToken.value.trim();
    const headers = Object.assign({}, opts?.headers || {});
    if (token) headers["Authorization"] = "Bearer " + token;
    const r = await fetch(url, Object.assign({}, opts, { headers }));
    return r;
}

async function loadGrowth() {
    growthCards.innerHTML = "";
    growthCharts.innerHTML = "";
    growthFrontier.innerHTML = "";
    growthRegistry.innerHTML = "";
    growthError.hidden = true;
    if (growthToken.value.trim()) localStorage.setItem("jin.adminToken", growthToken.value.trim());
    const hasToken = !!growthToken.value.trim();

    try {
        const [sr, br] = await Promise.all([
            fetchAdmin("/api/learn/status"),
            fetchAdmin("/api/bench/history"),
        ]);
        if (sr.status === 403 || br.status === 403) {
            growthErr(hasToken
                ? "Admin API rejected the token (or not on loopback)."
                : "Admin API is OFF — set JINGPT_TOKEN (and run on 127.0.0.1) to see the dashboard.");
            return;
        }
        const st = await sr.json();
        const bh = await br.json();
        renderGrowth(st, bh.runs || []);
    } catch (e) {
        growthErr("Dashboard load failed: " + e);
    }
}

function renderGrowth(st, runs) {
    const cards = [];
    if (st.memory) {
        cards.push({ label: "Memory entries", value: st.memory.total || 0,
                     sub: `${st.memory.semantic || 0} semantic · ${st.memory.disputed || 0} disputed` });
    }
    if (st.ledger && typeof st.ledger.total_cycles === "number") {
        cards.push({ label: "Learning cycles", value: st.ledger.total_cycles,
                     sub: `${st.ledger.merged || 0} merged · ${st.ledger.rolled_back || 0} rolled back` });
    }
    if (runs.length) {
        const last = runs[runs.length - 1];
        cards.push({ label: "Latest probe pass", value: fmtPct(last.probe_pass_rate),
                     sub: last.checkpoint + " · " + last.timestamp });
        cards.push({ label: "Latest decode tok/s", value: last.decode_tok_per_s,
                     sub: "peak VRAM " + (last.peak_vram_gb != null ? last.peak_vram_gb + " GB" : "n/a") });
    }
    if (st.budget) {
        cards.push({ label: "Curiosity budget", value: st.budget.max_cycles + " cycles",
                     sub: st.budget.max_requests + " req · " + st.budget.max_gb_day + " GB/day" });
    }
    cards.push({ label: "Cycle running", value: st.running ? "YES" : "no",
                 sub: "/api/learn/trigger" });

    growthCards.innerHTML = cards.map((c) =>
        `<div class="growth-card"><div class="growth-card-label">${c.label}</div>` +
        `<div class="growth-card-value">${c.value}</div>` +
        `<div class="growth-card-sub">${c.sub || ""}</div></div>`
    ).join("");

    // Line charts from bench history.
    const series = [
        { key: "heldout_loss", label: "Held-out loss (lower = better)" },
        { key: "probe_pass_rate", label: "Probe pass rate" },
        { key: "decode_tok_per_s", label: "Decode tok/s" },
    ];
    const chartHtml = series.map((s) =>
        `<div class="growth-chart"><div class="growth-chart-title">${s.label}</div>` +
        `<div class="growth-chart-svg" data-key="${s.key}"></div></div>`
    ).join("");
    growthCharts.innerHTML = chartHtml;
    series.forEach((s) => drawLineChart(
        document.querySelector(`.growth-chart-svg[data-key="${s.key}"]`), runs, s.key));

    // Frontier topics.
    const frontier = st.frontier || [];
    growthFrontier.innerHTML = `<div class="growth-col-title">Knowledge frontier (live)</div>` +
        (frontier.length
            ? frontier.map((f) =>
                `<div class="growth-row"><span>${esc(f.topic || f)}</span>` +
                `<span class="growth-score">${f.score != null ? f.score.toFixed(3) : ""}</span></div>`
              ).join("")
            : `<div class="growth-empty">No frontier yet — run a curiosity cycle.</div>`);

    // Registry + checkpoints.
    const reg = st.registry || {};
    const ck = reg.checkpoints || [];
    const gold = reg.golden;
    growthRegistry.innerHTML = `<div class="growth-col-title">Checkpoint registry</div>` +
        (gold ? `<div class="growth-row"><span>🛡 ${esc(gold.path)}</span><span class="growth-score">golden</span></div>` : "") +
        (ck.length
            ? ck.map((e) =>
                `<div class="growth-row"><span>${esc(e.path)}</span>` +
                `<span class="growth-score">${esc(e.changelog?.action || "promoted")}</span></div>`
              ).join("")
            : `<div class="growth-empty">No promoted checkpoints yet.</div>`);
}

function fmtPct(v) {
    if (v == null) return "—";
    return (v * 100).toFixed(0) + "%";
}
function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) =>
        ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function drawLineChart(el, runs, key) {
    const pts = runs.map((r, i) => ({ i, v: r[key] })).filter((p) => p.v != null);
    if (pts.length < 2) {
        el.innerHTML = `<div class="growth-empty">Not enough bench runs yet (need ≥2).</div>`;
        return;
    }
    const W = 520, H = 120, PAD = 8;
    const vs = pts.map((p) => p.v);
    const lo = Math.min(...vs), hi = Math.max(...vs);
    const span = (hi - lo) || 1;
    const x = (i) => PAD + (i / (pts.length - 1)) * (W - 2 * PAD);
    const y = (v) => H - PAD - ((v - lo) / span) * (H - 2 * PAD);
    const line = pts.map((p) => `${x(p.i).toFixed(1)},${y(p.v).toFixed(1)}`).join(" ");
    const last = pts[pts.length - 1];
    el.innerHTML =
        `<svg viewBox="0 0 ${W} ${H}" class="growth-svg">` +
        `<polyline points="${line}" fill="none" stroke="currentColor" stroke-width="2"/>` +
        `<circle cx="${x(last.i).toFixed(1)}" cy="${y(last.v).toFixed(1)}" r="3"/>` +
        `</svg>` +
        `<div class="growth-chart-meta">latest: ${last.v} (${runs[last.i].checkpoint})</div>`;
}

})();
