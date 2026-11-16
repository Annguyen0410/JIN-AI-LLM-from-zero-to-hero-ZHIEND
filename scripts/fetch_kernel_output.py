"""Tải output của một Kaggle kernel về ổ D mà KHÔNG nạp cả file vào RAM.

Vì sao script này tồn tại
------------------------
``kaggle kernels output`` (kaggle 2.2.4, ``kaggle/api/kaggle_api_extended.py:6718``)
làm đúng thế này::

    download_response = requests.get(item.url, stream=True)
    with open(outfile, "wb") as out:
        out.write(download_response.content)

``.content`` đọc trọn body vào RAM: một checkpoint 1.24GB cần ~2.5GB cấp phát liền
mạch. Tệ hơn, file được mở ở mode ``"wb"`` (truncate) TRƯỚC khi đọc, nên khi
``MemoryError`` nổ ra thì trên đĩa còn lại đúng một file **0 byte trông như thật**.
Chuyện đó đã xảy ra ngày 16/09 và suýt làm cháy một session Kaggle 12h: file rỗng
được copy vào ``D:\\jin-ckpt`` rồi upload lên dataset, console vẫn in
``Upload successful: jingpt_latest.pt (0B)``.

Script này lấy danh sách file bằng đúng request API mà CLI dùng, rồi ghi từng
chunk 4MB thẳng xuống đĩa (đỉnh RAM ~4MB thay vì ~2.5GB) vào ``<đích>.part`` và
chỉ ``os.replace`` khi tải xong hẳn — mạng đứt giữa chừng không bao giờ để lại
một file rỗng "trông như đã tải xong". Cuối cùng nó tự kiểm định: file nhỏ hơn
1GB coi như hỏng và script thoát với mã khác 0, nên không thể lỡ tay upload.

Yêu cầu: ``kaggle`` CLI (đã có trong .venv) — script dùng ``requests`` đi kèm.
Xác thực lấy từ ``KAGGLE_CONFIG_DIR`` (trên máy này là ``D:\\kaggle``); chạy
``. .\\scripts\\env.ps1`` trước cho chắc.

Ví dụ::

    # Xem Kaggle đang giữ file gì, khỏi tải nhầm (vài giây)
    python scripts/fetch_kernel_output.py annguyen0410/notebookc6ea0ca54d --list-only

    # Tải checkpoint 46k + 37k về scratch, tự bỏ qua file đã đủ
    python scripts/fetch_kernel_output.py annguyen0410/notebookc6ea0ca54d \\
        --out "D:\\JINAI Main\\scratch\\downloads\\out_latest" --require-step 46000
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

# Một checkpoint 147.74M tham số nặng ~1.24GB. Nhỏ hơn ngưỡng này gần như chắc
# chắn là tải hỏng (đúng ca 0-byte ngày 16/09) — và tải hỏng thì không được upload.
MIN_PLAUSIBLE_BYTES = 1_000_000_000

# Đỉnh RAM khi tải = đúng kích thước chunk này (thay vì bằng cả file).
CHUNK_BYTES = 4 * 1024 * 1024

# Đích mặc định là scratch của nhà mới (D:\JINAI Main\scratch\downloads), TUYỆT ĐỐI
# không mặc định vào một folder dataset: `kaggle datasets version -p <folder>`
# upload MỌI file trong folder đó, nên một checkpoint lỡ rơi vào
# D:\JINAI Main\kaggle\jin-ckpt (hoặc kéo về qua junction D:\jin-kaggle) sẽ bị
# đẩy lên dataset ngay lần upload kế tiếp.
DEFAULT_OUT = os.path.join("D:" + os.sep, "JINAI Main", "scratch", "downloads", "out_latest")
DEFAULT_PATTERN = r"jingpt_(latest|best)\.pt"


# ── Helpers thuần (test được, không cần mạng) ───────────────────────────────

def enable_utf8_stdout() -> None:
    """Console Windows mặc định cp1252 — in tiếng Việt là UnicodeEncodeError.

    Cùng cách train.py/chat.py xử lý; main() gọi hàm này trước khi in bất cứ gì.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def human_bytes(n: int) -> str:
    """1536 -> ``1.50 KiB``."""
    for unit, limit in (("GiB", 1024 ** 3), ("MiB", 1024 ** 2), ("KiB", 1024)):
        if n >= limit:
            return f"{n / limit:.2f} {unit}"
    return f"{n} B"


def match_files(files: list, pattern: str) -> list:
    """Lọc các mục output theo regex khớp trên ``file_name``."""
    rx = re.compile(pattern)
    return [f for f in files if rx.search(f.file_name or "")]


def size_gate(size: int) -> bool:
    """True nếu kích thước file đủ tin cậy để dùng (>= MIN_PLAUSIBLE_BYTES)."""
    return size >= MIN_PLAUSIBLE_BYTES


# ── Kaggle API ─────────────────────────────────────────────────────────────

def build_api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def list_output(api, kernel: str, page_size: int = 20):
    """Trả về ``(owner, slug, files, log)`` — cùng request CLI dùng, không tải gì."""
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiListKernelSessionOutputRequest,
    )

    owner, slug, _ = api.parse_kernel_string(kernel)
    files: list = []
    log = ""
    token = None

    with api.build_kaggle_client() as kaggle:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name = owner
            req.kernel_slug = slug
            req.page_size = page_size
            if token:
                req.page_token = token
            resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
            files.extend(f for f in (resp.files or []) if f is not None)
            if resp.log and not log:
                log = resp.log
            token = resp.next_page_token
            if not token:
                break

    return owner, slug, files, log


def remote_size(url: str) -> int:
    """Content-Length của file trên storage, chỉ đọc header (không tải body)."""
    import requests

    with requests.get(url, stream=True, timeout=(15, 60)) as resp:
        resp.raise_for_status()
        return int(resp.headers.get("Content-Length") or 0)


# ── Tải ────────────────────────────────────────────────────────────────────

def stream_download(url: str, dest: str, *, force: bool = False) -> int | None:
    """Ghi ``url`` ra ``dest`` theo chunk. Trả về số byte, hoặc None nếu bỏ qua."""
    import requests

    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from storage import output_path
    dest = output_path(dest)
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    part = dest + ".part"

    with requests.get(url, stream=True, timeout=(15, 300)) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length") or 0)

        if not force and total and os.path.exists(dest) and os.path.getsize(dest) == total:
            print(f"[skip] {os.path.basename(dest)} — đã đủ {human_bytes(total)}")
            return None

        written = 0
        started = time.time()
        shown = started
        with open(part, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=CHUNK_BYTES):
                if not chunk:
                    continue
                fh.write(chunk)
                written += len(chunk)
                now = time.time()
                if now - shown >= 2.0:
                    shown = now
                    speed = written / max(now - started, 1e-6) / 1024 ** 2
                    pct = f"{written / total * 100:5.1f}%" if total else "  ?  "
                    print(f"\r  {pct} {human_bytes(written):>12}  {speed:5.2f} MiB/s",
                          end="", flush=True)
        print()

    if total and written != total:
        os.remove(part)
        raise RuntimeError(f"tải thiếu {written:,}/{total:,} byte — đã xoá {part}")

    os.replace(part, dest)
    return written


# ── Kiểm định ──────────────────────────────────────────────────────────────

def inspect_checkpoint(path: str) -> dict:
    """Đọc metadata checkpoint. Ném ValueError nếu file không dùng được."""
    size = os.path.getsize(path)
    print(f"[verify] {path}")
    print(f"[verify]   size      = {human_bytes(size)} ({size:,} byte)")

    if not size_gate(size):
        raise ValueError(
            f"chỉ {size:,} byte, dưới ngưỡng {human_bytes(MIN_PLAUSIBLE_BYTES)} "
            f"— file rỗng hoặc tải hỏng. KHÔNG upload file này."
        )

    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError("checkpoint không phải dict — không phải file JIN AI.")

    meta = {
        "iteration": ckpt.get("iteration"),
        "val_loss": ckpt.get("val_loss"),
        "best_loss": ckpt.get("best_loss"),
        "model_config": ckpt.get("model_config") or {},
        "tokenizer_state": ckpt.get("tokenizer_state") or {},
    }
    cfg = meta["model_config"]
    tok = meta["tokenizer_state"]
    print(f"[verify]   iteration = {meta['iteration']:,}" if isinstance(meta["iteration"], int)
          else f"[verify]   iteration = {meta['iteration']}")
    print(f"[verify]   val_loss  = {meta['val_loss']}   best_loss = {meta['best_loss']}")
    if cfg:
        print(f"[verify]   model     = {cfg.get('n_layers')}L, {cfg.get('n_heads')}H, "
              f"dim {cfg.get('embed_dim')}, vocab {cfg.get('vocab_size')}")
    if tok:
        print(f"[verify]   tokenizer = {tok.get('kind')} (vocab {tok.get('vocab_size')})")

    del ckpt
    import gc

    gc.collect()
    return meta


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tải output Kaggle theo kiểu streaming (an toàn với RAM thấp).")
    p.add_argument("kernel", help="owner/slug của notebook (vd annguyen0410/notebookc6ea0ca54d)")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help=f"Thư mục đích (mặc định {DEFAULT_OUT}); cấu trúc thư mục "
                        f"output được giữ nguyên")
    p.add_argument("--pattern", default=DEFAULT_PATTERN,
                   help=f"Regex lọc tên file (mặc định {DEFAULT_PATTERN})")
    p.add_argument("--force", action="store_true",
                   help="Tải lại cả khi file đích đã đúng kích thước")
    p.add_argument("--list-only", action="store_true",
                   help="Chỉ in tên + kích thước Kaggle đang giữ, không tải")
    p.add_argument("--no-verify", action="store_true",
                   help="Bỏ qua bước kiểm định size/step sau khi tải")
    p.add_argument("--require-step", type=int, default=None,
                   help="Thoát lỗi nếu không file nào có iteration đúng bằng số này "
                        "(vd 46000 cho não 46k)")
    p.add_argument("--expect-step", type=int, default=None,
                   help="Chỉ áp cổng step cho đúng một file (dùng kèm --pattern hẹp)")
    p.add_argument("--no-log", action="store_true",
                   help="Không lưu log của kernel vào --out")
    return p.parse_args(argv)


def main(argv=None) -> int:
    enable_utf8_stdout()
    args = parse_args(argv)
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from storage import configure_runtime, output_path
    configure_runtime()
    args.out = output_path(args.out)
    print(f"[kaggle] config dir = {os.environ.get('KAGGLE_CONFIG_DIR', '(mặc định ~/.kaggle)')}")
    print(f"[kaggle] đích       = {args.out}")

    api = build_api()
    owner, slug, files, log = list_output(api, args.kernel)
    print(f"[kaggle] {owner}/{slug}: {len(files)} mục output")

    selected = match_files(files, args.pattern)
    if not selected:
        print(f"[kaggle] Không file nào khớp pattern {args.pattern!r}. Kaggle đang giữ:")
        for f in files:
            print(f"    {f.file_name}")
        return 2

    if args.list_only:
        for f in selected:
            size = remote_size(f.url)
            print(f"  {f.file_name:<46} {human_bytes(size):>14}  ({size:,} byte)")
        return 0

    failures: list[str] = []
    ok: list[tuple[str, dict]] = []

    for f in selected:
        dest = os.path.join(args.out, *str(f.file_name).split("/"))
        print(f"\n[get ] {f.file_name}")
        try:
            written = stream_download(f.url, dest, force=args.force)
        except Exception as exc:  # mạng, quyền, đĩa...
            print(f"[fatal] {f.file_name}: {exc}")
            failures.append(f.file_name)
            continue
        if written is not None:
            print(f"[get ] xong {human_bytes(written)} -> {dest}")

        if args.no_verify:
            ok.append((dest, {}))
            continue
        try:
            meta = inspect_checkpoint(dest)
        except Exception as exc:
            print(f"[verify] FAIL: {exc}")
            failures.append(f.file_name)
            continue
        ok.append((dest, meta))

    if log and not args.no_log:
        log_path = os.path.join(args.out, f"{slug}.log")
        os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(log)
        print(f"\n[log ] kernel log -> {log_path} ({human_bytes(os.path.getsize(log_path))})")

    print("\n" + "=" * 68)
    for dest, meta in ok:
        step = meta.get("iteration")
        print(f"  OK    {os.path.basename(dest):<24} {human_bytes(os.path.getsize(dest)):>12}"
              f"  step={step}")
    for name in failures:
        print(f"  FAIL  {name}")

    if failures:
        print(f"\n[X] {len(failures)} file KHÔNG đạt — đừng copy/upload những file này.")
        return 1

    if args.expect_step is not None and len(ok) == 1:
        step = ok[0][1].get("iteration")
        if step != args.expect_step:
            print(f"\n[X] mong đợi step {args.expect_step:,} nhưng file ghi {step}.")
            return 1

    if args.require_step is not None:
        steps = [m.get("iteration") for _, m in ok]
        if args.require_step not in steps:
            print(f"\n[X] không file nào có iteration = {args.require_step:,} "
                  f"(đã thấy: {steps}).")
            return 1

    print("\n[v] Tất cả file đều đạt cổng kiểm định.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
