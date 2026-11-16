"""
Unified data fetcher for JIN AI.

Merges the functionality of the old ``fetch_corpus.py`` and
``fetch_hf_datasets.py`` into one script.

Supports two catalog formats:
- ``sources.json`` (raw public-domain text corpora)
- ``hf_datasets.json`` (SLM-grade instruction/chat datasets)

Examples:
    python fetch_datasets.py --catalog sources.json
    python fetch_datasets.py --catalog hf_datasets.json --only tinystories-v2
    python fetch_datasets.py --catalog hf_datasets.json --list
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import random
import re
import hashlib
import shutil
import time
import urllib.parse
import sys
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
import zipfile
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

import config


USER_AGENT = "JINAI-Fetcher/2.0"
SEED_HF_DIR = os.path.join(config.ROOT_DIR, "data", "seed", "hf")

def resolve_hf_files(entry: dict) -> list[str]:
    """Discover real shard names, then pin downloads to the repository SHA.

    Never invent shard counts or download every shard by default. Dataset
    metadata is fetched via stdlib, no datasets/HF client dependency.
    """
    repo = entry["hf_repo"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Invalid HF repository ID")
    revision = entry.get("revision", "main")
    api = f"https://huggingface.co/api/datasets/{repo}/revision/{urllib.parse.quote(revision, safe='')}?blobs=true"
    with open_url(api) as response:
        raw = response.read((32 << 20) + 1)
    if len(raw) > (32 << 20):
        raise ValueError("Repository metadata too large")
    info = json.loads(raw)
    sha = info["sha"]
    pattern = re.compile(entry["file_pattern"])
    candidates = [item for item in info.get("siblings", []) if pattern.fullmatch(item["rfilename"])]
    candidates.sort(key=lambda item: item["rfilename"])
    candidates = candidates[:max(1, int(entry.get("max_shards", 1)))]
    if not candidates:
        raise ValueError(f"No train shards matched {entry['file_pattern']} in {repo}")
    entry["resolved_revision"] = sha
    entry["resolved_bytes"] = sum(item.get("size", 0) for item in candidates)
    entry["resolved_files"] = [{"path":i["rfilename"], "bytes":i.get("size"),
                                "sha256":i.get("lfs", {}).get("sha256")} for i in candidates]
    return [f"https://huggingface.co/datasets/{repo}/resolve/{sha}/{urllib.parse.quote(i['rfilename'], safe='/')}" for i in candidates]


def iter_parquet_qa(path: Path, entry: dict) -> Iterable[dict]:
    import pyarrow.parquet as pq
    fields = [entry["question_column"], entry["answer_column"]]
    quality_field = entry.get("quality_column")
    if quality_field:
        fields.append(quality_field)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=fields):
        for row in batch.to_pylist():
            if quality_field:
                try:
                    if float(row.get(quality_field) or 0) < entry.get("min_quality", 1):
                        continue
                except (ValueError, TypeError):
                    continue
            q, a = row.get(fields[0]), row.get(fields[1])
            if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
                yield _instruction_to_messages(q, a)


def fetch_row_sample(entry: dict, raw_root: Path, rows: int, budget: int | None) -> int:
    """Small real sample when train shards are several GB. Not full dataset.

    The viewer uses current revision (cannot pin via this endpoint); record
    actual fetched bytes/hash/time, and don't claim revision reproducibility.
    """
    if not 1 <= rows <= 10000:
        raise ValueError("sample-rows must be 1..10000; use shards for larger runs")
    total_bytes = 0
    requests = []
    def documents():
        nonlocal total_bytes
        for offset in range(0, rows, 100):
            url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode({
                "dataset":entry["hf_repo"],"config":entry["server_config"],"split":"train",
                "offset":offset,"length":min(100,rows-offset)})
            with open_url(url) as response:
                raw = response.read((16 << 20) + 1)
            if len(raw) > (16 << 20):raise ValueError("Sample response exceeds budget")
            total_bytes += len(raw)
            if budget is not None and total_bytes > budget:raise ValueError("Sample download budget exceeded")
            payload = json.loads(raw)
            requests.append({"url":url,"response_sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)})
            batch = payload.get("rows", [])
            if not batch:break
            for record in batch:
                text = record.get("row", {}).get("text")
                if isinstance(text, str) and text.strip():yield text
    name = f"rows-sample-{rows}"
    target = raw_root/f"hf-{entry['id']}"/(name+".txt")
    if target.exists():
        print(f"[Skip] Existing sample preserved: {target}");return 0
    count, size = write_raw_text_shard(raw_root,entry["id"],name,documents(),
                     max_bytes=int(entry.get("max_output_mb",128)*(1 << 20)))
    from lineage import atomic_json,sha256_file
    atomic_json(str(target)+".manifest.json",{"dataset_id":entry["id"],"kind":"training_row_sample",
        "license":entry.get("license"),"dataset_card":entry.get("dataset_card"),"requested_rows":rows,
        "actual_rows":count,"output_bytes":size,"network_bytes":total_bytes,"output_sha256":sha256_file(str(target)),
        "revision":"viewer current; not revision-pinned","requests":requests,
        "fetched_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})
    return total_bytes


# ── HTTP helpers ────────────────────────────────────────────────────────

def open_url(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            return urllib.request.urlopen(req, timeout=45)
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
            time.sleep(1 + attempt * 2)
        except (TimeoutError, urllib.error.URLError):
            if attempt == 2:raise
            time.sleep(1 + attempt * 2)
    raise RuntimeError("HTTP retries exhausted")


def download_with_progress(url: str, destination: Path, limit_bytes: int | None = None) -> Path:
    destination = Path(config.output_path(destination))
    os.makedirs(destination.parent, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing overwrite: {destination}")
    tmp = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    print(f"[GET] {url}")
    if tmp.exists():
        raise FileExistsError(f"Partial file already exists; inspect/recover it before retrying: {tmp}")
    try:
        with open_url(url) as resp, open(tmp, "xb") as out:
            total = resp.headers.get("Content-Length")
            total = int(total) if total and total.isdigit() else None
            if limit_bytes is not None and total is not None and total > limit_bytes:
                raise ValueError("Complete file exceeds download budget; choose fewer/smaller shards")
            if shutil.disk_usage(destination.parent).free < (total or limit_bytes or (1 << 30)) + (256 << 20):
                raise ValueError("Insufficient disk plus safety reserve")
            read = 0
            while True:
                buf = resp.read(65536)
                if not buf:
                    break
                if limit_bytes is not None and read + len(buf) > limit_bytes:
                    raise ValueError("Download budget exceeded; partial file NOT published")
                out.write(buf);digest.update(buf);read += len(buf)
                if read % (64 * 1024 * 1024) < 65536:
                    print(f"[Download] {read/1e6:.1f} MB", flush=True)
            if total is not None and read != total:
                raise ValueError(f"Truncated response: expected {total}, got {read}")
            if not read:
                raise ValueError("Empty download")
            out.flush();os.fsync(out.fileno())
        tmp.replace(destination)
        from lineage import atomic_json
        atomic_json(str(destination) + ".manifest.json", {"url":url,"final_url":resp.geturl(),
                    "bytes":read,"sha256":digest.hexdigest(),"status":"complete",
                    "fetched_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})
    except Exception:
        if tmp.exists():tmp.unlink()
        raise
    print(f"[Saved] {destination} ({destination.stat().st_size/1e6:.1f} MB)")
    return destination


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def decode_text(raw: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


# ── Sources catalog (raw text) ──────────────────────────────────────────

WIKI_TEXT_TAGS = {"abstract", "title"}


def parse_wikimedia_abstract_dump(raw: bytes) -> str:
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as handle:
        tree = ET.parse(handle)
    texts = [e.text.strip() for e in tree.iter() for tag in (e.tag.rsplit("}", 1)[-1],)
             if tag in WIKI_TEXT_TAGS and e.text and e.text.strip()]
    return "\n".join(texts)


def extract_zip_text(raw: bytes, suffixes: tuple[str, ...] = (".txt",)) -> str:
    texts = []
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for m in archive.namelist():
            if m.endswith("/") or not m.lower().endswith(suffixes):
                continue
            with archive.open(m) as f:
                texts.append(decode_text(f.read()))
    return "\n".join(texts)


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def write_source_text(output_dir: Path, source_id: str, source_name: str, text: str) -> Path:
    output_dir = Path(config.output_path(output_dir))
    target_dir = output_dir / source_id
    os.makedirs(target_dir, exist_ok=True)
    path = target_dir / f"{slugify(source_name) or source_id}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def load_sources(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(json.load(f).get("sources", []))


def fetch_source(source: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    sid = str(source["id"])
    name = str(source.get("name", sid))
    stype = str(source.get("type", "text"))
    url = str(source["url"])
    print(f"[Fetch] {sid}: {name}")
    raw = fetch_bytes(url)
    if stype == "text":
        text = decode_text(raw)
    elif stype == "gzip_text":
        text = decode_text(gzip.decompress(raw))
    elif stype == "wikimedia_abstract_gz":
        text = parse_wikimedia_abstract_dump(raw)
    elif stype == "zip_text":
        text = extract_zip_text(raw, tuple(source.get("member_suffixes", [".txt"])))
    elif stype == "mediawiki_xml_bz2":
        print(f"[Skip] {sid}: .bz2 parsing not yet supported.")
        return
    else:
        raise ValueError(f"Unknown source type: {stype}")
    text = text.strip()
    if not text:
        print(f"[Skip] {sid}: empty payload")
        return
    path = write_source_text(output_dir, sid, name, text)
    print(f"[Saved] {path} ({len(text):,} chars)")


# ── HF catalog (instruction / chat datasets) ────────────────────────────

def load_catalog(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload.get("datasets", []))


def _open_maybe_gzip(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _require_pyarrow():
    try:
        import pyarrow.parquet  # noqa: F401
        return True
    except ImportError:
        print("[Skip] Install pyarrow for parquet datasets: pip install pyarrow")
        return False


# -------- Pretrain text converters --------------------------------------

def iter_plain_text(path: Path) -> Iterable[str]:
    with _open_maybe_gzip(path) as f:
        buf = []
        for line in f:
            if line.strip() in ("<|endoftext|>", "<|end|>"):
                text = "".join(buf).strip()
                if text:
                    yield text
                buf = []
            else:
                buf.append(line)
        tail = "".join(buf).strip()
        if tail:
            yield tail


def iter_parquet_text(path: Path, text_columns: list[str]) -> Iterable[str]:
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=256, columns=text_columns):
        for row in batch.to_pylist():
            for col in text_columns:
                if row.get(col):
                    yield str(row[col])


# -------- Instruction format converters ---------------------------------

def _instruction_to_messages(instruction: str, response: str, input_: str | None = None) -> dict:
    user = instruction.strip()
    if input_ and input_.strip():
        user = f"{user}\n\n{input_.strip()}"
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": response.strip()}]}


def iter_alpaca_json(path: Path) -> Iterable[dict]:
    for row in _stream_json_documents(path):
        inst = row.get("instruction") or ""
        resp = row.get("output") or row.get("response") or ""
        if inst and resp:
            yield _instruction_to_messages(inst, resp, row.get("input"))


def iter_dolly_jsonl(path: Path) -> Iterable[dict]:
    with _open_maybe_gzip(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            inst = row.get("instruction") or ""
            resp = row.get("response") or row.get("output") or ""
            if inst and resp:
                yield _instruction_to_messages(inst, resp, row.get("context"))


def iter_bactrian_json(path: Path) -> Iterable[dict]:
    for row in _stream_json_documents(path):
        inst = row.get("instruction") or ""
        resp = row.get("output") or ""
        if inst and resp:
            yield _instruction_to_messages(inst, resp, row.get("input"))


def iter_lima_jsonl(path: Path) -> Iterable[dict]:
    with _open_maybe_gzip(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            convs = row.get("conversations") or []
            if len(convs) < 2:
                continue
            msgs = []
            for i, turn in enumerate(convs):
                role = "user" if i % 2 == 0 else "assistant"
                content = turn if isinstance(turn, str) else turn.get("value", "")
                if not content:
                    continue
                msgs.append({"role": role, "content": content})
            if len(msgs) >= 2:
                yield {"messages": msgs}


def iter_gsm8k_jsonl(path: Path) -> Iterable[dict]:
    with _open_maybe_gzip(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = row.get("question") or ""
            a = row.get("answer") or ""
            if q and a:
                yield _instruction_to_messages(q, a.replace("####", "Answer:"))


def iter_metamath_json(path: Path) -> Iterable[dict]:
    for row in _stream_json_documents(path):
        q = row.get("query") or row.get("question") or ""
        a = row.get("response") or ""
        if q and a:
            yield _instruction_to_messages(q, a)


def iter_oasst_trees(path: Path) -> Iterable[dict]:
    with _open_maybe_gzip(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tree = json.loads(line)
            root = tree.get("prompt") or tree
            for path_msgs in _oasst_walk(root, []):
                if len(path_msgs) >= 2 and path_msgs[-1]["role"] == "assistant":
                    yield {"messages": path_msgs}


def _oasst_walk(node: dict, acc: list[dict]):
    if not isinstance(node, dict):
        return
    role = node.get("role")
    text = node.get("text")
    mapped = None
    if role == "prompter":
        mapped = "user"
    elif role == "assistant":
        mapped = "assistant"
    new_acc = acc + [{"role": mapped, "content": text}] if mapped and text else acc
    replies = node.get("replies") or []
    if not replies:
        if new_acc:
            yield new_acc
        return
    for child in replies:
        yield from _oasst_walk(child, new_acc)


def _map_sharegpt_turns(convs) -> list[dict]:
    """Map ShareGPT-style turns (``from``/``value``) or chat turns
    (``role``/``content``) to the internal ``{role, content}`` format."""
    msgs = []
    for turn in convs:
        if not isinstance(turn, dict):
            continue
        role = turn.get("from") or turn.get("role")
        content = turn.get("value") or turn.get("content")
        if not role or not content:
            continue
        if role in {"human", "user", "prompter"}:
            role = "user"
        elif role in {"gpt", "assistant", "bot"}:
            role = "assistant"
        elif role in {"system"}:
            role = "system"
        else:
            continue
        msgs.append({"role": role, "content": content})
    return msgs


def iter_parquet_messages(path: Path, text_columns: list[str]) -> Iterable[dict]:
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=256, columns=text_columns):
        for row in batch.to_pylist():
            for col in text_columns:
                convs = row.get(col)
                if not convs:
                    continue
                msgs = _map_sharegpt_turns(convs)
                if len(msgs) >= 2:
                    yield {"messages": msgs}
                break


def _stream_json_documents(path: Path) -> Iterable[dict]:
    """Stream top-level dicts from a JSON array file or a JSONL file.

    Uses incremental ``raw_decode`` so a ~1 GB payload (OpenHermes 2.5
    ships as one big JSON array) never sits fully in memory.
    """
    with _open_maybe_gzip(path) as handle:
        head = handle.read(1 << 16)
        handle.seek(0)
        if not head.lstrip().startswith("["):
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj
            return

        decoder = json.JSONDecoder()
        buf = ""
        expect_value = True
        while True:
            buf = buf.lstrip()
            if not buf:
                chunk = handle.read(1 << 20)
                if not chunk:
                    return
                buf += chunk
                continue
            if expect_value:
                if buf[0] == "]":
                    return
                if buf[0] == "[":
                    buf = buf[1:]
                    continue
                if buf[0] == ",":
                    buf = buf[1:]
                    continue
                try:
                    obj, end = decoder.raw_decode(buf)
                except json.JSONDecodeError:
                    chunk = handle.read(1 << 20)
                    if not chunk:
                        return
                    buf += chunk
                    continue
                buf = buf[end:]
                expect_value = False
                if isinstance(obj, dict):
                    yield obj
            else:
                expect_value = True


def iter_json_messages(path: Path) -> Iterable[dict]:
    for row in _stream_json_documents(path):
        convs = row.get("conversations") or row.get("messages") or row.get("turns")
        if not convs or not isinstance(convs, list):
            continue
        msgs = _map_sharegpt_turns(convs)
        if len(msgs) >= 2:
            yield {"messages": msgs}


# -------- Writers --------------------------------------------------------

def write_raw_text_shard(output_root: Path, dataset_id: str, shard_name: str,
                         texts: Iterable[str], *, max_bytes: int | None = None) -> tuple[int, int]:
    output_root = Path(config.output_path(output_root))
    target_dir = output_root / f"hf-{dataset_id}"
    os.makedirs(target_dir, exist_ok=True)
    target_file = target_dir / f"{shard_name}.txt"
    if target_file.exists():
        raise FileExistsError(f"Converted output exists; use a new --output-dir: {target_file}")
    tmp = target_file.with_suffix(".txt.part")
    count = 0
    total_bytes = 0
    with open(tmp, "x", encoding="utf-8", newline="\n") as out:
        for text in texts:
            text = text.strip()
            if not text:
                continue
            record = text + "\n<|endoftext|>\n"
            size = len(record.encode("utf-8"))
            if max_bytes and total_bytes + size > max_bytes:
                break
            out.write(record)
            count += 1
            total_bytes += size
        out.flush();os.fsync(out.fileno())
    tmp.replace(target_file)
    print(f"[Raw ] {target_file} ({count:,} docs, {total_bytes/1e6:.2f} MB)")
    return count, total_bytes


def write_seed_chat_shard(dataset_id: str, shard_name: str, rows: Iterable[dict],
                        *, max_rows: int | None = None) -> tuple[int, int]:
    seed_dir = config.output_path(SEED_HF_DIR)
    os.makedirs(seed_dir, exist_ok=True)
    target_file = Path(seed_dir) / f"{dataset_id}-{shard_name}.jsonl"
    if target_file.exists():
        raise FileExistsError(f"Converted output exists; use a new --seed-dir: {target_file}")
    tmp = target_file.with_suffix(".jsonl.part")
    count = 0
    total_bytes = 0
    with open(tmp, "x", encoding="utf-8", newline="\n") as out:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False)
            out.write(line + "\n")
            count += 1
            total_bytes += len((line + "\n").encode("utf-8"))
            if max_rows and count >= max_rows:
                break
        out.flush();os.fsync(out.fileno())
    tmp.replace(target_file)
    print(f"[Seed] {target_file} ({count:,} conversations)")
    return count, total_bytes


def process_hf_dataset(entry: dict[str, Any], cache_dir: Path, raw_root: Path,
                       limit_bytes: int | None, max_rows: int | None) -> None:
    did = str(entry["id"])
    name = str(entry.get("name", did))
    kind = str(entry.get("kind", "pretrain_text"))
    fmt = str(entry.get("format", "text"))
    files = resolve_hf_files(entry) if entry.get("hf_repo") else list(entry.get("files", []))
    requires = list(entry.get("requires", []))
    text_columns = list(entry.get("text_columns", []))
    if not files:
        print(f"[Skip] {did}: no files")
        return
    if "pyarrow" in requires and not _require_pyarrow():
        return
    print(f"\n===== {did} ({name}) =====")
    remaining = entry.get("remaining_download_bytes")
    for url in files:
        shard_name = slugify(Path(url).name.split("?")[0])
        dl_path = cache_dir / f"hf-{did}" / Path(url).name.split("?")[0]
        expected_output = (raw_root / f"hf-{did}" / f"{shard_name}.txt") if kind.startswith("pretrain") else Path(SEED_HF_DIR) / f"{did}-{shard_name}.jsonl"
        if expected_output.exists():
            print(f"[Skip] Output already exists (preserved): {expected_output}")
            continue
        if not dl_path.exists():
            try:
                allowed = min(v for v in (limit_bytes,remaining) if v is not None) if any(v is not None for v in (limit_bytes,remaining)) else None
                download_with_progress(url, dl_path, limit_bytes=allowed)
                if remaining is not None:
                    remaining -= dl_path.stat().st_size
            except Exception as exc:
                raise RuntimeError(f"Download failed for {did}: {exc}") from exc
        else:
            print(f"[Cache] {dl_path}")
        try:
            _convert_one(entry, fmt, kind, text_columns, dl_path, shard_name, raw_root, max_rows)
            from lineage import atomic_json, sha256_file
            atomic_json(str(expected_output) + ".manifest.json", {"dataset_id":did,
                "license":entry.get("license"),"dataset_card":entry.get("dataset_card"),
                "revision":entry.get("resolved_revision"),"url":url,"source_sha256":sha256_file(str(dl_path)),
                "output_sha256":sha256_file(str(expected_output)),"output_bytes":expected_output.stat().st_size,
                "limits":{"max_rows":max_rows,"max_output_mb":entry.get("max_output_mb")},"status":"converted"})
        except Exception as exc:
            raise RuntimeError(f"Conversion failed for {did}: {exc}") from exc


def _convert_one(entry, fmt, kind, text_columns, dl_path, shard_name, raw_root, max_rows):
    did = str(entry["id"])
    if kind.startswith("pretrain"):
        if fmt == "text":
            write_raw_text_shard(raw_root, did, shard_name, iter_plain_text(dl_path),
                                 max_bytes=int(entry["max_output_mb"] * (1 << 20)) if entry.get("max_output_mb") else None)
        elif fmt == "parquet":
            write_raw_text_shard(raw_root, did, shard_name, iter_parquet_text(dl_path, text_columns or ["text"]),
                                 max_bytes=int(entry["max_output_mb"] * (1 << 20)) if entry.get("max_output_mb") else None)
        return
    if fmt == "parquet_qa":
        rows = iter_parquet_qa(dl_path, entry)
        write_seed_chat_shard(did, shard_name, rows, max_rows=max_rows)
        return
    converters = {
        "alpaca_json": iter_alpaca_json,
        "dolly_jsonl": iter_dolly_jsonl,
        "bactrian_json": iter_bactrian_json,
        "lima_jsonl": iter_lima_jsonl,
        "gsm8k_jsonl": iter_gsm8k_jsonl,
        "metamath_json": iter_metamath_json,
        "oasst_trees_jsonl_gz": iter_oasst_trees,
        "parquet_messages": iter_parquet_messages,
        "json_messages": iter_json_messages,
    }
    converter = converters.get(fmt)
    if converter is None:
        raise ValueError(f"Unknown format: {fmt}")
    if fmt == "parquet_messages":
        rows = converter(dl_path, text_columns or ["conversations"])
    else:
        rows = converter(dl_path)
    write_seed_chat_shard(did, shard_name, rows, max_rows=max_rows)


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch datasets for JIN AI")
    p.add_argument("--catalog", default=None,
                   help="Path to catalog JSON (sources.json or hf_datasets.json)")
    # Sources-specific
    p.add_argument("--output-dir", type=str, default=config.RAW_DATA_DIR)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--max-sources", type=int, default=None)
    # HF-specific
    p.add_argument("--cache-dir", default=os.path.join(config.CACHE_DIR, "hf"))
    p.add_argument("--limit-mb", type=float, default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--list", action="store_true")
    p.add_argument("--plan", action="store_true", help="Resolve pinned remote shards, report bytes; no payload downloads")
    p.add_argument("--max-shards", type=int, default=None)
    p.add_argument("--sample-rows", type=int, default=None, help="Fetch bounded training-row samples via HF viewer instead of multi-GB parquet files")
    p.add_argument("--seed-dir", default=SEED_HF_DIR)
    p.add_argument("--max-output-mb", type=float, default=None)
    p.add_argument("--total-download-mb", type=float, default=None, help="Cumulative NEW-download budget")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global SEED_HF_DIR
    args.seed_dir = config.output_path(args.seed_dir)
    args.output_dir = config.output_path(args.output_dir)
    args.cache_dir = config.output_path(args.cache_dir)
    SEED_HF_DIR = args.seed_dir
    for value in (args.limit_mb, args.max_output_mb, args.total_download_mb):
        if value is not None and value <= 0:
            raise SystemExit("Byte budgets must be positive")
    catalog_path = args.catalog
    if not catalog_path:
        # Auto-detect: prefer hf_datasets.json, fallback to sources.json
        for candidate in ["hf_datasets.json", "sources.json"]:
            full = os.path.join(config.ROOT_DIR, candidate)
            if os.path.exists(full):
                catalog_path = full
                break
    if not catalog_path or not os.path.exists(catalog_path):
        raise SystemExit("No catalog found. Specify --catalog or place hf_datasets.json / sources.json in project root.")

    with open(catalog_path, encoding="utf-8") as handle:
        is_hf = "datasets" in json.load(handle)

    if args.list:
        items = load_catalog(catalog_path) if is_hf else load_sources(catalog_path)
        for item in items:
            flag = "x" if item.get("enabled") else " "
            n = item.get("name", item["id"])
            k = item.get("kind", "?") if is_hf else "raw"
            print(f"[{flag}] {item['id']:30s} {k:20s} {n}")
        return

    if is_hf:
        catalog = load_catalog(catalog_path)
        selected_ids = set(args.only or [])
        unknown = selected_ids - {e["id"] for e in catalog}
        if unknown:
            raise SystemExit(f"Unknown dataset IDs: {sorted(unknown)}")
        selected = [e for e in catalog if e["id"] in selected_ids] if selected_ids else [e for e in catalog if e.get("enabled")]
        if not selected:
            print("[Info] No datasets selected. Use --only or enable entries in the catalog.")
            return
        cache_dir = Path(args.cache_dir)
        raw_root = Path(args.output_dir)
        limit_bytes = int(args.limit_mb * 1024 * 1024) if args.limit_mb else None
        print(f"[Catalog] {len(selected)} dataset(s)")
        plan = []
        failures = []
        remaining = int(args.total_download_mb * (1 << 20)) if args.total_download_mb else None
        for entry in selected:
            if args.max_shards is not None:
                entry["max_shards"] = max(1, args.max_shards)
            if args.max_output_mb is not None:
                entry["max_output_mb"] = args.max_output_mb
            try:
                if args.plan:
                    files = resolve_hf_files(entry) if entry.get("hf_repo") else entry.get("files", [])
                    plan.append({"id":entry["id"],"license":entry.get("license"),"revision":entry.get("resolved_revision"),
                                 "files":files,"known_bytes":entry.get("resolved_bytes"),"domain":entry.get("domain")})
                    continue
                # Budget includes new cached payloads, not already present files.
                if args.sample_rows:
                    if not entry.get("server_config") or not entry.get("hf_repo") or not str(entry.get("kind", "")).startswith("pretrain"):
                        raise ValueError("Row sampling requires pretrain repo + explicit server_config")
                    consumed = fetch_row_sample(entry, raw_root, args.sample_rows, remaining)
                    if remaining is not None:remaining -= consumed
                    continue
                before = sum(p.stat().st_size for p in cache_dir.rglob("*") if p.is_file()) if cache_dir.exists() else 0
                effective_limit = min(v for v in (limit_bytes,remaining) if v is not None) if any(v is not None for v in (limit_bytes,remaining)) else None
                if remaining is not None and remaining <= 0:
                    raise ValueError("Total download budget exhausted")
                entry["remaining_download_bytes"] = remaining
                process_hf_dataset(entry, cache_dir=cache_dir, raw_root=raw_root, limit_bytes=effective_limit, max_rows=args.max_rows)
                after = sum(p.stat().st_size for p in cache_dir.rglob("*") if p.is_file())
                if remaining is not None:
                    remaining -= max(0, after-before)
            except Exception as exc:
                failures.append({"id":entry["id"],"error":str(exc)})
                print(f"[Error] {entry['id']}: {exc}")
        if args.plan:
            from lineage import atomic_json
            atomic_json(os.path.join(config.ROOT_DIR,"bench_results","data_download_plan.json"), {"datasets":plan,"failures":failures})
            print(json.dumps(plan, ensure_ascii=True, indent=2))
        if failures:
            raise SystemExit(f"{len(failures)} dataset(s) failed; no success claimed")
        print("\nDone. Run:  python build_instruct_dataset.py")
    else:
        sources = load_sources(catalog_path)
        selected_ids = set(args.only or [])
        selected = [s for s in sources if s.get("enabled", True)]
        if selected_ids:
            selected = [s for s in selected if s["id"] in selected_ids]
        if args.max_sources:
            selected = selected[:args.max_sources]
        if not selected:
            raise RuntimeError("No enabled sources matched.")
        os.makedirs(args.output_dir, exist_ok=True)
        for src in selected:
            try:
                fetch_source(src, output_dir=args.output_dir)
            except Exception as exc:
                print(f"[Error] {src.get('id')}: {exc}")


if __name__ == "__main__":
    random.seed(1337)
    main()