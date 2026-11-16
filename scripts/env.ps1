# scripts/env.ps1 — dồn mọi thứ nặng về ổ D trước khi train / tải checkpoint.
#
# Vì sao cần: ổ C của máy này chỉ còn ~2.8GB (99% đầy) trong khi pagefile 15.8GB
# nằm trên chính ổ đó. Khi RAM vật lý cạn, Windows không giãn được pagefile nên
# các cấp phát bộ nhớ lớn nổ ngay:
#   MemoryError khi requests đọc trọn checkpoint 1.24GB, "OpenBLAS error: Memory
#   allocation still failed", và bench.py chết im lặng với exit code âm.
# TEMP/TMP của user cũng đang trỏ vào C:\Users\nguye\AppData\Local\Temp.
#
# Dùng (dấu chấm + khoảng trắng ở đầu là bắt buộc — dot-source thì biến mới sống
# trong phiên hiện tại, gọi .\scripts\env.ps1 không có dấu chấm thì biến chết
# theo subshell):
#
#     . .\scripts\env.ps1
#
# Không ghi gì vào registry. Muốn cố định vĩnh viễn (mở PowerShell mới là có),
# chạy tay phần dưới — thay đổi này nằm NGOÀI thư mục project nên tự bạn quyết:
#
#     [Environment]::SetEnvironmentVariable("TEMP",  "D:\tmp",        "User")
#     [Environment]::SetEnvironmentVariable("TMP",   "D:\tmp",        "User")
#     [Environment]::SetEnvironmentVariable("HF_HOME",    "D:\JINAI Main\cache\hf",    "User")
#     [Environment]::SetEnvironmentVariable("TORCH_HOME", "D:\JINAI Main\cache\torch", "User")
#
# Cache của pip đã nằm sẵn trên D (pip config: global.cache-dir = D:\pip-cache);
# biến dưới đây chỉ để chắc chắn, kể cả khi config bị reset.

# Nhà chung của dự án là D:\JINAI Main — xem README.md trong đó để biết bản đồ.
# Các đường dẫn cũ D:\jin-ckpt / D:\jin-kaggle / D:\jin-tokencache / D:\kaggle /
# D:\jin-loop-dl giờ là junction trỏ vào nhà mới, nên mọi lệnh trong
# KAGGLE_RUNBOOK.md / KAGGLE_GUIDE.md vẫn chạy y nguyên không phải sửa gì.
$targets = [ordered]@{
    "TEMP"              = "D:\tmp"
    "TMP"               = "D:\tmp"
    "HF_HOME"           = "D:\JINAI Main\cache\hf"
    "TORCH_HOME"        = "D:\JINAI Main\cache\torch"
    "XDG_CACHE_HOME"    = "D:\JINAI Main\cache"
    "PIP_CACHE_DIR"     = "D:\pip-cache"
    "KAGGLE_CONFIG_DIR" = "D:\JINAI Main\kaggle\config"
}

foreach ($name in $targets.Keys) {
    $path = $targets[$name]
    if (-not (Test-Path -LiteralPath $path)) {
        New-Item -ItemType Directory -Force -Path $path | Out-Null
    }
    Set-Item -Path "Env:$name" -Value $path
}

Write-Host "[env] Moi thu nang da ve o D:"
foreach ($name in $targets.Keys) {
    Write-Host ("  {0,-18} = {1}" -f $name, $targets[$name])
}

$freeC = [int]((Get-PSDrive C).Free / 1GB)
$freeD = [int]((Get-PSDrive D).Free / 1GB)
Write-Host "[env] Con trong: C = $freeC GB | D = $freeD GB"
if ($freeC -lt 10) {
    Write-Host "[env] CANH BAO: o C con duoi 10GB. Xoa bot C:\Users\nguye\AppData\Local\Temp va can nhac chuyen pagefile sang D:" -ForegroundColor Yellow
}
