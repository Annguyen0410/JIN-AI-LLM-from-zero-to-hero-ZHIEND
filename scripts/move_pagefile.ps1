# Chuyen pagefile tu o C (dang chi con ~11 GB trong) sang o D (con 627 GB).
# Thu tu an toan: TAO pagefile tren D TRUOC, xoa C SAU.
# Thay doi chi co hieu luc sau khi khoi dong lai may.

$ErrorActionPreference = 'Continue'
$mm = 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management'
$sizeMb = 24576

function Show-State([string]$label) {
    Write-Host ("=== " + $label + " ===")
    $cs = Get-CimInstance Win32_ComputerSystem
    Write-Host ("AutomaticManagedPagefile = " + $cs.AutomaticManagedPagefile)
    $settings = Get-CimInstance Win32_PageFileSetting
    if ($settings) {
        foreach ($s in $settings) {
            Write-Host ("  setting  " + $s.Name + "  initial=" + $s.InitialSize + "MB  max=" + $s.MaximumSize + "MB")
        }
    } else {
        Write-Host "  (khong co setting nao)"
    }
    foreach ($u in (Get-CimInstance Win32_PageFileUsage)) {
        Write-Host ("  in-use   " + $u.Name + "  alloc=" + $u.AllocatedBaseSize + "MB")
    }
    Write-Host ("  registry PagingFiles = " + ((Get-ItemProperty $mm -Name PagingFiles -EA SilentlyContinue).PagingFiles -join ' | '))
    Write-Host ""
}

Show-State 'TRUOC KHI DOI'

# 1. Tat che do Windows tu quan ly pagefile — neu khong tat thi moi thay doi se bi ghi de.
$cs = Get-CimInstance Win32_ComputerSystem
if ($cs.AutomaticManagedPagefile) {
    Set-CimInstance -InputObject $cs -Property @{ AutomaticManagedPagefile = $false }
    Write-Host "Da tat AutomaticManagedPagefile (truoc do BAT)."
} else {
    Write-Host "AutomaticManagedPagefile da tat tu truoc."
}
Write-Host ""

# 2. TAO pagefile tren D truoc tien.
$onD = Get-CimInstance Win32_PageFileSetting | Where-Object { $_.Name -like 'D:*' }
if ($onD) {
    Set-CimInstance -InputObject $onD -Property @{ InitialSize = $sizeMb; MaximumSize = $sizeMb }
    Write-Host ("Da cap nhat pagefile D: thanh " + $sizeMb + "MB co dinh.")
} else {
    try {
        $pf = ([WMIClass]'Win32_PageFileSetting').CreateInstance()
        $pf.Name = 'D:\pagefile.sys'
        $pf.InitialSize = $sizeMb
        $pf.MaximumSize = $sizeMb
        $pf.Put() | Out-Null
        Write-Host ("Da tao pagefile D:\pagefile.sys " + $sizeMb + "MB.")
    } catch {
        Write-Host ("LOI khi tao pagefile D: " + $_.Exception.Message)
    }
}
Write-Host ""

# 3. Chi khi D da chac chan co, moi xoa C.
$dOk = Get-CimInstance Win32_PageFileSetting | Where-Object { $_.Name -like 'D:*' }
if ($dOk) {
    $onC = Get-CimInstance Win32_PageFileSetting | Where-Object { $_.Name -like 'C:*' }
    if ($onC) {
        foreach ($c in $onC) {
            Write-Host ("Dang xoa pagefile tren C: " + $c.Name)
            Remove-CimInstance -InputObject $c -EA SilentlyContinue
        }
    } else {
        Write-Host "Khong thay pagefile nao tren C de xoa."
    }
} else {
    Write-Host "DUNG LAI: pagefile D: chua tao duoc, KHONG xoa pagefile tren C (an toan hon la mat ca hai)."
}
Write-Host ""

# 4. Chan cuoi: ghi thang vao registry neu WMI khong dat duoc dung trang thai mong muon.
$want = 'D:\pagefile.sys ' + $sizeMb + ' ' + $sizeMb
$now = (Get-ItemProperty $mm -Name PagingFiles -EA SilentlyContinue).PagingFiles
if (-not ($now -join '|') -like '*D:\pagefile.sys*') {
    Set-ItemProperty -Path $mm -Name PagingFiles -Value @($want) -Type MultiString
    Write-Host ("Da ghi registry PagingFiles = " + $want)
} else {
    Write-Host "Registry da tro dung D:, khong can ghi them."
}
Write-Host ""

Show-State 'SAU KHI DOI'
Write-Host "XONG. Thay doi chi ap dung sau khi khoi dong lai may."
Write-Host "Sau khi reboot, C:\pagefile.sys se bien mat va C: duoc tra lai ~11,3 GB."
