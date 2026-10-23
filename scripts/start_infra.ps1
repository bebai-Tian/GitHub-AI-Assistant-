$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$composeFile = Join-Path $projectRoot "docker-compose.yml"
$infraServices = @("postgres", "etcd", "minio", "milvus")

function Test-DockerReady {
    & docker info --format "{{.ServerVersion}}" *> $null
    return ($LASTEXITCODE -eq 0)
}

if (-not (Test-DockerReady)) {
    $dockerDesktop = Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\Docker Desktop.exe"
    if (-not (Test-Path -LiteralPath $dockerDesktop)) {
        $dockerDesktop = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    }
    if (Test-Path -LiteralPath $dockerDesktop) {
        Write-Host "Docker 守护进程未运行，正在启动 Docker Desktop..."
        Start-Process -FilePath $dockerDesktop
    }
    else {
        Write-Host "未找到 Docker Desktop，可执行文件不存在。"
    }

    Write-Host "等待 Docker 守护进程就绪（最多 5 分钟）..."
    $ready = $false
    for ($attempt = 0; $attempt -lt 100; $attempt++) {
        if (Test-DockerReady) {
            $ready = $true
            break
        }
        Start-Sleep -Seconds 3
    }
    if (-not $ready) {
        throw "Docker 守护进程在 5 分钟内没有就绪，请手动打开 Docker Desktop 后重试。"
    }
}

Write-Host "Docker 已就绪。"
Write-Host "启动基础设施容器：$($infraServices -join ', ')"
& docker compose -f $composeFile up -d @infraServices
if ($LASTEXITCODE -ne 0) {
    throw "docker compose up 失败，退出码 $LASTEXITCODE。"
}

Write-Host "等待容器进入健康状态（最多 3 分钟）..."
$deadline = (Get-Date).AddMinutes(3)
$healthy = $false
while ((Get-Date) -lt $deadline) {
    $running = & docker compose -f $composeFile ps --status running --services 2>$null
    $missing = @($infraServices | Where-Object { $running -notcontains $_ })
    if ($missing.Count -eq 0) {
        $healthy = $true
        break
    }
    Start-Sleep -Seconds 5
}

& docker compose -f $composeFile ps
if (-not $healthy) {
    Write-Host "警告：部分容器未在 3 分钟内进入运行状态，请检查上面的输出。"
}

Write-Host ""
Write-Host "基础设施已启动："
Write-Host "  PostgreSQL : localhost:5432 (devflow/devflow)"
Write-Host "  Milvus     : localhost:19530"
Write-Host "  MinIO      : localhost:9000 (控制台 :9001)"
Write-Host "  停止命令   : docker compose down"
