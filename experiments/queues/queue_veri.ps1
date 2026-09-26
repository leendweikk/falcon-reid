# VeRi queue (26 Sep): runs the jobs one after another, logs each to runs\queue_logs\<name>.txt
# Safe to restart: jobs whose log already says they finished are skipped.
#   powershell -ExecutionPolicy Bypass -File queue_veri.ps1
$ErrorActionPreference = "Continue"
$VIT = "vit_base_patch16_dinov3.lvd1689m"

$jobs = @(
    # A9: our training split + VeRi (plates blurred by experiments/prepare_veri.py, checked by eye first).
    # Same ViT recipe as the final model. Gate: fast-mode mAP@10 >= +0.7 on seed 42 (ref 79.14)
    # AND still a win on seed 7 (ref 78.00).
    @{ name = "veri_s42";      args = @("train_final.py", "--model", "vit", "--lr-backbone", "5e-5", "--epochs", "30",
                                        "--stop-epoch", "20", "--train-csv", "data/splits/train_split.csv", "data/veri.csv",
                                        "--out", "runs/veri_s42") },
    @{ name = "eval_veri_s42"; args = @("experiments/multisize_test.py", "--weights", "runs/veri_s42/vit.pth",
                                        "--backbone", $VIT, "--sizes", "256") },
    @{ name = "veri_s7";       args = @("train_final.py", "--model", "vit", "--lr-backbone", "5e-5", "--epochs", "30",
                                        "--stop-epoch", "20", "--train-csv", "data/splits_seed7/train_split.csv", "data/veri.csv",
                                        "--out", "runs/veri_s7") },
    @{ name = "eval_veri_s7";  args = @("experiments/multisize_test.py", "--weights", "runs/veri_s7/vit.pth",
                                        "--backbone", $VIT, "--splits", "data/splits_seed7", "--sizes", "256") }
)

New-Item -ItemType Directory -Force "runs\queue_logs" | Out-Null
foreach ($j in $jobs) {
    $log = "runs\queue_logs\$($j.name).txt"
    if ((Test-Path $log) -and (Select-String -Path $log -Pattern "^done\.|saved final EMA|^saved ->" -Quiet)) {
        Write-Host "=== skip $($j.name) (already finished)"
        continue
    }
    Write-Host "=== $(Get-Date -Format 'HH:mm') START $($j.name)"
    & python -u @($j.args) 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
    Write-Host "=== $(Get-Date -Format 'HH:mm') END   $($j.name)"
}
Write-Host "=== queue finished $(Get-Date -Format 'HH:mm')"
