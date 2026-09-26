# Night queue (25-26 Sep): runs the jobs one after another, logs each to runs\queue_logs\<name>.txt
# Safe to restart: jobs whose log already says they finished are skipped.
#   powershell -ExecutionPolicy Bypass -File queue_night.ps1
$ErrorActionPreference = "Continue"
$VIT = "vit_base_patch16_dinov3.lvd1689m"

$jobs = @(
    # 1) seed-7 check of the 30-epoch ViT recipe (it won fast mode on seed 42: 80.27 vs 79.14)
    @{ name = "vit_e30_s7";         args = @("train_final.py", "--model", "vit", "--lr-backbone", "5e-5", "--epochs", "30", "--stop-epoch", "30",
                                             "--train-csv", "data/splits_seed7/train_split.csv", "--out", "runs/split7_vit_e30") },
    @{ name = "eval_vitfast_s7_e20"; args = @("experiments/multisize_test.py", "--weights", "runs/split7_vit/vit.pth", "--backbone", $VIT,
                                             "--splits", "data/splits_seed7", "--sizes", "256") },
    @{ name = "eval_vitfast_s7_e30"; args = @("experiments/multisize_test.py", "--weights", "runs/split7_vit_e30/vit.pth", "--backbone", $VIT,
                                             "--splits", "data/splits_seed7", "--sizes", "256") },
    # 2) final ViT on ALL 1,541 cars, both recipes, so the morning decision needs no extra GPU time
    @{ name = "final_vit_e30";      args = @("train_final.py", "--model", "vit", "--lr-backbone", "5e-5", "--epochs", "30", "--stop-epoch", "30",
                                             "--out", "runs/final_vit_e30") },
    @{ name = "final_vit_e20";      args = @("train_final.py", "--model", "vit", "--lr-backbone", "5e-5", "--epochs", "30", "--stop-epoch", "20",
                                             "--out", "runs/final_vit_e20") }
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
