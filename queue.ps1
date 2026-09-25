# Experiment queue: runs the jobs one after another, logs each to runs\queue_logs\<name>.txt
# Safe to restart: jobs whose log already says they finished are skipped.
#   powershell -ExecutionPolicy Bypass -File queue.ps1
$ErrorActionPreference = "Continue"
$VIT = "vit_base_patch16_dinov3.lvd1689m"

$jobs = @(
    # 0) is the ViT alone a better FAST model than Base? (no flip + re-rank; Base = 78.98)
    @{ name = "eval_vit_fast";  args = @("experiments/multisize_test.py", "--weights", "runs/val_vit/best_ema.pth", "--backbone", $VIT, "--sizes", "256") },
    # 1) confirm the ViT ensemble on seed 7 (same recipe as val_vit: lr 5e-5, stop at 20)
    @{ name = "vit_s7";         args = @("train_final.py", "--model", "vit", "--lr-backbone", "5e-5", "--stop-epoch", "20",
                                         "--train-csv", "data/splits_seed7/train_split.csv", "--out", "runs/split7_vit") },
    @{ name = "eval_ens_s7";    args = @("experiments/ensemble_test.py", "--splits", "data/splits_seed7",
                                         "--base", "runs/split7/base.pth", "--small", "runs/split7/small.pth", "--vit", "runs/split7_vit/vit.pth") },
    # 2) A2: ViT recipe (seed 42): full 30-epoch schedule, then + layer-wise LR decay
    @{ name = "vit_e30";        args = @("train.py", "--model", "vit", "--lr-backbone", "5e-5", "--stop-epoch", "30", "--run-name", "val_vit_e30") },
    @{ name = "vit_llrd";       args = @("train.py", "--model", "vit", "--lr-backbone", "1e-4", "--llrd", "0.75", "--stop-epoch", "30", "--run-name", "val_vit_llrd") },
    # 3) A3-A5: Base recipe upgrades (seed 42), one change each
    @{ name = "base_cam";       args = @("train.py", "--model", "base", "--camera-aware", "--run-name", "val_base_cam") },
    @{ name = "base_gem";       args = @("train.py", "--model", "base", "--pool", "gem", "--run-name", "val_base_gem") },
    @{ name = "base_p32";       args = @("train.py", "--model", "base", "--P", "32", "--run-name", "val_base_p32") },
    # 4) real-pipeline scores (fast mode: no flip + re-rank) for every new model
    @{ name = "eval_base_cam";  args = @("experiments/multisize_test.py", "--weights", "runs/val_base_cam/best_ema.pth", "--sizes", "256") },
    @{ name = "eval_base_gem";  args = @("experiments/multisize_test.py", "--weights", "runs/val_base_gem/best_ema.pth", "--sizes", "256") },
    @{ name = "eval_base_p32";  args = @("experiments/multisize_test.py", "--weights", "runs/val_base_p32/best_ema.pth", "--sizes", "256") },
    @{ name = "eval_vit_e30";   args = @("experiments/multisize_test.py", "--weights", "runs/val_vit_e30/best_ema.pth", "--backbone", $VIT, "--sizes", "256") },
    @{ name = "eval_vit_llrd";  args = @("experiments/multisize_test.py", "--weights", "runs/val_vit_llrd/best_ema.pth", "--backbone", $VIT, "--sizes", "256") },
    @{ name = "eval_ens_e30";   args = @("experiments/ensemble_test.py", "--base", "runs/convnext_b_v6_ema/best_ema.pth",
                                         "--small", "runs/convnext_s_v2_ema/best_ema.pth", "--vit", "runs/val_vit_e30/best_ema.pth") },
    @{ name = "eval_ens_llrd";  args = @("experiments/ensemble_test.py", "--base", "runs/convnext_b_v6_ema/best_ema.pth",
                                         "--small", "runs/convnext_s_v2_ema/best_ema.pth", "--vit", "runs/val_vit_llrd/best_ema.pth") }
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
