# Distillation queue (26 Sep): runs the jobs one after another, logs each to runs\queue_logs\<name>.txt
# Safe to restart: jobs whose log already says they finished are skipped.
#   powershell -ExecutionPolicy Bypass -File queue_distill.ps1
$ErrorActionPreference = "Continue"
$VIT = "vit_base_patch16_dinov3.lvd1689m"

$jobs = @(
    # A11 distillation: the ensemble (teacher) teaches the fast ViT (student). Same ViT recipe as the final
    # model (lr 5e-5, 30-epoch schedule stopped at 20) + relational KD. Gate: fast-mode mAP@10 must beat
    # the plain ViT by >= 0.7 on seed 42 (ref 79.14) AND still win on seed 7 (ref 78.00).
    @{ name = "teacher_s42";      args = @("tools/teacher_embed.py", "--train-csv", "data/splits/train_split.csv",
                                           "--base", "runs/convnext_b_v6_ema/best_ema.pth", "--vit", "runs/val_vit/best_ema.pth",
                                           "--out", "runs/teacher_s42.npz") },
    @{ name = "distill_s42";      args = @("train_final.py", "--model", "vit", "--lr-backbone", "5e-5", "--epochs", "30",
                                           "--stop-epoch", "20", "--train-csv", "data/splits/train_split.csv",
                                           "--out", "runs/distill_s42", "--distill", "runs/teacher_s42.npz") },
    @{ name = "eval_distill_s42"; args = @("experiments/multisize_test.py", "--weights", "runs/distill_s42/vit.pth",
                                           "--backbone", $VIT, "--sizes", "256") },
    @{ name = "teacher_s7";       args = @("tools/teacher_embed.py", "--train-csv", "data/splits_seed7/train_split.csv",
                                           "--base", "runs/split7/base.pth", "--vit", "runs/split7_vit/vit.pth",
                                           "--out", "runs/teacher_s7.npz") },
    @{ name = "distill_s7";       args = @("train_final.py", "--model", "vit", "--lr-backbone", "5e-5", "--epochs", "30",
                                           "--stop-epoch", "20", "--train-csv", "data/splits_seed7/train_split.csv",
                                           "--out", "runs/distill_s7", "--distill", "runs/teacher_s7.npz") },
    @{ name = "eval_distill_s7";  args = @("experiments/multisize_test.py", "--weights", "runs/distill_s7/vit.pth",
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
