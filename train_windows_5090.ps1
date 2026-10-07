# Run in the project's existing CUDA-enabled Python environment.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$CkptDir,
    [string]$Fp8Weights = 'weights/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors',
    [Parameter(Mandatory = $true)][string]$DataDir,
    [string]$OutputDir = 'output/lora-5090',
    [string]$Python = 'python',
    [ValidateRange(0, 40)][int]$CpuOffloadBlocks = 32,
    [ValidateRange(0, 96)][double]$CpuCacheGiB = 8,
    [ValidateRange(1, 1000000)][int]$MaxSteps = 5000,
    [switch]$PinMemory,
    [string]$ResumeFrom,
    [switch]$Preview
)
$ErrorActionPreference = 'Stop'
$trainingArgs = @(
    'train_lora.py', '--ckpt_dir', $CkptDir,
    '--fp8_checkpoint', $Fp8Weights,
    '--data_dir', $DataDir, '--output_dir', $OutputDir,
    '--frame_num', '81', '--quant', 'fp8',
    '--lora_rank', '64', '--lora_alpha', '64',
    '--lr', '1e-4', '--audio_lr', '1e-4',
    '--cpu_offload_blocks', "$CpuOffloadBlocks",
    '--cpu_cache_gb', "$CpuCacheGiB",
    '--gradient_checkpointing', '--activation_offload', '--vae_cpu_offload',
    '--attention_backend', 'sdpa', '--use_amp', '--use_8bit_optim',
    '--num_workers', '0', '--max_steps', "$MaxSteps",
    '--save_every', '250', '--log_every', '10', '--debug_assert_shapes'
)
if ($PinMemory) { $trainingArgs += '--offload_pin_memory' }
if ($ResumeFrom) { $trainingArgs += @('--resume_from', $ResumeFrom) }
if ($Preview) {
    $trainingArgs
    return
}
Push-Location $PSScriptRoot
try {
    & $Python @trainingArgs
    if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
