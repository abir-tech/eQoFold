<#
.SYNOPSIS
    Windows equivalent of the Makefile. GNU make is not installed by default on
    Windows and Git for Windows does not ship it, so this shim exposes the same
    targets with the same behaviour.

.EXAMPLE
    ./make.ps1 env
    ./make.ps1 reference
    ./make.ps1 verify
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('all', 'help', 'env', 'install', 'sequences', 'reference', 'experiments', 'circuits', 'figures', 'test', 'lint', 'verify', 'clean', 'distclean')]
    [string]$Target = 'all'
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Venv = Join-Path $Root '.venv'
$VPy = Join-Path $Venv 'Scripts\python.exe'
$Reference = Join-Path $Root 'data\references\vienna_reference.csv'

function Assert-Venv {
    if (-not (Test-Path $VPy)) {
        Write-Host '.venv not found; run "./make.ps1 env" first.' -ForegroundColor Yellow
        exit 1
    }
}

function Invoke-Step([string]$Label, [scriptblock]$Body) {
    Write-Host "==> $Label" -ForegroundColor Cyan
    & $Body
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($Target) {
    'help' {
        Write-Host 'targets: env sequences reference experiments circuits figures test lint verify clean distclean all'
    }
    { $_ -in 'env', 'install' } {
        if (-not (Test-Path $VPy)) {
            Invoke-Step 'create venv' { python -m venv $Venv }
        }
        Invoke-Step 'upgrade pip' { & $VPy -m pip install --upgrade pip --quiet }
        Invoke-Step 'install package' { & $VPy -m pip install -e "$Root[dev]" --quiet }
        Write-Host 'environment ready' -ForegroundColor Green
    }
    'sequences' {
        Assert-Venv
        Invoke-Step 'generate sequences' { & $VPy (Join-Path $Root 'scripts\generate_sequences.py') }
    }
    'reference' {
        Assert-Venv
        Invoke-Step 'build reference' { & $VPy (Join-Path $Root 'scripts\build_reference.py') }
    }
    'experiments' {
        Assert-Venv
        $E = Join-Path $Root 'experiments'
        Invoke-Step 'enumeration ablation' { & $VPy (Join-Path $E 'ablate_enumeration.py') --tiers A,M }
        Invoke-Step 'encoding gap' { & $VPy (Join-Path $E 'run_encoding_gap.py') --tiers A,M --quiet }
        Invoke-Step 'fidelity ladder' { & $VPy (Join-Path $E 'run_fidelity_ladder.py') --tiers A,M --max-stems 45 }
        Invoke-Step 'dirac-3 study' { & $VPy (Join-Path $E 'run_dirac3_study.py') --tiers A,M --max-stems 35 --seeds 3 }
        Invoke-Step 'solver comparison' { & $VPy (Join-Path $E 'run_solvers.py') --tiers A,M --max-stems 16 --budget 2.0 }
        Invoke-Step 'advanced tasks' { & $VPy (Join-Path $E 'run_advanced.py') --max-stems 12 }
    }
    'circuits' {
        Assert-Venv
        $E = Join-Path $Root 'experiments'
        Invoke-Step 'scaling sweep' { & $VPy (Join-Path $E 'scaling_sweep.py') }
        Invoke-Step 'scaling sweep (calibrated)' { & $VPy (Join-Path $E 'scaling_sweep_calibrated.py') }
        Invoke-Step 'flagship deep dive' { & $VPy (Join-Path $E 'flagship_deep_dive.py') }
        Invoke-Step 'hardware-aware demo' { & $VPy (Join-Path $E 'hardware_aware_demo.py') }
        Invoke-Step 'shot-noise robustness' { & $VPy (Join-Path $E 'noise_robustness.py') }
        Invoke-Step 'pseudoknot illustration' { & $VPy (Join-Path $E 'pseudoknot_illustration.py') }
    }
    'figures' {
        Assert-Venv
        Invoke-Step 'ladder figures' { & $VPy (Join-Path $Root 'experiments\make_ladder_figures.py') }
        Invoke-Step 'paper figures' { & $VPy (Join-Path $Root 'experiments\make_figures.py') }
    }
    'test' {
        Assert-Venv
        Invoke-Step 'pytest' { & $VPy -m pytest }
    }
    'lint' {
        Assert-Venv
        Invoke-Step 'ruff' { & $VPy -m ruff check (Join-Path $Root 'src') (Join-Path $Root 'tests') (Join-Path $Root 'scripts') }
    }
    'verify' {
        Assert-Venv
        $tmp = Join-Path $env:TEMP 'vienna_reference_check.csv'
        Invoke-Step 'rebuild reference into temp' { & $VPy (Join-Path $Root 'scripts\build_reference.py') --out $tmp }
        $a = Get-FileHash $Reference -Algorithm SHA256
        $b = Get-FileHash $tmp -Algorithm SHA256
        if ($a.Hash -eq $b.Hash) {
            Write-Host "OK: reference table reproduces exactly (sha256 $($a.Hash.Substring(0,16))...)" -ForegroundColor Green
        }
        else {
            Write-Host 'FAIL: regenerated reference differs from the committed table' -ForegroundColor Red
            Compare-Object (Get-Content $Reference) (Get-Content $tmp) | Select-Object -First 20
            exit 1
        }
    }
    'clean' {
        Get-ChildItem -Path $Root -Include '__pycache__', '.pytest_cache' -Recurse -Directory -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force
        Write-Host 'cleaned' -ForegroundColor Green
    }
    'distclean' {
        & $PSCommandPath clean
        if (Test-Path $Venv) { Remove-Item $Venv -Recurse -Force }
        Write-Host 'removed venv' -ForegroundColor Green
    }
    'all' {
        & $PSCommandPath sequences
        & $PSCommandPath reference
        & $PSCommandPath experiments
        & $PSCommandPath figures
        & $PSCommandPath test
    }
}
