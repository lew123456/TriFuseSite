param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('ppi','smpi','dpi','rpi')]
    [string]$Task,
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$TrainingArgs
)
$repositoryRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repositoryRoot
try {
    python main.py train --config "config/$Task.yaml" @TrainingArgs
    $trainingExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $trainingExitCode
