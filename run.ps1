param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('ppi','smpi','dpi','rpi')]
    [string]$Task,
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$TrainingArgs
)
python main_train.py --config "config/$Task.yaml" @TrainingArgs
exit $LASTEXITCODE
