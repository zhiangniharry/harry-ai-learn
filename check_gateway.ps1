# Check if gateway is running
$proc = Get-Process node -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like '*openclaw*gateway*' }
if ($proc) {
    Write-Host "Gateway running, PID: $($proc.Id)"
} else {
    Write-Host "Gateway not running, starting..."
    Start-Process powershell -ArgumentList '-NoExit', '-Command', "& 'C:\Users\Administrator\AppData\Roaming\npm\openclaw.ps1' gateway" -WindowStyle Hidden
    Start-Sleep 3
    $newProc = Get-Process node -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like '*openclaw*gateway*' }
    if ($newProc) {
        Write-Host "Gateway started, PID: $($newProc.Id)"
    } else {
        Write-Host "Gateway start failed"
    }
}
