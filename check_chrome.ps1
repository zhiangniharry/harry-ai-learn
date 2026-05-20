Get-Process chrome -ErrorAction SilentlyContinue | Select-Object Id | ForEach-Object {
    $pid = $_.Id
    $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$pid").CommandLine
    [PSCustomObject]@{Id=$pid; HasRemoteDebugging=($cmd -like '*--remote-debugging*')}
} | Format-Table -AutoSize
