Set-Location $PSScriptRoot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements-client.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 300 --retries 10
Write-Host "Client dependencies installed. Run: python -m sp4cpg_client.app"
