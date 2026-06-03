# Setup Environment for TensorFlow Edge Impulse Project
# Run this script as Administrator if npm global install fails

Write-Host "Iniciando configuración del entorno para TensorFlow y Edge Impulse..." -ForegroundColor Cyan

# 1. Instalar dependencias de Python
Write-Host "Instalando requerimientos de Python (requirements.txt)..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "Hubo un error instalando las dependencias de Python. Verifica el entorno." -ForegroundColor Red
    exit $LASTEXITCODE
}

# 2. Verificar NodeJS para el Edge Impulse CLI
Write-Host "Comprobando NodeJS..." -ForegroundColor Yellow
try {
    node -v
} catch {
    Write-Host "NodeJS no está instalado. Por favor instala NodeJS desde https://nodejs.org/ e intenta nuevamente." -ForegroundColor Red
    exit 1
}

# 3. Instalar Edge Impulse CLI
Write-Host "Instalando edge-impulse-cli de manera global..." -ForegroundColor Yellow
npm install -g edge-impulse-cli
if ($LASTEXITCODE -ne 0) {
    Write-Host "Hubo un error instalando edge-impulse-cli. Verifica los permisos." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "¡Entorno configurado correctamente! Todo está listo para procesar y cargar los datos." -ForegroundColor Green
