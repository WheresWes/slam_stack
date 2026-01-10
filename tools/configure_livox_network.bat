@echo off
:: Livox Mid-360 Network Configuration Script
:: Run this script as Administrator (right-click -> Run as administrator)

echo ============================================
echo  Livox Mid-360 Network Configuration
echo ============================================
echo.

:: Check for admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: This script requires Administrator privileges.
    echo.
    echo Please right-click this file and select "Run as administrator"
    echo.
    pause
    exit /b 1
)

echo Current Ethernet configuration:
netsh interface ip show config name="Ethernet"
echo.

:: Ask for confirmation
echo This will set your Ethernet adapter to:
echo   IP Address:  192.168.1.50
echo   Subnet Mask: 255.255.255.0
echo   Gateway:     (none)
echo.
echo This is required for communication with the Livox Mid-360 LiDAR.
echo.
set /p confirm="Continue? (Y/N): "
if /i not "%confirm%"=="Y" (
    echo Cancelled.
    pause
    exit /b 0
)

echo.
echo Configuring Ethernet adapter...
netsh interface ip set address name="Ethernet" static 192.168.1.50 255.255.255.0

if %errorLevel% equ 0 (
    echo.
    echo SUCCESS! Ethernet adapter configured.
    echo.
    echo New configuration:
    netsh interface ip show config name="Ethernet"
    echo.
    echo You can now run the Livox connection test.
    echo NOTE: This configuration is persistent - Windows will remember it.
) else (
    echo.
    echo ERROR: Failed to configure Ethernet adapter.
    echo Make sure the adapter name is "Ethernet" - check with: netsh interface show interface
)

echo.
pause
