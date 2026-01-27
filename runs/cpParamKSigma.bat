@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM 设置根目录
set "ROOT_FOLDER=D:\users\xiaoyaopan\PxyAI\PMRID_OFFICIAL\PMRID"

echo 正在检查根目录是否存在...
if not exist "%ROOT_FOLDER%" (
    echo 错误：指定的根目录不存在！
    echo 根目录：%ROOT_FOLDER%
    pause
    exit /b 1
)

echo 根目录检查通过！
echo.

REM 获取当前脚本所在目录作为目标路径
set "TARGET_DIR=%~dp0"

echo 开始拷贝文件...
echo.

REM 定义要拷贝的文件数组
set FILE_COUNT=4
set "FILE_LIST[1]=data\RawDataset.py"
set "FILE_LIST[2]=engine\train.py"
set "FILE_LIST[3]=utils\KSigma.py"
set "FILE_LIST[4]=models\net_torch.py"

set ALL_FILES_EXIST=1
set SUCCESS_COUNT=0
set FAIL_COUNT=0

REM 首先检查所有源文件是否存在
for /l %%i in (1,1,%FILE_COUNT%) do (
    set "REL_PATH=!FILE_LIST[%%i]!"
    set "SOURCE_FILE=%ROOT_FOLDER%\!REL_PATH!"
    
    if not exist "!SOURCE_FILE!" (
        echo 错误：找不到源文件
        echo 源文件：!SOURCE_FILE!
        echo.
        set ALL_FILES_EXIST=0
        set /a FAIL_COUNT+=1
    )
)

if %ALL_FILES_EXIST% equ 0 (
    echo 部分源文件不存在，请检查路径！
    pause
    exit /b 1
)

echo 所有源文件检查通过，开始拷贝...
echo.

REM 拷贝文件
for /l %%i in (1,1,%FILE_COUNT%) do (
    set "REL_PATH=!FILE_LIST[%%i]!"
    set "SOURCE_FILE=%ROOT_FOLDER%\!REL_PATH!"
    
    REM 从相对路径中提取文件名
    for %%F in ("!REL_PATH!") do set "FILENAME=%%~nxF"
    
    echo 正在拷贝: !FILENAME!
    echo   from: !SOURCE_FILE!
    echo   to:   %TARGET_DIR%!FILENAME!
    
    REM 执行拷贝
    copy "!SOURCE_FILE!" "%TARGET_DIR%!FILENAME!" >nul
    
    if errorlevel 1 (
        echo 错误：拷贝失败！
        echo.
        set /a FAIL_COUNT+=1
    ) else (
        echo 成功：文件已拷贝
        echo.
        set /a SUCCESS_COUNT+=1
    )
)

echo ====================
echo 拷贝完成!
echo 成功: %SUCCESS_COUNT% 个文件
if %FAIL_COUNT% gtr 0 (
    echo 失败：%FAIL_COUNT% 个文件
)

REM 显示拷贝的文件列表
echo.
echo 已拷贝到当前目录的文件：
dir /b "%TARGET_DIR%*.py" | findstr /i "RawDataset train KSigma"

echo.
echo 按任意键退出...
pause >nul