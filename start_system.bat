@echo off
echo CLIP RAG 系统启动脚本
echo.

title CLIP RAG 系统

echo 步骤1: 激活Python虚拟环境
cd backend
call venv\Scripts\activate

echo.
echo 步骤2: 检查向量存储
if not exist "..\vectorstore\chroma.sqlite3" (
    echo 正在初始化向量存储...
    python initialize_vectorstore.py
    if errorlevel 1 (
        echo 向量存储初始化失败
        pause
        exit /b 1
    )
) else (
    echo ✓ 向量存储已存在
)

echo.
echo 步骤3: 启动后端服务器
echo 正在启动后端服务器...
start cmd /k "cd backend && venv\Scripts\activate && python start_server.py"

echo 等待5秒让后端启动...
timeout /t 5 /nobreak > nul

echo.
echo 步骤4: 测试后端接口
echo 正在测试后端接口...
python test_langgraph.py

echo.
echo 步骤5: 启动前端
echo 前端将在新窗口中启动...
timeout /t 3 /nobreak > nul

echo 请手动在前端目录中启动前端：
echo 1. 打开新的命令行窗口
echo 2. cd frontend
echo 3. pnpm install
echo 4. pnpm dev
echo.
echo 或者按任意键自动打开浏览器到前端...
pause > nul
start http://localhost:3000

echo.
echo ✅ 系统启动完成！
echo 后端: http://localhost:8000
echo 前端: http://localhost:3000
echo API文档: http://localhost:8000/docs
echo.
echo 按任意键退出...
pause > nul