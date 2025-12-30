import subprocess
import time
import os

print("启动CLIP RAG系统...")

# 启动后端
print("1. 启动后端服务器...")
subprocess.Popen('start cmd /k "cd backend && venv\\Scripts\\activate && python start_server.py"', shell=True)
time.sleep(3)

# 启动前端
print("2. 启动前端界面...")
subprocess.Popen('start cmd /k "cd frontend && pnpm dev"', shell=True)
time.sleep(5)

# 打开浏览器
print("3. 打开浏览器...")
os.system('start http://localhost:3000')
os.system('start http://localhost:8000/docs')

print("\n✅ 启动完成!")
print("前端: http://localhost:3000")
print("后端: http://localhost:8000")
print("API文档: http://localhost:8000/docs")
print("\n按回车键退出...")
input()