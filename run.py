import os
import subprocess
import sys

# 设置环境变量，确保Streamlit能找到你的utils和pages
os.environ["PYTHONPATH"] = os.getcwd()

print("--- PowerTool Overseas Version (EN) Local Test ---")
print("Launching Streamlit...")

# 使用 subprocess 运行，比 os.system 更稳定，能看到详细报错
try:
    subprocess.run([sys.executable, "-m", "streamlit", "run", "Home.py"])
except KeyboardInterrupt:
    print("\nStopping local server...")