#!/usr/bin/env bash
# Kiem tra moi truong Jetson truoc khi bat dau IVID
# Chay:  bash check_jetson.sh 2>&1 | tee jetson_env.txt

sec() { printf '\n===== %s =====\n' "$1"; }

sec "1. THIET BI / L4T"
cat /etc/nv_tegra_release 2>/dev/null || echo "(khong co /etc/nv_tegra_release)"
cat /proc/device-tree/model 2>/dev/null; echo
uname -a

sec "2. JETPACK"
apt-cache show nvidia-jetpack 2>/dev/null | grep -E '^(Package|Version)' | head -4 \
  || echo "(khong tim thay goi nvidia-jetpack)"
dpkg-query --show nvidia-l4t-core 2>/dev/null

sec "3. CUDA"
nvcc --version 2>/dev/null || echo "(nvcc khong co trong PATH - thu /usr/local/cuda/bin/nvcc)"
/usr/local/cuda/bin/nvcc --version 2>/dev/null
ls -d /usr/local/cuda* 2>/dev/null
cat /usr/local/cuda/version.json 2>/dev/null | head -12

sec "4. cuDNN"
dpkg -l 2>/dev/null | grep -i cudnn | awk '{print $2, $3}'

sec "5. TENSORRT"
dpkg -l 2>/dev/null | grep -iE 'tensorrt|libnvinfer' | awk '{print $2, $3}'
python3 -c "import tensorrt; print('python tensorrt:', tensorrt.__version__)" 2>&1 | tail -1
/usr/src/tensorrt/bin/trtexec --version 2>&1 | head -3 || echo "(khong co trtexec)"

sec "6. PYTHON / PIP"
python3 --version
python3 -c "import sys; print(sys.executable)"
pip3 --version 2>/dev/null
pip3 list 2>/dev/null | grep -iE '^(torch|torchvision|onnx|onnxruntime|numpy|opencv|ultralytics|pycuda)' || echo "(chua cai package nao lien quan)"

sec "7. PYTORCH + CUDA"
python3 - <<'PY' 2>&1 | tail -8
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda version:", torch.version.cuda)
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch chua cai hoac loi:", e)
PY

sec "8. ONNXRUNTIME"
python3 - <<'PY' 2>&1 | tail -5
try:
    import onnxruntime as ort
    print("onnxruntime:", ort.__version__)
    print("providers:", ort.get_available_providers())
except Exception as e:
    print("onnxruntime chua cai:", e)
PY

sec "9. CHE DO NGUON / XUNG NHIP  (FR-14)"
sudo nvpmodel -q 2>/dev/null || nvpmodel -q 2>/dev/null || echo "(can sudo)"
sudo jetson_clocks --show 2>/dev/null | head -20 || echo "(can sudo hoac khong co jetson_clocks)"

sec "10. NHIET DO"
for z in /sys/devices/virtual/thermal/thermal_zone*; do
  [ -f "$z/type" ] && echo "$(cat $z/type): $(( $(cat $z/temp) / 1000 )) C"
done 2>/dev/null

sec "11. BO NHO / DIA"
free -h
df -h / /home 2>/dev/null | sort -u
swapon --show 2>/dev/null

sec "12. DOCKER"
docker --version 2>/dev/null || echo "(chua co docker)"
docker info 2>/dev/null | grep -iE 'runtime|default runtime'
cat /etc/docker/daemon.json 2>/dev/null

sec "13. jetson-stats (khuyen nghi cai: sudo pip3 install -U jetson-stats)"
jetson_release 2>/dev/null || echo "(chua cai jetson-stats)"

sec "14. MANG (de setup ssh tu may dev)"
hostname
ip -4 addr show 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v 127.0.0.1

echo
echo "===== XONG. Gui file jetson_env.txt cho Claude ====="
