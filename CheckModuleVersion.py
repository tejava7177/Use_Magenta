import platform
import importlib.metadata as importlib_metadata
from tabulate import tabulate
import subprocess

# 권장 버전 리스트
requirements = {
    "Python": "3.8",
    "magenta": "2.1.3",
    "note-seq": "latest",
    "numpy": "1.24.4",
    "librosa": "0.10.1",
    "numba": "0.57.1",
    "llvmlite": "0.40.1",
    "python-rtmidi": "1.5.8",
    "llvm-openmp": "conda",
}

def get_conda_version(package_name):
    try:
        result = subprocess.run(['conda', 'list', package_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in result.stdout.splitlines():
            if line.startswith(package_name):
                return line.split()[1]  # 버전 정보 추출
        return "❌ Not installed"
    except Exception:
        return "❌ Error checking"

# Python 버전
python_version = platform.python_version()
results = [["Python", python_version, requirements["Python"]]]

# 나머지 패키지들 확인
for package, expected in requirements.items():
    if package == "Python":
        continue
    if package == "llvm-openmp":
        version = get_conda_version(package)
    else:
        try:
            version = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            version = "❌ Not installed"
    results.append([package, version, expected])

# 표 출력
print(tabulate(results, headers=["패키지", "설치된 버전", "권장 버전"], tablefmt="github"))