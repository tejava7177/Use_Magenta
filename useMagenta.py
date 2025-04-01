import importlib
import pkg_resources

# 공식 HiFi-GAN 권장 버전
required_versions = {
    "torch": "1.4.0",
    "numpy": "1.17.4",
    "librosa": "0.7.2",
    "scipy": "1.4.1",
    "tensorboard": "2.0",
    "soundfile": "0.10.3.post1",
    "matplotlib": "3.1.3",
}

print("🔍 Checking HiFi-GAN compatible environment...\n")

for pkg, expected_version in required_versions.items():
    try:
        module = importlib.import_module(pkg)
        installed_version = pkg_resources.get_distribution(pkg).version
        status = "✅ MATCH" if installed_version == expected_version else "⚠️ MISMATCH"
        print(f"{pkg:12} | Installed: {installed_version:15} | Required: {expected_version:15} → {status}")
    except ImportError:
        print(f"{pkg:12} | ❌ Not Installed")
    except Exception as e:
        print(f"{pkg:12} | ⚠️ Error checking version: {e}")

print("\n✅ 환경 점검 완료.")