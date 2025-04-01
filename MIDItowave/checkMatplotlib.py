import numpy as np
import matplotlib.pyplot as plt

mel_path = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/MIDItowave/melspec/01d158f380bc294eae670ec45e45e761.npy"
mel = np.load(mel_path)

print(f"Mel shape: {mel.shape}")
print(f"Mel value range: min={mel.min()}, max={mel.max()}")

plt.imshow(mel, aspect='auto', origin='lower')
plt.title("Mel-spectrogram")
plt.xlabel("Time")
plt.ylabel("Mel bins")
plt.colorbar()
plt.show()

