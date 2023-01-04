import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt('output_yt_split_1.txt',dtype=int)

# plt.subplot(1,2,1)
f = plt.figure(1)
plt.plot(data[:,0], data[:,1],'k')
plt.title("Upload Packets")
plt.xlabel("packet id")
plt.ylabel("nº Upload packets")

# plt.subplot(1,2,1)
f = plt.figure(2)
plt.plot(data[:,0], data[:,2],'k')
plt.title("Upload Bytes")
plt.xlabel("packet id")
plt.ylabel("nº Upload Bytes")

# plt.subplot(1,2,1)
f = plt.figure(3)
plt.plot(data[:,0], data[:,3],'k')
plt.title("Download Packets")
plt.xlabel("packet id")
plt.ylabel("nº download packets")

# plt.subplot(1,2,1)
f = plt.figure(4)
plt.plot(data[:,0], data[:,4],'k')
plt.title("Download Bytes")
plt.xlabel("packet id")
plt.ylabel("nº download Bytes")


plt.show()
