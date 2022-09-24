pattern = "BGGR"
ID = 31
path = f"./1_pics/{ID}/"
height = 296
width = 246

assert isinstance(path, str), "Path must be a string"


import numpy as np

data = []
with open(path+'img.txt', "r") as file:
    for line in file:
        if line.isspace():
            continue
        line = line.rstrip()
        line_data = int(line, 16)
        data.append(line_data)
        
data = np.array(data, dtype=np.uint8).reshape((height, width))

out_file = open(path+'std.txt', "w")
for i in range(height):
    for j in range(width):
        print('{:02x}'.format(data[i,j]), file=out_file, end=" ")
    print(file=out_file)


r_ = np.zeros((height>>1, width>>1), dtype=np.uint8)
g_ = np.zeros((height>>1, width>>1), dtype=np.uint8)
b_ = np.zeros((height>>1, width>>1), dtype=np.uint8)

if pattern == "RGGB":
    r_[::, ::] = data[0::2, 0::2]
    g_[::, ::] = (data[0::2, 1::2] / 2 + data[1::2, 0::2] / 2).astype(np.uint8)
    b_[::, ::] = data[1::2, 1::2]
elif pattern == "GRBG":
    r_[::, ::] = data[1::2, 0::2]
    g_[::, ::] = (data[0::2, 0::2] / 2 + data[1::2, 1::2] / 2).astype(np.uint8)
    b_[::, ::] = data[0::2, 1::2]
elif pattern == "GBRG":
    r_[::, ::] = data[0::2, 1::2]
    g_[::, ::] = (data[0::2, 0::2] / 2 + data[1::2, 1::2] / 2).astype(np.uint8)
    b_[::, ::] = data[1::2, 0::2]
elif pattern == "BGGR":
    r_[::, ::] = data[1::2, 1::2]
    g_[::, ::] = (data[0::2, 1::2] / 2 + data[1::2, 0::2] / 2).astype(np.uint8)
    b_[::, ::] = data[0::2, 0::2]

out_file = open(path+'std_R.txt', "w")
for i in range(height):
    for j in range(width):
        print('{:02x}'.format(r_[i>>1,j>>1]), file=out_file)
        
out_file.close()
    
out_file = open(path+'std_G.txt', "w")
for i in range(height):
    for j in range(width):
        print('{:02x}'.format(g_[i>>1,j>>1]), file=out_file)
out_file.close()
        
out_file = open(path+'std_B.txt', "w")
for i in range(height):
    for j in range(width):
        print('{:02x}'.format(b_[i>>1,j>>1]), file=out_file)
        
out_file.close()