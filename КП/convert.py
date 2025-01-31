from PIL import Image
import struct
import ctypes

s = 125
for i1 in range(s):
	i = i1 + 1
	s1 = i // 100
	s2 = (i // 10) % 10
	s3 = i % 10
	name = 'res/' + str(s1) + str(s2) + str(s3) + '.data'
	fin = open(name, 'rb')
	(w, h) = struct.unpack('hi', fin.read(8))
	buff = ctypes.create_string_buffer(4 * w * h)
	fin.readinto(buff)
	fin.close()
	img = Image.new('RGBA', (w, h))
	pix = img.load()
	offset = 0
	for j in range(h):
		for i in range(w):
			(r, g, b, a) = struct.unpack_from('cccc', buff, offset)
			pix[i, j] = (ord(r), ord(g), ord(b), ord(a))
			offset += 4
	name1 = 'out/' + str(s1) + str(s2) + str(s3) + '.png'
	img.save(name1)
