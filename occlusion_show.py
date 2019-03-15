import os
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors

occ_map=np.loadtxt('occlusion_map.txt',dtype=np.float64)

plt.imshow(abs(occ_map), cmap=cm.hot)
plt.colorbar()
plt.show()

occ_map_img = Image.fromarray(np.uint8(cm.hot(abs(occ_map))*255))

tf_img = Image.open("demo_data/sob_lateral.bmp")
target_img = Image.open("demo_data/Doc3_lateral.bmp")
ori_img = Image.new('RGB',(320,256),255)
ori_img.paste(tf_img,(0,0))
ori_img.paste(target_img,(0,128))
#ori_img = ImageOps.invert(ori_img)
#ori_img.show()


blend_img = Image.blend(ori_img,occ_map_img.convert('RGB'),0.5)
blend_img.show()
blend_img.save("Sob-Doc3_blend.bmp")

