import numpy as np
import vispy
import sys
# from plyfile import PlyData, PlyElement
import pdb
import vispy.scene
from vispy.scene import visuals
from sklearn.preprocessing import normalize

pcl = np.load('first_sample.npy')
# br  = birds_eye_view()
# bp  = br.get_birds_eye_view(pcl)
# plt.imshow(bp)
# plt.show()
# pdb.set_trace()



def normalize(data):
	data_ = data/255
	return data_

#
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
#


pcl= normalize(pcl)


# # create scatter object and fill in the data
scatter = visuals.Markers()
scatter.set_data(pcl, size = 5)
view.add(scatter)
#
view.camera = 'turntable'
#
axis = visuals.XYZAxis(parent=view.scene)
vispy.app.run()
