import sys
import scipy.io
import numpy as np

matlab_forest_file = sys.argv[1]
forest_file = sys.argv[2]
m_dict = scipy.io.loadmat(matlab_forest_file)
matlab_forest = m_dict['T'][0]
forest_array = np.zeros((len(matlab_forest),), dtype=np.object)
for i in xrange(len(matlab_forest)):
    tree = matlab_forest[i]
    conv_tree = -np.ones(tree.shape, dtype=np.float64)
    conv_tree[:, :tree.shape[1] - 1] = tree[:, 1:]
    conv_tree[:, -1] = tree[:, 0]
    conv_tree[:, 0] = -tree[:, 3]
    conv_tree[:, 1] = tree[:, 2]
    conv_tree[:, 2] = -tree[:, 5]
    conv_tree[:, 3] = tree[:, 4]
    forest_array[i] = conv_tree
m_dict['forest'] = forest_array
scipy.io.savemat(forest_file, m_dict)

