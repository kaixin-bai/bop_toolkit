# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Configuration of the BOP Toolkit."""

######## Basic ########

# Folder with the BOP datasets.
# datasets_path = r'/path/to/bop/datasets'
datasets_path = r'/data/hdd1/kb/agile/bkx_master/6dofbkx/datasets/ycbv'

# Folder with pose results to be evaluated.
results_path = r'/data/hdd1/kb/agile/bkx_master/6dofbkx/datasets/ycbv'

# Folder for the calculated pose errors and performance scores.
eval_path = r'/data/hdd1/kb/agile/bkx_master/6dofbkx/datasets/ycbv'

######## Extended ########

# Folder for outputs (e.g. visualizations).
output_path = r'output_path'

# For offscreen C++ rendering: Path to the build folder of bop_renderer (github.com/thodan/bop_renderer).
bop_renderer_path = r'/path/to/bop_renderer/build'

# Executable of the MeshLab server.
meshlab_server_path = r'/path/to/meshlabserver.exe'
