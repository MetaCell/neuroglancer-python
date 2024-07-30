from neuroglancer_utils.create_datasets.create_full_cube import create_cube
from neuroglancer_utils.create_datasets.create_sphere import create_sphere
from neuroglancer_utils.create_datasets.create_allen_multi import create_allen_multi
from neuroglancer_utils.create_datasets.create_cdf_example import create_cdf_example
from neuroglancer_utils.local_server import create_server

create_cube()
create_sphere()
create_allen_multi()
create_cdf_example()
create_server(directory="datasets")
input("Press enter to continue")
