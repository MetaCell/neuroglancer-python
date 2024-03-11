from neuroglancer_utils.create_datasets.create_full_cube import create_cube
from neuroglancer_utils.create_datasets.create_sphere import create_sphere
from neuroglancer_utils.local_server import create_server

create_cube()
create_sphere()
create_server(directory="datasets")
input("Press enter to continue")
