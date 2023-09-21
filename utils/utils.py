from typing import Tuple
import numpy as np
import os
import os.path as osp
import logging

import torch
from torch_geometric.data import Data
import pygmsh
from pygmsh.occ.dummy import Dummy
import gmsh
from lightning.fabric.utilities.cloud_io import get_filesystem


def train_val_test_split(
        path: str,
        n: int,
        val_size: float,
        test_size: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the dataset into train, validation and test sets."""
    indices = np.random.permutation(n)

    if not os.path.exists(osp.join(path, 'indices')):
        os.makedirs(osp.join(path, 'indices'))
        train_index, val_index, test_index = indices[:int(n*(1-(val_size+test_size)))], indices[int(n*(1-(val_size+test_size))):int(n*(1-test_size))],  indices[int(n*(1-test_size)):]
        np.savetxt(osp.join(path, 'indices', 'train_index.txt'), train_index, fmt='%i')
        np.savetxt(osp.join(path, 'indices', 'val_index.txt'), val_index, fmt='%i')
        np.savetxt(osp.join(path, 'indices', 'test_index.txt'), test_index, fmt='%i')
        return train_index, val_index, test_index
    else:
        return load_train_val_test_index(path)
    

def load_train_val_test_index(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the train, validation and test sets indices."""
    return np.loadtxt(osp.join(path, 'indices', 'train_index.txt'), dtype=int), np.loadtxt(osp.join(path, 'indices', 'val_index.txt'), dtype=int), np.loadtxt(osp.join(path, 'indices', 'test_index.txt'), dtype=int)


def get_next_version(path: str) -> int:
    """Get the next version number for the logger."""
    log = logging.getLogger(__name__)
    fs = get_filesystem(path)

    try:
        listdir_info = fs.listdir(path)
    except OSError:
        log.warning("Missing logger folder: %s", path)
        return 0

    existing_versions = []
    for listing in listdir_info:
        d = listing["name"]
        bn = os.path.basename(d)
        if fs.isdir(d) and bn.startswith("version_"):
            dir_ver = bn.split("_")[1].replace("/", "")
            existing_versions.append(int(dir_ver))
    if len(existing_versions) == 0:
        return 0

    return max(existing_versions) + 1

def generate_mesh_2d(
        cad_path: str,
        batch: Data,
        pred: torch.Tensor,
        save_dir: str
) -> None:
    """Generate mesh for 2D geometry."""
    with open (cad_path, 'r+') as f:
        # read the file
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if not (line.startswith('//') or line.startswith('SetFactory'))]

        # extract points, lines and circles
        points = [line for line in lines if line.startswith('Point')]
        lines__ = [line for line in lines if line.startswith('Line')]
        circles = [line for line in lines if line.startswith('Ellipse')]
        curve_loops = [line for line in lines if line.startswith('Curve Loop')]
        extrudes = [line for line in lines if line.startswith('Extrude')]

        # extract coordinates and mesh size
        points_dict = {}
        for line in points:
            id = int(line.split('(')[1].split(')')[0])
            points_dict[id] = [float(p) for p in line.split('{')[1].split('}')[0].split(',')][:3]
        points_dict = dict(sorted(points_dict.items()))

        # extract edges
        lines__ = torch.Tensor([[int(p) for p in line.split('{')[1].split('}')[0].split(',')] for line in lines__]).long()
        circles = torch.Tensor([[int(p) for p in line.split('{')[1].split('}')[0].split(',')] for line in circles]).long()[:,[0,2]]
        edges = torch.cat([lines__, circles], dim=0)

        # list center points
        center_points = []
        for id in list(points_dict.keys()):
            if not id in edges:
                center_points.append(id)

        # extract curve loops
        curve_loops_dict = {}
        for curve_loop in curve_loops:
            id = int(curve_loop.split('(')[1].split(')')[0])
            curve = [int(p) for p in curve_loop.split('{')[1].split('}')[0].split(',')]
            curve_loops_dict[id] = curve

        # Initialize empty geometry using the build in kernel in GMSH
        geometry = pygmsh.geo.Geometry()
        # Fetch model we would like to add data to
        model = geometry.__enter__()

        # Add points
        points_gmsh = []
        count = 0
        for id, point in points_dict.items():
            if id not in center_points:
                points_gmsh.append(model.add_point(x=point, mesh_size=pred[(id-1)-count].cpu().item()))
            else:
                points_gmsh.append(model.add_point(x=point, mesh_size=1.0))
                count += 1

        # Add edges
        channel_lines = []
        for i in range(0, 4):
            channel_lines.append(model.add_line(p0=points_gmsh[edges[i][0]-1], p1=points_gmsh[edges[i][1]-1]))

        start = 4
        while (start+4 <= len(points_dict)):
            channel_lines.append(model.add_ellipse_arc(
                start=points_gmsh[start+1],
                center=points_gmsh[start],
                point_on_major_axis=points_gmsh[start+2],
                end=points_gmsh[start+2]
            ))
            channel_lines.append(model.add_ellipse_arc(
                start=points_gmsh[start+2],
                center=points_gmsh[start],
                point_on_major_axis=points_gmsh[start+3],
                end=points_gmsh[start+3]
            ))
            channel_lines.append(model.add_ellipse_arc(
                start=points_gmsh[start+3],
                center=points_gmsh[start],
                point_on_major_axis=points_gmsh[start+1],
                end=points_gmsh[start+1]
            ))
            start += 4

        # Add curve loops
        channel_loop = []
        for id, curve_loop in curve_loops_dict.items():
            channel_loop.append(model.add_curve_loop(curves=[channel_lines[i-1] for i in curve_loop]))

        # Create a plane surface for meshing
        plane_surface = model.add_plane_surface(curve_loop=channel_loop[0], holes=channel_loop[1:])

        # Call gmsh kernel before add physical entities
        model.synchronize()

        # Add physical entities
        model.add_physical(entities=[plane_surface], label="FLUID")
        model.add_physical(entities=[channel_lines[0]], label="INFLOW")
        model.add_physical(entities=[channel_lines[2]], label="OUTFLOW")
        model.add_physical(entities=[channel_lines[1], channel_lines[3]], label="WALL_BOUNDARY")
        model.add_physical(entities=channel_lines[4:], label="OBSTACLE")

        geometry.generate_mesh(dim=2)
        gmsh.write(osp.join(save_dir, "vtk", 'cad_{:03d}.vtk'.format(batch.name[0])))
        gmsh.write(osp.join(save_dir, "mesh", 'cad_{:03d}.msh2'.format(batch.name[0])))
        
        gmsh.clear()
        geometry.__exit__()

def generate_mesh_3d(
        cad_path: str,
        batch: Data,
        pred: torch.Tensor,
        save_dir: str
) -> None:
    """Generate mesh for 3D geometry."""
    with open (cad_path, 'r+') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if not (line.startswith('//') or line.startswith('SetFactory'))]

        # extract geometries
        box = [line for line in lines if line.startswith('Box')]
        cylinders = [line for line in lines if line.startswith('Cylinder')]

        # extract coordinates from geometries
        box = [float(box[0].split('{')[1].split('}')[0].split(', ')[i]) for i in range(len(box[0].split('{')[1].split('}')[0].split(', ')))] # [xs, ys, zs, dx, dy, dz]
        cylinders = [[float(cylinders[j].split('{')[1].split('}')[0].split(', ')[i]) for i in range(len(cylinders[0].split('{')[1].split('}')[0].split(', '))-1)] for j in range(len(cylinders))] # [xs, ys, zs, dx, dy, dz, r]

        # Initialize empty geometry using the build in kernel in GMSH
        geometry = pygmsh.occ.geometry.Geometry()

        # Fetch model we would like to add data to
        model = geometry.__enter__()

        # Add box to model
        box = model.add_box(box[:3], box[3:])
        model.synchronize()

        # Add cylinders to model
        cyl = []
        for cylinder in cylinders:
            cyl.append(model.add_cylinder(x0=cylinder[:3], axis=cylinder[3:6], radius=cylinder[6], angle=2*np.pi))
            model.synchronize()

        # Add boolean difference of box and cylinder to model
        vol = model.boolean_difference([box], cyl, delete_first=True, delete_other=True)
        model.synchronize()

        # Set mesh size for box points
        for i in range(len(gmsh.model.getEntities(0))):
            gmsh.model.mesh.setSize([gmsh.model.getEntities(0)[i]], pred[i].cpu().item())

        # Set physical labels
        model.add_physical(vol, label='FLUID')
        model.add_physical(Dummy(gmsh.model.getEntities(2)[0][0], gmsh.model.getEntities(2)[0][1]), label='INFLOW')
        model.add_physical(Dummy(gmsh.model.getEntities(2)[6][0], gmsh.model.getEntities(2)[5][1]), label='OUTFLOW')
        model.add_physical([Dummy(gmsh.model.getEntities(2)[i][0], gmsh.model.getEntities(2)[i][1]) for i in range(1,5)], label='WALL_BOUNDARY')
        model.add_physical([Dummy(gmsh.model.getEntities(2)[i][0], gmsh.model.getEntities(2)[i][1]) for i in range(6,6+len(cyl))], label='OBSTACLE')

        # Generate 3d mesh
        geometry.generate_mesh(dim=3)

        gmsh.write(osp.join(save_dir, "vtk", 'cad_{:03d}.vtk'.format(batch.name[0])))
        gmsh.write(osp.join(save_dir, "mesh", 'cad_{:03d}.msh2'.format(batch.name[0])))
        
        gmsh.clear()
        geometry.__exit__()
