import os
import os.path as osp
import glob
from typing import Optional, Callable
import torch
import numpy as np
from alive_progress import alive_bar
from torch_geometric.data import Dataset, Data
import enum

class NodeType(enum.IntEnum):
    """
    Define the code for the one-hot vector representing the node types.
    Note that this is consistent with the codes provided in the original
    MeshGraphNets study: 
    https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
    """
    NORMAL = 0
    INFLOW = 1
    OUTFLOW = 2
    WALL_BOUNDARY = 3
    OBSTACLE = 4
    SIZE = 5


class CAD(Dataset):
    """CAD dataset."""
    def __init__(
            self,
            root: str,
            dim: int,
            split: str,
            idx: np.ndarray,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None
    ) -> None:
        self.root = root
        self.dim = dim
        self.split = split
        self.idx = idx

        self.eps = torch.tensor(1e-8)

        # mean and std of the node features are calculated
        self.mean_vec_x = torch.zeros(5)
        self.std_vec_x = torch.zeros(5)

        # mean and std of the edge features are calculated
        self.mean_vec_edge = torch.zeros(8)
        self.std_vec_edge = torch.zeros(8)

        # mean and std of the output parameters are calculated
        self.mean_vec_y = torch.zeros(1)
        self.std_vec_y = torch.zeros(1)

        # define counters used in normalization
        self.num_accs_x  =  0
        self.num_accs_edge = 0
        self.num_accs_y = 0

        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self) -> list:
        return ["cad_{:03d}".format(i) for i in self.idx]

    @property
    def processed_file_names(self) -> list:
        return glob.glob(os.path.join(self.processed_dir, self.split, 'cad_*.pt'))

    def download(self):
        pass

    def node_type(self, label: str) -> int:
        """Return the code for the one-hot vector representing the node type."""
        if label == 'INFLOW':
            return NodeType.INFLOW
        elif label == 'OUTFLOW':
            return NodeType.OUTFLOW
        elif label == 'WALL_BOUNDARY':
            return NodeType.WALL_BOUNDARY
        elif label == 'OBSTACLE':
            return NodeType.OBSTACLE
        else:
            return NodeType.NORMAL
        
    def update_stats(self, x: torch.Tensor, edge_attr: torch.Tensor, y: torch.Tensor) -> None:
        """Update the mean and std of the node features, edge features, and output parameters."""
        self.mean_vec_x += torch.sum(x, dim = 0)
        self.std_vec_x += torch.sum(x**2, dim = 0)
        self.num_accs_x += x.shape[0]

        self.mean_vec_edge += torch.sum(edge_attr, dim=0)
        self.std_vec_edge += torch.sum(edge_attr**2, dim=0)
        self.num_accs_edge += edge_attr.shape[0]

        self.mean_vec_y += torch.sum(y, dim=0)
        self.std_vec_y += torch.sum(y**2, dim=0)
        self.num_accs_y += y.shape[0]

    def save_stats(self) -> None:
        """Save the mean and std of the node features, edge features, and output parameters."""
        self.mean_vec_x = self.mean_vec_x / self.num_accs_x
        self.std_vec_x = torch.maximum(torch.sqrt(self.std_vec_x / self.num_accs_x - self.mean_vec_x**2), self.eps)

        self.mean_vec_edge = self.mean_vec_edge / self.num_accs_edge
        self.std_vec_edge = torch.maximum(torch.sqrt(self.std_vec_edge / self.num_accs_edge - self.mean_vec_edge**2), self.eps)

        self.mean_vec_y = self.mean_vec_y / self.num_accs_y
        self.std_vec_y = torch.maximum(torch.sqrt(self.std_vec_y / self.num_accs_y - self.mean_vec_y**2), self.eps)

        save_dir = osp.join(self.processed_dir, 'stats', self.split)
        os.makedirs(save_dir, exist_ok=True)

        torch.save(self.mean_vec_x, osp.join(save_dir, 'mean_vec_x.pt'))
        torch.save(self.std_vec_x, osp.join(save_dir, 'std_vec_x.pt'))

        torch.save(self.mean_vec_edge, osp.join(save_dir, 'mean_vec_edge.pt'))
        torch.save(self.std_vec_edge, osp.join(save_dir, 'std_vec_edge.pt'))

        torch.save(self.mean_vec_y, osp.join(save_dir, 'mean_vec_y.pt'))
        torch.save(self.std_vec_y, osp.join(save_dir, 'std_vec_y.pt'))

    def process_file_2d(
            self,
            name: str
    ) -> None:
        with open(osp.join(self.raw_dir, f'{name}.geo_unrolled'), 'r') as f:
            # read lines and remove comments
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            lines = [line for line in lines if not (line.startswith('//') or line.startswith('SetFactory'))]

            # extract points, lines and circles
            points = [line for line in lines if line.startswith('Point')]
            lines__ = [line for line in lines if line.startswith('Line')]
            circles = [line for line in lines if line.startswith('Ellipse')]
            extrudes = [line for line in lines if line.startswith('Extrude')]
            physical_curves = [line for line in lines if line.startswith('Physical Curve')]

            # extract coordinates and mesh size
            points_id = torch.Tensor([int(line.split('(')[1].split(')')[0]) for line in points]).long()
            _, indices = torch.sort(points_id)
            points = torch.Tensor([[float(p) for p in line.split('{')[1].split('}')[0].split(',')] for line in points])
            points = points[indices]
            y = points[:, -1]
            points = points[:, :-1]

            # extract edges
            lines_id = torch.Tensor([int(line.split('(')[1].split(')')[0]) for line in lines__]).long()
            lines__ = torch.Tensor([[int(p) for p in line.split('{')[1].split('}')[0].split(',')] for line in lines__]).long()
            circles_id = torch.Tensor([int(line.split('(')[1].split(')')[0]) for line in circles]).long()
            circles = torch.Tensor([[int(p) for p in line.split('{')[1].split('}')[0].split(',')] for line in circles]).long()[:,[0,2]]
            edges_id = torch.cat([lines_id, circles_id], dim=0) - 1
            _, indices = torch.sort(edges_id)
            edges = torch.cat([lines__, circles], dim=0)-1
            edges = edges[indices]

            # add extruded points and edges
            for extrude in extrudes:
                z_extrude = float(extrude.split('}')[0].split(',')[-1])
                extruded_curves_id = torch.Tensor([int(extrude.split('{')[3:][i].split('}')[0]) for i in range(len(extrude.split('{')[3:]))]).long() - 1
                extruded_points_id = []
                new_extruded_points_id = []
                for id in extruded_curves_id:
                    for i in edges[id]:
                        if not i in extruded_points_id:
                            extruded_points_id.append(i)
                            new_extruded_points_id.append(len(points))
                            points = torch.cat([points, torch.Tensor([points[i,0], points[i,1], z_extrude]).unsqueeze(0)], dim=0)
                            y = torch.cat([y, torch.Tensor([y[i]])], dim=0)
                extruded_points_id = torch.Tensor(extruded_points_id).long()
                new_extruded_points_id = torch.Tensor(new_extruded_points_id).long()

                new_extruded_curves = edges[extruded_curves_id]
                for i in range(len(extruded_points_id)):
                    new_extruded_curves = torch.where(new_extruded_curves == extruded_points_id[i], new_extruded_points_id[i], new_extruded_curves)
                extruded_connexion = torch.cat([extruded_points_id.unsqueeze(dim=1), new_extruded_points_id.unsqueeze(dim=1)], dim=1)
                edges = torch.cat([edges, extruded_connexion, new_extruded_curves], dim=0)

            count = 0
            for i in range(points.shape[0]):
                if not (i-count) in edges:
                    points = torch.cat([points[:i-count], points[i-count+1:]], dim=0)
                    y = torch.cat([y[:i-count], y[i-count+1:]], dim=0)
                    edges = edges - 1*(edges>(i-count))
                    count += 1

            receivers = torch.min(edges, dim=1).values
            senders = torch.max(edges, dim=1).values
            packed_edges = torch.stack([senders, receivers], dim=1)
            # remove duplicates and unpack
            unique_edges, permutation = torch.unique(packed_edges, return_inverse=True, dim=0)
            senders, receivers = unique_edges[:, 0], unique_edges[:, 1]
            # create two-way connectivity
            edge_index = torch.stack([torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0)], dim=0)

            # extract node types
            edge_types = torch.zeros(edges.shape[0], dtype=torch.long)
            for curve in physical_curves:
                label = curve.split('(')[1].split('"')[1]
                lines = curve.split('{')[1].split('}')[0].split(',')
                for line in lines:
                    edge_types[int(line)-1] = self.node_type(label)
            tmp = torch.zeros(edges.shape[0], dtype=torch.long)
            for i in range(len(permutation)):
                tmp[permutation[i]] = edge_types[i]
            edge_types = torch.cat((tmp, tmp), dim=0)
            edge_types_one_hot = torch.nn.functional.one_hot(edge_types.long(), num_classes=NodeType.SIZE)

            # get edge attributes
            u_i = points[edge_index[0]][:,:2]
            u_j = points[edge_index[1]][:,:2]
            u_ij = torch.Tensor(u_i - u_j)
            u_ij_norm = torch.norm(u_ij, p=2, dim=1, keepdim=True)
            edge_attr = torch.cat((u_ij, u_ij_norm, edge_types_one_hot),dim=-1).type(torch.float)

            # get node attributes
            x = torch.zeros(points.shape[0], NodeType.SIZE)
            for i in range(edge_index.shape[0]):
                for j in range(edge_index.shape[1]):
                    x[edge_index[i,j], edge_types[j]] = 1.0

            self.update_stats(x, edge_attr, y)

            torch.save(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, name=torch.tensor(int(name[-3:]), dtype=torch.long)), osp.join(self.processed_dir, self.split, f'{name}.pt'))

    def process_file_3d(
            self,
            name: str
    ) -> None:
        with open(osp.join(self.raw_dir, f'{name}.geo_unrolled'), 'r') as f:
            # read lines and remove comments
            lines = f.readlines()
            lines = [line.replace(' ', '') for line in lines]

            # Extract mesh sizes variables
            mesh_sizes_variables = {}
            tmp = [line for line in lines if line.startswith("cl__")]
            for line in tmp:
                key = line.split('=')[0]
                value = float(line.split('=')[-1].split(';')[0])
                mesh_sizes_variables[key] = value
            
            # Extract coordinates and mesh sizes
            convert_points = {}
            coo = {}
            mesh_sizes = {}
            tmp = [line for line in lines if line.startswith("Point(")]
            i=0
            for line in tmp:
                key = line.split('(')[1].split(')')[0]
                convert_points[key] = i
                value = line.split('{')[1].split('}')[0].split(',')
                coo[key] = [float(value[i]) for i in range(3)]
                if (len(value)>3):
                    mesh_sizes[key] = mesh_sizes_variables[value[-1]]
                i+=1
            points = torch.Tensor(list(coo.values()))

            # Convert mesh sizes to tensor
            y = torch.Tensor(list(mesh_sizes.values()))
            indices = torch.Tensor([convert_points[key] for key in mesh_sizes.keys()]).long()
            points = points[indices]

            # Extract edges
            edges = {}
            tmp = [line for line in lines if line.startswith("Line(") or line.startswith("Spline(")]
            n_cyl = len(tmp)/2
            for line in tmp:
                key = line.split('(')[1].split(')')[0]
                value = line.split('{')[1].split('}')[0].split(',')
                edges[key] = value
            
            # Connectivity matrix
            convert_edges = {}
            connectivity = []
            for key, value in edges.items():
                convert_edges[key] = [len(connectivity)+i for i in range(len(value)-1)]
                for i in range(len(value)-1):
                    connectivity.append([convert_points[value[i]], convert_points[value[i+1]]])
            edges = torch.Tensor(connectivity).long()

            # Identify edges to keep
            keep_edges = []
            for i in range(edges.shape[0]):
                if ((edges[i,0] in indices) and (edges[i,1] in indices)):
                    keep_edges.append(i)
            edges = edges[keep_edges]

            # Add control edges
            for i in range(n_cyl):
                edges = torch.cat((edges, torch.Tensor([[7+i,0]]).long()))
                edges = torch.cat((edges, torch.Tensor([[7+i,2]]).long()))
                edges = torch.cat((edges, torch.Tensor([[7+i,5]]).long()))
                edges = torch.cat((edges, torch.Tensor([[7+i,6]]).long()))

                edges = torch.cat((edges, torch.Tensor([[8+n_cyl+i,1]]).long()))
                edges = torch.cat((edges, torch.Tensor([[8+n_cyl+i,3]]).long()))
                edges = torch.cat((edges, torch.Tensor([[8+n_cyl+i,4]]).long()))
                edges = torch.cat((edges, torch.Tensor([[8+n_cyl+i,9]]).long()))  

            receivers = torch.min(edges, dim=1).values
            senders = torch.max(edges, dim=1).values
            packed_edges = torch.stack([senders, receivers], dim=1)
            # Remove duplicates and unpack
            unique_edges, permutation = torch.unique(packed_edges, return_inverse=True, dim=0)
            senders, receivers = unique_edges[:, 0], unique_edges[:, 1]
            # Create two-way connectivity
            edge_index = torch.stack([torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0)], dim=0).long()

            # Extract curves
            curves = {}
            tmp = [line for line in lines if line.startswith("CurveLoop(")]
            tmp = [line.replace('-', '') for line in tmp]
            for line in tmp:
                key = line.split('(')[1].split(')')[0]
                value = line.split('{')[1].split('}')[0].split(',')
                curves[key] = value

            # Extract surfaces
            surfaces_edges = {}
            tmp = [line for line in lines if line.startswith("PlaneSurface(") or line.startswith("Surface(")]
            for line in tmp:
                key = line.split('(')[1].split(')')[0]
                value = line.split('{')[1].split('}')[0].split(',')
                for i in range(len(value)):
                    if (i==0):
                        surfaces_edges[key] = curves[value[i]][:]
                    else:
                        surfaces_edges[key] += curves[value[i]][:]

            # Extract physical groups
            physical_groups = {}
            tmp = [line for line in lines if line.startswith("PhysicalSurface")]
            for line in tmp:
                key = line.split('"')[1].split('"')[0]
                value = line.split('{')[1].split('}')[0].split(',')
                for i in range(len(value)):
                    for j in range(len(surfaces_edges[value[i]])):
                        if (i==0 and j==0):
                            physical_groups[key] = convert_edges[surfaces_edges[value[i]][j]][:]
                        else:
                            physical_groups[key] += convert_edges[surfaces_edges[value[i]][j]][:]

            # Extract edge physical groups
            physical_groups_order = ['WALL_Y', 'WALL_Z', 'OUTFLOW', 'INFLOW', 'OBSTACLE']
            physical_groups_edges = [0 for i in range(len(connectivity))]
            for key in physical_groups_order:
                for i in range(len(physical_groups[key])):
                    if (key=='INFLOW'):
                        physical_groups_edges[physical_groups[key][i]] = NodeType.INFLOW
                    elif (key=='OUTFLOW'):
                        physical_groups_edges[physical_groups[key][i]] = NodeType.OUTFLOW
                    elif (key=='WALL_Y' or key=='WALL_Z'):
                        physical_groups_edges[physical_groups[key][i]] = NodeType.WALL_BOUNDARY
                    elif (key=='OBSTACLE'):
                        physical_groups_edges[physical_groups[key][i]] = NodeType.OBSTACLE
                    else:
                        raise ValueError('Physical group not recognized.')
            edge_types = torch.Tensor(physical_groups_edges).long().long()
            edge_types = edge_types[keep_edges]
            edge_types = torch.cat((edge_types, (NodeType.WALL_BOUNDARY*torch.ones(edges.shape[0]-edge_types.shape[0])).long().long()))
            
            # convert edges labels to edge_index format
            tmp = torch.zeros(edges.shape[0], dtype=torch.long)
            for i in range(len(permutation)):
                tmp[permutation[i]] = edge_types[i]
            edge_types = torch.cat((tmp, tmp), dim=0)

            # convert edge labels to one-hot vector
            edge_types_one_hot = torch.nn.functional.one_hot(edge_types.long(), num_classes=NodeType.SIZE)

            # construct edge attributes
            u_i = points[edge_index[0]][:,:2]
            u_j = points[edge_index[1]][:,:2]
            u_ij = torch.Tensor(u_i - u_j)
            u_ij_norm = torch.norm(u_ij, p=2, dim=1, keepdim=True)
            edge_attr = torch.cat((u_ij, u_ij_norm, edge_types_one_hot),dim=-1).type(torch.float)

            # get node labels
            x = torch.zeros(points.shape[0], NodeType.SIZE)
            for i in range(edge_index.shape[0]):
                for j in range(edge_index.shape[1]):
                    x[edge_index[i,j], edge_types[j]] = 1.0

            self.update_stats(x, edge_attr, y)

            torch.save(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, name=torch.tensor(int(name[-3:]), dtype=torch.long)), osp.join(self.processed_dir, self.split, f'{name}.pt'))

    def process(self) -> None:
        """Process the dataset."""
        os.makedirs(os.path.join(self.processed_dir, self.split), exist_ok=True)
        print(f'{self.split} dataset')
        with alive_bar(total=len(self.processed_file_names)) as bar:
            for name in self.raw_file_names:
                if self.dim == 2:
                    self.process_file_2d(name)
                elif self.dim == 3:
                    self.process_file_3d(name)
                else:
                    raise ValueError(f'Invalid dimension {self.dim}')
                bar()
        self.save_stats()

    def len(self) -> int:
        return len(self.processed_file_names)
    
    def get(self, idx: int) -> Data:
        data = torch.load(os.path.join(self.processed_dir, self.split, "cad_{:03d}.pt".format(self.idx[idx])))
        return data