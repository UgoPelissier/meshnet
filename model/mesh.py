import os.path as osp
import os
import shutil
import numpy as np
import meshio
import vtk
from matplotlib.tri import Triangulation, LinearTriInterpolator
import torch
from torch import Tensor
from pyfreefem import FreeFemRunner


def cell2point(
        file: str,
        field: Tensor
) -> np.ndarray:
    """Convert cell data to point data."""
    reader = vtk.vtkXMLUnstructuredGridReader()  
    reader.SetFileName(file) 
    reader.Update()

    converter = vtk.vtkCellDataToPointData()
    converter.ProcessAllArraysOn()
    converter.SetInputConnection(reader.GetOutputPort())
    converter.Update()
    return np.array(converter.GetOutput().GetPointData().GetArray(field))

def mesh(
        pred: Tensor,
        x: Tensor,
        pos: Tensor,
        path: str,
        save_folder: str,
        name: str
) -> None:
    """Save the mesh prediction in a temporary file."""
    data = torch.hstack((x, pos, pred))

    columns=['type', 'tstart', 'tend', 'radius1', 'radius2', 'length', 'orientation', 'xstart', 'ystart', 'zstart', 'xend', 'yend', 'zend', 'pred']
    type = data[:,0].long()
    tstart = data[:,1]
    tend = data[:,2]
    radius1 = data[:,3]
    radius2 = data[:,4]
    length = data[:,5]
    orientation = data[:,6]
    xstart = data[:,7]
    ystart = data[:,8]
    zstart = data[:,9]
    xend = data[:,10]
    yend = data[:,11]
    zend = data[:,12]
    pred = data[:,13]

    points = orientation*length/pred
    
    runner = FreeFemRunner(script=osp.join(path, 'model', 'prim2mesh.edp'), run_dir=osp.join(save_folder, "tmp", name))
    runner.import_variables(
        path=save_folder,
        name=name,
        type=type,
        tstart=tstart,
        tend=tend,
        xs=xstart,
        ys=ystart,
        zs=zstart,
        xe=xend,
        ye=yend,
        ze=zend,
        r1=radius1,
        r2=radius2,
        points=points.long()
        )
    runner.execute()
    shutil.rmtree(osp.join(save_folder, "tmp", name))

def projection(
        dataset: str,
        val_folder: str,
        name: str,
        epoch: int
)-> float:
    """Project the prediction on the mesh and compute the error."""
    pred = meshio.read(osp.join(val_folder, "tmp",  f'{name}_pred_{epoch:03}.vtk'))
    triPred = Triangulation(x=pred.points[:,0], y=pred.points[:,1], triangles=pred.cells[0].data)

    sol = meshio.read(osp.join(dataset, "raw/sol", f'{name}.vtu'))
    triSol = Triangulation(x=sol.points[:,0], y=sol.points[:,1], triangles=sol.cells[0].data)

    error = sol
    error_field = np.zeros(1)

    for key in sol.cell_data_dict.keys():
        field = cell2point(file=osp.join(dataset, "raw/sol", f'{name}.vtu'), field=key)
        sol.point_data[key] = field

        interp_sol = LinearTriInterpolator(triSol, field)
        pred_field = interp_sol(pred.points[:,0], pred.points[:,1])
        pred_field.fill_value = 0.
        pred.point_data[key] = pred_field

        interp_pred = LinearTriInterpolator(triPred, pred_field)
        error_field = abs(interp_pred(error.points[:,0], error.points[:,1]) - field)
        error_field.fill_value = 0.
        error.point_data[key] = error_field

    pred.write(osp.join(val_folder, "pred", name, f'{name}_pred_{epoch:03}.vtu'))
    error.write(osp.join(val_folder, "error", name, f'{name}_error_{epoch:03}.vtu'))
    os.remove(osp.join(val_folder, "tmp", name, f'{name}_pred_{epoch:03}.vtk'))

    return error_field.sum()

def post_process(
        pred: Tensor,
        x: Tensor,
        pos: Tensor,
        name: str,
        wdir: str,
        save_folder: str
) -> None:
    """Compute the projection loss."""
    if (torch.sum(pred<=0)==0):
        try:
            mesh(pred, x, pos, wdir, save_folder, name)
        except Exception:
            pass