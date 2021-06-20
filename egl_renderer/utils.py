import os
import trimesh
import numpy as np
import fnmatch
from zipfile import ZipFile
from io import BytesIO

def open_from_zip(zippath, datapath, return_zip_path = False):
    input_zip = ZipFile(zippath)
    match_fn = lambda x: fnmatch.fnmatch(x, datapath)
    filenames = list(filter(match_fn, input_zip.namelist()))
    if len(filenames) == 0:
        raise FileNotFoundError("No file matching '{}' in archive".format(datapath))
    elif len(filenames) > 1:
        raise FileNotFoundError("More than one file matching '{}' exists in archive: {}".format(datapath, filenames))
    else:
        filename = filenames[0]
        filehandler = BytesIO(input_zip.read(filename))
        if return_zip_path:
            return filehandler, filename
        return filehandler


def load_pc_from_zip(zippath, datapath):
    filehandler, filename = open_from_zip(zippath, datapath, return_zip_path = True)
    ext = os.path.splitext(filename)[1][1:]
    mesh = trimesh.load(filehandler, ext, process=False)
    return mesh


def get_camera_position(xyz_ang, pos):
    camera_pose = np.array([
        [1.0, 0, 0, pos[0]],
        [0.0, 1.0, 0.0, pos[1]],
        [0.0, 0, 1.0, pos[2]],
        [0.0, 0.0, 0.0, 1.0],
    ])
    sin, cos = [np.sin(a) for a in xyz_ang], [np.cos(a) for a in xyz_ang]
    x_rot = np.array(
        [
            [1.0, 0, 0, 0.0],
            [0.0, cos[0], -sin[0], 0.0],
            [0.0, sin[0], cos[0], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    y_rot = np.array(
        [
            [cos[1], 0, sin[1], 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sin[1], 0, cos[1], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    z_rot = np.array(
        [
            [cos[2], -sin[2], 0, 0.0],
            [sin[2], cos[2], 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    return camera_pose.dot(z_rot.dot(y_rot.dot(x_rot)))
