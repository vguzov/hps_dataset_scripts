import numpy as np
from abc import ABC, abstractmethod
from OpenGL.GL import *
import glm
from scipy.spatial.transform import Rotation


class BaseCameraModel(ABC):
    def __init__(self, context, shader, name):
        self.context = context
        self.shader = shader
        self.model = name

    def init_extrinsics(self, quat, pose):
        R = Rotation.from_quat(np.roll(quat, -1)).as_matrix()
        t = np.array([pose]).T
        RT_inv = np.vstack([np.hstack([R.T, -np.matmul(R.T, t)]), [[0, 0, 0, 1]]])

        self.context.View = glm.mat4(RT_inv.T.astype(np.float32).copy())
        self.context.Model = glm.mat4(1.0)
        self.context.MV = self.context.View * self.context.Model

        self.context.shader_ids.update({'MV': glGetUniformLocation(self.shader.program, 'MV')})

    @abstractmethod
    def init_intrinsics(self, **kwargs):
        pass

    def upload_extrinsics(self):
        glUniformMatrix4fv(self.context.shader_ids['MV'], 1, GL_FALSE, glm.value_ptr(self.context.MV))

    @abstractmethod
    def upload_intrinsics(self):
        pass

    def upload(self):
        self.upload_extrinsics()
        self.upload_intrinsics()

    def locate_uniforms(self, keys):
        self.context.shader_ids = {k: glGetUniformLocation(self.shader.program, k) for k in keys}


class OcamModel(BaseCameraModel):
    def __init__(self, context, shader):
        super().__init__(context, shader, "ocam")

    def init_intrinsics(self, cameramodel_dict, fov=360, far=20.):
        ocammodel_dict = cameramodel_dict['OCamModel']
        # polynomial coefficients for the direct mapping function
        ocam_pol = [float(x) for x in ocammodel_dict['cam2world']['coeff']]
        # polynomial coefficients for the inverse mapping function
        ocam_invpol = np.array([float(x) for x in ocammodel_dict['world2cam']['coeff']])
        # center: "row" and "column", starting from 0 (C convention)
        ocam_xy_center = np.array((float(ocammodel_dict['cx']), float(ocammodel_dict['cy'])))
        # _affine parameters "c", "d", "e"
        ocam_affine = np.array([float(ocammodel_dict[x]) for x in ['c', 'd', 'e']])
        # image size: "height" and "width"
        ocam_imsize = cameramodel_dict['ImageSize']
        ocam_img_size = np.array((int(ocam_imsize['Width']), int(ocam_imsize['Height'])))

        self.context.ocam_invpol = ocam_invpol / ocam_img_size[0] * 2
        self.context.ocam_center_off = ocam_xy_center / ocam_img_size[::-1] * 2 - 1
        self.context.ocam_theta_thresh = np.deg2rad(fov / 2) - np.pi / 2
        self.context.ocam_affine = ocam_affine.copy()
        self.context.ocam_affine[:2] *= ocam_img_size[0] / ocam_img_size[1]
        self.context.far = far

        self.locate_uniforms(['ocam_invpol', 'ocam_affine', 'ocam_center_off', 'ocam_theta_thresh', 'far'])

    def upload_intrinsics(self):
        glUniform1dv(self.context.shader_ids['ocam_invpol'], 18, self.context.ocam_invpol.astype(np.float64).copy())
        glUniform3dv(self.context.shader_ids['ocam_affine'], 1, self.context.ocam_affine.astype(np.float64).copy())
        glUniform2dv(self.context.shader_ids['ocam_center_off'], 1,
                     self.context.ocam_center_off.astype(np.float64).copy())
        glUniform1f(self.context.shader_ids['ocam_theta_thresh'], float(self.context.ocam_theta_thresh))
        glUniform1f(self.context.shader_ids['far'], float(self.context.far))


class OpenCVModel(BaseCameraModel):
    def __init__(self, context, shader):
        super().__init__(context, shader, "opencv")

    def init_intrinsics(self, image_size, focal_dist, center, distorsion_coeffs, far=20.):
        assert len(distorsion_coeffs) == 5
        image_size = np.array(image_size)
        focal_dist = np.array(focal_dist)
        center = np.array(center)
        distorsion_coeffs = np.array(distorsion_coeffs)
        self.context.focal_dist = (focal_dist/image_size*2).astype(np.float32).copy()
        self.context.center_off = (center/image_size*2 - 1).astype(np.float32).copy()
        self.context.distorsion_coeffs = distorsion_coeffs.astype(np.float32).copy()
        self.context.far = np.array(far).astype(np.float32).copy()
        self.locate_uniforms(['distorsion_coeff', 'center_off', 'focal_dist', 'far'])

    def upload_intrinsics(self):
        glUniform1fv(self.context.shader_ids['distorsion_coeff'], 5, self.context.distorsion_coeffs)
        glUniform2fv(self.context.shader_ids['center_off'], 1, self.context.center_off)
        glUniform2fv(self.context.shader_ids['focal_dist'], 1, self.context.focal_dist)
        glUniform1f(self.context.shader_ids['far'], self.context.far)


class PerspectiveModel(BaseCameraModel):
    def __init__(self, context, shader):
        super().__init__(context, shader, "perspective")

    def init_intrinsics(self, image_size, fov=45., far=20., near=0.05):
        width,height = image_size
        self.context.Projection = glm.perspective(glm.radians(fov),float(width)/float(height),near,far)
        self.locate_uniforms(['P'])

    def upload_intrinsics(self):
        glUniformMatrix4fv(self.context.shader_ids['P'], 1, GL_FALSE, glm.value_ptr(self.context.Projection))


camera_models = {'ocam': OcamModel, 'opencv': OpenCVModel, 'perspective': PerspectiveModel}
vertex_shader_models = {'ocam': 'vertex_ocam.glsl', 'opencv': 'vertex_opencv.glsl',
                        'perspective': 'vertex_perspective.glsl'}
