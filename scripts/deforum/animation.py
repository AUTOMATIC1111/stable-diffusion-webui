import numpy as np
import cv2
from functools import reduce
import math
import py3d_tools as p3d
import torch
from einops import rearrange
import re
import pathlib
import os
import pandas as pd

def check_is_number(value):
    float_pattern = r'^(?=.)([+-]?([0-9]*)(\.([0-9]+))?)$'
    return re.match(float_pattern, value)

def sample_from_cv2(sample: np.ndarray) -> torch.Tensor:
    sample = ((sample.astype(float) / 255.0) * 2) - 1
    sample = sample[None].transpose(0, 3, 1, 2).astype(np.float16)
    sample = torch.from_numpy(sample)
    return sample

def sample_to_cv2(sample: torch.Tensor, type=np.uint8) -> np.ndarray:
    sample_f32 = rearrange(sample.squeeze().cpu().numpy(), "c h w -> h w c").astype(np.float32)
    sample_f32 = ((sample_f32 * 0.5) + 0.5).clip(0, 1)
    sample_int8 = (sample_f32 * 255)
    return sample_int8.astype(type)

def construct_RotationMatrixHomogenous(rotation_angles):
    assert(type(rotation_angles)==list and len(rotation_angles)==3)
    RH = np.eye(4,4)
    cv2.Rodrigues(np.array(rotation_angles), RH[0:3, 0:3])
    return RH

def vid2frames(video_path, video_in_frame_path, n=1, overwrite=True):      
    input_content = os.listdir(video_in_frame_path)
    if len(input_content) == 0 or overwrite:
        try:
            for f in pathlib.Path(video_path).glob('*.jpg'):
                f.unlink()
        except:
            pass
        assert os.path.exists(video_path), f"Video input {video_path} does not exist"
          
        vidcap = cv2.VideoCapture(video_path)
        success,image = vidcap.read()
        count = 0
        t=1
        success = True
        while success:
            if count % n == 0:
                cv2.imwrite(video_in_frame_path + os.path.sep + f"{t:05}.jpg" , image)     # save frame as JPEG file
                t += 1
            success,image = vidcap.read()
            count += 1
        print("Converted %d frames" % count)
    else: print("Frames already unpacked")

# https://en.wikipedia.org/wiki/Rotation_matrix
def getRotationMatrixManual(rotation_angles):
	
    rotation_angles = [np.deg2rad(x) for x in rotation_angles]
    
    phi         = rotation_angles[0] # around x
    gamma       = rotation_angles[1] # around y
    theta       = rotation_angles[2] # around z
    
    # X rotation
    Rphi        = np.eye(4,4)
    sp          = np.sin(phi)
    cp          = np.cos(phi)
    Rphi[1,1]   = cp
    Rphi[2,2]   = Rphi[1,1]
    Rphi[1,2]   = -sp
    Rphi[2,1]   = sp
    
    # Y rotation
    Rgamma        = np.eye(4,4)
    sg            = np.sin(gamma)
    cg            = np.cos(gamma)
    Rgamma[0,0]   = cg
    Rgamma[2,2]   = Rgamma[0,0]
    Rgamma[0,2]   = sg
    Rgamma[2,0]   = -sg
    
    # Z rotation (in-image-plane)
    Rtheta      = np.eye(4,4)
    st          = np.sin(theta)
    ct          = np.cos(theta)
    Rtheta[0,0] = ct
    Rtheta[1,1] = Rtheta[0,0]
    Rtheta[0,1] = -st
    Rtheta[1,0] = st
    
    R           = reduce(lambda x,y : np.matmul(x,y), [Rphi, Rgamma, Rtheta]) 
    
    return R

def getPoints_for_PerspectiveTranformEstimation(ptsIn, ptsOut, W, H, sidelength):
    
    ptsIn2D      =  ptsIn[0,:]
    ptsOut2D     =  ptsOut[0,:]
    ptsOut2Dlist =  []
    ptsIn2Dlist  =  []
    
    for i in range(0,4):
        ptsOut2Dlist.append([ptsOut2D[i,0], ptsOut2D[i,1]])
        ptsIn2Dlist.append([ptsIn2D[i,0], ptsIn2D[i,1]])
    
    pin  =  np.array(ptsIn2Dlist)   +  [W/2.,H/2.]
    pout = (np.array(ptsOut2Dlist)  +  [1.,1.]) * (0.5*sidelength)
    pin  = pin.astype(np.float32)
    pout = pout.astype(np.float32)
    
    return pin, pout


def warpMatrix(W, H, theta, phi, gamma, scale, fV):
    
    # M is to be estimated
    M          = np.eye(4, 4)
    
    fVhalf     = np.deg2rad(fV/2.)
    d          = np.sqrt(W*W+H*H)
    sideLength = scale*d/np.cos(fVhalf)
    h          = d/(2.0*np.sin(fVhalf))
    n          = h-(d/2.0)
    f          = h+(d/2.0)
    
    # Translation along Z-axis by -h
    T       = np.eye(4,4)
    T[2,3]  = -h
    
    # Rotation matrices around x,y,z
    R = getRotationMatrixManual([phi, gamma, theta])
    
    
    # Projection Matrix 
    P       = np.eye(4,4)
    P[0,0]  = 1.0/np.tan(fVhalf)
    P[1,1]  = P[0,0]
    P[2,2]  = -(f+n)/(f-n)
    P[2,3]  = -(2.0*f*n)/(f-n)
    P[3,2]  = -1.0
    
    # pythonic matrix multiplication
    F       = reduce(lambda x,y : np.matmul(x,y), [P, T, R]) 
    
    # shape should be 1,4,3 for ptsIn and ptsOut since perspectiveTransform() expects data in this way. 
    # In C++, this can be achieved by Mat ptsIn(1,4,CV_64FC3);
    ptsIn = np.array([[
                 [-W/2., H/2., 0.],[ W/2., H/2., 0.],[ W/2.,-H/2., 0.],[-W/2.,-H/2., 0.]
                 ]])
    ptsOut  = np.array(np.zeros((ptsIn.shape), dtype=ptsIn.dtype))
    ptsOut  = cv2.perspectiveTransform(ptsIn, F)
    
    ptsInPt2f, ptsOutPt2f = getPoints_for_PerspectiveTranformEstimation(ptsIn, ptsOut, W, H, sideLength)
    
    # check float32 otherwise OpenCV throws an error
    assert(ptsInPt2f.dtype  == np.float32)
    assert(ptsOutPt2f.dtype == np.float32)
    M33 = cv2.getPerspectiveTransform(ptsInPt2f,ptsOutPt2f)

    return M33, sideLength

def anim_frame_warp_2d(prev_img_cv2, args, anim_args, keys, frame_idx):
    angle = keys.angle_series[frame_idx]
    zoom = keys.zoom_series[frame_idx]
    translation_x = keys.translation_x_series[frame_idx]
    translation_y = keys.translation_y_series[frame_idx]

    center = (args.W // 2, args.H // 2)
    trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    rot_mat = cv2.getRotationMatrix2D(center, angle, zoom)
    trans_mat = np.vstack([trans_mat, [0,0,1]])
    rot_mat = np.vstack([rot_mat, [0,0,1]])
    if anim_args.flip_2d_perspective:
        perspective_flip_theta = keys.perspective_flip_theta_series[frame_idx]
        perspective_flip_phi = keys.perspective_flip_phi_series[frame_idx]
        perspective_flip_gamma = keys.perspective_flip_gamma_series[frame_idx]
        perspective_flip_fv = keys.perspective_flip_fv_series[frame_idx]
        M,sl = warpMatrix(args.W, args.H, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, 1., perspective_flip_fv);
        post_trans_mat = np.float32([[1, 0, (args.W-sl)/2], [0, 1, (args.H-sl)/2]])
        post_trans_mat = np.vstack([post_trans_mat, [0,0,1]])
        bM = np.matmul(M, post_trans_mat)
        xform = np.matmul(bM, rot_mat, trans_mat)
    else:
        xform = np.matmul(rot_mat, trans_mat)

    return cv2.warpPerspective(
        prev_img_cv2,
        xform,
        (prev_img_cv2.shape[1], prev_img_cv2.shape[0]),
        borderMode=cv2.BORDER_WRAP if anim_args.border == 'wrap' else cv2.BORDER_REPLICATE
    )

def anim_frame_warp_3d(device, prev_img_cv2, depth, anim_args, keys, frame_idx):
    TRANSLATION_SCALE = 1.0/200.0 # matches Disco
    translate_xyz = [
        -keys.translation_x_series[frame_idx] * TRANSLATION_SCALE, 
        keys.translation_y_series[frame_idx] * TRANSLATION_SCALE, 
        -keys.translation_z_series[frame_idx] * TRANSLATION_SCALE
    ]
    rotate_xyz = [
        math.radians(keys.rotation_3d_x_series[frame_idx]), 
        math.radians(keys.rotation_3d_y_series[frame_idx]), 
        math.radians(keys.rotation_3d_z_series[frame_idx])
    ]
    rot_mat = p3d.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=device), "XYZ").unsqueeze(0)
    result = transform_image_3d(device, prev_img_cv2, depth, rot_mat, translate_xyz, anim_args, keys, frame_idx)
    torch.cuda.empty_cache()
    return result

def transform_image_3d(device, prev_img_cv2, depth_tensor, rot_mat, translate, anim_args, keys, frame_idx):
    # adapted and optimized version of transform_image_3d from Disco Diffusion https://github.com/alembics/disco-diffusion 
    w, h = prev_img_cv2.shape[1], prev_img_cv2.shape[0]

    aspect_ratio = float(w)/float(h)
    near = keys.near_series[frame_idx]
    far = keys.far_series[frame_idx]
    fov_deg = keys.fov_series[frame_idx]
    persp_cam_old = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov_deg, degrees=True, device=device)
    persp_cam_new = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov_deg, degrees=True, R=rot_mat, T=torch.tensor([translate]), device=device)

    # range of [-1,1] is important to torch grid_sample's padding handling
    y,x = torch.meshgrid(torch.linspace(-1.,1.,h,dtype=torch.float32,device=device),torch.linspace(-1.,1.,w,dtype=torch.float32,device=device))
    z = torch.as_tensor(depth_tensor, dtype=torch.float32, device=device)
    xyz_old_world = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)

    xyz_old_cam_xy = persp_cam_old.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]
    xyz_new_cam_xy = persp_cam_new.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]

    offset_xy = xyz_new_cam_xy - xyz_old_cam_xy
    # affine_grid theta param expects a batch of 2D mats. Each is 2x3 to do rotation+translation.
    identity_2d_batch = torch.tensor([[1.,0.,0.],[0.,1.,0.]], device=device).unsqueeze(0)
    # coords_2d will have shape (N,H,W,2).. which is also what grid_sample needs.
    coords_2d = torch.nn.functional.affine_grid(identity_2d_batch, [1,1,h,w], align_corners=False)
    offset_coords_2d = coords_2d - torch.reshape(offset_xy, (h,w,2)).unsqueeze(0)

    image_tensor = rearrange(torch.from_numpy(prev_img_cv2.astype(np.float32)), 'h w c -> c h w').to(device)
    new_image = torch.nn.functional.grid_sample(
        image_tensor.add(1/512 - 0.0001).unsqueeze(0), 
        offset_coords_2d, 
        mode=anim_args.sampling_mode, 
        padding_mode=anim_args.padding_mode, 
        align_corners=False
    )

    # convert back to cv2 style numpy array
    result = rearrange(
        new_image.squeeze().clamp(0,255), 
        'c h w -> h w c'
    ).cpu().numpy().astype(prev_img_cv2.dtype)
    return result

class DeformAnimKeys():
    def __init__(self, anim_args):
        self.angle_series = get_inbetweens(parse_key_frames(anim_args.angle), anim_args.max_frames)
        self.zoom_series = get_inbetweens(parse_key_frames(anim_args.zoom), anim_args.max_frames)
        self.translation_x_series = get_inbetweens(parse_key_frames(anim_args.translation_x), anim_args.max_frames)
        self.translation_y_series = get_inbetweens(parse_key_frames(anim_args.translation_y), anim_args.max_frames)
        self.translation_z_series = get_inbetweens(parse_key_frames(anim_args.translation_z), anim_args.max_frames)
        self.rotation_3d_x_series = get_inbetweens(parse_key_frames(anim_args.rotation_3d_x), anim_args.max_frames)
        self.rotation_3d_y_series = get_inbetweens(parse_key_frames(anim_args.rotation_3d_y), anim_args.max_frames)
        self.rotation_3d_z_series = get_inbetweens(parse_key_frames(anim_args.rotation_3d_z), anim_args.max_frames)
        self.perspective_flip_theta_series = get_inbetweens(parse_key_frames(anim_args.perspective_flip_theta), anim_args.max_frames)
        self.perspective_flip_phi_series = get_inbetweens(parse_key_frames(anim_args.perspective_flip_phi), anim_args.max_frames)
        self.perspective_flip_gamma_series = get_inbetweens(parse_key_frames(anim_args.perspective_flip_gamma), anim_args.max_frames)
        self.perspective_flip_fv_series = get_inbetweens(parse_key_frames(anim_args.perspective_flip_fv), anim_args.max_frames)
        self.noise_schedule_series = get_inbetweens(parse_key_frames(anim_args.noise_schedule), anim_args.max_frames)
        self.strength_schedule_series = get_inbetweens(parse_key_frames(anim_args.strength_schedule), anim_args.max_frames)
        self.contrast_schedule_series = get_inbetweens(parse_key_frames(anim_args.contrast_schedule), anim_args.max_frames)
        self.cfg_scale_schedule_series = get_inbetweens(parse_key_frames(anim_args.cfg_scale_schedule), anim_args.max_frames)
        self.seed_schedule_series = get_inbetweens(parse_key_frames(anim_args.seed_schedule), anim_args.max_frames)
        self.fov_series = get_inbetweens(parse_key_frames(anim_args.fov_schedule), anim_args.max_frames)
        self.near_series = get_inbetweens(parse_key_frames(anim_args.near_schedule), anim_args.max_frames)
        self.far_series = get_inbetweens(parse_key_frames(anim_args.far_schedule), anim_args.max_frames)

def get_inbetweens(key_frames, max_frames, integer=False, interp_method='Linear'):
    import numexpr
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])
    
    for i in range(0, max_frames):
        if i in key_frames:
            value = key_frames[i]
            value_is_number = check_is_number(value)
            # if it's only a number, leave the rest for the default interpolation
            if value_is_number:
                t = i
                key_frame_series[i] = value
        if not value_is_number:
            t = i
            key_frame_series[i] = numexpr.evaluate(value)
    key_frame_series = key_frame_series.astype(float)
    
    if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
        interp_method = 'Quadratic'    
    if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
        interp_method = 'Linear'
          
    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[max_frames-1] = key_frame_series[key_frame_series.last_valid_index()]
    key_frame_series = key_frame_series.interpolate(method=interp_method.lower(), limit_direction='both')
    if integer:
        return key_frame_series.astype(int)
    return key_frame_series

def parse_key_frames(string, prompt_parser=None):
    # because math functions (i.e. sin(t)) can utilize brackets 
    # it extracts the value in form of some stuff
    # which has previously been enclosed with brackets and
    # with a comma or end of line existing after the closing one
    pattern = r'((?P<frame>[0-9]+):[\s]*\((?P<param>[\S\s]*?)\)([,][\s]?|[\s]?$))'
    frames = dict()
    for match_object in re.finditer(pattern, string):
        frame = int(match_object.groupdict()['frame'])
        param = match_object.groupdict()['param']
        if prompt_parser:
            frames[frame] = prompt_parser(param)
        else:
            frames[frame] = param
    if frames == {} and len(string) != 0:
        raise RuntimeError('Key Frame string not correctly formatted')
    return frames
