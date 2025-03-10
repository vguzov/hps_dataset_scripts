import json
import numpy as np
from tqdm import trange
from queue import SimpleQueue
from argparse import ArgumentParser
from videoio import VideoWriter, VideoReader, read_video_params

from egl_renderer import PointCloudRenderer
from egl_renderer.libegl import EGLContext
from egl_renderer.utils import load_pc_from_zip

known_cameras = {
    '029756': {'camera_model': 'opencv',
               'camera_params': [871.8002191615716, 885.5799798967853, 961.5442456700675, 550.6864687995145,
                                 -0.2546258499129524, 0.08039095012755905, 0.00014583290360426732, -1.397345667125021e-05],
               'resolution': (1920, 1080)},
    '029757': {'camera_model': 'opencv',
               'camera_params': [870.3980232039048, 883.7777267137334, 977.2412305921678, 550.4215456406043,
                                 -0.26512391289441484, 0.09675221734814766, 6.0420687185190274e-06, 0.00012467424426572157],
               'resolution': (1920, 1080)}
}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("input_loc", help="Localization file")
    parser.add_argument("input_pczip", help="3D scan zip file")
    parser.add_argument("output", help="Output video")
    parser.add_argument("-iv", "--input_video", help="Input video from the camera")
    parser.add_argument("-res", "--resolution", nargs=2, type=int, help="Overwrite rendering resolution")
    parser.add_argument("-c", "--camera", choices=list(known_cameras.keys()), required=True,
                        help="Camera model (available choises: "+", ".join(sorted(known_cameras.keys()))+")")
    parser.add_argument('-tf', '--total_frames', default=None, type=int, help="Maximum amount of frames to render")
    parser.add_argument('-sf', '--starting_frame', default=0, type=int, help="Staring frame number")
    parser.add_argument('--far', type=float, default=100., help="Maximum rendering distance")
    parser.add_argument("--split_videoside", choices=['l', 'r', 'left', 'right'], default='l',
                        help="Input video side on the split view")

    args = parser.parse_args()

    nosplit = args.input_video is None
    try:
        pointcloud = load_pc_from_zip(args.input_pczip, "pointcloud.ply")
    except FileNotFoundError:
        pointcloud = load_pc_from_zip(args.input_pczip, "*/pointcloud.ply")
    camera = known_cameras[args.camera]

    resolution = args.resolution
    if not resolution:
        resolution = camera['resolution']
        if not nosplit:
            videoparams = read_video_params(args.input_video)
            video_resolution = (videoparams['width'], videoparams['height'])
            video_scaling_required = any([x!=y for x,y in zip(resolution,video_resolution)])

    print(f"Rendering at {resolution} resolution")

    ctx = EGLContext()

    if not ctx.initialize(*resolution):
        print('Could not initialize OpenGL context.')

    opencv_renderer = PointCloudRenderer(*resolution)
    opencv_renderer.init_opengl()


    focal_dist = camera['camera_params'][:2]
    center = camera['camera_params'][2:4]
    dist_coeffs = camera['camera_params'][4:]+[0.]

    opencv_renderer.init_context(pointcloud, camera['camera_model'], image_size=resolution, focal_dist=focal_dist,
                                 center=center,
                                 distorsion_coeffs=dist_coeffs, far=args.far)
    s_results = json.load(open(args.input_loc))
    max_frame_number = max((int(k) for k in s_results.keys()))
    if args.total_frames:
        max_frame_number = min(max_frame_number, args.starting_frame + args.total_frames - 1)

    if not nosplit:
        if video_scaling_required:
            video_iterator = iter(VideoReader(args.input_video, output_resolution=resolution,
                                              start_frame=args.starting_frame))
        else:
            video_iterator = iter(VideoReader(args.input_video, start_frame=args.starting_frame))
        max_frame_number = min(max_frame_number, args.starting_frame+len(video_iterator)-1)

    tqdm_iter = trange(args.starting_frame, max_frame_number+1)

    last_frame = None
    frame_queue = SimpleQueue()
    queue_size = 50
    with VideoWriter(args.output, resolution=resolution, fps=30, preset='veryfast') as vw:
        def process_frame(delete_pbo = True):
            if frame_queue.qsize() >= queue_size:
                last_frame = frame_queue.get()
            else:
                last_frame = None
            if last_frame is not None:
                prev_orig_color, pbo, active = last_frame
                if not active:
                    color = np.zeros(resolution[::-1] + (3,), dtype=np.uint8)
                else:
                    color = opencv_renderer.get_requested_color(pbo, delete_pbo=delete_pbo)
                    color = color[::-1]
                if not nosplit:
                    if args.split_videoside[0] == 'r':
                        color = np.hstack([color[:, :resolution[0] // 2],
                                           prev_orig_color[:, resolution[0] // 2:]])
                    else:
                        color = np.hstack([prev_orig_color[:, :resolution[0] // 2],
                                           color[:, resolution[0] // 2:]])
                vw.write(color)
                return pbo
            else:
                return None

        for frame_ind in tqdm_iter:
            pbo = process_frame(False)
            imname = str(frame_ind)
            impos = s_results[imname] if imname in s_results else None
            if impos is not None:
                pos = np.array(impos['position'])
                quat = np.array(impos['quaternion'])
                opencv_renderer.locate_camera(quat, pos)
                opencv_renderer.draw()
                pbo = opencv_renderer.request_color_async(pbo)
                active = True
            else:
                active = False
            if nosplit:
                orig_color = None
            else:
                try:
                    orig_color = next(video_iterator)
                except StopIteration:
                    orig_color = np.zeros(resolution+(3,), dtype=np.uint8)
            frame_queue.put((orig_color, pbo, active))
        queue_size = 0
        while not frame_queue.empty():
            process_frame(True)