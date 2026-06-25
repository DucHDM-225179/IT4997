import bpy
import numpy as np
import os
from mathutils import Matrix, Vector

# --- CONFIGURATION ---
NPZ_PATH = "TO_BE_REPLACED_NPZ_PATH"
PC_NPZ_PATH = "TO_BE_REPLACED_PC_NPZ_PATH"
VIDEO_PATH = "TO_BE_REPLACED_VIDEO_PATH" 
ORIGINAL_RES = TO_BE_REPLACED_ORIGINAL_RES # (Width, Height)
FRAME_STRIDE = TO_BE_REPLACED_FRAME_STRIDE
START_FRAME = TO_BE_REPLACED_START_FRAME
# ---------------------

def create_blender_camera(data):
    intrinsics = data['intrinsics'][0] 
    W_orig, H_orig = ORIGINAL_RES
    
    # Create camera
    cam_data = bpy.data.cameras.new("SpaTrackCam")
    cam_obj = bpy.data.objects.new("SpaTrackCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    
    # 1. Handle Focal Length (FOV)
    fy = intrinsics[1, 1]
    fov_v = 2 * np.arctan(H_orig / (2 * fy))
    
    cam_data.lens_unit = 'FOV'
    cam_data.sensor_fit = 'VERTICAL'
    cam_data.angle = fov_v
    
    # 2. Principal Point Shift
    cam_data.shift_x = (intrinsics[0, 2] - W_orig / 2) / H_orig
    cam_data.shift_y = (H_orig / 2 - intrinsics[1, 2]) / H_orig
    
    # Setup Background Video
    clip = None
    if os.path.exists(VIDEO_PATH):
        cam_data.show_background_images = True
        bg = cam_data.background_images.new()
        bg.source = 'MOVIE_CLIP'
        clip = bpy.data.movieclips.load(VIDEO_PATH)
        bg.clip = clip
        bg.frame_method = 'STRETCH'
        bg.alpha = 1.0

    return cam_obj, clip

def create_animated_point_cloud(pc_npz_path, offset):
    if not os.path.exists(pc_npz_path):
        print(f"Point cloud file not found: {pc_npz_path}")
        return
        
    pc_data = np.load(pc_npz_path)
    points_seq = pc_data['points'] # (T, M, 3)
    colors_seq = pc_data['colors'] # (T, M, 3)
    T, M, _ = points_seq.shape
    
    # Create parent empty object to group all frame point clouds
    pc_parent = bpy.data.objects.new("SpaTrackPointCloud", None)
    bpy.context.collection.objects.link(pc_parent)
    
    end_scene_frame = START_FRAME + (T - 1 - offset) * FRAME_STRIDE
    
    for t in range(T):
        pts = points_seq[t]
        cols = colors_seq[t]
        
        # Filter out invalid points (placed at camera position / black)
        valid_mask = np.any(cols != 0, axis=-1)
        valid_pts = pts[valid_mask]
        valid_cols = cols[valid_mask] / 255.0
        
        M_valid = len(valid_pts)
        if M_valid == 0:
            continue
            
        # Create mesh and object for this frame's point cloud
        mesh = bpy.data.meshes.new(f"SpaTrackPC_F{t:03d}")
        obj_frame = bpy.data.objects.new(f"SpaTrackPC_F{t:03d}", mesh)
        obj_frame.parent = pc_parent
        bpy.context.collection.objects.link(obj_frame)
        
        mesh.from_pydata(valid_pts.tolist(), [], [])
        mesh.update()
        
        # Set vertex colors
        mesh.color_attributes.new(
            name="Color",
            type="FLOAT_COLOR",
            domain="POINT"
        )
        if "Color" in mesh.color_attributes:
            color_attr = mesh.color_attributes["Color"]
            rgba = np.ones((M_valid, 4), dtype=np.float32)
            rgba[:, :3] = valid_cols
            try:
                color_attr.data.foreach_set("color", rgba.ravel())
            except Exception:
                for i in range(M_valid):
                    color_attr.data[i].color = (rgba[i][0], rgba[i][1], rgba[i][2], 1.0)
        mesh.update()
        
        # Keyframe visibility to only show on the active frame
        blender_frame = START_FRAME + (t - offset) * FRAME_STRIDE
        
        # 1. Hide at START_FRAME if active frame is later
        if blender_frame > START_FRAME:
            obj_frame.hide_viewport = True
            obj_frame.hide_render = True
            obj_frame.keyframe_insert(data_path="hide_viewport", frame=START_FRAME)
            obj_frame.keyframe_insert(data_path="hide_render", frame=START_FRAME)
            
            # Hide right up to the frame before
            obj_frame.hide_viewport = True
            obj_frame.hide_render = True
            obj_frame.keyframe_insert(data_path="hide_viewport", frame=blender_frame - 1)
            obj_frame.keyframe_insert(data_path="hide_render", frame=blender_frame - 1)
            
        # 2. Show at the active frame
        obj_frame.hide_viewport = False
        obj_frame.hide_render = False
        obj_frame.keyframe_insert(data_path="hide_viewport", frame=blender_frame)
        obj_frame.keyframe_insert(data_path="hide_render", frame=blender_frame)
        
        # 3. Hide from the next frame onwards
        if blender_frame < end_scene_frame:
            obj_frame.hide_viewport = True
            obj_frame.hide_render = True
            obj_frame.keyframe_insert(data_path="hide_viewport", frame=blender_frame + 1)
            obj_frame.keyframe_insert(data_path="hide_render", frame=blender_frame + 1)
            
            obj_frame.hide_viewport = True
            obj_frame.hide_render = True
            obj_frame.keyframe_insert(data_path="hide_viewport", frame=end_scene_frame)
            obj_frame.keyframe_insert(data_path="hide_render", frame=end_scene_frame)
            
    print(f"Import Finished. Loaded {T} frame point clouds grouped under {pc_parent.name}.")

def load_spatrack_data(npz_path):
    data = np.load(npz_path)
    coords_3d = data['coords']      # (T, N, 3) World space
    extrinsics = data['extrinsics'] # (T, 4, 4) W2C
    T, N, _ = coords_3d.shape

    # Camera
    cam_obj, clip = create_blender_camera(data)
    
    # Calculate offset if Blender video frames differ from PyAV frames
    video_frames = clip.frame_duration if clip else T
    offset = max(0, T - video_frames)

    # 1. Setup Scene
    # Total frames is determined by data, but we offset by START_FRAME
    bpy.context.scene.frame_start = START_FRAME
    bpy.context.scene.frame_end = START_FRAME + (T - 1 - offset) * FRAME_STRIDE
    bpy.context.scene.render.resolution_x = ORIGINAL_RES[0]
    bpy.context.scene.render.resolution_y = ORIGINAL_RES[1]
    
    # 3. Coordinate Transformation Matrices
    m_world_cv_to_bl = Matrix(((1,0,0,0),(0,0,1,0),(0,-1,0,0),(0,0,0,1)))
    m_cam_cv_to_bl = Matrix(((1,0,0,0),(0,-1,0,0),(0,0,-1,0),(0,0,0,1)))

    # 4. Animate Camera
    for t in range(T):
        blender_frame = START_FRAME + (t - offset) * FRAME_STRIDE
        if blender_frame < START_FRAME:
            continue
        w2c = Matrix(extrinsics[t].tolist())
        c2w_cv = w2c.inverted()
        cam_obj.matrix_world = m_world_cv_to_bl @ c2w_cv @ m_cam_cv_to_bl
        cam_obj.keyframe_insert(data_path="location", frame=blender_frame)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=blender_frame)

    # 5. Create Point Tracks (Raw 3D Coordinates)
    points_parent = bpy.data.objects.new("Tracks", None)
    bpy.context.collection.objects.link(points_parent)
    
    for n in range(N):
        empty = bpy.data.objects.new(f"Track_{n:04d}", None)
        empty.parent = points_parent
        empty.empty_display_size = 0.05
        empty.empty_display_type = 'PLAIN_AXES'
        bpy.context.collection.objects.link(empty)
        
        for t in range(T):
            blender_frame = START_FRAME + (t - offset) * FRAME_STRIDE
            if blender_frame < START_FRAME:
                continue
            p_world = Vector(coords_3d[t, n])
            
            # Map CV coordinates to Blender Space
            empty.location = (p_world.x, p_world.z, -p_world.y)
            empty.keyframe_insert(data_path="location", frame=blender_frame)

    # 6. Import Point Cloud (.npz) if exists
    if PC_NPZ_PATH and os.path.exists(PC_NPZ_PATH):
        create_animated_point_cloud(PC_NPZ_PATH, offset)

    print(f"Import Finished. Loaded {N} raw 3D tracks (offset={offset}).")

if __name__ == "__main__":
    load_spatrack_data(NPZ_PATH)

if __name__ == "__main__":
    load_spatrack_data(NPZ_PATH)
