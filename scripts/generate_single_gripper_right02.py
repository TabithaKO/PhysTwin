import pickle, json, glob, torch, numpy as np, warp as wp, cv2, os, copy
from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg

# === Helper to create single-gripper data ===
def make_single_gripper_data(original_pkl, side='right'):
    """Create a new final_data.pkl with only one gripper's controller points"""
    with open(original_pkl, 'rb') as f:
        data = pickle.load(f)
    
    ctrl_pts = data['controller_points']  # (T, N, 3)
    n_ctrl = ctrl_pts.shape[1]  # 10
    
    # KMeans to split
    from sklearn.cluster import KMeans
    pts0 = ctrl_pts[0]  # first frame
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    labels = kmeans.fit_predict(pts0)
    c0 = pts0[labels==0].mean(axis=0)
    c1 = pts0[labels==1].mean(axis=0)
    
    # mask0 = left (lower X), mask1 = right (higher X)
    if c0[0] > c1[0]:
        left_mask = labels == 1
        right_mask = labels == 0
    else:
        left_mask = labels == 0
        right_mask = labels == 1
    
    mask = right_mask if side == 'right' else left_mask
    
    new_data = copy.deepcopy(data)
    new_data['controller_points'] = ctrl_pts[:, mask, :]
    new_data['controller_mask'] = data['controller_mask']  # keep original mask
    
    out_path = original_pkl.replace('final_data.pkl', f'final_data_{side}_only.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(new_data, f)
    
    n_new = new_data['controller_points'].shape[1]
    print(f'Created {out_path}: {n_new} controller points ({side} side)')
    return out_path

# === Create both single-gripper data files ===
orig_pkl = 'data/so101_cloth/traj_right_02/final_data.pkl'
right_pkl = make_single_gripper_data(orig_pkl, 'right')
left_pkl = make_single_gripper_data(orig_pkl, 'left')

# === Load config (shared) ===
cfg.load_from_yaml('configs/cloth.yaml')
with open('experiments_optimization/traj_right_02/optimal_params.pkl','rb') as f:
    cfg.set_optimal_params(pickle.load(f))
with open('data/so101_cloth/traj_right_02/calibrate.pkl','rb') as f:
    c2ws = pickle.load(f)
cfg.c2ws = np.array(c2ws)
cfg.w2cs = np.array([np.linalg.inv(c) for c in c2ws])
with open('data/so101_cloth/traj_right_02/metadata.json','r') as f:
    d = json.load(f)
cfg.intrinsics = np.array(d['intrinsics'])
cfg.WH = d['WH']

K0 = np.array(d['intrinsics'])[0]
K1 = np.array(d['intrinsics'])[1]
c2w0 = np.array(c2ws[0])
c2w1 = np.array(c2ws[1])

ckpt = torch.load(glob.glob('experiments/traj_right_02/train/best_*.pth')[0])

# Motion definitions: (name, direction, steps, spring_Y, settle, stab, data_pkl)
Z = [0,0,0]
motions = [
    # RIGHT gripper only — all use Y=1000 + settle+stabilize
    ('R lift',        [0,0,-0.0003], 500, 1000, 200, 200, right_pkl),
    ('R push right',  [0.0003,0,0],  300, 1000, 200, 200, right_pkl),
    ('R push left',   [-0.0003,0,0], 300, 1000, 200, 200, right_pkl),
    ('R drag fwd',    [0,0.0003,0],  300, 1000, 200, 200, right_pkl),
    ('R drag back',   [0,-0.0003,0], 300, 1000, 200, 200, right_pkl),
    ('R fold over L', [-0.0005,0,-0.0003], 300, 1000, 200, 200, right_pkl),
    # LEFT gripper only — all use Y=1000 + settle+stabilize
    ('L lift',        [0,0,-0.0003], 500, 1000, 200, 200, left_pkl),
    ('L push left',   [-0.0003,0,0], 300, 1000, 200, 200, left_pkl),
    ('L push right',  [0.0003,0,0],  300, 1000, 200, 200, left_pkl),
    ('L drag fwd',    [0,0.0003,0],  300, 1000, 200, 200, left_pkl),
    ('L drag back',   [0,-0.0003,0], 300, 1000, 200, 200, left_pkl),
    ('L fold over R', [0.0005,0,-0.0003], 300, 1000, 200, 200, left_pkl),
]

# === Rendering helpers ===
PW, PH = 1280, 720

def project(pts, K, w2c):
    p = pts.copy(); p[:, 2] *= -1
    ph = np.hstack([p, np.ones((len(p), 1))])
    pc = (w2c @ ph.T).T[:, :3]
    v = pc[:, 2] > 0.01
    px = (K @ pc.T).T
    uv = px[:, :2] / (px[:, 2:3] + 1e-8)
    return uv.astype(int), v

def hcolors(z):
    real = z[z < -0.01]
    if len(real) < 10:
        return np.full((len(z), 3), 150, dtype=np.uint8)
    zn, zx = real.min(), real.max()
    if zx - zn < 0.01: zn, zx = zn - 0.05, zx + 0.05
    zn_ = np.clip(1.0 - (z - zn) / (zx - zn), 0, 1)
    c = np.zeros((len(z), 3), dtype=np.uint8)
    c[:, 2] = (zn_ * 255).astype(np.uint8)
    c[:, 0] = ((1 - zn_) * 255).astype(np.uint8)
    c[:, 1] = 30
    c[z > -0.01] = [120, 120, 120]
    return c

# === Run each motion ===
all_results = []
prev_data_pkl = None

for motion_name, direction, num_steps, sy_override, settle_n, stab_n, data_pkl in motions:
    print(f'\n--- {motion_name}: {num_steps} steps ---')
    
    # Reinitialize trainer if data file changed
    if data_pkl != prev_data_pkl:
        print(f'  Loading {data_pkl}...')
        trainer = InvPhyTrainerWarp(
            data_path=data_pkl,
            base_dir=f'./temp_experiments/single_{os.path.basename(data_pkl)}',
            pure_inference_mode=True,
        )
        sim = trainer.simulator
        prev_data_pkl = data_pkl
    
    # Set springs
    if sy_override is not None:
        n_springs = ckpt['spring_Y'].shape[0]
        fake_Y = torch.full((n_springs,), sy_override, device='cuda')
        sim.set_spring_Y(torch.log(fake_Y).detach().clone())
        print(f'  Spring Y: {sy_override}')
    else:
        sim.set_spring_Y(torch.log(ckpt['spring_Y']).detach().clone())
        print(f'  Spring Y: learned')
    
    sim.set_collide(ckpt['collide_elas'].detach().clone(), ckpt['collide_fric'].detach().clone())
    sim.set_collide_object(ckpt['collide_object_elas'].detach().clone(), ckpt['collide_object_fric'].detach().clone())
    
    n_obj = sim.num_object_points
    ctrl_init = sim.controller_points[0].clone()
    n_ctrl = ctrl_init.shape[0]
    print(f'  {n_ctrl} controller points')
    
    # Reset
    sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)
    current_target = ctrl_init.clone()
    
    # Settle
    if settle_n > 0:
        print(f'  Settling {settle_n} steps...')
        for _ in range(settle_n):
            sim.set_controller_interactive(ctrl_init, ctrl_init)
            if sim.object_collision_flag: sim.update_collision_graph()
            wp.capture_launch(sim.forward_graph)
            sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v, pure_inference=True)
    
    # Stabilize
    if stab_n > 0:
        sim.set_init_state(sim.wp_states[-1].wp_x, wp.zeros(sim.wp_states[-1].wp_v.shape[0], dtype=wp.vec3, device='cuda'), pure_inference=True)
        print(f'  Stabilizing {stab_n} steps...')
        for _ in range(stab_n):
            sim.set_controller_interactive(ctrl_init, ctrl_init)
            if sim.object_collision_flag: sim.update_collision_graph()
            wp.capture_launch(sim.forward_graph)
            v_torch = wp.to_torch(sim.wp_states[-1].wp_v, requires_grad=False)
            v_torch *= 0.1
            damped_v = wp.from_torch(v_torch.contiguous(), dtype=wp.vec3)
            sim.set_init_state(sim.wp_states[-1].wp_x, damped_v, pure_inference=True)
        sim.set_init_state(sim.wp_states[-1].wp_x, wp.zeros(sim.wp_states[-1].wp_v.shape[0], dtype=wp.vec3, device='cuda'), pure_inference=True)
    
    # Collect frames
    move_dir = torch.tensor(direction, dtype=torch.float32, device='cuda')
    frames_cloth = []
    frames_grip = []
    
    x0 = wp.to_torch(sim.wp_states[-1].wp_x if (settle_n > 0 or stab_n > 0) else sim.wp_states[0].wp_x, requires_grad=False)
    frames_cloth.append(x0[:n_obj].detach().cpu().numpy().copy())
    frames_grip.append(current_target.mean(dim=0).detach().cpu().numpy().copy())
    
    nan_hit = False
    prev_target = current_target.clone()
    for st in range(num_steps):
        prev_target = current_target.clone()
        current_target += move_dir
        sim.set_controller_interactive(prev_target, current_target)
        if sim.object_collision_flag: sim.update_collision_graph()
        wp.capture_launch(sim.forward_graph)
        x = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False)
        sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v, pure_inference=True)
        if torch.isnan(x).any():
            print(f'  NaN at step {st+1}')
            nan_hit = True
            break
        frames_cloth.append(x[:n_obj].detach().cpu().numpy().copy())
        frames_grip.append(current_target.mean(dim=0).detach().cpu().numpy().copy())
    
    if nan_hit:
        continue
    
    # Determine gripper side for coloring
    grip_side = 'R' if motion_name.startswith('R') else 'L'
    grip_color = (0, 0, 255) if grip_side == 'R' else (0, 255, 0)
    
    print(f'  Done: {len(frames_cloth)} frames')
    all_results.append((motion_name, frames_cloth, frames_grip, num_steps, grip_side, grip_color))

# === Cameras ===
cloth_world = all_results[0][1][0].copy()
cloth_world[:, 2] *= -1
cloth_center = cloth_world.mean(axis=0)

def rotate_c2w(c2w, center, axis, angle_deg):
    a = np.radians(angle_deg)
    co, si = np.cos(a), np.sin(a)
    if axis == 'x': R = np.array([[1,0,0],[0,co,-si],[0,si,co]])
    elif axis == 'y': R = np.array([[co,0,si],[0,1,0],[-si,0,co]])
    elif axis == 'z': R = np.array([[co,-si,0],[si,co,0],[0,0,1]])
    new = np.eye(4)
    new[:3,3] = R @ (c2w[:3,3] - center) + center
    new[:3,:3] = R @ c2w[:3,:3]
    return new

cameras = [
    ('TopAngle', K0, np.linalg.inv(rotate_c2w(c2w0, cloth_center, 'x', -45))),
    ('Side', K0, np.linalg.inv(rotate_c2w(c2w0, cloth_center, 'x', -70))),
    ('ThreeQ', K0, np.linalg.inv(rotate_c2w(rotate_c2w(c2w0, cloth_center, 'x', -50), cloth_center, 'z', 35))),
    ('Cam1', K1, np.linalg.inv(c2w1)),
]

# Table surface (use original data for table)
trainer_orig = InvPhyTrainerWarp(
    data_path=orig_pkl,
    base_dir='./temp_experiments/table_ref',
    pure_inference_mode=True,
)
x_init = wp.to_torch(trainer_orig.simulator.wp_init_vertices, requires_grad=False)[:trainer_orig.simulator.num_object_points]
sz_ = x_init[:, 2].detach().cpu().numpy()
table_z_val = sz_[sz_ < -0.01].max()
cx_ = x_init[:, 0].detach().cpu().numpy()
cy_ = x_init[:, 1].detach().cpu().numpy()
rm = sz_ < -0.01
xmn, xmx = cx_[rm].min() - 0.12, cx_[rm].max() + 0.12
ymn, ymx = cy_[rm].min() - 0.12, cy_[rm].max() + 0.12
gx = np.arange(xmn, xmx, 0.006)
gy = np.arange(ymn, ymx, 0.006)
gxx, gyy = np.meshgrid(gx, gy)
table_pts = np.zeros((gxx.size, 3), dtype=np.float32)
table_pts[:, 0] = gxx.ravel()
table_pts[:, 1] = gyy.ravel()
table_pts[:, 2] = table_z_val

def draw_table(panel, K, w2c):
    uv, valid = project(table_pts, K, w2c)
    for k in range(len(uv)):
        if valid[k]:
            x, y = uv[k, 0], uv[k, 1]
            if 0 <= x < PW and 0 <= y < PH:
                cv2.circle(panel, (x, y), 3, (70, 65, 55), -1)

# === Render videos ===
out_dir = 'experiments/traj_right_02/single_gripper_v2'
os.makedirs(out_dir, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
FPS = 30

for motion_name, frames_cloth, frames_grip, num_steps, grip_side, grip_color in all_results:
    safe_name = motion_name.replace(' ', '_').replace('+', 'plus').lower()
    out_path = f'{out_dir}/{safe_name}.mp4'
    out = cv2.VideoWriter(out_path, fourcc, FPS, (PW * 2, PH * 2))
    n_frames = len(frames_cloth)
    duration = n_frames / FPS
    print(f'Rendering {motion_name} ({n_frames}f @ {FPS}fps = {duration:.1f}s)...')
    
    for i in range(n_frames):
        cloth = frames_cloth[i]
        gc = np.array(frames_grip[i], dtype=np.float64)
        hc = hcolors(cloth[:, 2])
        panels = []
        
        for cam_name, K, w2c in cameras:
            panel = np.full((PH, PW, 3), 20, dtype=np.uint8)
            draw_table(panel, K, w2c)
            uv, valid = project(cloth, K, w2c)
            for k in range(len(uv)):
                if valid[k]:
                    x, y = uv[k, 0], uv[k, 1]
                    if 0 <= x < PW and 0 <= y < PH:
                        cv2.circle(panel, (x, y), 5, hc[k].tolist(), -1)
            
            # Single gripper marker
            uv_g, v_g = project(gc.reshape(1, 3).astype(np.float32), K, w2c)
            if v_g[0]:
                cx, cy = uv_g[0, 0], uv_g[0, 1]
                if 0 <= cx < PW and 0 <= cy < PH:
                    cv2.circle(panel, (cx, cy), 20, grip_color, -1)
                    cv2.circle(panel, (cx, cy), 23, (255, 255, 255), 3)
                    cv2.putText(panel, grip_side, (cx - 8, cy + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            g0 = np.array(frames_grip[0], dtype=np.float64)
            disp = np.linalg.norm(gc - g0) * 100
            cv2.putText(panel, f'{cam_name} | {motion_name}', (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.putText(panel, f'Step {i}/{num_steps} | {grip_side}:{disp:.1f}cm', (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 2)
            cv2.rectangle(panel, (0, 0), (PW - 1, PH - 1), (60, 60, 60), 1)
            panels.append(panel)
        
        top = np.hstack([panels[0], panels[1]])
        bot = np.hstack([panels[2], panels[3]])
        out.write(np.vstack([top, bot]))
    
    out.release()
    print(f'  Saved: {out_path}')

print('\n=== ALL DONE ===')
for f in sorted(glob.glob(f'{out_dir}/*.mp4')):
    sz = os.path.getsize(f) / 1024 / 1024
    print(f'  {f} ({sz:.1f} MB)')
