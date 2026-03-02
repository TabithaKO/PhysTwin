import pickle, json, glob, torch, numpy as np, warp as wp, cv2, os, warnings
warnings.filterwarnings('ignore')
from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg

cfg.load_from_yaml('configs/cloth.yaml')
with open('experiments_optimization/trajectory_01/optimal_params.pkl','rb') as f:
    cfg.set_optimal_params(pickle.load(f))
with open('data/so101_cloth/trajectory_01/calibrate.pkl','rb') as f:
    c2ws = pickle.load(f)
cfg.c2ws = np.array(c2ws)
cfg.w2cs = np.array([np.linalg.inv(c) for c in c2ws])
with open('data/so101_cloth/trajectory_01/metadata.json','r') as f:
    d = json.load(f)
cfg.intrinsics = np.array(d['intrinsics'])
cfg.WH = d['WH']

SPRING_Y = 1000  # Soft enough for visible drape

trainer = InvPhyTrainerWarp(
    data_path='data/so101_cloth/trajectory_01/final_data.pkl',
    base_dir='./temp_experiments/trajectory_01_v17',
    pure_inference_mode=True,
)
sim = trainer.simulator
ckpt = torch.load(glob.glob('experiments/trajectory_01/train/best_*.pth')[0])

# Override spring stiffness for visible drape
n_springs = ckpt['spring_Y'].shape[0]
fake_Y = torch.full((n_springs,), SPRING_Y, device='cuda')
sim.set_spring_Y(torch.log(fake_Y).detach().clone())
sim.set_collide(ckpt['collide_elas'].detach().clone(), ckpt['collide_fric'].detach().clone())
sim.set_collide_object(ckpt['collide_object_elas'].detach().clone(), ckpt['collide_object_fric'].detach().clone())
print(f'Spring Y override: {SPRING_Y}')

K0 = np.array(d['intrinsics'])[0]
K1 = np.array(d['intrinsics'])[1]
c2w0 = np.array(c2ws[0])
c2w1 = np.array(c2ws[1])
n_obj = sim.num_object_points

ctrl_init = sim.controller_points[0].clone()

# Flatten cloth to table surface
init_x = wp.to_torch(sim.wp_init_vertices, requires_grad=False)
coords = init_x.clone()
table_z = coords[:n_obj, 2].max().item()
print(f'Flattening cloth to table Z={table_z:.4f}')
coords[:n_obj, 2] = table_z
flat_verts = wp.from_torch(coords.contiguous(), dtype=wp.vec3)
ctrl_init[:, 2] = table_z

# KMeans split into L/R groups
from sklearn.cluster import KMeans
vis_ctrl = ctrl_init.cpu().numpy()
kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
labels = kmeans.fit_predict(vis_ctrl)
mask0 = torch.from_numpy(labels == 0).to('cuda')
mask1 = torch.from_numpy(labels == 1).to('cuda')
c0 = vis_ctrl[labels==0].mean(axis=0)
c1 = vis_ctrl[labels==1].mean(axis=0)
if c0[0] > c1[0]:
    mask0, mask1 = mask1, mask0
    c0, c1 = c1, c0
print(f'Left group ({mask0.sum()} pts) centroid: {c0}')
print(f'Right group ({mask1.sum()} pts) centroid: {c1}')

def settle_and_stabilize(sim, ctrl):
    """300 settle + 200 damping steps to eliminate jitter"""
    sim.set_init_state(flat_verts, sim.wp_init_velocities)
    for _ in range(20):
        sim.set_controller_interactive(ctrl, ctrl)
        if sim.object_collision_flag: sim.update_collision_graph()
        wp.capture_launch(sim.forward_graph)
        sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v, pure_inference=True)
    for _ in range(30):
        sim.set_controller_interactive(ctrl, ctrl)
        if sim.object_collision_flag: sim.update_collision_graph()
        wp.capture_launch(sim.forward_graph)
        v_torch = wp.to_torch(sim.wp_states[-1].wp_v, requires_grad=False)
        v_torch *= 0.1
        damped_v = wp.from_torch(v_torch.contiguous(), dtype=wp.vec3)
        sim.set_init_state(sim.wp_states[-1].wp_x, damped_v, pure_inference=True)
    sim.set_init_state(sim.wp_states[-1].wp_x,
                       wp.zeros(sim.wp_states[-1].wp_v.shape[0], dtype=wp.vec3, device='cuda'),
                       pure_inference=True)

# 0.3mm/step, -Z = up
# Motions tuned from test_lift experiments
motions = [
    ('Both lift',       [0,0,-0.0003],      [0,0,-0.0003],      400),
    ('Pull apart',      [0.0003,0,0],       [-0.0003,0,0],      400),
    ('Fold R over L',   [-0.0006,0,-0.0003],[0,0,0],            400),
    ('Fold L over R',   [0,0,0],            [0.0006,0,-0.0003], 400),
    ('Both drag fwd',   [0,0.0003,0],       [0,0.0003,0],       400),
    ('Twist',           [0,0.0003,0],       [0,-0.0003,0],      400),
    ('Push together',   [-0.0003,0,0],      [0.0003,0,0],       400),
    ('Lift + stretch',  [0.0003,0,-0.0002], [-0.0003,0,-0.0002],400),
]

all_results = []
for motion_name, rd, ld, num_steps in motions:
    right_dir = torch.tensor(rd, dtype=torch.float32, device='cuda')
    left_dir = torch.tensor(ld, dtype=torch.float32, device='cuda')
    total_r = np.linalg.norm(rd) * num_steps * 100
    total_l = np.linalg.norm(ld) * num_steps * 100
    print(f'\n--- {motion_name}: {num_steps} steps, R={total_r:.1f}cm, L={total_l:.1f}cm ---')

    # Reset and settle fresh for each motion
    print('  Settling...')
    settle_and_stabilize(sim, ctrl_init)
    current_target = ctrl_init.clone()
    prev_target = current_target.clone()

    x0 = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False)
    z0 = x0[:n_obj, 2].cpu().numpy(); r0 = z0[z0 != 0]
    print(f'  After settle Z: {r0.min():.4f} to {r0.max():.4f} = {(r0.max()-r0.min())*100:.1f}cm')

    frames_cloth = []
    frames_right = []
    frames_left = []

    frames_cloth.append(x0[:n_obj].detach().cpu().numpy().copy())
    frames_right.append(current_target[mask1].mean(dim=0).detach().cpu().numpy().copy())
    frames_left.append(current_target[mask0].mean(dim=0).detach().cpu().numpy().copy())

    nan_hit = False
    for st in range(num_steps):
        prev_target = current_target.clone()
        current_target[mask1] += right_dir
        current_target[mask0] += left_dir

        sim.set_controller_interactive(prev_target, current_target)
        if sim.object_collision_flag:
            sim.update_collision_graph()
        wp.capture_launch(sim.forward_graph)

        x = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False)
        sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v, pure_inference=True)

        if torch.isnan(x).any():
            print(f'  NaN at step {st+1}')
            nan_hit = True
            break

        frames_cloth.append(x[:n_obj].detach().cpu().numpy().copy())
        frames_right.append(current_target[mask1].mean(dim=0).detach().cpu().numpy().copy())
        frames_left.append(current_target[mask0].mean(dim=0).detach().cpu().numpy().copy())

    if nan_hit:
        continue
    print(f'  Done: {len(frames_cloth)} frames')
    all_results.append((motion_name, frames_cloth, frames_right, frames_left, num_steps))

# === Cameras (same as v5) ===
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
    ('Front', K0, np.linalg.inv(c2w0)),
    ('Side', K0, np.linalg.inv(rotate_c2w(c2w0, cloth_center, 'z', 90))),
    ('Angled', K0, np.linalg.inv(rotate_c2w(c2w0, cloth_center, 'y', 60))),
    ('Alt', K1, np.linalg.inv(rotate_c2w(c2w1, cloth_center, 'x', 90))),
]

PW, PH = 1280, 720

# Table surface
x_init = wp.to_torch(sim.wp_init_vertices, requires_grad=False)[:n_obj]
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

def project(pts, K, w2c):
    p = pts.copy()
    p[:, 2] *= -1
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
    if zx - zn < 0.01:
        zn, zx = zn - 0.05, zx + 0.05
    zn_ = np.clip(1.0 - (z - zn) / (zx - zn), 0, 1)
    c = np.zeros((len(z), 3), dtype=np.uint8)
    c[:, 2] = (zn_ * 255).astype(np.uint8)
    c[:, 0] = ((1 - zn_) * 255).astype(np.uint8)
    c[:, 1] = 30
    c[z > -0.01] = [120, 120, 120]
    return c

def draw_table(panel, K, w2c):
    uv, valid = project(table_pts, K, w2c)
    for k in range(len(uv)):
        if valid[k]:
            x, y = uv[k, 0], uv[k, 1]
            if 0 <= x < PW and 0 <= y < PH:
                cv2.circle(panel, (x, y), 3, (70, 65, 55), -1)

out_dir = 'experiments/trajectory_01/dual_gripper_v17'
os.makedirs(out_dir, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
FPS = 30

for motion_name, frames_cloth, frames_right, frames_left, num_steps in all_results:
    safe_name = motion_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus').lower()
    out_path = f'{out_dir}/{safe_name}.mp4'
    out = cv2.VideoWriter(out_path, fourcc, FPS, (PW * 2, PH * 2))
    n_frames = len(frames_cloth)
    duration = n_frames / FPS
    print(f'Rendering {motion_name} ({n_frames}f @ {FPS}fps = {duration:.1f}s)...')
    for i in range(n_frames):
        cloth = frames_cloth[i]
        rc = np.array(frames_right[i], dtype=np.float64)
        lc = np.array(frames_left[i], dtype=np.float64)
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
            uv_r, v_r = project(rc.reshape(1, 3).astype(np.float32), K, w2c)
            if v_r[0]:
                cx, cy = uv_r[0, 0], uv_r[0, 1]
                if 0 <= cx < PW and 0 <= cy < PH:
                    cv2.circle(panel, (cx, cy), 20, (0, 0, 255), -1)
                    cv2.circle(panel, (cx, cy), 23, (255, 255, 255), 3)
                    cv2.putText(panel, 'R', (cx - 8, cy + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            uv_l, v_l = project(lc.reshape(1, 3).astype(np.float32), K, w2c)
            if v_l[0]:
                cx, cy = uv_l[0, 0], uv_l[0, 1]
                if 0 <= cx < PW and 0 <= cy < PH:
                    cv2.circle(panel, (cx, cy), 20, (0, 255, 0), -1)
                    cv2.circle(panel, (cx, cy), 23, (255, 255, 255), 3)
                    cv2.putText(panel, 'L', (cx - 8, cy + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            r0 = np.array(frames_right[0], dtype=np.float64)
            l0 = np.array(frames_left[0], dtype=np.float64)
            r_disp = np.linalg.norm(rc - r0) * 100
            l_disp = np.linalg.norm(lc - l0) * 100
            cv2.putText(panel, f'{cam_name} | {motion_name}', (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.putText(panel, f'Step {i}/{num_steps} | R:{r_disp:.1f}cm L:{l_disp:.1f}cm', (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 2)
            cv2.rectangle(panel, (0, 0), (PW - 1, PH - 1), (60, 60, 60), 1)
            panels.append(panel)
        top = np.hstack([panels[0], panels[1]])
        bot = np.hstack([panels[2], panels[3]])
        frame = np.vstack([top, bot])
        out.write(frame)
    out.release()
    print(f'  Saved: {out_path}')

print('\n=== ALL DONE ===')
for f in sorted(glob.glob(f'{out_dir}/*.mp4')):
    sz2 = os.path.getsize(f) / 1024 / 1024
    print(f'  {f} ({sz2:.1f} MB)')
