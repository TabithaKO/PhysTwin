import pickle, json, glob, torch, numpy as np, warp as wp, cv2, sys, warnings
warnings.filterwarnings('ignore')
from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg

# Setup
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

K0 = np.array(d['intrinsics'])[0]
c2w0 = np.array(c2ws[0])
w2c0 = np.linalg.inv(c2w0)

PW, PH = 640, 480

def make_virtual_cam(eye, target, up):
    fwd = target - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    u = np.cross(right, fwd)
    R = np.array([right, -u, fwd])
    t = -R @ eye
    w2c = np.eye(4); w2c[:3,:3] = R; w2c[:3,3] = t
    return w2c

def project(pts, K, w2c):
    p = pts.copy(); p[:,2] *= -1
    ph = np.hstack([p, np.ones((len(p),1))])
    pc = (w2c @ ph.T).T[:,:3]
    v = pc[:,2] > 0.01
    px = (K @ pc.T).T
    uv = px[:,:2] / (px[:,2:3]+1e-8)
    return uv.astype(int), v

def init_sim(spring_y):
    trainer = InvPhyTrainerWarp(
        data_path='data/so101_cloth/trajectory_01/final_data.pkl',
        base_dir='./temp_experiments/lift_test',
        pure_inference_mode=True,
    )
    sim = trainer.simulator
    ckpt = torch.load(glob.glob('experiments/trajectory_01/train/best_*.pth')[0])
    n_springs = ckpt['spring_Y'].shape[0]
    fake_Y = torch.full((n_springs,), spring_y, device='cuda')
    sim.set_spring_Y(torch.log(fake_Y).detach().clone())
    sim.set_collide(ckpt['collide_elas'].detach().clone(), ckpt['collide_fric'].detach().clone())
    sim.set_collide_object(ckpt['collide_object_elas'].detach().clone(), ckpt['collide_object_fric'].detach().clone())
    return sim, trainer

def settle_and_stabilize(sim, ctrl_init):
    """Heavy settle + damping to eliminate jitter"""
    sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)
    # 300 settle steps
    print('  Settling 300 steps...')
    for _ in range(300):
        sim.set_controller_interactive(ctrl_init, ctrl_init)
        if sim.object_collision_flag: sim.update_collision_graph()
        wp.capture_launch(sim.forward_graph)
        sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v, pure_inference=True)
    # 200 damping steps
    print('  Damping 200 steps...')
    for _ in range(200):
        sim.set_controller_interactive(ctrl_init, ctrl_init)
        if sim.object_collision_flag: sim.update_collision_graph()
        wp.capture_launch(sim.forward_graph)
        v_torch = wp.to_torch(sim.wp_states[-1].wp_v, requires_grad=False)
        v_torch *= 0.05
        damped_v = wp.from_torch(v_torch.contiguous(), dtype=wp.vec3)
        sim.set_init_state(sim.wp_states[-1].wp_x, damped_v, pure_inference=True)
    # Final zero velocity
    sim.set_init_state(sim.wp_states[-1].wp_x, 
                       wp.zeros(sim.wp_states[-1].wp_v.shape[0], dtype=wp.vec3, device='cuda'),
                       pure_inference=True)
    x0 = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False)[:sim.num_object_points].cpu().numpy()
    z0 = x0[:,2]; r0 = z0[z0!=0]
    print(f'  After settle: Z {r0.min():.4f} to {r0.max():.4f} = {(r0.max()-r0.min())*100:.1f}cm spread')

def get_gripper_masks(ctrl_init):
    from sklearn.cluster import KMeans
    vis = ctrl_init.cpu().numpy()
    km = KMeans(n_clusters=2, random_state=0, n_init=10)
    labels = km.fit_predict(vis)
    m0 = torch.from_numpy(labels == 0).to('cuda')
    m1 = torch.from_numpy(labels == 1).to('cuda')
    c0 = vis[labels==0].mean(axis=0)
    c1 = vis[labels==1].mean(axis=0)
    # m0 = left (smaller X), m1 = right (larger X)
    if c0[0] > c1[0]:
        m0, m1 = m1, m0
    return m0, m1  # left_mask, right_mask

def get_center_mask(sim, ctrl_init):
    """Find particles near center X (between grippers) for center-lift"""
    x = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False)[:sim.num_object_points]
    xcoords = x[:, 0]
    vis = ctrl_init.cpu().numpy()
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, random_state=0, n_init=10)
    labels = km.fit_predict(vis)
    c0 = vis[labels==0].mean(axis=0)
    c1 = vis[labels==1].mean(axis=0)
    center_x = (c0[0] + c1[0]) / 2.0
    # Controller points nearest to center
    dists = torch.abs(ctrl_init[:, 0] - center_x)
    # Move ALL controllers but weight by proximity to center
    return dists

def render_frame(cloth, rc, lc, cameras, n_obj, step, total, label):
    z = cloth[:,2]; real = z[z!=0]
    spread = 0
    if len(real) > 10:
        zn, zx = real.min(), real.max(); spread = zx - zn
        if spread < 0.005: zn, zx = zn - 0.05, zx + 0.05
        t = np.clip((z - zn) / (zx - zn), 0, 1)
        hc = np.zeros((len(z), 3), dtype=np.uint8)
        hc[:,2] = (t * 255).astype(np.uint8)
        hc[:,0] = ((1-t) * 255).astype(np.uint8)
        hc[:,1] = 30
    else:
        hc = np.full((len(z), 3), 150, dtype=np.uint8)

    full = np.full((PH*2, PW*2, 3), 20, dtype=np.uint8)
    for ci, (cname, cw2c, cK) in enumerate(cameras):
        row, col = ci // 2, ci % 2
        panel = np.full((PH, PW, 3), 30, dtype=np.uint8)
        uv, valid = project(cloth, cK, cw2c)
        for k in range(len(uv)):
            if valid[k]:
                px, py = uv[k,0]*PW//1280, uv[k,1]*PH//720
                if 0 <= px < PW and 0 <= py < PH:
                    cv2.circle(panel, (px, py), 3, hc[k].tolist(), -1)
        for gc, color, lbl in [(rc,(0,0,255),'R'), (lc,(0,255,0),'L')]:
            uv_g, v_g = project(gc.reshape(1,3).astype(np.float32), cK, cw2c)
            if v_g[0]:
                gx, gy = uv_g[0,0]*PW//1280, uv_g[0,1]*PH//720
                if 0 <= gx < PW and 0 <= gy < PH:
                    cv2.circle(panel, (gx, gy), 10, color, -1)
                    cv2.putText(panel, lbl, (gx-5, gy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        cv2.putText(panel, cname, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(panel, f'{step}/{total} {label} spr:{spread*100:.1f}cm',
                    (10, PH-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150,150,150), 1)
        full[row*PH:(row+1)*PH, col*PW:(col+1)*PW] = panel
    return full

def run_motion(sim, ctrl_init, left_mask, right_mask, right_dir, left_dir, num_steps, cameras, outpath, label):
    n_obj = sim.num_object_points
    current = ctrl_init.clone()
    prev = current.clone()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outpath, fourcc, 30, (PW*2, PH*2))
    rd = right_dir; ld = left_dir

    for st in range(num_steps):
        prev = current.clone()
        current[right_mask] += rd
        current[left_mask] += ld
        sim.set_controller_interactive(prev, current)
        if sim.object_collision_flag: sim.update_collision_graph()
        wp.capture_launch(sim.forward_graph)
        sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v, pure_inference=True)

        x = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False)
        cloth = x[:n_obj].detach().cpu().numpy()
        rc = current[right_mask].mean(dim=0).detach().cpu().numpy()
        lc = current[left_mask].mean(dim=0).detach().cpu().numpy()
        frame = render_frame(cloth, rc, lc, cameras, n_obj, st+1, num_steps, label)
        out.write(frame)

    out.release()
    xf = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False)[:n_obj].cpu().numpy()
    zf = xf[:,2]; rf = zf[zf!=0]
    print(f'  Final Z: {rf.min():.4f} to {rf.max():.4f} = {(rf.max()-rf.min())*100:.1f}cm')
    print(f'  Saved: {outpath}')

# ============ MAIN ============
SPRING_Y = 1000
STEP_SIZE = 0.0003
NUM_STEPS = 400

print('='*60)
print('LIFT TYPE 1: Edge lift (both grippers lift equally)')
print('='*60)
sim1, _ = init_sim(SPRING_Y)
ctrl1 = sim1.controller_points[0].clone()
settle_and_stabilize(sim1, ctrl1)
left_mask, right_mask = get_gripper_masks(ctrl1)

# Setup cameras from settled cloth position
x0 = wp.to_torch(sim1.wp_states[-1].wp_x, requires_grad=False)[:sim1.num_object_points].cpu().numpy()
cc = x0.mean(axis=0).copy(); cc[2] *= -1
cameras = [
    ('Front', w2c0, K0),
    ('Side', make_virtual_cam(cc + np.array([0.5,0,0]), cc, np.array([0,0,1])), K0),
    ('Top', make_virtual_cam(cc + np.array([0,0,0.5]), cc, np.array([0,1,0])), K0),
    ('Angled', make_virtual_cam(cc + np.array([0.3,-0.3,0.3]), cc, np.array([0,0,1])), K0),
]

lift_dir = torch.tensor([0, 0, -STEP_SIZE], dtype=torch.float32, device='cuda')
zero_dir = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda')
run_motion(sim1, ctrl1, left_mask, right_mask, lift_dir, lift_dir, NUM_STEPS, cameras,
           'experiments/trajectory_01/lift_edge.mp4', f'Edge lift Y={SPRING_Y}')

print()
print('='*60)
print('LIFT TYPE 2: Center sag (grippers hold, gravity drapes center)')
print('='*60)
# This is: lift edges slightly, then hold — letting center sag dynamically
sim2, _ = init_sim(SPRING_Y)
ctrl2 = sim2.controller_points[0].clone()
settle_and_stabilize(sim2, ctrl2)
left_mask2, right_mask2 = get_gripper_masks(ctrl2)

# Lift edges then pull apart to emphasize center drape
lift_spread = torch.tensor([0.0002, 0, -STEP_SIZE], dtype=torch.float32, device='cuda')
lift_spread_l = torch.tensor([-0.0002, 0, -STEP_SIZE], dtype=torch.float32, device='cuda')
run_motion(sim2, ctrl2, left_mask2, right_mask2, lift_spread, lift_spread_l, NUM_STEPS, cameras,
           'experiments/trajectory_01/lift_center_sag.mp4', f'Lift+spread Y={SPRING_Y}')

print('\nDone! Playing edge lift...')
