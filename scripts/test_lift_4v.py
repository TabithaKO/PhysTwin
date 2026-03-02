import pickle, json, glob, torch, numpy as np, warp as wp, cv2, os, sys, warnings
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

trainer = InvPhyTrainerWarp(
    data_path='data/so101_cloth/trajectory_01/final_data.pkl',
    base_dir='./temp_experiments/lift_test',
    pure_inference_mode=True,
)
sim = trainer.simulator
ckpt = torch.load(glob.glob('experiments/trajectory_01/train/best_*.pth')[0])

spring_y_override = float(sys.argv[4]) if len(sys.argv) > 4 else None
flat_init = len(sys.argv) > 5 and sys.argv[5] == 'flat' 
flat_init = len(sys.argv) > 5 and sys.argv[5] == 'flat' 
if spring_y_override is not None:
    n_springs = ckpt['spring_Y'].shape[0]
    fake_Y = torch.full((n_springs,), spring_y_override, device='cuda')
    sim.set_spring_Y(torch.log(fake_Y).detach().clone())
    print(f'Spring Y override: {spring_y_override}')
else:
    sim.set_spring_Y(torch.log(ckpt['spring_Y']).detach().clone())
    print(f'Using learned spring Y')

sim.set_collide(ckpt['collide_elas'].detach().clone(), ckpt['collide_fric'].detach().clone())
sim.set_collide_object(ckpt['collide_object_elas'].detach().clone(), ckpt['collide_object_fric'].detach().clone())

K0 = np.array(d['intrinsics'])[0]
c2w0 = np.array(c2ws[0])
w2c0 = np.linalg.inv(c2w0)
n_obj = sim.num_object_points

# Optionally flatten cloth to table
ctrl_init = sim.controller_points[0].clone()
if flat_init:
    init_x = wp.to_torch(sim.wp_init_vertices, requires_grad=False).clone()
    n_pts = init_x.shape[0]
    coords = init_x.reshape(n_pts, 3) if init_x.dim()==1 else init_x
    # Find table Z (most positive Z = closest to table in this convention)
    table_z = coords[:n_obj, 2].max().item()
    print(f"Flattening cloth to table Z={table_z:.4f}")
    coords[:n_obj, 2] = table_z
    flat_verts = wp.from_torch(coords.contiguous(), dtype=wp.vec3)
    sim.set_init_state(flat_verts, sim.wp_init_velocities)
    # Also flatten controller points to same Z
    ctrl_init[:, 2] = table_z
else:
    sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)
settle_steps = 5 if flat_init else (200 if spring_y_override and spring_y_override < 10000 else 50)
print(f'Settling {settle_steps} steps...')
for _ in range(settle_steps):
    sim.set_controller_interactive(ctrl_init, ctrl_init)
    if sim.object_collision_flag: sim.update_collision_graph()
    wp.capture_launch(sim.forward_graph)
    sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v, pure_inference=True)
# Zero velocity and do extra hold-still steps to fully stabilize
sim.set_init_state(sim.wp_states[-1].wp_x, wp.zeros(sim.wp_states[-1].wp_v.shape[0], dtype=wp.vec3, device='cuda'), pure_inference=True)
stab_steps = 10 if flat_init else 200
print(f'Stabilizing ({stab_steps} hold-still steps with velocity damping)...')
for _ in range(stab_steps):
    sim.set_controller_interactive(ctrl_init, ctrl_init)
    if sim.object_collision_flag: sim.update_collision_graph()
    wp.capture_launch(sim.forward_graph)
    # Dampen velocity each step
    v_torch = wp.to_torch(sim.wp_states[-1].wp_v, requires_grad=False)
    v_torch *= 0.1  # near-total damping
    damped_v = wp.from_torch(v_torch.contiguous(), dtype=wp.vec3)
    sim.set_init_state(sim.wp_states[-1].wp_x, damped_v, pure_inference=True)
# Final zero velocity
sim.set_init_state(sim.wp_states[-1].wp_x, wp.zeros(sim.wp_states[-1].wp_v.shape[0], dtype=wp.vec3, device='cuda'), pure_inference=True)

x0 = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False)[:n_obj].detach().cpu().numpy()
z0 = x0[:,2]; r0 = z0[z0!=0]
print(f'After settle Z: {r0.min():.4f} to {r0.max():.4f} = {(r0.max()-r0.min())*100:.1f}cm spread')

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

PW, PH = 640, 480

def make_virtual_cam(eye, target, up, K):
    """Create w2c from eye position, target, up vector"""
    fwd = target - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    u = np.cross(right, fwd)
    R = np.array([right, -u, fwd])  # OpenCV convention
    t = -R @ eye
    w2c = np.eye(4)
    w2c[:3,:3] = R; w2c[:3,3] = t
    return w2c

# Camera setups: front (cam0), side, top-down, angled
cloth_center = x0.mean(axis=0)
# Flip Z for camera world (PhysTwin Z is negated for camera)
cc = cloth_center.copy(); cc[2] *= -1

# Side camera (from +X looking at cloth)
side_w2c = make_virtual_cam(
    eye=cc + np.array([0.5, 0, 0]),
    target=cc, up=np.array([0, 0, 1]), K=K0)

# Top-down camera (from above)
top_w2c = make_virtual_cam(
    eye=cc + np.array([0, 0, 0.5]),
    target=cc, up=np.array([0, 1, 0]), K=K0)

# Angled camera (45 deg from front-right)
ang_w2c = make_virtual_cam(
    eye=cc + np.array([0.3, -0.3, 0.3]),
    target=cc, up=np.array([0, 0, 1]), K=K0)

cameras = [
    ('Front', w2c0, K0),
    ('Side', side_w2c, K0),
    ('Top', top_w2c, K0),
    ('Angled', ang_w2c, K0),
]

def project(pts, K, w2c):
    p = pts.copy(); p[:,2] *= -1
    ph = np.hstack([p, np.ones((len(p),1))])
    pc = (w2c @ ph.T).T[:,:3]
    v = pc[:,2] > 0.01
    px = (K @ pc.T).T
    uv = px[:,:2] / (px[:,2:3]+1e-8)
    return uv.astype(int), v

rd = [float(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [0,0,0.001]
ld = [float(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0,0,0]
num_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 80

right_dir = torch.tensor(rd, dtype=torch.float32, device='cuda')
left_dir = torch.tensor(ld, dtype=torch.float32, device='cuda')
sy_label = f'Y={int(spring_y_override)}' if spring_y_override else 'Y=learned'
print(f'R: {rd}, L: {ld}, steps: {num_steps}, {sy_label}')

current_target = ctrl_init.clone()
prev_target = current_target.clone()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
FULL_W, FULL_H = PW*2, PH*2
out = cv2.VideoWriter('experiments/trajectory_01/test_lift.mp4', fourcc, 30, (FULL_W, FULL_H))

for st in range(num_steps):
    prev_target = current_target.clone()
    current_target[mask1] += right_dir
    current_target[mask0] += left_dir
    sim.set_controller_interactive(prev_target, current_target)
    if sim.object_collision_flag: sim.update_collision_graph()
    wp.capture_launch(sim.forward_graph)
    sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v, pure_inference=True)
    
    x = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False)
    cloth = x[:n_obj].detach().cpu().numpy()
    rc = current_target[mask1].mean(dim=0).detach().cpu().numpy()
    lc = current_target[mask0].mean(dim=0).detach().cpu().numpy()
    
    z = cloth[:,2]; real = z[z!=0]
    spread = 0
    if len(real)>10:
        zn,zx = real.min(),real.max(); spread = zx-zn
        if spread<0.01: zn,zx=zn-0.05,zx+0.05
        t = np.clip((z-zn)/(zx-zn),0,1)
        hc = np.zeros((len(z),3),dtype=np.uint8)
        hc[:,2]=(t*255).astype(np.uint8); hc[:,0]=((1-t)*255).astype(np.uint8); hc[:,1]=30
    else:
        hc = np.full((len(z),3),150,dtype=np.uint8)
    
    full = np.full((FULL_H,FULL_W,3),20,dtype=np.uint8)
    
    for ci,(cname,cw2c,cK) in enumerate(cameras):
        row, col = ci//2, ci%2
        panel = np.full((PH,PW,3),30,dtype=np.uint8)
        
        uv,valid = project(cloth, cK, cw2c)
        for k in range(len(uv)):
            if valid[k]:
                px,py = uv[k,0]*PW//1280, uv[k,1]*PH//720
                if 0<=px<PW and 0<=py<PH:
                    cv2.circle(panel,(px,py),3,hc[k].tolist(),-1)
        
        for gc,color,label in [(rc,(0,0,255),'R'),(lc,(0,255,0),'L')]:
            uv_g,v_g = project(gc.reshape(1,3).astype(np.float32), cK, cw2c)
            if v_g[0]:
                gx,gy = uv_g[0,0]*PW//1280, uv_g[0,1]*PH//720
                if 0<=gx<PW and 0<=gy<PH:
                    cv2.circle(panel,(gx,gy),10,color,-1)
                    cv2.putText(panel,label,(gx-5,gy+5),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,255,255),1)
        
        cv2.putText(panel,cname,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
        cv2.putText(panel,f'{st+1}/{num_steps} {sy_label} spr:{spread*100:.1f}cm',
                    (10,PH-10),cv2.FONT_HERSHEY_SIMPLEX,0.35,(150,150,150),1)
        
        full[row*PH:(row+1)*PH, col*PW:(col+1)*PW] = panel
    
    out.write(full)

out.release()
zf = cloth[:,2]; rf = zf[zf!=0]
rf = zf[zf!=0]
print(f'Final Z: {rf.min():.4f} to {rf.max():.4f} = {(rf.max()-rf.min())*100:.1f}cm spread')
print(f'Saved: experiments/trajectory_01/test_lift.mp4')
