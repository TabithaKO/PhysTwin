import pickle, json, glob, torch, numpy as np, warp as wp, cv2, sys, warnings, copy
warnings.filterwarnings('ignore')
from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg

# Args: direction steps spring_Y side
# e.g.: python test_single.py 0,0,-0.0003 300 100 right
rd = [float(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [0,0,-0.0003]
num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 300
sy = float(sys.argv[3]) if len(sys.argv) > 3 else 1000
side = sys.argv[4] if len(sys.argv) > 4 else 'right'

# Create single-gripper data if needed
orig_pkl = 'data/so101_cloth/trajectory_01/final_data.pkl'
side_pkl = orig_pkl.replace('final_data.pkl', f'final_data_{side}_only.pkl')

import os
if not os.path.exists(side_pkl):
    from sklearn.cluster import KMeans
    with open(orig_pkl, 'rb') as f:
        data = pickle.load(f)
    ctrl_pts = data['controller_points']
    pts0 = ctrl_pts[0]
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    labels = kmeans.fit_predict(pts0)
    c0 = pts0[labels==0].mean(axis=0)
    c1 = pts0[labels==1].mean(axis=0)
    if c0[0] > c1[0]:
        left_mask, right_mask = labels==1, labels==0
    else:
        left_mask, right_mask = labels==0, labels==1
    mask = right_mask if side == 'right' else left_mask
    new_data = copy.deepcopy(data)
    new_data['controller_points'] = ctrl_pts[:, mask, :]
    with open(side_pkl, 'wb') as f:
        pickle.dump(new_data, f)
    print(f'Created {side_pkl}')

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
    data_path=side_pkl,
    base_dir=f'./temp_experiments/test_single_{side}',
    pure_inference_mode=True,
)
sim = trainer.simulator
ckpt = torch.load(glob.glob('experiments/trajectory_01/train/best_*.pth')[0])

n_springs = ckpt['spring_Y'].shape[0]
fake_Y = torch.full((n_springs,), sy, device='cuda')
sim.set_spring_Y(torch.log(fake_Y).detach().clone())
sim.set_collide(ckpt['collide_elas'].detach().clone(), ckpt['collide_fric'].detach().clone())
sim.set_collide_object(ckpt['collide_object_elas'].detach().clone(), ckpt['collide_object_fric'].detach().clone())

K0 = np.array(d['intrinsics'])[0]
c2w0 = np.array(c2ws[0])
w2c0 = np.linalg.inv(c2w0)
n_obj = sim.num_object_points
ctrl_init = sim.controller_points[0].clone()
n_ctrl = ctrl_init.shape[0]
print(f'{side} gripper: {n_ctrl} ctrl pts, Y={sy}, dir={rd}, steps={num_steps}')

# Settle
sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)
print('Settling 200 steps...')
for _ in range(200):
    sim.set_controller_interactive(ctrl_init, ctrl_init)
    if sim.object_collision_flag: sim.update_collision_graph()
    wp.capture_launch(sim.forward_graph)
    sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v, pure_inference=True)

# Stabilize
sim.set_init_state(sim.wp_states[-1].wp_x, wp.zeros(sim.wp_states[-1].wp_v.shape[0], dtype=wp.vec3, device='cuda'), pure_inference=True)
print('Stabilizing 200 steps...')
for _ in range(200):
    sim.set_controller_interactive(ctrl_init, ctrl_init)
    if sim.object_collision_flag: sim.update_collision_graph()
    wp.capture_launch(sim.forward_graph)
    v_torch = wp.to_torch(sim.wp_states[-1].wp_v, requires_grad=False)
    v_torch *= 0.1
    damped_v = wp.from_torch(v_torch.contiguous(), dtype=wp.vec3)
    sim.set_init_state(sim.wp_states[-1].wp_x, damped_v, pure_inference=True)
sim.set_init_state(sim.wp_states[-1].wp_x, wp.zeros(sim.wp_states[-1].wp_v.shape[0], dtype=wp.vec3, device='cuda'), pure_inference=True)

x0 = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False)[:n_obj].detach().cpu().numpy()
z0 = x0[:,2]; r0 = z0[z0!=0]
print(f'After settle Z: {r0.min():.4f} to {r0.max():.4f} = {(r0.max()-r0.min())*100:.1f}cm')

# Virtual cameras
cc = x0.mean(axis=0).copy(); cc[2] *= -1
def make_cam(eye, target, up):
    fwd = target - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    u = np.cross(right, fwd)
    R = np.array([right, -u, fwd])
    t = -R @ eye
    w2c = np.eye(4); w2c[:3,:3] = R; w2c[:3,3] = t
    return w2c

PW, PH = 640, 480
cameras = [
    ('Front', w2c0, K0),
    ('Side', make_cam(cc+np.array([0.5,0,0]), cc, np.array([0,0,1])), K0),
    ('Top', make_cam(cc+np.array([0,0,0.5]), cc, np.array([0,1,0])), K0),
    ('Angled', make_cam(cc+np.array([0.3,-0.3,0.3]), cc, np.array([0,0,1])), K0),
]

def project(pts, K, w2c):
    p = pts.copy(); p[:,2] *= -1
    ph = np.hstack([p, np.ones((len(p),1))])
    pc = (w2c @ ph.T).T[:,:3]
    v = pc[:,2] > 0.01
    px = (K @ pc.T).T
    uv = px[:,:2] / (px[:,2:3]+1e-8)
    return uv.astype(int), v

move_dir = torch.tensor(rd, dtype=torch.float32, device='cuda')
current_target = ctrl_init.clone()
grip_color = (0,0,255) if side == 'right' else (0,255,0)
grip_label = 'R' if side == 'right' else 'L'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
FULL_W, FULL_H = PW*2, PH*2
out = cv2.VideoWriter('experiments/trajectory_01/test_single.mp4', fourcc, 30, (FULL_W, FULL_H))

for st in range(num_steps):
    prev_target = current_target.clone()
    current_target += move_dir
    sim.set_controller_interactive(prev_target, current_target)
    if sim.object_collision_flag: sim.update_collision_graph()
    wp.capture_launch(sim.forward_graph)
    sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v, pure_inference=True)
    
    x = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False)
    if torch.isnan(x).any():
        print(f'NaN at step {st+1}')
        break
    cloth = x[:n_obj].detach().cpu().numpy()
    gc = current_target.mean(dim=0).detach().cpu().numpy()
    
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
        uv_g,v_g = project(gc.reshape(1,3).astype(np.float32), cK, cw2c)
        if v_g[0]:
            gx,gy = uv_g[0,0]*PW//1280, uv_g[0,1]*PH//720
            if 0<=gx<PW and 0<=gy<PH:
                cv2.circle(panel,(gx,gy),10,grip_color,-1)
                cv2.putText(panel,grip_label,(gx-5,gy+5),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,255,255),1)
        cv2.putText(panel,cname,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
        cv2.putText(panel,f'{st+1}/{num_steps} Y={int(sy)} spr:{spread*100:.1f}cm',
                    (10,PH-10),cv2.FONT_HERSHEY_SIMPLEX,0.35,(150,150,150),1)
        full[row*PH:(row+1)*PH, col*PW:(col+1)*PW] = panel
    out.write(full)

out.release()
zf = cloth[:,2]; rf = zf[zf!=0]
print(f'Final Z: {rf.min():.4f} to {rf.max():.4f} = {(rf.max()-rf.min())*100:.1f}cm')
print(f'Saved: experiments/trajectory_01/test_single.mp4')
