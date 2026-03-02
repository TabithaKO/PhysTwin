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

# Parse spring_Y override
spring_y_override = None
if len(sys.argv) > 4:
    spring_y_override = float(sys.argv[4])

if spring_y_override is not None:
    n_springs = ckpt['spring_Y'].shape[0]
    fake_Y = torch.full((n_springs,), spring_y_override, device='cuda')
    sim.set_spring_Y(torch.log(fake_Y).detach().clone())
    print(f'Spring Y override: {spring_y_override}')
else:
    sim.set_spring_Y(torch.log(ckpt['spring_Y']).detach().clone())
    print(f'Using learned spring Y (median ~75k)')

sim.set_collide(ckpt['collide_elas'].detach().clone(), ckpt['collide_fric'].detach().clone())
sim.set_collide_object(ckpt['collide_object_elas'].detach().clone(), ckpt['collide_object_fric'].detach().clone())

K0 = np.array(d['intrinsics'])[0]
c2w0 = np.array(c2ws[0])
w2c0 = np.linalg.inv(c2w0)
n_obj = sim.num_object_points

# Settle phase
sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)
ctrl_init = sim.controller_points[0].clone()
for _ in range(50):
    sim.set_controller_interactive(ctrl_init, ctrl_init)
    if sim.object_collision_flag: sim.update_collision_graph()
    wp.capture_launch(sim.forward_graph)
    sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v, pure_inference=True)

# Zero velocity after settle
sim.set_init_state(sim.wp_states[-1].wp_x, wp.zeros(sim.wp_states[-1].wp_v.shape[0], dtype=wp.vec3, device='cuda'), pure_inference=True)

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

def project(pts, K, w2c):
    p = pts.copy(); p[:,2] *= -1
    ph = np.hstack([p, np.ones((len(p),1))])
    pc = (w2c @ ph.T).T[:,:3]
    v = pc[:,2] > 0.01
    px = (K @ pc.T).T
    uv = px[:,:2] / (px[:,2:3]+1e-8)
    return uv.astype(int), v

# Parse args
rd = [float(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [0,0,0.001]
ld = [float(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else rd.copy()
num_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 80

right_dir = torch.tensor(rd, dtype=torch.float32, device='cuda')
left_dir = torch.tensor(ld, dtype=torch.float32, device='cuda')
sy_label = f'Y={int(spring_y_override)}' if spring_y_override else 'Y=learned'
print(f'R dir: {rd}, L dir: {ld}, steps: {num_steps}, {sy_label}')

current_target = ctrl_init.clone()
prev_target = current_target.clone()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('experiments/trajectory_01/test_lift.mp4', fourcc, 30, (PW, PH))

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
    
    z = cloth[:,2]; real = z[z != 0]
    if len(real)>10:
        zn,zx = real.min(),real.max()
        spread = zx - zn
        if spread < 0.01: zn,zx = zn-0.05, zx+0.05
        t = np.clip((z - zn)/(zx - zn), 0, 1)
        hc = np.zeros((len(z),3),dtype=np.uint8)
        hc[:,2] = (t*255).astype(np.uint8)  # blue=low(table), red=high(lifted)
        hc[:,0] = ((1-t)*255).astype(np.uint8)
        hc[:,1] = 30
    else:
        hc = np.full((len(z),3),150,dtype=np.uint8)
    
    panel = np.full((PH,PW,3),20,dtype=np.uint8)
    uv,valid = project(cloth, K0, w2c0)
    for k in range(len(uv)):
        if valid[k]:
            px,py = uv[k,0]*PW//1280, uv[k,1]*PH//720
            if 0<=px<PW and 0<=py<PH:
                cv2.circle(panel,(px,py),3,hc[k].tolist(),-1)
    
    for gc,color,label in [(rc,(0,0,255),'R'),(lc,(0,255,0),'L')]:
        uv_g,v_g = project(gc.reshape(1,3).astype(np.float32), K0, w2c0)
        if v_g[0]:
            gx,gy = uv_g[0,0]*PW//1280, uv_g[0,1]*PH//720
            if 0<=gx<PW and 0<=gy<PH:
                cv2.circle(panel,(gx,gy),12,color,-1)
                cv2.putText(panel,label,(gx-5,gy+5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
    
    cv2.putText(panel,f'Step {st+1}/{num_steps} {sy_label} spread:{spread*100:.1f}cm' if 'spread' in dir() else f'Step {st+1}',
                (10,25),cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,200),1)
    out.write(panel)

out.release()
print(f'Saved: experiments/trajectory_01/test_lift.mp4')
# Print final stats
x_final = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False)[:n_obj].detach().cpu().numpy()
zf = x_final[:,2]; rf = zf[zf!=0]
if len(rf)>0:
    print(f'Final Z range: {rf.min():.4f} to {rf.max():.4f} = spread {(rf.max()-rf.min())*100:.1f}cm')
