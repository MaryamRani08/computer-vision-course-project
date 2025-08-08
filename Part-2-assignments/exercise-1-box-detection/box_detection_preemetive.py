import time
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import label
import cv2

def visualize_image(img, title="Image", cmap=None):
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')

def visualize_point_cloud(ax, pc, sample_rate=10):
    subs = pc[::sample_rate, ::sample_rate, :]
    X, Y, Z = subs[:,:,0].ravel(), subs[:,:,1].ravel(), subs[:,:,2].ravel()
    mask = Z != 0
    ax.scatter(X[mask], Y[mask], Z[mask], s=1, alpha=0.3)

def preemptive_ransac(pc, M=1000, B=100, thres=0.05, use_mlesac=False, gamma=None):

    flat = pc.reshape(-1, 3)
    v_mask = (flat[:,2] != 0)
    valid = flat[v_mask]
    valid_idx = np.nonzero(v_mask)[0]
    N = valid.shape[0]

    # M hypotheses Generation
    normals, ds = [], []
    while len(normals) < M:
        i,j,k = np.random.choice(N,3, replace=False)
        p1,p2,p3 = valid[[i,j,k]]
        n = np.cross(p2-p1, p3-p1)
        norm = np.linalg.norm(n)
        if norm < 1e-6: continue
        n /= norm
        d = n.dot(p1)
        normals.append(n); ds.append(d)
    normals = np.vstack(normals)   
    ds = np.array(ds)             

    # Preemptive Pruning
    alive = np.arange(M)
    order = np.random.permutation(N)
    ptr = 0
    while len(alive) > 1 and ptr < N:
        batch = order[ptr:ptr+B]
        pts = valid[batch]  # (B,3)
        dists = np.abs(pts.dot(normals[alive].T) - ds[alive])
        if use_mlesac:
            γ = thres if gamma is None else gamma
            costs = np.sum(np.where(dists < thres, dists, γ), axis=0)
        else:
            costs = -np.sum(dists < thres, axis=0)

        keep = costs.argsort()[: max(1, len(alive)//2) ]
        alive = alive[keep]
        ptr += B

    best_p = alive[0]
    best_n, best_d = normals[best_p], ds[best_p]

    # Calculate final inliers
    all_d = np.abs(valid.dot(best_n) - best_d)
    inliers = np.nonzero(all_d < thres)[0]

    return best_n, best_d, inliers, valid_idx

def create_clean_mask(H, W, inlier_idxs, valid_idx):
    mask = np.zeros(H*W, np.uint8)
    mask[ valid_idx[inlier_idxs] ] = 1
    mask = mask.reshape((H, W))
    
    # morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    # keep only largest component
    comp, no = label(mask)
    if no > 0:
        sizes = np.bincount(comp.ravel())
        largest = sizes[1:].argmax() + 1
        mask = (comp == largest).astype(np.uint8)

    return mask

def convert_mask_to_points(pc, mask):
    idxs = np.nonzero(mask.ravel())[0]
    pts = pc.reshape(-1,3)[idxs]
    return pts

def measure_box_dimensions(box_pts, box_plane, floor_plane):
    mn, mx = box_pts.min(axis=0), box_pts.max(axis=0)
    width  = mx[0] - mn[0]
    length = mx[1] - mn[1]
    n_box, d_box = box_plane
    _, d_floor = floor_plane
    height = abs(d_box - d_floor) / np.linalg.norm(n_box)
    return width, length, height

def plot_planes(pc, Ms, B=100, thr_floor=0.05, thr_box=0.01):
    H, W, _ = pc.shape
    fig = plt.figure(figsize=(5*len(Ms), 8))

    subs = pc[::5, ::5, :]
    xs, ys, zs = subs[:,:,0].ravel(), subs[:,:,1].ravel(), subs[:,:,2].ravel()
    valid_mask = zs != 0

    all_pts = pc.reshape(-1,3)
    all_pts = all_pts[all_pts[:,2]!=0]
    xmin, xmax = all_pts[:,0].min(), all_pts[:,0].max()
    ymin, ymax = all_pts[:,1].min(), all_pts[:,1].max()
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 10),
                         np.linspace(ymin, ymax, 10))

    for col, M in enumerate(Ms, start=1):
        axf = fig.add_subplot(2, len(Ms), col, projection='3d')
        axf.scatter(xs[valid_mask], ys[valid_mask], zs[valid_mask], s=1, alpha=0.2)
        n_f, d_f, in_f, vidx_f = preemptive_ransac(pc, M=M, B=B, thres=thr_floor)
        zf = (d_f - n_f[0]*xx - n_f[1]*yy) / n_f[2]
        axf.plot_surface(xx, yy, zf, alpha=0.4)
        axf.set_title(f"Floor, M={M}\n inliers={len(in_f)}")
        axf.set_xticks([]); axf.set_yticks([]); axf.set_zticks([])

        m_floor = create_clean_mask(H, W, in_f, vidx_f)
        pts_nonfloor = convert_mask_to_points(pc, m_floor==0)
        fake_pc = pts_nonfloor.reshape(-1,1,3)

        axb = fig.add_subplot(2, len(Ms), len(Ms)+col, projection='3d')
        axb.scatter(xs[valid_mask], ys[valid_mask], zs[valid_mask], s=1, alpha=0.2)

        n_b, d_b, in_b, vidx_b = preemptive_ransac(
            fake_pc, M=M, B=B, thres=thr_box
        )
        zb = (d_b - n_b[0]*xx - n_b[1]*yy) / n_b[2]
        axb.plot_surface(xx, yy, zb, alpha=0.4)
        axb.set_title(f"Box top, M={M}\n inliers={len(in_b)}")
        axb.set_xticks([]); axb.set_yticks([]); axb.set_zticks([])

    plt.tight_layout()
    plt.show()

def main():
    # Data loading
    data = scipy.io.loadmat('example1kinect.mat')
    pc = data['cloud1']
    H, W, _ = pc.shape

    visualize_image(data['amplitudes1'], title="Amplitude")
    visualize_image(data['distances1'], cmap='hot', title="Distance")
    plt.show()

    print("use_mlesac  M    B   floor_inliers   box_w   box_l   box_h   time(s)")
    for use_m in (False, True):
        for M in (500, 1000, 2000):
            for B in (50, 100, 200):
                t0 = time.time()
                n_f, d_f, in_f, vidx_f = preemptive_ransac(
                    pc, M=M, B=B,
                    thres=0.05,
                    use_mlesac=use_m,
                    gamma=0.1
                )
                m_floor = create_clean_mask(H, W, in_f, vidx_f)
                pts_nonfloor = convert_mask_to_points(pc, m_floor==0)
                fake_pc = pts_nonfloor.reshape(-1,1,3)
                n_b, d_b, in_b, vidx_b = preemptive_ransac(
                    fake_pc, M=M, B=B,
                    thres=0.01,
                    use_mlesac=use_m,
                    gamma=0.02
                )
                pts_box = pts_nonfloor[in_b]
                w, l, h = measure_box_dimensions(
                    pts_box,
                    (n_b, d_b),
                    (n_f, d_f)
                )
                t1 = time.time()

                print(f"{str(use_m):>10} {M:4} {B:4} {len(in_f):14} "
                      f"{w:6.3f} {l:6.3f} {h:6.3f} {t1-t0:8.2f}")
    plot_planes(pc, Ms=[500, 1000, 2000], B=100,
                thr_floor=0.05, thr_box=0.01)

if __name__ == "__main__":
    main()
