
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import scipy.sparse as sp
import scipy.sparse.linalg as spla

st.set_page_config(page_title="Via Topology (Simple GUI)", layout="wide")
st.title("Via Topology Designer — Simple GUI")

st.sidebar.header("Geometry")
Lx = st.sidebar.number_input("Block X (µm)", 20.0, 200.0, 50.0, 1.0)
Ly = st.sidebar.number_input("Block Y (µm)", 20.0, 200.0, 50.0, 1.0)
Lz = st.sidebar.number_input("Thickness Z (µm)", 10.0, 400.0, 100.0, 1.0)
via_diam = st.sidebar.number_input("Via diameter (µm)", 0.2, 10.0, 1.0, 0.1)
via_radius = via_diam/2
pitch = st.sidebar.number_input("Via pitch (µm)", 2.0, 50.0, 10.0, 0.5)

st.sidebar.header("Mesh")
nx = st.sidebar.slider("Grid X", 10, 60, 28, 1)
ny = st.sidebar.slider("Grid Y", 10, 60, 28, 1)
nz = st.sidebar.slider("Grid Z", 8, 40, 18, 1)

st.sidebar.header("Material (host glass)")
E_GPa = st.sidebar.number_input("Young’s modulus E (GPa)", 10.0, 200.0, 64.0, 1.0)
nu = st.sidebar.number_input("Poisson’s ratio ν", 0.05, 0.49, 0.20, 0.01, format="%.2f")
eps_glass = st.sidebar.number_input("Relative permittivity εr", 2.0, 12.0, 3.9, 0.1)
tan_delta_glass = st.sidebar.number_input("Loss tangent tanδ", 0.0, 0.1, 0.002, 0.0001, format="%.4f")

st.sidebar.header("Loading")
pressure = st.sidebar.number_input("Uniform pressure on top (MPa)", 0.01, 10.0, 0.5, 0.01)

st.sidebar.header("Dielectric mixing")
window_r = st.sidebar.slider("Local window radius (µm)", 0.5, 5.0, 2.0, 0.5)

air_eps = 1.0006
air_tand = 0.0

def node_id(i,j,k,ny,nz):
    return (i*ny + j)*nz + k

def build_grid(Lx,Ly,Lz,nx,ny,nz):
    xs = np.linspace(0, Lx, nx); ys = np.linspace(0, Ly, ny); zs = np.linspace(0, Lz, nz)
    dx,dy,dz = xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]
    X,Y,Z = np.meshgrid(xs,ys,zs, indexing='ij')
    return xs,ys,zs,dx,dy,dz,X,Y,Z

def run_model():
    xs,ys,zs,dx,dy,dz,X,Y,Z = build_grid(Lx,Ly,Lz,nx,ny,nz)

    # Via centres (square)
    cx = np.arange(pitch/2, Lx, pitch)
    cy = np.arange(pitch/2, Ly, pitch)
    centres = np.array([(a,b) for a in cx for b in cy])

    # Hole mask
    Rmin = np.full(X.shape, np.inf)
    for (a,b) in centres:
        R = np.sqrt((X-a)**2 + (Y-b)**2)
        Rmin = np.minimum(Rmin, R)
    hole = Rmin <= via_radius
    active = ~hole

    num_nodes = nx*ny*nz
    id_map = -np.ones(num_nodes, dtype=int)
    active_ids = np.where(active.ravel())[0]
    for new, old in enumerate(active_ids):
        id_map[old] = new
    ndof = len(active_ids)*3
    if ndof == 0:
        st.error("Mesh eliminated by via size; reduce via diameter or increase pitch.")
        return None, None

    # Stiffness assembly (use extend, not += to avoid local rebinding problems)
    spring_scale = E_GPa*1e3  # MPa
    kx = spring_scale / (dx+1e-12)
    ky = spring_scale / (dy+1e-12)
    kz = spring_scale / (dz+1e-12)
    kf_xy = spring_scale / np.sqrt(dx*dx+dy*dy) * 0.5
    kf_xz = spring_scale / np.sqrt(dx*dx+dz*dz) * 0.5
    kf_yz = spring_scale / np.sqrt(dy*dy+dz*dz) * 0.5
    kb = spring_scale / np.sqrt(dx*dx+dy*dy+dz*dz) * 0.25

    rows, cols, data = [], [], []

    def add_spring(n1,n2,k):
        i1 = id_map[n1]; i2 = id_map[n2]
        if i1 < 0 or i2 < 0: return
        i1v = [i1*3+0, i1*3+1, i1*3+2]
        i2v = [i2*3+0, i2*3+1, i2*3+2]
        p1 = np.array([X.ravel()[n1], Y.ravel()[n1], Z.ravel()[n1]])
        p2 = np.array([X.ravel()[n2], Y.ravel()[n2], Z.ravel()[n2]])
        d = p2-p1; L = np.linalg.norm(d)
        if L == 0: return
        n = d / L
        Kloc = k * np.outer(n,n)
        for a in range(3):
            for b in range(3):
                rows.extend([i1v[a], i1v[a], i2v[a], i2v[a]])
                cols.extend([i1v[b], i2v[b], i1v[b], i2v[b]])
                data.extend([ Kloc[a,b], -Kloc[a,b], -Kloc[a,b], Kloc[a,b] ])

    def N(i,j,k): return node_id(i,j,k,ny,nz)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n = N(i,j,k)
                if id_map[n] < 0: continue
                if i+1 < nx: add_spring(n, N(i+1,j,k), kx)
                if j+1 < ny: add_spring(n, N(i,j+1,k), ky)
                if k+1 < nz: add_spring(n, N(i,j,k+1), kz)
                if i+1 < nx and j+1 < ny:
                    add_spring(n, N(i+1,j+1,k), kf_xy)
                    if j-1 >= 0: add_spring(n, N(i+1,j-1,k), kf_xy)
                if i+1 < nx and k+1 < nz:
                    add_spring(n, N(i+1,j,k+1), kf_xz)
                    if k-1 >= 0: add_spring(n, N(i+1,j,k-1), kf_xz)
                if j+1 < ny and k+1 < nz:
                    add_spring(n, N(i,j+1,k+1), kf_yz)
                    if k-1 >= 0: add_spring(n, N(i,j+1,k-1), kf_yz)
                if i+1 < nx and j+1 < ny and k+1 < nz: add_spring(n, N(i+1,j+1,k+1), kb)
                if i+1 < nx and j+1 < ny and k-1 >= 0: add_spring(n, N(i+1,j+1,k-1), kb)
                if i+1 < nx and j-1 >= 0 and k+1 < nz: add_spring(n, N(i+1,j-1,k+1), kb)
                if i+1 < nx and j-1 >= 0 and k-1 >= 0: add_spring(n, N(i+1,j-1,k-1), kb)

    K = sp.coo_matrix((data,(rows,cols)), shape=(ndof,ndof)).tocsr()

    # BCs and load
    fixed = []
    for i in range(nx):
        for j in range(ny):
            n = N(i,j,0)
            cid = id_map[n]
            if cid >= 0: fixed.extend([cid*3+0, cid*3+1, cid*3+2])
    fixed = np.array(sorted(set(fixed)), dtype=int)

    F = np.zeros(ndof)
    ax_area = (Lx/(nx-1))*(Ly/(ny-1))
    for i in range(nx):
        for j in range(ny):
            n = N(i,j,nz-1)
            cid = id_map[n]
            if cid >= 0: F[cid*3+2] += pressure * ax_area

    free = np.setdiff1d(np.arange(ndof), fixed)
    if free.size == 0:
        st.error("No free DOFs — check constraints/mesh.")
        return None, None

    uf = spla.spsolve(K[free][:,free], F[free])
    U = np.zeros(ndof); U[free] = uf

    # Collect fields
    coords = np.column_stack((
        np.linspace(0,Lx,nx).repeat(ny*nz)[active.ravel()],
        np.tile(np.linspace(0,Ly,ny).repeat(nz), nx)[active.ravel()],
        np.tile(np.linspace(0,Lz,nz), nx*ny)[active.ravel()]
    ))
    Ux = U[0::3]; Uy = U[1::3]; Uz = U[2::3]
    u_mag = np.sqrt(Ux**2 + Uy**2 + Uz**2)

    # Energy density
    springs = []
    def rec(n1,n2,k):
        i1,i2 = id_map[n1], id_map[n2]
        if i1>=0 and i2>=0: springs.append((i1,i2,k))

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n = N(i,j,k)
                if id_map[n] < 0: continue
                if i+1 < nx: rec(n, N(i+1,j,k), kx)
                if j+1 < ny: rec(n, N(i,j+1,k), ky)
                if k+1 < nz: rec(n, N(i,j,k+1), kz)
                if i+1 < nx and j+1 < ny:
                    rec(n, N(i+1,j+1,k), kf_xy)
                    if j-1>=0: rec(n, N(i+1,j-1,k), kf_xy)
                if i+1 < nx and k+1 < nz:
                    rec(n, N(i+1,j,k+1), kf_xz)
                    if k-1>=0: rec(n, N(i+1,j,k-1), kf_xz)
                if j+1 < ny and k+1 < nz:
                    rec(n, N(i,j+1,k+1), kf_yz)
                    if k-1>=0: rec(n, N(i,j+1,k-1), kf_yz)
                if i+1 < nx and j+1 < ny and k+1 < nz: rec(n, N(i+1,j+1,k+1), kb)
                if i+1 < nx and j+1 < ny and k-1 >= 0: rec(n, N(i+1,j+1,k-1), kb)
                if i+1 < nx and j-1 >= 0 and k+1 < nz: rec(n, N(i+1,j-1,k+1), kb)
                if i+1 < nx and j-1 >= 0 and k-1 >= 0: rec(n, N(i+1,j-1,k-1), kb)

    Vnode = (Lx/(nx-1))*(Ly/(ny-1))*(Lz/(nz-1))
    Uvec = np.column_stack((Ux,Uy,Uz))
    def spring_energy(i1,i2,k):
        p1 = coords[i1]; p2 = coords[i2]; d = p2-p1; L = np.linalg.norm(d)
        if L==0: return 0.0
        n = d/L; du = Uvec[i2]-Uvec[i1]; ext = np.dot(du,n)
        return 0.5*k*ext**2
    nodal_E = np.zeros(len(active_ids))
    for (a,b,kst) in springs:
        e = spring_energy(a,b,kst)
        nodal_E[a] += 0.5*e; nodal_E[b] += 0.5*e
    energy_density = nodal_E / Vnode

    # Macro strain & E proxy
    top_ids = []
    for i in range(nx):
        for j in range(ny):
            n = N(i,j,nz-1); cid = id_map[n]
            if cid>=0: top_ids.append(cid)
    top_ids = np.array(top_ids, dtype=int)
    avg_uz_top = np.mean(Uz[top_ids]) if top_ids.size>0 else 0.0
    macro_strain = avg_uz_top / (Lz if Lz>0 else 1.0)
    E_eff_proxy = 2.0 * energy_density / (macro_strain**2 + 1e-30)

    # Local void fraction estimate in XY window
    def f_local(idx):
        g = active_ids[idx]
        k = g % nz; j = (g//nz) % ny; i = g//(ny*nz)
        rx = max(1, int(np.ceil(window_r/(Lx/(nx-1)))))
        ry = max(1, int(np.ceil(window_r/(Ly/(ny-1)))))
        i0=max(0,i-rx); i1=min(nx-1,i+rx)
        j0=max(0,j-ry); j1=min(ny-1,j+ry)
        total=0; solid=0
        for ii in range(i0,i1+1):
            for jj in range(j0,j1+1):
                total += 1
                if not hole[ii,jj,k]: solid += 1
        return 1.0 - (solid/total if total>0 else 1.0)
    f = np.array([f_local(idx) for idx in range(len(active_ids))])
    f = np.clip(f, 0.0, 0.6)

    # Dielectric mixing (complex MG)
    eps_m = eps_glass*(1 - 1j*tan_delta_glass)
    eps_i = air_eps*(1 - 1j*air_tand)
    num = (eps_i + 2*eps_m) + 2*f*(eps_i - eps_m)
    den = (eps_i + 2*eps_m) - f*(eps_i - eps_m)
    eps_eff_c = eps_m * (num/den)
    eps_eff_real = np.real(eps_eff_c)
    tan_delta_eff = np.abs(np.imag(eps_eff_c)/np.maximum(eps_eff_real,1e-12))

    # Distance to nearest via wall
    Rflat = np.full(len(active_ids), np.inf)
    xy = coords[:,:2]
    for (a,b) in centres:
        Rflat = np.minimum(Rflat, np.sqrt((xy[:,0]-a)**2 + (xy[:,1]-b)**2))
    dist_to_wall = Rflat - via_radius

    # Scale proxy to E_local
    bulk_mask = dist_to_wall >= 3.0
    bulk_mean = np.mean(E_eff_proxy[bulk_mask]) if np.any(bulk_mask) else np.mean(E_eff_proxy)
    E_local_GPa = E_GPa * (E_eff_proxy / (bulk_mean + 1e-30))

    df = pd.DataFrame({
        "x_um": coords[:,0], "y_um": coords[:,1], "z_um": coords[:,2],
        "u_mag": u_mag, "E_eff_proxy": E_eff_proxy, "E_local_GPa": E_local_GPa,
        "eps_eff_real": eps_eff_real, "tan_delta_eff": tan_delta_eff,
        "f_void_local": f, "dist_to_wall": dist_to_wall
    })
    return df, dict(nx=nx,ny=ny,nz=nz,Lx=Lx,Ly=Ly,Lz=Lz)

if st.button("Run / Update model", type="primary"):
    with st.spinner("Solving..."):
        out = run_model()
    if out[0] is None:
        st.stop()
    df, meta = out
    st.success("Done.")
    st.download_button("Download dataset (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="via_topology_dataset.csv", mime="text/csv")

    # Slices
    def nearest(vals, target): 
        vals = np.array(vals); i = np.argmin(np.abs(vals-target)); return vals[i]
    z_mid = nearest(df["z_um"].values, Lz/2)
    y_mid = nearest(df["y_um"].values, Ly/2)
    sl_planar = df[np.isclose(df["z_um"], z_mid)]
    sl_side   = df[np.isclose(df["y_um"], y_mid)]

    c1,c2 = st.columns(2)
    with c1:
        st.subheader(f"Planar tanδ (z ≈ {z_mid:.1f} µm)")
        fig,ax = plt.subplots()
        p = ax.scatter(sl_planar["x_um"], sl_planar["y_um"], c=sl_planar["tan_delta_eff"], s=16)
        ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
        cb = fig.colorbar(p, ax=ax); cb.set_label("tanδ")
        st.pyplot(fig)
    with c2:
        st.subheader(f"Planar Young’s modulus (z ≈ {z_mid:.1f} µm)")
        fig,ax = plt.subplots()
        p = ax.scatter(sl_planar["x_um"], sl_planar["y_um"], c=sl_planar["E_local_GPa"], s=16)
        ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
        cb = fig.colorbar(p, ax=ax); cb.set_label("E_local (GPa)")
        st.pyplot(fig)

    c3,c4 = st.columns(2)
    with c3:
        st.subheader(f"Side tanδ (y ≈ {y_mid:.1f} µm)")
        fig,ax = plt.subplots()
        p = ax.scatter(sl_side["x_um"], sl_side["z_um"], c=sl_side["tan_delta_eff"], s=16)
        ax.set_xlabel("x (µm)"); ax.set_ylabel("z (µm)")
        cb = fig.colorbar(p, ax=ax); cb.set_label("tanδ")
        st.pyplot(fig)
    with c4:
        st.subheader(f"Side Young’s modulus (y ≈ {y_mid:.1f} µm)")
        fig,ax = plt.subplots()
        p = ax.scatter(sl_side["x_um"], sl_side["z_um"], c=sl_side["E_local_GPa"], s=16)
        ax.set_xlabel("x (µm)"); ax.set_ylabel("z (µm)")
        cb = fig.colorbar(p, ax=ax); cb.set_label("E_local (GPa)")
        st.pyplot(fig)

    st.subheader("3D maps")
    c5,c6 = st.columns(2)
    sample = np.linspace(0, len(df)-1, min(15000, len(df))).astype(int)
    with c5:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        s = ax.scatter(df["x_um"].values[sample], df["y_um"].values[sample], df["z_um"].values[sample],
                       c=df["tan_delta_eff"].values[sample], s=5)
        ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)"); ax.set_zlabel("z (µm)")
        ax.set_title("3D tanδ")
        cb = fig.colorbar(s, ax=ax, shrink=0.7); cb.set_label("tanδ")
        st.pyplot(fig)
    with c6:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        s = ax.scatter(df["x_um"].values[sample], df["y_um"].values[sample], df["z_um"].values[sample],
                       c=df["E_local_GPa"].values[sample], s=5)
        ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)"); ax.set_zlabel("z (µm)")
        ax.set_title("3D Young’s modulus")
        cb = fig.colorbar(s, ax=ax, shrink=0.7); cb.set_label("E_local (GPa)")
        st.pyplot(fig)

else:
    st.info("Set parameters and press **Run / Update model**.")
