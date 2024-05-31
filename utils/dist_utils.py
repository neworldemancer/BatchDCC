import numpy as np
from matplotlib.pyplot import cm
from matplotlib import pyplot as plt
from scipy import optimize as opt
import os

xy_fact = 2
n_phase = 8
mtr_inv = np.array([])

# lambda_grad = 1


# find inverted coefs for D
def get_inv_mtr(n):
    mtr = 4 * np.eye(n) + np.eye(n, k=1) + np.eye(n, k=-1)
    mtr[0, n-1] = mtr[n-1, 0] = 1

    mtr_inv = np.linalg.inv(mtr)
    return mtr_inv

def rotate_arr(a, shft):
    # shft: b[0] = a[shft]
    
    if shft==0:
        return a.copy()
    
    n = a.shape[0]
    b = np.zeros_like(a)
    
    while shft<0:
        shft = n + shft
    while shft>n:
        shft = shft - n

    b[:n - shft] = a[shft:]
    b[n - shft:] = a[:shft]
    return b
    
def get_spline(x):
    while x<0:
        x += n_pts
    while x>=n_pts:
        x -= n_pts
    
    i = int(np.floor(x))
    t = x - i
    t2 = t*t
    t3 = t2*t
    
    return a[i] + b[i] * t + c[i] * t2 + d[i] * t3
    
def get_spline_at(x, period, a, b, c, d):
    while x<0:
        x += period
    while x>=period:
        x -= period
        
    n_pts = a.shape[0]
    
    step = period / n_pts
    i_float = x / step
    i = int(i_float)
    
    t = i_float - i
    t2 = t*t
    t3 = t2*t
    
    return a[i] + b[i] * t + c[i] * t2 + d[i] * t3

def get_spline_grad_at(x, period, a, b, c, d):
    while x<0:
        x += period
    while x>=period:
        x -= period
        
    n_pts = a.shape[0]
    
    step = period / n_pts
    i_float = x / step
    i = int(i_float)
    i_p1 = 0 if (i==n_pts-1) else i+1
    
    return a[i_p1] - a[i]

def get_abcd(pars):
    # find D
    P_r_p1 = rotate_arr(pars, 1)
    P_r_m1 = rotate_arr(pars, -1)
    DP = (P_r_p1 - P_r_m1) * 3
    
    global mtr_inv
    if mtr_inv.shape[0] != pars.shape[0]:
        mtr_inv = get_inv_mtr(pars.shape[0])
    
    D = np.matmul(mtr_inv, DP)
    
    # find a,b,c,d
    D_r_p1 = rotate_arr(D, 1)
    a = pars
    b = D
    c = 3 * (P_r_p1 - pars) - 2 * D - D_r_p1
    d = 2 * (pars - P_r_p1) + D + D_r_p1
    
    return a,b,c,d

def loss_spline_1(pars, pos, period):
    """
    pars: spline points
    pos: list of x,y,sy triples
    period: function period
    """
    
    a,b,c,d = get_abcd(pars)
    # get loss for all pos
    losses = [(get_spline_at(x, period, a,b,c,d)-y)/sy for x,y,sy in pos]
    losses = np.array(losses)

    return losses

def get_curve_amplitude_sigma(pos):
    """
    returns curve amplitude estimated as 3* std(y) /2 and mean value of measurement precision mean(sy)
    """
    y = pos[:,1]
    sy = pos[:,2]
    
    sy_lim = np.percentile(sy, 5) * 20
    sy_mean = sy[sy<sy_lim].mean()
    return 3*y.std()/2, sy_mean

def loss_spline_2(pars, pos, period):
    """
    pars: spline points
    pos: list of x,y,sy triples
    period: function period
    """
    
    a,b,c,d = get_abcd(pars)
    n_pts = a.shape[0]

    losses = [get_spline_grad_at(x, period, a,b,c,d) for x,*_ in pos]
    losses = np.array(losses)
    coef = n_pts / sy_mean / y_ampl / 8  # normalized for 8 pts
    losses = losses * coef
    return losses

y_ampl, sy_mean = (1, 1)  # y_ampl, sy_mean = get_curve_amplitude_sigma(pos)

def loss_spline(pars, pos, period):
    """
    pars: spline points
    pos: list of x,y,sy triples
    period: function period
    """
    
    l1 = loss_spline_1(pars, pos, period)
    l2 = loss_spline_2(pars, pos, period)
    #print(l1[0], l2[0])
    return np.sqrt(l1**2+ lambda_grad * l2**2)

def get_last_k(path):
    arr = []
    for itr in range(30):
        try:
            filename = os.path.join(path, f'itr_{itr:02d}', 'ia', 'distortion', 'FitSin.dat')
            fit_pars = np.fromfile(filename, sep=' ')
            k = fit_pars[1]
            arr.append(k)
        except:
            break
    return arr[-1]

def get_orig_img_size(path):
    for itr in range(30):
        try:
            filename = os.path.join(path, f'itr_{itr:02d}', 'ia', 'distortion', 'orig_img_sizes.par')
            sizes = np.fromfile(filename, sep=' ', dtype=np.int32)
            
            return sizes
        except:
            break
    return None

def plot_last_spline_px(path, idx, ax, label, dataset_idx, pxsz, npix):
    fontdictL={'size':16}
    fontdictT={'size':18, 'weight':'bold'}
    arr = []
    for itr in range(30):
        try:
            p = os.path.join(path, f'itr_{itr:02d}', 'ia', 'distortion', 'FitSpline.dat')
            #print(p)
            fit_pars = np.fromfile(p, sep=' ')
            fit_pars = fit_pars.reshape((-1, 11))
            arr.append(fit_pars)
        except:
            break
    if itr == 0:  # no records found
        print(f"can't plot spline, no spline fit data present in {path}")
        return False
    #print('arr=', arr)
    arr = np.stack(arr)
    
    k = get_last_k(path)
    
    if pxsz is None:
        pxsz = (1,1,1)
    
    period = np.pi*2
    
    if npix is None:
        npix = abs(int(period/k/pxsz[1]))
        
    #print(npix)
    y_range = np.arange(npix) * abs(pxsz[1])
    ph_range = y_range * k
    #plt.plot(ph_range)
    #plt.show()
    
    colors = ['green', 'blue', 'red']
    tit = ['X', 'Y', 'Z']
    
    data_it = arr[idx]
    n_ph = data_it[:, 1]

    colids = list(range(len(colors)))
    colid = colids[dataset_idx]
        
    for axi in range(3):
        p_ph = data_it[:, 2 + 3*axi]
        s_d_ph = data_it[:, 3 + 3*axi]
        s_m_ph = data_it[:, 4 + 3*axi]
        
        s = np.sqrt((s_d_ph**2 + s_m_ph**2) / n_ph)
        
        px = abs(1. if pxsz is None else pxsz[axi])

        a,b,c,d = get_abcd(p_ph)
        dist = np.array([get_spline_at(x, period, a,b,c,d) for x in ph_range])
        a,b,c,d = get_abcd(p_ph+s)
        distt = np.array([get_spline_at(x, period, a,b,c,d) for x in ph_range])
        a,b,c,d = get_abcd(p_ph-s)
        distb = np.array([get_spline_at(x, period, a,b,c,d) for x in ph_range])
        ax[axi].plot(y_range, dist*px, color=colors[colid], label=label)
        ax[axi].fill_between(y_range, distt*px, distb*px, color=colors[colid], alpha=0.2)
        #plt.legend(ttl)
        
        a,b,c,d = get_abcd(p_ph+s_d_ph)
        dist_t_sgm = np.array([get_spline_at(x, period, a,b,c,d) for x in ph_range])
        ax[axi].set_xlabel('y, $\mu m$',fontdict=fontdictL)
        ax[axi].tick_params(axis='x', labelsize=16)
        ax[axi].tick_params(axis='y', labelsize=16)
        if axi==0:
            ax[axi].set_ylabel('offset, $\mu m$', fontdict=fontdictL)

        ax[axi].set_title(tit[axi],fontdict=fontdictT)
    return True

def gen_dist_map(path, pxsz, orig_sz, skip_norm=False, xyz_min=None, xyz_max=None):
    arr = []
    for itr in range(30):
        try:
            filename = os.path.join(path, f'itr_{itr:02d}', 'ia', 'distortion', 'FitSpline.dat')

            fit_pars = np.fromfile(filename, sep=' ')
            fit_pars = fit_pars.reshape((-1, 11))
            arr.append(fit_pars)
        except:
            break
    if len(arr) == 0:  # no records found
        print(f"can't generate distortion map: no spline fit data present in {path}")
        return None
    #print(path, itr, arr)
    data_it = arr[-1]
    
    if orig_sz is None:
        orig_sz = get_orig_img_size(path)
    
    k = get_last_k(path)
    
    if pxsz is None:
        print('assuming default processing mode with pix size set to 1.')
        pxsz = (1,1,1)
    
    period = np.pi*2
    
    # 1.  get limits
    if xyz_min is None or xyz_max is None:
        ph_range = np.linspace(0, period, 1024, endpoint=False)
        xyz_min = []
        xyz_max = []
        for axi in range(3):
            p_ph = data_it[:, 2 + 3*axi]*pxsz[axi]

            a,b,c,d = get_abcd(p_ph)
            dist = np.array([get_spline_at(x, period, a,b,c,d) for x in ph_range])
            xyz_min.append(dist.min())
            xyz_max.append(dist.max())


    extra_pix = [int( np.ceil((i_max-i_min) * (1 if axi == 2 else xy_fact)) ) 
                 for axi, (i_max, i_min) in enumerate(zip(xyz_max, xyz_min))]
    
    #npix = abs(int(period/k/pxsz[1]))
    
    # 2. set new size
    corr_sz = [v+dv for v, dv in zip(orig_sz, extra_pix)]
    
    n_pix = corr_sz[1]
    y_range = np.arange(n_pix)
    ph_range = y_range * k
    
    
    
    n_ph = len(data_it)
    dists = {}
    for ph in range(n_ph):
        phases = ph_range + ph/n_ph * period
        #print(phases.shape)
        for axi in range(3):
            p_ph = data_it[:, 2 + 3*axi]*pxsz[axi]
            
 
            a,b,c,d = get_abcd(p_ph)
            dist = np.array([get_spline_at(x, period, a,b,c,d) for x in phases])
            min_ax = xyz_min[axi]
            f = 1 if axi == 2 else xy_fact
            
            if not skip_norm:
                dist -= min_ax
                dist *= f
            
            dists[(ph,axi)] = dist

    return xyz_min, xyz_max, extra_pix, corr_sz, dists

def save_sz(sz, path):
    with open(path, 'wt') as f:
        d,h,w = sz
        f.write(f'{d} {h} {w}\n')
        
def save_dist_file(dists, path, n_ph):
    with open(path, 'wt') as f:
        for ph in range(n_ph):
            for axi in range(3):
                f.write(f'{ph} {axi} ')
                
                dist = dists[(ph,axi)]
                s = [f'{v:.4f}' for v in dist]
                
                f.write(' '.join(s))
                f.write('\n')
        
def save_dist_map(orig_sz, corr_sz, dists, path):
    #1. save size:
    save_sz(orig_sz, os.path.join(path, 'orig_img_sizes.par'))
    save_sz(corr_sz, os.path.join(path, 'corr_img_sizes.par'))
    save_dist_file(dists, os.path.join(path, 'dist_map.par'), n_ph=n_phase)
    
def gen_save_dist_map(path, out_path, pxsz=None, orig_sz=None):
    orig_sz = orig_sz or get_orig_img_size(path)
    gdm_res = gen_dist_map(path, pxsz=pxsz, orig_sz=orig_sz)
    if gdm_res is None:
        return False
    xyz_min, xyz_max, extra_pix, corr_sz, dists = gdm_res
    save_dist_map(orig_sz, corr_sz, dists, out_path)
    return True

def interpolate_save_dist_map(path1, path2, coord1, coord2, coord, out_path, orig_sz=None, pxsz=None):
    if orig_sz is None:
        print('target size not set, trying original sizes of maping datasets')
        
        orig_sz1 = get_orig_img_size(path1)
        orig_sz2 = get_orig_img_size(path2)
        assert orig_sz1[0]==orig_sz2[0] and orig_sz1[1]==orig_sz2[1]
        
        orig_sz = orig_sz1
        
    if pxsz is None:
        print('assuming default processing mode with pix size set to 1.')
        pxsz = (1,1,1)
                
    xyz_min1, xyz_max1, extra_pix1, corr_sz1, dists1 = gen_dist_map(path1, pxsz, orig_sz=orig_sz, skip_norm=True)
    xyz_min2, xyz_max2, extra_pix2, corr_sz2, dists2 = gen_dist_map(path2, pxsz, orig_sz=orig_sz, skip_norm=True)
    
    xyz_min = [min(v1,v2) for v1, v2 in zip(xyz_min1, xyz_min2)]
    xyz_max = [max(v1,v2) for v1, v2 in zip(xyz_max1, xyz_max2)]

    xyz_min1, xyz_max1, extra_pix1, corr_sz1, dists1 = gen_dist_map(path1, pxsz, orig_sz=orig_sz, xyz_min=xyz_min, xyz_max=xyz_max)
    xyz_min2, xyz_max2, extra_pix2, corr_sz2, dists2 = gen_dist_map(path2, pxsz, orig_sz=orig_sz, xyz_min=xyz_min, xyz_max=xyz_max)
    
    assert(corr_sz1 == corr_sz2)
    corr_sz = corr_sz1
        
    # interp coefs:
    a1 = (coord2 - coord) / (coord2 - coord1)
    a2 = 1-a1
    
    dists = {}
    for k in dists1.keys():
        dist1 = dists1[k]
        dist2 = dists2[k]
        dist = a1 * dist1 + a2 * dist2
        dists[k] = dist
    
    save_dist_map(orig_sz, corr_sz, dists, out_path)

def compare_px( pathes, labels, comp_name, pxsz, fig_size=(12, 2.8), npix = 512, show=True):
    fig, axs = plt.subplots(1, 3, figsize=fig_size, sharex='all', sharey='all')
    
    plotting_res = []
    for i, pl in enumerate(zip(pathes, labels)):
        p,l=pl
        res = plot_last_spline_px(p, -1, ax=axs, label=l, dataset_idx=i, pxsz=pxsz, npix=npix)
        plotting_res.append(res)
    if not np.any(plotting_res):
        plt.close()
        return
    
    #ymin = ymax = 0
    #for x in axs:
    #    y1,y2 = x.get_ylim()
    #    ymin = min(ymin,y1)
    #    ymax = max(ymax,y2)
    #    
    #for axi, x in enumerate(axs):
    #    x.set_ylim(ymin*1.1, ymax*1.1)
        
    _=axs[0].legend(prop={'size':15})

    plt.tight_layout(pad=0, h_pad=0, w_pad=1)
    if comp_name:
        fig.savefig(comp_name+'.png')
        fig.savefig(comp_name+'.pdf')
    if show:
        plt.plot()
    plt.close()
    
def plot_last_spline_ph(path, idx, ax, label, dataset_idx, pxsz, npix):
    fontdictL={'size':15}
    fontdictT={'size':17, 'weight':'bold'}
    arr = []
    for itr in range(30):
        try:
            filename = os.path.join(path, f'itr_{itr:02d}', 'ia', 'distortion', 'FitSpline.dat')
            fit_pars = np.fromfile(filename, sep=' ')
            fit_pars = fit_pars.reshape((-1, 11))
            arr.append(fit_pars)
        except:
            break
    arr = np.stack(arr)
    
    k = get_last_k(path)
    
    ph_range = np.linspace(0, np.pi*2, npix)
    
    colors = ['green', 'blue', 'red']
    tit = ['X', 'Y', 'Z']
    tits = ['x', 'y', 'z']
   
    data_it = arr[idx]
    n_ph = data_it[:, 1]

    colids = list(range(len(colors)))
    colid = colids[dataset_idx]
        
    for axi in range(3):
        p_ph = data_it[:, 2 + 3*axi]
        s_d_ph = data_it[:, 3 + 3*axi]
        s_m_ph = data_it[:, 4 + 3*axi]
        
        s = np.sqrt((s_d_ph**2 + s_m_ph**2) / n_ph)
        
        px = abs(1. if pxsz is None else pxsz[axi])

        a,b,c,d = get_abcd(p_ph)
        dist = np.array([get_spline_at(x, period, a,b,c,d) for x in ph_range])
        a,b,c,d = get_abcd(p_ph+s)
        distt = np.array([get_spline_at(x, period, a,b,c,d) for x in ph_range])
        a,b,c,d = get_abcd(p_ph-s)
        distb = np.array([get_spline_at(x, period, a,b,c,d) for x in ph_range])
        ax[axi].plot(ph_range, dist*px, color=colors[colid], label=label)
        ax[axi].fill_between(ph_range, distt*px, distb*px, color=colors[colid], alpha=0.2)
        #plt.legend(ttl)
        
        a,b,c,d = get_abcd(p_ph+s_d_ph)
        dist_t_sgm = np.array([get_spline_at(x, period, a,b,c,d) for x in ph_range])
        ax[axi].set_xlabel('$\phi, rad$',fontdict=fontdictL)
        ax[axi].set_xticks(np.linspace(0, np.pi*2, 5))
        ax[axi].set_xticklabels(['$0$', '$\pi /2 $', '$\pi$', '$3 \pi/2$', '$2 \pi$'])

        ax[axi].tick_params(axis='x', labelsize=15)
        ax[axi].tick_params(axis='y', labelsize=15)
        ax[axi].set_ylabel('$%s(\phi), \mu m$'%tits[axi], fontdict=fontdictL)

        #ax[axi].set_title(tit[axi],fontdict=fontdictT)
        pass

def compare_ph( pathes, labels, comp_name, pxsz, fig_size=(12, 2.8), npix = 512):
    fontdictT={'size':27, 'weight':'bold'}
    fig, axs = plt.subplots(1, 3, figsize=fig_size, sharex='all')
    for i, pl in enumerate(zip(pathes, labels)):
        p,l=pl
        plot_last_spline_ph(p, -1, ax=axs, label=l, dataset_idx=i, pxsz=pxsz, npix=npix)
    
    #ymin = ymax = 0
    #for x in axs:
    #    y1,y2 = x.get_ylim()
    #    ymin = min(ymin,y1)
    #    ymax = max(ymax,y2)
    #    
    #for axi, x in enumerate(axs):
    #    x.set_ylim(ymin*1.1, ymax*1.1)
        
    #_=axs[0].legend(prop={'size':15})

    plt.suptitle('3$^{rd}$ order spline fit of the periodic motion', y=0.99, fontsize=17, fontweight='bold')
    plt.tight_layout(pad=2.5, h_pad=0, w_pad=1)
    if comp_name:
        fig.savefig(comp_name+'.png')
        fig.savefig(comp_name+'.pdf')
    plt.show()
    
def compare_iterations(path, save_name, pxsz, fig_size=(12, 2.8), npix = 512):
    fontdictL={'size':15}
    fontdictT={'size':17, 'weight':'bold'}
    arr = []
    for itr in range(30):
        try:
            filename = os.path.join(path, f'itr_{itr:02d}', 'ia', 'distortion', 'FitSpline.dat')
            fit_pars = np.fromfile(filename, sep=' ')
            fit_pars = fit_pars.reshape((-1, 11))
            arr.append(fit_pars)
        except:
            break
    arr = np.stack(arr)
    
    ph_range = np.linspace(0, np.pi*2, npix)
    ttl = ['itr '+str(v) for v in range(nv_a.shape[0])]
    tit = ['X', 'Y', 'Z']
    tits = ['x', 'y', 'z']
    colors = ["#cd4473","#9cd855","#6d68d2","#54a940","#b551b3","#66dd93","#cd4f36","#78d8cb","#d28535","#8ba7e0",
              "#cbbb45","#6d6193","#557c2e","#d797c7","#b1ca88","#ae6365","#4898b1","#826b36","#4d8767","#dbaf84"]
    
    fig, axs = plt.subplots(1, 3, figsize=fig_size)
    for ax in range(3):
        for itr, data_it in enumerate(arr):
            p_it = data_it[:, 2 + 3*ax]
            a,b,c,d = get_abcd(p_it)
            dist = np.array([get_spline_at(x, period, a,b,c,d) for x in ph_range]) * abs(pxsz[ax])
            axs[ax].plot(ph_range, dist, color=colors[itr])
            axs[ax].grid(True)
            #axs[ax].set_title(tit[ax], fontdict=fontdictT)
            if ax == 0:
                axs[ax].legend(ttl, prop={'size':15})
        
        axs[ax].set_xlabel('$\phi, rad$',fontdict=fontdictL)
        axs[ax].set_ylabel('$%s(\phi), \mu m$'%tits[ax], fontdict=fontdictL)
        
        axs[ax].set_xticks(np.linspace(0, np.pi*2, 5))
        axs[ax].set_xticklabels(['$0$', '$\pi /2 $', '$\pi$', '$3 \pi/2$', '$2 \pi$'])
        axs[ax].tick_params(axis='x', labelsize=15)
        axs[ax].tick_params(axis='y', labelsize=15)
        
    plt.suptitle('Iterative motion pattern improvement', y=0.99, fontsize=17, fontweight='bold')
    plt.tight_layout(pad=2.5, h_pad=0, w_pad=1)
    if save_name:
        fig.savefig(save_name+'.png')
        fig.savefig(save_name+'.pdf')
    plt.show()
    
def load_dm(path):
    p = os.path.join(path, 'dist_map.par')
    dm = np.fromfile(p, sep=' ').reshape((8, 3, -1))
    for ph, dm_ph in enumerate(dm):
        for ax, dm_ph_ax in enumerate(dm_ph):
            assert ph == dm_ph_ax[0]
            assert ax == dm_ph_ax[1]
    dm = dm[..., 2:]
    dm -= dm.mean(axis=(0, 2), keepdims=True)
    return dm