import astropy.units as u
import astropy.coordinates as coord
import matplotlib.pyplot as plt
import numpy as np

from dustmaps.bayestar import BayestarQuery

bayestar = None

def get_MG_BPRP(g, dust_correct=False):
    global bayestar
    
    if dust_correct:
        if bayestar is None:
            bayestar = BayestarQuery()
            
        c = g.get_skycoord()
        ebv = bayestar.query(c)
        mg = g.get_G0(ebv=ebv) - g.distmod
        bprp = g.get_BP0(ebv=ebv) - g.get_RP0(ebv=ebv)
    else:
        mg = g.phot_g_mean_mag - g.distmod
        bprp = g.phot_bp_mean_mag - g.phot_rp_mean_mag
    
    return mg, bprp


def make_cmd(g, ax=None, dust_correct=False,
             cbar_label='', app_mag_twinx=True, add_labels=True,
             **kwargs):
    
    mg, bprp = get_MG_BPRP(g, dust_correct=dust_correct)
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7), 
                               constrained_layout=True)
    else:
        fig = ax.figure
    
    kwargs.setdefault('edgecolor', '#666666')
    kwargs.setdefault('linewidth', 0.5)
    kwargs.setdefault('cmap', 'cividis')
    cs = ax.scatter(bprp, mg, **kwargs)
    
#     if label_OG:
#         ax.annotate('TIC 2749',
#                     (bprp[g.source_id == 1490845584382687232].value, 
#                      mg[g.source_id == 1490845584382687232].value),
#                     xytext=(bprp[g.source_id == 1490845584382687232].value + 0.2, 
#                             mg[g.source_id == 1490845584382687232].value - 0.4),
#                     arrowprops=dict(arrowstyle='->', color='#777777'), fontsize=14,
#                     zorder=100)

#         ax.annotate('TOI 1807',
#                     (bprp[g.source_id == 1476485996883837184].value, 
#                      mg[g.source_id == 1476485996883837184].value),
#                     xytext=(bprp[g.source_id == 1476485996883837184].value + 0.2, 
#                             mg[g.source_id == 1476485996883837184].value - 0.4),
#                     arrowprops=dict(arrowstyle='->', color='#777777'), fontsize=14,
#                     zorder=100)

    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(14, -4)
    
    if add_labels:
        ax.set_xlabel(r'$G_{\rm BP}-G_{\rm RP}$')
        ax.set_ylabel('$M_G$')
    
    if app_mag_twinx:
        ax2 = ax.twinx()
        ylim = ax.get_ylim()
        # fid_dm = coord.Distance(40*u.pc).distmod.value
        fid_dm = np.nanmedian(g.distmod.value)
        ax2.set_ylim(ylim[0] + fid_dm, 
                     ylim[1] + fid_dm)
        
        if add_labels:
            ax2.set_ylabel('$G$ [mag] (at median DM)')
    
    if 'c' in kwargs:
        cb = fig.colorbar(cs, ax=ax, aspect=80)
        cb.set_label(cbar_label)
    
    fig.set_facecolor('w')
    
    return fig, ax