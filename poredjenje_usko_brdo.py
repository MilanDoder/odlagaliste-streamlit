import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from geometrija_v2 import Kupa, Teren, presek_kupe_i_terena
from test_buvac_podaci import ucitaj_teren

WX, WY, WZ, K = 6412400.0, 4970900.0, 198.0, 120.0

t_bez = Teren.iz_tacaka(ucitaj_teren("podaci/001-Teren-3-Buvac.txt"))
t_sa  = Teren.iz_tacaka(ucitaj_teren("podaci/001-Teren-3-Buvac-v2-usko-brdo.txt"))
kupa = Kupa(wx=WX, wy=WY, wz=WZ, k=K, ugao=37.0, profil="krug")

r_bez = presek_kupe_i_terena(kupa, t_bez, 512, 2)
r_sa  = presek_kupe_i_terena(kupa, t_sa, 512, 2)

x0,x1,y0,y1 = r_bez.granice_racuna
pad = 0.18*(x1-x0)
vx0,vx1,vy0,vy1 = x0-pad, x1+pad, y0-pad, y1+pad
n=200
GX,GY = np.meshgrid(np.linspace(vx0,vx1,n), np.linspace(vy0,vy1,n))
ZK = kupa.z(GX,GY)

fig = plt.figure(figsize=(17.5, 10.5))
gs = GridSpec(2, 3, figure=fig, width_ratios=[1,1,0.95], hspace=0.12, wspace=0.16)

zlim = (60, 235)
for col,(t,r,nas) in enumerate([(t_bez,r_bez,"BEZ uskog brda"),(t_sa,r_sa,"SA uskim brdom")]):
    ZT = t.z(GX,GY)
    tijelo = np.where(ZK>ZT+1e-9, ZK, np.nan)

    ax = fig.add_subplot(gs[0,col], projection="3d")
    ax.plot_surface(GX,GY,ZT, cmap="terrain", alpha=0.92, linewidth=0, vmin=60, vmax=225)
    ax.plot_surface(GX,GY,tijelo, color="peru", alpha=0.62, linewidth=0)
    for kk in r.konture:
        ax.plot(kk[:,0],kk[:,1],kk[:,2]+0.6,"r-",lw=2.2)
    ax.set_zlim(*zlim)
    ax.set_box_aspect((1,1,0.42)); ax.view_init(elev=34, azim=-58)
    ax.set_title(f"{nas}\nV = {r.zapremina:,.0f} m³   |   petlji: {r.broj_petlji}",
                 fontsize=13, fontweight="bold", pad=2)
    ax.tick_params(labelsize=7)

    ax2 = fig.add_subplot(gs[1,col])
    cf = ax2.contourf(GX,GY,ZT, levels=np.linspace(60,225,24), cmap="terrain", extend="both")
    kx,ky = kupa.gornja_kontura()
    ax2.plot(kx,ky,"b-",lw=1.3,label="gornji plato kupe")
    for i,kk in enumerate(r.konture):
        ax2.plot(kk[:,0],kk[:,1],"r-",lw=2.4,label="presjek kupa–teren" if i==0 else None)
    ax2.axhline(WY, color="k", ls=":", lw=0.8)
    ax2.set_aspect("equal"); ax2.tick_params(labelsize=7)
    ax2.set_title(f"osnova = {r.povrsina_osnove:,.0f} m²", fontsize=11)
    if col==0: ax2.legend(fontsize=8, loc="upper left")

fig.colorbar(cf, ax=fig.axes[1::2][:2], location="bottom", fraction=0.03, pad=0.06, label="kota terena (m)")

# --- profil kroz vrh (I-Z) ---
axp = fig.add_subplot(gs[0,2])
R = 0.62*(x1-x0)
px = np.linspace(WX-R, WX+R, 900); py = np.full_like(px, WY)
zt_b, zt_s, zk = t_bez.z(px,py), t_sa.z(px,py), kupa.z(px,py)
d = px-WX
axp.fill_between(d, np.maximum(zt_b,zk*0+zt_b), np.where(zk>zt_b,zk,zt_b),
                 color="peru", alpha=0.35, label="tijelo bez brda")
axp.fill_between(d, zt_s, np.where(zk>zt_s,zk,zt_s), color="peru", alpha=0.85,
                 label="tijelo sa brdom")
axp.plot(d, zt_b, "g--", lw=1.4, label="teren bez brda")
axp.plot(d, zt_s, "k-", lw=1.6, label="teren sa brdom")
axp.plot(d, zk, "b-", lw=1.6, label="površina kupe")
axp.set_xlabel("rastojanje od vrha kupe duž I–Z (m)", fontsize=9)
axp.set_ylabel("kota (m)", fontsize=9)
axp.set_title("Poprečni profil kroz vrh kupe", fontsize=12, fontweight="bold")
axp.legend(fontsize=8, loc="upper left"); axp.grid(alpha=0.25); axp.tick_params(labelsize=8)
axp.set_ylim(140, 230)

# --- tabela poredjenja ---
axt = fig.add_subplot(gs[1,2]); axt.axis("off")
dV = r_bez.zapremina - r_sa.zapremina
dA = r_bez.povrsina_osnove - r_sa.povrsina_osnove
rows = [
    ["", "bez brda", "sa brdom", "razlika"],
    ["zapremina (m³)", f"{r_bez.zapremina:,.0f}", f"{r_sa.zapremina:,.0f}",
     f"−{dV:,.0f}\n(−{dV/r_bez.zapremina*100:.1f} %)"],
    ["osnova (m²)", f"{r_bez.povrsina_osnove:,.0f}", f"{r_sa.povrsina_osnove:,.0f}",
     f"−{dA:,.0f}\n(−{dA/r_bez.povrsina_osnove*100:.1f} %)"],
    ["presječnih petlji", f"{r_bez.broj_petlji}", f"{r_sa.broj_petlji}", "tijelo podijeljeno"],
    ["max debljina (m)", f"{r_bez.max_debljina:.1f}", f"{r_sa.max_debljina:.1f}", ""],
    ["num. greška (m³)", f"{r_bez.procjena_greske:,.0f}", f"{r_sa.procjena_greske:,.0f}", ""],
    ["teren u vrhu (m)", f"{float(t_bez.z(WX,WY)):.1f}", f"{float(t_sa.z(WX,WY)):.1f}", "krijest +59.9"],
]
tab = axt.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
tab.auto_set_font_size(False); tab.set_fontsize(9.5); tab.scale(1, 2.35)
for j in range(4): tab[0,j].set_facecolor("#dcdcdc"); tab[0,j].set_text_props(weight="bold")
for i in range(1,len(rows)): tab[i,0].set_text_props(weight="bold")
tab[1,3].set_facecolor("#ffe0e0"); tab[3,3].set_facecolor("#ffe0e0")
axt.set_title(f"Kupa: vrh ({WX:.0f}, {WY:.0f}, {WZ:.0f}), k = {K:.0f} m, ugao = 37°",
              fontsize=11, pad=14)

fig.suptitle("Uticaj uskog brda (grebena) na zapreminu odlagališta — isti teren, ista kupa",
             fontsize=15, fontweight="bold", y=0.975)
plt.savefig("poredjenje_usko_brdo.png", dpi=120, bbox_inches="tight")
print("ok", r_bez.zapremina, r_sa.zapremina, dV, r_bez.povrsina_osnove, r_sa.povrsina_osnove)
