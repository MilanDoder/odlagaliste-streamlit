"""
demo_v2.py — vizuelna demonstracija presjeka kupe i BRDOVITOG terena.

Pokretanje:  python3 demo_v2.py
Izlaz:       demo_v2_presjek.png  (3D prikaz + tlocrt sa konturama)
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from geometrija_v2 import Kupa, Teren, presek_kupe_i_terena

# --- brdovit teren sa dva brda koja probijaju padinu kupe -------------------
def teren_fn(x, y):
    valovi = 4.0 * np.sin((x - 1000.0) / 55.0) * np.cos((y - 2000.0) / 40.0)
    b1 = 45.0 * np.exp(-(((x - 1130.0) / 20.0) ** 2 + ((y - 2000.0) / 20.0) ** 2))
    b2 = 45.0 * np.exp(-(((x - 870.0) / 20.0) ** 2 + ((y - 2000.0) / 20.0) ** 2))
    return 140.0 + valovi + b1 + b2

teren = Teren.analiticki(teren_fn, (600, 1400, 1600, 2400), (131.0, 186.0))
kupa = Kupa(wx=1000, wy=2000, wz=185, k=120, ugao=37, profil="krug")

rez = presek_kupe_i_terena(kupa, teren, rezolucija=512, rafiniranje=2)
print(f"Zapremina tijela:      {rez.zapremina:,.0f} m³")
print(f"Površina osnove:       {rez.povrsina_osnove:,.0f} m²")
print(f"Broj presječnih petlji: {rez.broj_petlji}")
print(f"Max debljina nasipa:   {rez.max_debljina:.1f} m")
print(f"Procjena num. greške:  {rez.procjena_greske:,.0f} m³ "
      f"({rez.procjena_greske / rez.zapremina * 100:.4f} %)")

# --- crtanje ----------------------------------------------------------------
n = 160
x0, x1, y0, y1 = rez.granice_racuna
gx = np.linspace(x0, x1, n)
gy = np.linspace(y0, y1, n)
GX, GY = np.meshgrid(gx, gy)
ZT = teren.z(GX, GY)
ZK = kupa.z(GX, GY)
tijelo = np.where(ZK > ZT, ZK, np.nan)     # kupa samo iznad terena

fig = plt.figure(figsize=(15, 6.5))

ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.plot_surface(GX, GY, ZT, cmap="terrain", alpha=0.85,
                linewidth=0, antialiased=True)
ax.plot_surface(GX, GY, tijelo, color="peru", alpha=0.55, linewidth=0)
for kont in rez.konture:
    ax.plot(kont[:, 0], kont[:, 1], kont[:, 2] + 0.3, "r-", lw=2)
ax.set_title(f"Kupa na brdovitom terenu — V = {rez.zapremina:,.0f} m³, "
             f"{rez.broj_petlji} petlje presjeka")
ax.set_box_aspect((1, 1, 0.35))
ax.view_init(elev=35, azim=-60)

ax2 = fig.add_subplot(1, 2, 2)
cf = ax2.contourf(GX, GY, ZT, levels=24, cmap="terrain", alpha=0.9)
plt.colorbar(cf, ax=ax2, label="kota terena (m)")
kx, ky = kupa.gornja_kontura()
ax2.plot(kx, ky, "b--", lw=1.2, label="gornji plato kupe")
for i, kont in enumerate(rez.konture):
    ax2.plot(kont[:, 0], kont[:, 1], "r-", lw=2,
             label="presjek kupa–teren" if i == 0 else None)
ax2.set_aspect("equal")
ax2.legend(loc="upper right")
ax2.set_title("Tlocrt: presječne konture (rupe = brda probijaju kupu)")

plt.tight_layout()
plt.savefig("demo_v2_presjek.png", dpi=130)
print("\nSačuvano: demo_v2_presjek.png")
