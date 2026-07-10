"""
napravi_teren_usko_brdo.py — pravi novi teren fajl: stvarni teren Buvac
+ USKI GREBEN ("usko brdo"), uz očuvanu uvalu (kop) i sav ostali reljef.

Greben je aditivan: z_novo = z_staro + h(x, y), pa se ništa ne briše —
uvala, kosine i etaže ostaju netaknute.

Oblik grebena: Gausov profil oko duži (linijski greben), sa mekim
opadanjem na krajevima (smoothstep) da ne nastane vještačka litica.

    h(p) = H · exp(−(s/σ)²) · taper(t)

    t = projekcija (p − C) na pravac grebena
    s = normalno rastojanje od ose grebena

Podrazumijevano je greben postavljen na ravan plato SZ od kopa
(6412400, 4970900; teren ≈ 158 m, raspon ±2 m u krugu r = 150 m) — tako
da kupa postavljena tu bude presječena grebenom na dva dijela, kao u
primjeru sa slike.

Pokretanje:

    py napravi_teren_usko_brdo.py
    py napravi_teren_usko_brdo.py --visina 70 --sigma 20 --azimut 30

Izlaz:
    podaci/001-Teren-3-Buvac-v2-usko-brdo.txt   (isti CSV format: X,Y,Z)
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from geometrija_v2 import Kupa, Teren, presek_kupe_i_terena
from test_buvac_podaci import ucitaj_teren, nadji_fajl


def greben_visina(x, y, cx, cy, azimut_deg, H, sigma, poluduzina, taper):
    """Aditivna visina uskog grebena u tački (x, y).

    Args:
        cx, cy:      centar grebena
        azimut_deg:  pravac ose grebena (0° = +X, 90° = +Y)
        H:           visina krijesta (m)
        sigma:       poluširina Gausovog profila (m); ukupna širina ≈ 4σ
        poluduzina:  polovina dužine grebena (m)
        taper:       dužina mekog opadanja na krajevima (m)
    """
    a = np.radians(azimut_deg)
    ux, uy = np.cos(a), np.sin(a)

    dx = np.asarray(x, float) - cx
    dy = np.asarray(y, float) - cy

    t = dx * ux + dy * uy               # duž ose
    s = dx * (-uy) + dy * ux            # normalno na osu

    poprecni = np.exp(-(s / sigma) ** 2)

    # smoothstep opadanje na krajevima: 1 unutra, 0 izvan poludužine
    at = np.abs(t)
    u = np.clip((poluduzina - at) / taper, 0.0, 1.0)
    uzduzni = u * u * (3.0 - 2.0 * u)

    return H * poprecni * uzduzni


def main():
    ap = argparse.ArgumentParser(description="Dodaj uski greben na Buvac teren")
    ap.add_argument("--teren", help="ulazni teren (default: podaci/001-Teren-3-Buvac.txt)")
    ap.add_argument("--cx", type=float, default=6412400.0)
    ap.add_argument("--cy", type=float, default=4970900.0)
    ap.add_argument("--azimut", type=float, default=90.0, help="pravac ose (° od +X)")
    ap.add_argument("--visina", type=float, default=60.0, help="visina krijesta (m)")
    ap.add_argument("--sigma", type=float, default=25.0, help="poluširina (m)")
    ap.add_argument("--poluduzina", type=float, default=400.0)
    ap.add_argument("--taper", type=float, default=120.0)
    ap.add_argument("--gustina", type=float, default=6.0,
                    help="razmak dodatnih tačaka duž grebena (m); 0 = bez dodavanja")
    ap.add_argument("--izlaz", default="podaci/001-Teren-3-Buvac-v2-usko-brdo.txt")
    a = ap.parse_args()

    f_in = nadji_fajl(a.teren, "podaci/001-Teren-3-Buvac.txt",
                      "001-Teren-3-Buvac.txt")
    xyz = ucitaj_teren(f_in)
    print(f"Ulazni teren: {len(xyz):,} tačaka, "
          f"Z ∈ [{xyz[:, 2].min():.1f}, {xyz[:, 2].max():.1f}] m")

    def h(px, py):
        return greben_visina(px, py, a.cx, a.cy, a.azimut,
                             a.visina, a.sigma, a.poluduzina, a.taper)

    # --- 1) podigni postojeće tačke ------------------------------------------
    novi = xyz.copy()
    novi[:, 2] = novi[:, 2] + h(novi[:, 0], novi[:, 1])

    # --- 2) zgusni tačke u koridoru grebena ----------------------------------
    # (medijan razmaka ulaznih tačaka je ~4 m; dodatne tačke daju oštar krijest)
    if a.gustina > 0:
        baza = Teren.iz_tacaka(xyz)          # interpolator IZVORNOG terena
        rad = np.radians(a.azimut)
        ux, uy = np.cos(rad), np.sin(rad)
        L = a.poluduzina
        W = 2.5 * a.sigma                    # koridor ±2.5σ
        tt = np.arange(-L, L + 1e-9, a.gustina)
        ss = np.arange(-W, W + 1e-9, a.gustina / 2.0)
        T, S = np.meshgrid(tt, ss)
        gx = a.cx + T * ux + S * (-uy)
        gy = a.cy + T * uy + S * ux
        gx, gy = gx.ravel(), gy.ravel()

        # ostani unutar granica izvornog terena
        x0, x1, y0, y1 = baza.xy_granice
        u = (gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1)
        gx, gy = gx[u], gy[u]
        gz = baza.z(gx, gy) + h(gx, gy)
        dodate = np.column_stack([gx, gy, gz])
        novi = np.vstack([novi, dodate])
        print(f"Dodato {len(dodate):,} tačaka u koridoru grebena "
              f"(razmak {a.gustina:.0f} m)")

    print(f"Novi teren:   {len(novi):,} tačaka, "
          f"Z ∈ [{novi[:, 2].min():.1f}, {novi[:, 2].max():.1f}] m")

    # --- 3) snimi -------------------------------------------------------------
    os.makedirs(os.path.dirname(a.izlaz) or ".", exist_ok=True)
    np.savetxt(a.izlaz, novi, delimiter=",", fmt="%.6f")
    mb = os.path.getsize(a.izlaz) / 1e6
    print(f"Sačuvano: {a.izlaz}  ({mb:.1f} MB)")

    # --- 4) brza provjera: kupa na grebenu se dijeli na dva dijela -----------
    print("\nProvjera — kupa postavljena na sam greben:")
    t_novi = Teren.iz_tacaka(novi)
    t_baza = Teren.iz_tacaka(xyz)
    z_baza = float(t_baza.z(a.cx, a.cy))
    z_greben = float(t_novi.z(a.cx, a.cy))
    print(f"  teren u centru: prije {z_baza:.1f} m, poslije {z_greben:.1f} m "
          f"(krijest +{z_greben - z_baza:.1f} m)")

    wz = z_baza + 40.0                     # plato ISPOD krijesta grebena
    kupa = Kupa(wx=a.cx, wy=a.cy, wz=wz, k=120.0, ugao=37.0, profil="krug")

    r_bez = presek_kupe_i_terena(kupa, t_baza, 512, 2)
    r_sa = presek_kupe_i_terena(kupa, t_novi, 512, 2)
    print(f"  bez grebena:  V = {r_bez.zapremina:12,.0f} m³, petlji = {r_bez.broj_petlji}")
    print(f"  sa grebenom:  V = {r_sa.zapremina:12,.0f} m³, petlji = {r_sa.broj_petlji}")
    print(f"  greben odnio {r_bez.zapremina - r_sa.zapremina:,.0f} m³ "
          f"({(1 - r_sa.zapremina / r_bez.zapremina) * 100:.1f} %)")
    if r_sa.broj_petlji >= 2:
        print("  ✓ tijelo je razdvojeno grebenom na više dijelova — kao u primjeru")
    else:
        print("  ! samo jedna petlja — povećaj --visina ili smanji --sigma")

    print(f"\nProbaj prikaz:")
    print(f"  py prikaz_3d.py --teren {a.izlaz} "
          f"--wx {a.cx:.0f} --wy {a.cy:.0f} --wz {wz:.0f} --k 120 "
          f"--profil krug --izlaz usko_brdo_3d.html")


if __name__ == "__main__":
    main()
