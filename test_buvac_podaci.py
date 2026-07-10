"""
test_buvac_podaci.py — test i vizuelizacija NOVE geometrije na STVARNIM
podacima kopa Buvac (teren od ~42.000 tačaka, granica zone, centar masa).

Ne mijenja geometrija_v2.py — samo ga koristi.

Pokretanje (fajlovi u podaci/ ili zadaj putanje):

    py test_buvac_podaci.py
    py test_buvac_podaci.py --teren podaci/001-Teren-3-Buvac.txt ^
        --granice podaci/001GranicaZonaBuvac.txt ^
        --cm podaci/001CentarMasaBuvac.txt --params podaci/DodatniUlazniParametri.txt

Opciono zadaj kupu ručno:  --wx 6412550 --wy 4970870 --wz 195 --k 150 --ugao 37

Izlaz:
    buvac_presjek.png   — 3D prikaz terena + kupe + presjeka i tlocrt
    ispis zapremine, površine osnove, broja petlji + sanity provjere
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from geometrija_v2 import Kupa, Teren, presek_kupe_i_terena


# ---------------------------------------------------------------------------
# Učitavanje ulaznih fajlova (isti formati kao loaders.py / MATLAB)
# ---------------------------------------------------------------------------

def ucitaj_teren(putanja: str) -> np.ndarray:
    """Teren: CSV bez zaglavlja, kolone X, Y, Z."""
    xyz = np.loadtxt(putanja, delimiter=",")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Teren mora imati 3 kolone (X,Y,Z): {putanja}")
    return xyz


def ucitaj_granice(putanja: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Granica zone: 1. linija min (X,Y,Z), 2. linija max, ostalo poligon."""
    linije = [l.strip() for l in open(putanja, encoding="utf-8")
              if l.strip()]
    podaci = np.array([[float(v) for v in l.split(",")] for l in linije])
    bbox_min, bbox_max = podaci[0], podaci[1]
    poligon = podaci[2:, :2]
    return bbox_min, bbox_max, poligon


def ucitaj_centar_masa(putanja: str) -> np.ndarray:
    return np.loadtxt(putanja, delimiter=",").ravel()


def ucitaj_parametre(putanja: str) -> dict:
    """DodatniUlazniParametri.txt: linije '%% opis' pa vrijednost."""
    vrijednosti = []
    for l in open(putanja, encoding="utf-8"):
        l = l.strip()
        if l and not l.startswith("%"):
            vrijednosti.append(float(l))
    kljucevi = ["mnv", "broj_generacija", "uslov_distance"]
    return dict(zip(kljucevi, vrijednosti))


def nadji_fajl(zadato: str | None, *kandidati: str) -> str:
    """Vrati zadatu putanju ili prvi postojeći kandidat."""
    if zadato:
        if not os.path.exists(zadato):
            sys.exit(f"Fajl ne postoji: {zadato}")
        return zadato
    for k in kandidati:
        if os.path.exists(k):
            return k
    sys.exit(f"Nije pronađen nijedan od: {kandidati} — zadaj putanju argumentom.")


# ---------------------------------------------------------------------------
# Automatski izbor lokacije kupe (ako nije zadata)
# ---------------------------------------------------------------------------

def auto_lokacija(teren: Teren, poligon: np.ndarray,
                  odmak_od_ruba: float = 500.0) -> tuple[float, float, float]:
    """Bira tačku UNUTAR interesne zone, dovoljno daleko od ruba, na
    relativno visokom terenu (da kupa lijepo 'sjedne' i presjek bude vidljiv).

    Vraća (wx, wy, z_terena_u_toj_tački).
    """
    from matplotlib.path import Path as MplPath

    x0, x1, y0, y1 = teren.xy_granice
    gx = np.linspace(x0, x1, 140)
    gy = np.linspace(y0, y1, 140)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX.ravel(), GY.ravel()])

    poly = poligon
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    path = MplPath(poly)
    unutra = path.contains_points(pts)

    # udaljenost od ruba poligona — grubo: min rastojanje do tjemena/segmenata
    # (za izbor lokacije dovoljna je udaljenost do tjemena gusto uzorkovanog ruba)
    rub = []
    for a, b in zip(poly[:-1], poly[1:]):
        t = np.linspace(0, 1, 25)[:, None]
        rub.append(a + t * (b - a))
    rub = np.vstack(rub)
    from scipy.spatial import cKDTree
    d_rub, _ = cKDTree(rub).query(pts)

    kandidati = unutra & (d_rub > odmak_od_ruba)
    if not np.any(kandidati):
        kandidati = unutra
    z = teren.z(pts[kandidati, 0], pts[kandidati, 1])
    najbolji = np.argmax(z)             # najviši teren → kupa na uzvišenju
    wx, wy = pts[kandidati][najbolji]
    return float(wx), float(wy), float(z[najbolji])


# ---------------------------------------------------------------------------
# Glavni tok
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Presjek kupe i stvarnog terena Buvac")
    ap.add_argument("--teren")
    ap.add_argument("--granice")
    ap.add_argument("--cm")
    ap.add_argument("--params")
    ap.add_argument("--wx", type=float)
    ap.add_argument("--wy", type=float)
    ap.add_argument("--wz", type=float)
    ap.add_argument("--k", type=float, default=150.0)
    ap.add_argument("--ugao", type=float, default=37.0)
    ap.add_argument("--profil", default="matlab", choices=["matlab", "krug"])
    ap.add_argument("--rezolucija", type=int, default=384)
    ap.add_argument("--izlaz", default="buvac_presjek.png")
    a = ap.parse_args()

    f_teren = nadji_fajl(a.teren, "podaci/001-Teren-3-Buvac.txt",
                         "001-Teren-3-Buvac.txt")
    f_gran = nadji_fajl(a.granice, "podaci/001GranicaZonaBuvac.txt",
                        "001GranicaZonaBuvac.txt")
    f_cm = nadji_fajl(a.cm, "podaci/001CentarMasaBuvac.txt",
                      "001CentarMasaBuvac.txt")
    f_par = nadji_fajl(a.params, "podaci/DodatniUlazniParametri.txt",
                       "DodatniUlazniParametri.txt")

    print("=" * 70)
    print("UČITAVANJE STVARNIH PODATAKA")
    print("=" * 70)

    xyz = ucitaj_teren(f_teren)
    print(f"  Teren:        {len(xyz):,} tačaka   "
          f"Z ∈ [{xyz[:, 2].min():.1f}, {xyz[:, 2].max():.1f}] m")

    bbox_min, bbox_max, poligon = ucitaj_granice(f_gran)
    print(f"  Zona:         poligon sa {len(poligon)} tjemena, "
          f"bbox Z ∈ [{bbox_min[2]:.0f}, {bbox_max[2]:.0f}]")

    cm = ucitaj_centar_masa(f_cm)
    print(f"  Centar masa:  ({cm[0]:.0f}, {cm[1]:.0f}, {cm[2]:.0f})")

    par = ucitaj_parametre(f_par)
    print(f"  Parametri:    mnv = {par['mnv']:.0f} m, "
          f"uslov distance = {par.get('uslov_distance', 0):.0f} m")

    t0 = time.perf_counter()
    teren = Teren.iz_tacaka(xyz)        # gradi se JEDNOM
    print(f"  Interpolator terena izgrađen za {time.perf_counter()-t0:.2f} s "
          f"(radi se jednom, dijeli se kroz sve evaluacije)")

    # --- kupa ---------------------------------------------------------------
    if a.wx is not None and a.wy is not None:
        wx, wy = a.wx, a.wy
        z_tu = float(teren.z(wx, wy))
    else:
        wx, wy, z_tu = auto_lokacija(teren, poligon)
        print(f"\n  Auto-lokacija kupe: ({wx:.0f}, {wy:.0f}), "
              f"teren tu = {z_tu:.1f} m")

    wz = a.wz if a.wz is not None else z_tu + 40.0
    kupa = Kupa(wx=wx, wy=wy, wz=wz, k=a.k, ugao=a.ugao, profil=a.profil)
    print(f"  Kupa: vrh ({wx:.0f}, {wy:.0f}, {wz:.1f}), k = {a.k:.0f} m, "
          f"ugao = {a.ugao:.0f}°, profil = {a.profil}")

    # --- presjek + zapremina -------------------------------------------------
    print("\n" + "=" * 70)
    print("PRESJEK KUPE I TERENA")
    print("=" * 70)
    t0 = time.perf_counter()
    rez = presek_kupe_i_terena(kupa, teren, rezolucija=a.rezolucija,
                               rafiniranje=2)
    dt = time.perf_counter() - t0

    if not rez.ima_preseka:
        sys.exit("  Kupa nema presjeka s terenom — spusti wz ili pomjeri lokaciju.")

    print(f"  Zapremina tijela:        {rez.zapremina:>14,.0f} m³")
    print(f"  Površina osnove:         {rez.povrsina_osnove:>14,.0f} m²")
    print(f"  Broj presječnih petlji:  {rez.broj_petlji}")
    print(f"  Max debljina nasipa:     {rez.max_debljina:>10.1f} m")
    print(f"  Procjena num. greške:    {rez.procjena_greske:>14,.0f} m³ "
          f"({rez.procjena_greske / rez.zapremina * 100:.4f} %)")
    print(f"  Vrijeme evaluacije:      {dt*1000:>10.1f} ms")

    # --- sanity provjere ------------------------------------------------------
    print("\n  Provjere:")
    ok = True

    def check(naziv, uslov):
        nonlocal ok
        print(f"    {'✓' if uslov else '✗'} {naziv}")
        ok = ok and uslov

    check("zapremina > 0 i konačna",
          0 < rez.zapremina < 1e12 and np.isfinite(rez.zapremina))
    check("bar jedna zatvorena presječna petlja", rez.broj_petlji >= 1)
    kont = rez.kontura_glavna
    z_na_konturi = teren.z(kont[:, 0], kont[:, 1])
    check("kontura leži na terenu (|Δz| < 0.5 m)",
          bool(np.all(np.abs(kont[:, 2] - z_na_konturi) < 0.5)))
    check("max debljina ≤ visina vrha iznad najniže tačke terena",
          rez.max_debljina <= (wz - teren.z_min) + 1e-6)
    check("numerička greška < 0.5 % zapremine",
          rez.procjena_greske < 0.005 * rez.zapremina)

    # --- vizuelizacija --------------------------------------------------------
    print("\n  Crtanje...")
    x0, x1, y0, y1 = rez.granice_racuna
    # prošireni prozor da se vidi okolina
    pad = 0.25 * (x1 - x0)
    vx0, vx1, vy0, vy1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad

    n = 220
    GX, GY = np.meshgrid(np.linspace(vx0, vx1, n), np.linspace(vy0, vy1, n))
    ZT = teren.z(GX, GY)
    ZK = kupa.z(GX, GY)
    tijelo = np.where(ZK > ZT, ZK, np.nan)

    fig = plt.figure(figsize=(16, 7))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(GX, GY, ZT, cmap="terrain", alpha=0.9,
                    linewidth=0, antialiased=True)
    ax.plot_surface(GX, GY, tijelo, color="peru", alpha=0.6, linewidth=0)
    for kont_i in rez.konture:
        ax.plot(kont_i[:, 0], kont_i[:, 1], kont_i[:, 2] + 0.5, "r-", lw=2)
    ax.set_title(f"Buvac — kupa na stvarnom terenu\n"
                 f"V = {rez.zapremina:,.0f} m³, "
                 f"petlji presjeka: {rez.broj_petlji}")
    ax.set_box_aspect((1, (vy1 - vy0) / (vx1 - vx0), 0.30))
    ax.view_init(elev=40, azim=-55)

    ax2 = fig.add_subplot(1, 2, 2)
    cf = ax2.contourf(GX, GY, ZT, levels=28, cmap="terrain")
    plt.colorbar(cf, ax=ax2, label="kota terena (m)")
    p = poligon if np.allclose(poligon[0], poligon[-1]) \
        else np.vstack([poligon, poligon[0]])
    ax2.plot(p[:, 0], p[:, 1], "k--", lw=1.2, label="interesna zona")
    kx, ky = kupa.gornja_kontura()
    ax2.plot(kx, ky, "b-", lw=1.3, label="gornji plato kupe")
    for i, kont_i in enumerate(rez.konture):
        ax2.plot(kont_i[:, 0], kont_i[:, 1], "r-", lw=2,
                 label="presjek kupa–teren" if i == 0 else None)
    ax2.plot(cm[0], cm[1], "m*", ms=14, label="centar masa")
    ax2.set_xlim(vx0, vx1); ax2.set_ylim(vy0, vy1)
    ax2.set_aspect("equal")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_title("Tlocrt: presječne konture na stvarnom terenu")

    plt.tight_layout()
    plt.savefig(a.izlaz, dpi=130)
    print(f"  Sačuvano: {a.izlaz}")

    print("\n" + "=" * 70)
    print("SVE PROVJERE PROŠLE" if ok else "NEKE PROVJERE PALE")
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
