"""
validacija_v2.py — poređenje STARE (geometry.py, ConvexHull) i NOVE
(geometrija_v2.py, visinska polja) metode presjeka i zapremine.

Pokretanje:  python3 validacija_v2.py

Ispisuje:
  • tačnost obje metode naspram egzaktne / referentne vrijednosti
    na ravnom, kosom i brdovitom terenu;
  • brzinu jedne evaluacije (relevantno za GA petlju).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial import Delaunay

from geometry import Surface, zapremina_kupe
from geometrija_v2 import (
    Kupa, Teren, presek_kupe_i_terena,
    zapremina_zarubljene_kupe_ravan_teren, zapremina_mesh_boolean,
)

# ---------------------------------------------------------------------------
# Parametri test kupe (tipični za Buvac): vrh 180 m, k = 100 m, ugao 37°
# ---------------------------------------------------------------------------
WX, WY, WZ, K, UGAO, MNV = 1000.0, 2000.0, 180.0, 100.0, 37.0, 140.0
ZONA_X = np.array([0.0, 2000.0, 2000.0, 0.0])
ZONA_Y = np.array([1000.0, 1000.0, 3000.0, 3000.0])


def teren_surface(fn, n=80):
    """Surface objekat terena (za staru metodu) iz funkcije z = fn(x, y)."""
    gx = np.linspace(400, 1600, n)
    gy = np.linspace(1400, 2600, n)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX.ravel(), GY.ravel(), fn(GX, GY).ravel()])
    return Surface(vertices=pts, faces=Delaunay(pts[:, :2]).simplices)


def uporedi(naziv, fn_terena, V_ref, opis_ref):
    print(f"\n{'='*72}\n{naziv}\n{'='*72}")

    surf = teren_surface(fn_terena)
    teren_v2 = Teren.iz_surface(surf)   # ISTI podaci za obje metode
    kupa = Kupa(WX, WY, WZ, K, UGAO, profil="krug")

    # --- stara metoda ------------------------------------------------------
    t0 = time.perf_counter()
    rez_stara = zapremina_kupe(WX, WY, WZ, UGAO, K, MNV, surf, ZONA_X, ZONA_Y)
    t_stara = time.perf_counter() - t0

    # --- nova metoda (teren izgrađen JEDNOM, kao u GA) ---------------------
    t0 = time.perf_counter()
    rez_nova = presek_kupe_i_terena(kupa, teren_v2, rezolucija=192, rafiniranje=1)
    t_nova = time.perf_counter() - t0

    def red(ime, V, t):
        if V >= 39_000_000:
            print(f"  {ime:<28} NEUSPJEH (kaznena vrijednost)")
            return
        gr = (V - V_ref) / V_ref * 100
        print(f"  {ime:<28} V = {V:>14,.0f} m³   greška {gr:>+7.2f}%   t = {t*1000:7.1f} ms")

    print(f"  Referenca ({opis_ref}):        V = {V_ref:>14,.0f} m³")
    red("STARA (ConvexHull)", rez_stara.zapremina, t_stara)
    red("NOVA  (visinska polja)", rez_nova.zapremina, t_nova)
    print(f"  Nova: petlji presjeka = {rez_nova.broj_petlji}, "
          f"površina osnove = {rez_nova.povrsina_osnove:,.0f} m², "
          f"procjena greške = {rez_nova.procjena_greske:,.0f} m³")
    return rez_stara, rez_nova


# ===========================================================================
# 1. RAVAN TEREN — egzaktna formula postoji
# ===========================================================================
kupa_ref = Kupa(WX, WY, WZ, K, UGAO, profil="krug")
V_egz = zapremina_zarubljene_kupe_ravan_teren(kupa_ref, MNV)
uporedi("1. RAVAN TEREN (z = 140)",
        lambda x, y: np.full(np.shape(x), 140.0), V_egz, "egzaktna formula")

# ===========================================================================
# 2. KOSI TEREN — referenca: nova metoda na rezoluciji 2048
# ===========================================================================
kos = lambda x, y: 140.0 + 0.08 * (x - 1000.0)
t_kos = Teren.analiticki(kos, (0, 2000, 1000, 3000), (60.0, 220.0))
V_ref_kos = presek_kupe_i_terena(kupa_ref, t_kos, 2048, 2).zapremina
uporedi("2. KOSI TEREN (nagib 8%)", kos, V_ref_kos, "rezolucija 2048")

# ===========================================================================
# 3. BRDOVIT TEREN — glavni slučaj koji je motivisao novu metodu
# ===========================================================================
def brdo(x, y):
    return (145.0
            + 8.0 * np.sin((x - 1000.0) / 60.0) * np.cos((y - 2000.0) / 45.0)
            + 5.0 * np.sin((x - 1000.0) / 23.0 + 1.0))

t_brdo = Teren.analiticki(brdo, (0, 2000, 1000, 3000), (130.0, 160.0))
V_ref_brdo = presek_kupe_i_terena(kupa_ref, t_brdo, 2048, 2).zapremina
uporedi("3. BRDOVIT TEREN (sinusni reljef ±13 m)", brdo, V_ref_brdo,
        "rezolucija 2048")

V_mesh = zapremina_mesh_boolean(kupa_ref, t_brdo)
if V_mesh is not None:
    print(f"  Nezavisna mesh-boolean verifikacija (trimesh): "
          f"V = {V_mesh:,.0f} m³ ({(V_mesh - V_ref_brdo)/V_ref_brdo*100:+.2f}%)")

# ===========================================================================
# 4. EKSTREMNO NEPRAVILAN TEREN — brda probijaju kupu (više petlji)
# ===========================================================================
def dva_brda(x, y):
    b1 = 45.0 * np.exp(-(((x - 1130.0) / 20.0) ** 2 + ((y - 2000.0) / 20.0) ** 2))
    b2 = 45.0 * np.exp(-(((x - 870.0) / 20.0) ** 2 + ((y - 2000.0) / 20.0) ** 2))
    return 140.0 + b1 + b2

t_2b = Teren.analiticki(dva_brda, (0, 2000, 1000, 3000), (140.0, 185.1))
kupa_2b = Kupa(WX, WY, 185.0, 120.0, UGAO, profil="krug")
V_ref_2b = presek_kupe_i_terena(kupa_2b, t_2b, 2048, 2).zapremina

surf_2b = teren_surface(dva_brda, n=100)
rez_st = zapremina_kupe(WX, WY, 185.0, UGAO, 120.0, MNV, surf_2b, ZONA_X, ZONA_Y)
rez_nv = presek_kupe_i_terena(kupa_2b, Teren.iz_surface(surf_2b), 256, 2)

print(f"\n{'='*72}\n4. DVA BRDA PROBIJAJU KUPU (footprint sa rupama)\n{'='*72}")
print(f"  Referenca (rez. 2048):        V = {V_ref_2b:>14,.0f} m³")
for ime, V in [("STARA (ConvexHull)", rez_st.zapremina),
               ("NOVA  (visinska polja)", rez_nv.zapremina)]:
    gr = (V - V_ref_2b) / V_ref_2b * 100
    print(f"  {ime:<28} V = {V:>14,.0f} m³   greška {gr:>+7.2f}%")
print(f"  Nova metoda detektuje {rez_nv.broj_petlji} presječne petlje "
      f"(vanjska granica + rupe oko brda)")

# ===========================================================================
# 5. BENCHMARK — brzina po evaluaciji, kao u GA petlji
# ===========================================================================
print(f"\n{'='*72}\n5. BENCHMARK (100 evaluacija — simulacija GA petlje)\n{'='*72}")

surf_b = teren_surface(brdo, n=100)
teren_b = Teren.iz_surface(surf_b)   # gradi se JEDNOM
rng = np.random.default_rng(7)
wzs = rng.uniform(165, 195, 100)
ks = rng.uniform(70, 130, 100)

t0 = time.perf_counter()
for wz_i, k_i in zip(wzs, ks):
    zapremina_kupe(WX, WY, wz_i, UGAO, k_i, MNV, surf_b, ZONA_X, ZONA_Y)
t_st = time.perf_counter() - t0

t0 = time.perf_counter()
for wz_i, k_i in zip(wzs, ks):
    kupa_i = Kupa(WX, WY, wz_i, k_i, UGAO, profil="krug")
    presek_kupe_i_terena(kupa_i, teren_b, rezolucija=192, rafiniranje=1)
t_nv = time.perf_counter() - t0

print(f"  STARA:  {t_st:.2f} s ukupno  ({t_st*10:.1f} ms/evaluaciji)")
print(f"  NOVA:   {t_nv:.2f} s ukupno  ({t_nv*10:.1f} ms/evaluaciji)")
print(f"  Ubrzanje: ×{t_st/t_nv:.1f}")
print()
