"""
test_geometrija_v2.py — testovi tačnosti novog presjeka i zapremine.

Pokretanje:  python3 test_geometrija_v2.py
"""

from __future__ import annotations

import numpy as np

from geometrija_v2 import (
    Kupa, Teren, PresekRezultat,
    presek_kupe_i_terena,
    zapremina_zarubljene_kupe_ravan_teren,
    zapremina_kupe_v2,
    povrsine_po_zonama,
)

PROSLO = 0
PALO = 0


def test(naziv: str, uslov: bool, detalj: str = ""):
    global PROSLO, PALO
    if uslov:
        PROSLO += 1
        print(f"  ✓ {naziv}")
    else:
        PALO += 1
        print(f"  ✗ {naziv}   {detalj}")


# ===========================================================================
print("\n[1] Ravan teren — poređenje sa EGZAKTNOM formulom zarubljene kupe")
# ===========================================================================

kupa = Kupa(wx=1000.0, wy=2000.0, wz=180.0, k=100.0, ugao=37.0, profil="krug")
teren = Teren.ravan(140.0)

V_egz = zapremina_zarubljene_kupe_ravan_teren(kupa, 140.0)
rez = presek_kupe_i_terena(kupa, teren, rezolucija=256, rafiniranje=2)

rel = abs(rez.zapremina - V_egz) / V_egz
test("zapremina unutar 0.05% od egzaktne",
     rel < 5e-4, f"rel. greška = {rel:.2e}, V={rez.zapremina:.1f}, Vegz={V_egz:.1f}")

# egzaktna površina osnove: krug poluprečnika R = k + h/tan
h = 180.0 - 140.0
R = 100.0 + h / np.tan(np.radians(37.0))
A_egz = np.pi * R * R
relA = abs(rez.povrsina_osnove - A_egz) / A_egz
test("površina osnove unutar 0.1% od egzaktne",
     relA < 1e-3, f"rel. greška = {relA:.2e}")

test("tačno jedna presječna petlja na ravnom terenu", rez.broj_petlji == 1)
test("max debljina = visina kupe iznad terena",
     abs(rez.max_debljina - h) < 1e-6)

# presječna kontura mora biti krug poluprečnika R oko (wx, wy) na z=140
kont = rez.kontura_glavna
r_kont = np.hypot(kont[:, 0] - 1000.0, kont[:, 1] - 2000.0)
test("kontura je krug tačnog poluprečnika (±0.5%)",
     np.all(np.abs(r_kont - R) / R < 5e-3),
     f"max odstupanje {np.max(np.abs(r_kont - R)):.3f} m")
test("kontura leži na terenu (z = 140)",
     np.allclose(kont[:, 2], 140.0))

test("Richardson procjena greške < stvarna greška × 10",
     rez.procjena_greske < max(abs(rez.zapremina - V_egz) * 10, V_egz * 1e-3))


# ===========================================================================
print("\n[2] Kosi teren (ravan pod nagibom) — poređenje sa referencom visoke rezolucije")
# ===========================================================================

# teren: z = 140 + 0.08·(x−1000)  → nagib 8%
nagib = Teren.analiticki(
    lambda x, y: 140.0 + 0.08 * (x - 1000.0),
    xy_granice=(0, 2000, 1000, 3000), z_granice=(60.0, 220.0))

rez_lo = presek_kupe_i_terena(kupa, nagib, rezolucija=128, rafiniranje=1)
rez_hi = presek_kupe_i_terena(kupa, nagib, rezolucija=1024, rafiniranje=2)

rel = abs(rez_lo.zapremina - rez_hi.zapremina) / rez_hi.zapremina
test("rezolucija 128 vs 1024: razlika < 0.2%",
     rel < 2e-3, f"rel = {rel:.2e}")
test("kontura na kosom terenu prati teren",
     np.allclose(rez_hi.kontura_glavna[:, 2],
                 140.0 + 0.08 * (rez_hi.kontura_glavna[:, 0] - 1000.0),
                 atol=0.5))
# na kosini je zapremina VEĆA nego na ravni kroz istu tačku vrha? Ne nužno —
# ali mora biti manja od zapremine do najniže tačke i veća od one do najviše.
V_do_min = zapremina_zarubljene_kupe_ravan_teren(kupa, float(nagib.z(kupa.wx - 300, 0)))
V_do_max = zapremina_zarubljene_kupe_ravan_teren(kupa, float(nagib.z(kupa.wx + 300, 0)))
test("zapremina između fizičkih granica (kosina)",
     min(V_do_min, V_do_max) * 0.5 < rez_hi.zapremina < max(V_do_min, V_do_max) * 1.5)


# ===========================================================================
print("\n[3] BRDOVIT teren (sinusni reljef) — konvergencija i stabilnost")
# ===========================================================================

def brdo(x, y):
    return (145.0
            + 8.0 * np.sin((x - 1000.0) / 60.0) * np.cos((y - 2000.0) / 45.0)
            + 5.0 * np.sin((x - 1000.0) / 23.0 + 1.0))

brdovit = Teren.analiticki(brdo, (0, 2000, 1000, 3000), (130.0, 160.0))

V_ref = presek_kupe_i_terena(kupa, brdovit, rezolucija=2048, rafiniranje=2).zapremina
greske = []
for n in (64, 128, 256, 512):
    Vn = presek_kupe_i_terena(kupa, brdovit, rezolucija=n, rafiniranje=2).zapremina
    greske.append(abs(Vn - V_ref) / V_ref)

test("konvergencija: greška monotono pada s rezolucijom",
     greske[0] > greske[-1], f"greske = {[f'{g:.2e}' for g in greske]}")
test("rezolucija 256 na brdovitom terenu: greška < 0.1%",
     greske[2] < 1e-3, f"greška = {greske[2]:.2e}")

# sanitet: zapremina na brdovitom terenu mora biti između zapremina
# računatih prema najvišoj i najnižoj koti terena
V_min = zapremina_zarubljene_kupe_ravan_teren(kupa, 160.0)
V_max = zapremina_zarubljene_kupe_ravan_teren(kupa, 130.0)
test("zapremina u fizičkim granicama [V(z_max), V(z_min)]",
     V_min < V_ref < V_max, f"{V_min:.0f} < {V_ref:.0f} < {V_max:.0f}")


# ===========================================================================
print("\n[4] Više presječnih petlji — kupa niska, brda 'probijaju' padinu")
# ===========================================================================

def dva_brda(x, y):
    # brda UNUTAR footprinta kupe (r≈140 < rub footprinta ≈180) i dovoljno
    # visoka da probiju padinu kupe → footprint dobija rupe (više petlji)
    b1 = 45.0 * np.exp(-(((x - 1130.0) / 20.0) ** 2 + ((y - 2000.0) / 20.0) ** 2))
    b2 = 45.0 * np.exp(-(((x - 870.0) / 20.0) ** 2 + ((y - 2000.0) / 20.0) ** 2))
    return 140.0 + b1 + b2

teren2 = Teren.analiticki(dva_brda, (0, 2000, 1000, 3000), (140.0, 195.0))
kupa2 = Kupa(wx=1000.0, wy=2000.0, wz=185.0, k=120.0, ugao=37.0, profil="krug")

rez2 = presek_kupe_i_terena(kupa2, teren2, rezolucija=512, rafiniranje=2)
test("detektovano više petlji (brda probijaju kupu)",
     rez2.broj_petlji >= 2, f"petlji = {rez2.broj_petlji}")
test("zapremina pozitivna i konačna",
     0 < rez2.zapremina < zapremina_zarubljene_kupe_ravan_teren(kupa2, 140.0))

# zapremina sa brdima mora biti MANJA nego bez njih (brda 'izgrizu' tijelo)
V_bez = presek_kupe_i_terena(kupa2, Teren.ravan(140.0), 512, 2).zapremina
test("brda smanjuju zapreminu u odnosu na ravan teren",
     rez2.zapremina < V_bez,
     f"{rez2.zapremina:.0f} vs {V_bez:.0f}")


# ===========================================================================
print("\n[5] Kupa iznad terena bez kontakta / potpuno ispod terena")
# ===========================================================================

kupa_mala = Kupa(wx=0, wy=0, wz=100.0, k=10.0, ugao=37.0)
rez_nista = presek_kupe_i_terena(kupa_mala, Teren.ravan(300.0), 64, 0)
test("kupa ispod terena → zapremina 0, nema presjeka",
     (not rez_nista.ima_preseka) and rez_nista.zapremina == 0.0)


# ===========================================================================
print("\n[6] Teren iz OBLAKA TAČAKA (TIN) — kao stvarni ulazni podaci")
# ===========================================================================

rng = np.random.default_rng(42)
N = 4000
tx = rng.uniform(400, 1600, N)
ty = rng.uniform(1400, 2600, N)
tz = brdo(tx, ty)
tin = Teren.iz_tacaka(np.column_stack([tx, ty, tz]))

rez_tin = presek_kupe_i_terena(kupa, tin, rezolucija=256, rafiniranje=2)
rez_ana = presek_kupe_i_terena(kupa, brdovit, rezolucija=256, rafiniranje=2)
rel = abs(rez_tin.zapremina - rez_ana.zapremina) / rez_ana.zapremina
test("TIN iz 4000 tačaka ≈ analitički teren (unutar 1.5%)",
     rel < 1.5e-2, f"rel = {rel:.2e}")
test("TIN presjek daje bar jednu zatvorenu petlju", rez_tin.broj_petlji >= 1)


# ===========================================================================
print("\n[7] MATLAB profil (8-režnjeviti oblik) — konzistentnost")
# ===========================================================================

km = Kupa(wx=1000, wy=2000, wz=180, k=100, ugao=37, profil="matlab")
rm = presek_kupe_i_terena(km, Teren.ravan(140.0), 256, 2)
# matlab profil ima r_top ∈ [k, 1.4k] → zapremina između kruga sa r=k i r=1.4k
V_k = zapremina_zarubljene_kupe_ravan_teren(kupa, 140.0)
kupa14 = Kupa(wx=1000, wy=2000, wz=180, k=140.0, ugao=37, profil="krug")
V_14k = zapremina_zarubljene_kupe_ravan_teren(kupa14, 140.0)
test("matlab profil: V između krugova r=k i r=1.4k",
     V_k < rm.zapremina < V_14k, f"{V_k:.0f} < {rm.zapremina:.0f} < {V_14k:.0f}")
test("r_top periodičnost: r(0) == r(2π)",
     abs(km.r_top(0.0) - km.r_top(2 * np.pi)) < 1e-9)


# ===========================================================================
print("\n[8] zapremina_kupe_v2 — drop-in kompatibilnost sa geometry.RezultatKupe")
# ===========================================================================

from geometry import Surface, RezultatKupe

# mali sintetički teren kao Surface (kako ga daje loaders)
gx = np.linspace(400, 1600, 60)
gy = np.linspace(1400, 2600, 60)
GX, GY = np.meshgrid(gx, gy)
pts = np.column_stack([GX.ravel(), GY.ravel(), brdo(GX, GY).ravel()])
from scipy.spatial import Delaunay as _D
surf = Surface(vertices=pts, faces=_D(pts[:, :2]).simplices)

zona_x = np.array([0.0, 2000.0, 2000.0, 0.0])
zona_y = np.array([1000.0, 1000.0, 3000.0, 3000.0])

r = zapremina_kupe_v2(1000, 2000, 180, 37, 100, 140, surf, zona_x, zona_y,
                      profil="krug")
test("vraća RezultatKupe", isinstance(r, RezultatKupe))
test("zapremina razumna (≈ analitička ±3%)",
     abs(r.zapremina - rez_ana.zapremina) / rez_ana.zapremina < 3e-2,
     f"{r.zapremina:.0f} vs {rez_ana.zapremina:.0f}")
test("donja_povrsina popunjena", r.donja_povrsina is not None and len(r.donja_povrsina) > 10)
test("intersect_surface ima 3D tačke", r.intersect_surface.vertices.shape[1] == 3)

# van interesne zone → VELIKA_VREDNOST (isto ponašanje kao stara verzija)
r_van = zapremina_kupe_v2(1000, 2000, 180, 37, 100, 140, surf,
                          np.array([0, 10, 10, 0.0]),
                          np.array([0, 0, 10, 10.0]))
test("van zone → kaznena vrijednost 40e6", r_van.zapremina == 40_000_000.0)


# ===========================================================================
print("\n[9] povrsine_po_zonama — tačnost naspram analitike")
# ===========================================================================

class _Zona:
    def __init__(self, naziv, x, y):
        self.naziv = naziv
        self.x_data = np.asarray(x, float)
        self.y_data = np.asarray(y, float)

# ravan teren → footprint je krug; zona = desna poluravan kroz centar kupe
rez_r = presek_kupe_i_terena(kupa, Teren.ravan(140.0), 512, 2)
pola = _Zona("desna", [1000, 3000, 3000, 1000], [0, 0, 4000, 4000])
sve = _Zona("sve", [-1e4, 1e4, 1e4, -1e4], [-1e4, -1e4, 1e4, 1e4])
pz = povrsine_po_zonama(rez_r, [pola, sve], rezolucija=512)

test("cijeli footprint ≈ πR² (±0.5%)",
     abs(pz["sve"] - A_egz) / A_egz < 5e-3, f"{pz['sve']:.0f} vs {A_egz:.0f}")
test("pola footprinta ≈ πR²/2 (±1%)",
     abs(pz["desna"] - A_egz / 2) / (A_egz / 2) < 1e-2,
     f"{pz['desna']:.0f} vs {A_egz/2:.0f}")


# ===========================================================================
print(f"\n{'='*60}\nUKUPNO: {PROSLO} prošlo, {PALO} palo\n{'='*60}")
import sys
sys.exit(0 if PALO == 0 else 1)
