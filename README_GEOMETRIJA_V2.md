# geometrija_v2 — tačan presjek kupe i (nepravilnog) terena

Novi modul koji zamjenjuje proračun presjeka i zapremine iz `geometry.py`
(`surface_intersection` + `zapremina_kupe`). Motivacija: stara metoda daje
**pogrešnu zapreminu na svakom terenu koji nije savršeno ravan**, a i na
ravnom terenu griješi zbog grube diskretizacije kupe.

---

## Zašto je stara metoda netačna

Izmjerene greške stare metode (`validacija_v2.py`, referenca = egzaktna
formula ili rezolucija 2048 + nezavisna mesh-boolean verifikacija):

| Teren | STARA (ConvexHull) | NOVA (visinska polja) |
|---|---|---|
| Ravan (z = 140) | **+14.4 %** | −0.01 % |
| Kosina 8 % | **−25.9 %** | −0.01 % |
| Brdovit (sinusni reljef ±13 m) | **+29.0 %** | −0.10 % |
| Dva brda probijaju kupu | **+18.8 %** | −0.07 % |

Uzroci grešaka stare metode:

1. **ConvexHull kao zapremina.** Tijelo odlagališta je odozdo ograničeno
   *terenom* — konkavnom, nepravilnom površinom. ConvexHull "popuni" sve
   doline ispod kupe i ne vidi reljef, pa sistematski precjenjuje na
   brdovitom terenu.
2. **Kupa od 18 tačaka.** Gornja i donja kontura su imale po 9 tačaka,
   pa presječna kriva praktično nije pratila stvarni oblik.
3. **Möller presjek O(N×M) po evaluaciji.** Cijeli teren se obrađivao
   ispočetka za svaku jedinku GA, a rezultat su bili nepovezani segmenti —
   ne zatvorena kontura, ne footprint, ne teren ispod kupe.
4. **Više petlji presjeka** (padina kupe presiječe više brda) uopšte nije
   bilo modelovano.

## Novi pristup: visinska polja

I kupa i teren su 2.5D površine `z = f(x, y)`. Definišemo razliku:

```
d(x, y) = z_kupa(x, y) − z_teren(x, y)
```

* **Presječna kriva** = nivo-kriva `d = 0` (marching squares, `contourpy`).
  Automatski daje zatvorene, orijentisane petlje — i više njih kad je teren
  nepravilan (vanjska granica + "rupe" oko brda koja probijaju kupu).
* **Zapremina** = `∬ max(0, d) dA` — *matematički tačna* definicija tijela
  između kupe i terena, bez ikakvih pretpostavki o konveksnosti.
  Integracija: trapezno pravilo na mreži + **adaptivno rafiniranje** (4×4,
  rekurzivno) ćelija kroz koje prolazi granica `d = 0`.
* **Površina osnove** = mjera skupa `{d > 0}` (stvarni footprint na terenu).
* **Procjena greške** = Richardsonova ekstrapolacija (razlika integrala na
  mreži koraka h i 2h).

Kupa je **analitička** (nema mesh-a): plato na visini `wz` poluprečnika
`r_top(θ)` i padina pod uglom `ugao`. Podržana su dva profila:
`"krug"` (pravilna zarubljena kupa — za nju postoje egzaktne formule) i
`"matlab"` (originalni 8-režnjeviti oblik `[s, k, u, k, …]`, s = 1.4k,
u = 1.25k, sa kontinualnom interpolacijom po uglu).

Teren se gradi **jednom** (`Delaunay` + barycentrična linearna interpolacija,
van omotača najbliži susjed) i dijeli kroz sve GA evaluacije — otud i
ubrzanje **×2.5** po evaluaciji uz višestruko veću tačnost (i to je na malom
terenu od 10 000 tačaka; na velikim terenima razlika raste, jer stara metoda
skalira sa brojem trouglova terena po *svakoj* evaluaciji, a nova ne).

## Verifikacija

* `test_geometrija_v2.py` — **28/28 testova**: egzaktna formula zarubljene
  kupe, konvergencija na kosom i brdovitom terenu, više petlji, TIN iz
  oblaka tačaka, MATLAB profil, drop-in kompatibilnost, površine po zonama.
* Nezavisna verifikacija: egzaktni **mesh-boolean** presjek čvrstih tijela
  (`trimesh` + `manifold3d`) slaže se sa grid metodom na **0.01 %** na
  brdovitom terenu.

## Upotreba

### Direktno (preporučeno za novi kod)

```python
from geometrija_v2 import Kupa, Teren, presek_kupe_i_terena

teren = Teren.iz_tacaka(xyz)              # (N, 3) — gradi se JEDNOM
kupa  = Kupa(wx=1000, wy=2000, wz=185, k=120, ugao=37, profil="matlab")

rez = presek_kupe_i_terena(kupa, teren, rezolucija=256, rafiniranje=2)

rez.zapremina         # m³ — tijelo između kupe i terena
rez.povrsina_osnove   # m² — footprint na terenu
rez.konture           # lista (K, 3) zatvorenih 3D petlji (za DXF/prikaz)
rez.broj_petlji       # > 1 kad brda probijaju kupu
rez.procjena_greske   # m³ — Richardsonova procjena numeričke greške
```

### Drop-in zamjena u postojećem kodu

`zapremina_kupe_v2()` ima isti potpis i vraća isti `RezultatKupe` kao
`geometry.zapremina_kupe` — dovoljno je zamijeniti import u
`ga_funkcije.py`, `ga_pokretac.py`, `main.py` i `app.py`:

```python
# from geometry import zapremina_kupe
from geometrija_v2 import zapremina_kupe_v2 as zapremina_kupe
```

Za maksimalnu brzinu u GA petlji, jednom napraviti
`teren_v2 = Teren.iz_surface(surface)` i proslijediti taj objekat umjesto
`Surface` — izbjegava se ponovna izgradnja Delaunay-a terena.

### Tačnije površine po ekonomskim zonama

`povrsine_po_zonama(rez, zone)` rasterizuje stvarni footprint (sa rupama)
po zonama — zamjena za ConvexHull aproksimaciju iz `ekonomija.py`.

## Parametri tačnost/brzina

| rezolucija | rafiniranje | tipična greška | vrijeme/evaluaciji* |
|---|---|---|---|
| 128 | 1 | < 0.3 % | ~5 ms |
| 192 | 1 | < 0.15 % | ~11 ms |
| 256 | 2 | < 0.05 % | ~25 ms |
| 512 | 2 | < 0.01 % | ~90 ms |

\* brdovit teren od 10 000 tačaka; `Teren` izgrađen unaprijed.
Za GA je 192/1 sasvim dovoljno; za finalni izvještaj koristiti 512/2.

## Fajlovi

| Fajl | Sadržaj |
|---|---|
| `geometrija_v2.py` | Novi modul (Teren, Kupa, presjek, zapremina, zone) |
| `test_geometrija_v2.py` | 28 testova tačnosti |
| `validacija_v2.py` | Poređenje stara vs nova metoda + benchmark |
| `demo_v2.py` | 3D vizuelizacija presjeka na brdovitom terenu |

## Zavisnosti

Obavezno: `numpy`, `scipy`, `contourpy` (dolazi uz matplotlib).
Opciono: `trimesh` + `manifold3d` — samo za `zapremina_mesh_boolean()`
(nezavisna verifikacija; ne koristi se u GA).
