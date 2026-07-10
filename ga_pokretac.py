"""
ga_pokretac.py  –  Korak 4: pokretanje genetskog algoritma

Zamjenjuje MATLAB poziv:
  ga(@funkcijaCiljaGenetskiAlgoritam, 4, [], [], [], [],
     [175 80 pointX pointY], [280 350 pointX pointY], nonlcon, opts)

Koristi scipy.optimize.differential_evolution — slobodan ekvivalent
MATLAB ga() iz Global Optimization Toolbox-a.

GA varijable (identično MATLAB-u):
  x[0] = wz   — visina vrha kupe  (175–280 m za Buvac)
  x[1] = k    — širina kupe       (80–350 m)
  x[2] = wx   — X koordinata      (fiksna — ista kao ulaz)
  x[3] = wy   — Y koordinata      (fiksna — ista kao ulaz)

MATLAB parametri → scipy ekvivalenti:
  PopulationSize  = 30    → popsize  (scipy množi sa N_var, koristimo maxiter)
  MaxGeneration   = N     → maxiter
  FunctionTolerance = 1e-7 → tol
  EliteCount = 2          → nema direktnog — handled by DE mutation/recombination
  CrossoverFraction = 0.8  → recombination=0.8
  mutation='adaptfeasible' → mutation=(0.5, 1.0) + strategy='best1bin'
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import differential_evolution, OptimizeResult

from ga_funkcije import GAKontekst, funkcija_cilja, get_bounds
from ekonomija import (
    distanca_od_centra_masa,
    ekonomska_cijena,
    racunaj_troskove,
)
from geometry import zapremina_kupe, unutar_interesne_zone


# ---------------------------------------------------------------------------
# Rezultat za jednu tačku
# ---------------------------------------------------------------------------

@dataclass
class RezultatTacke:
    """Kompletan izlaz GA optimizacije za jednu kandidat-tačku.

    Odgovara jednom redu u MATLAB FinalArray + ArrayZapremine.
    13 kolona iz headerFinal:
      Naziv_tacke, X_koordinata, Y_koordinata, Z_koordinata, K,
      Funkcija_cilja, Zapremina, Ugao, distanca, c1, c2, c3, Zone
    """
    naziv: str
    wx: float
    wy: float
    wz: float
    k: float
    f_vrednost: float
    zapremina: float
    ugao: float
    distanca: float
    c1: float
    c2: float
    c3: float
    zone: str
    unutar_zone: bool = True    # provjera unutarInteresneZone
    # Geometrija za DXF
    xx1: Optional[np.ndarray] = None   # gornja kontura X
    yy1: Optional[np.ndarray] = None   # gornja kontura Y
    zz1: Optional[np.ndarray] = None   # gornja kontura Z
    xx2: Optional[np.ndarray] = None   # donja kontura X
    yy2: Optional[np.ndarray] = None   # donja kontura Y
    zz2: Optional[np.ndarray] = None   # donja kontura Z

    def kao_red(self) -> list:
        """Vraća red za pandas DataFrame / Excel export."""
        return [
            self.naziv, self.wx, self.wy, self.wz, self.k,
            self.f_vrednost, self.zapremina, self.ugao,
            self.distanca, self.c1, self.c2, self.c3, self.zone,
        ]

    ZAGLAVLJE = [
        "Naziv_tacke", "X_koordinata", "Y_koordinata", "Z_koordinata",
        "K", "Funkcija_cilja", "Zapremina", "Ugao",
        "distanca", "c1", "c2", "c3", "Zone",
    ]


# ---------------------------------------------------------------------------
# GA opcije  (MATLAB opts ekvivalent)
# ---------------------------------------------------------------------------

@dataclass
class GAOpcije:
    """Parametri genetskog algoritma.

    MATLAB ekvivalent: optimoptions('ga', ...)
    """
    populacija: int = 30          # PopulationSize
    max_generacija: int = 3       # MaxGeneration  (iz DodatniUlazniParametri)
    tolerancija: float = 1e-7     # FunctionTolerance
    rekombinacija: float = 0.8    # CrossoverFraction
    mutacija: tuple = (0.5, 1.0)  # adaptfeasible aproksimacija
    strategija: str = "best1bin"  # MATLAB default selekcija
    seed: Optional[int] = None    # za reproducibilnost


# ---------------------------------------------------------------------------
# Optimizacija jedne tačke
# ---------------------------------------------------------------------------

def optimizuj_tacku(
    naziv: str,
    wx: float,
    wy: float,
    ctx: GAKontekst,
    opcije: GAOpcije,
    verzija: str = "buvac",
) -> Optional[RezultatTacke]:
    """Pokreće GA za jednu kandidat-tačku i vraća rezultat.

    MATLAB ekvivalent (petlja for i=1:size(listaTacaka)):
        [x, fval, exitflag, ...] = ga(@funkcijaCiljaGenetskiAlgoritam,
            4, [], [], [], [],
            [175 80 pointX pointY], [280 350 pointX pointY],
            nonlcon, opts)

    Args:
        naziv: ime tačke (npr. "point_3_1")
        wx, wy: koordinate kandidat-tačke (fiksirani)
        ctx:    GAKontekst sa svim parametrima
        opcije: GA parametri
        verzija: "buvac" ili "v1" za bounds

    Returns:
        RezultatTacke ako GA nađe rješenje (fval > 0),
        None ako GA ne konvergira ili fval <= 0
    """
    # Bounds izvedeni iz terena — radi za bilo koji teren
    import numpy as _np
    z_min = float(ctx.teren.vertices[:, 2].min()) if ctx.teren is not None else None
    z_max = float(ctx.teren.vertices[:, 2].max()) if ctx.teren is not None else None
    # Dijagonala zone interesa
    if ctx.zona_x is not None and len(ctx.zona_x) >= 2:
        dx = ctx.zona_x.max() - ctx.zona_x.min()
        dy = ctx.zona_y.max() - ctx.zona_y.min()
        zona_dijagonala = float(_np.sqrt(dx**2 + dy**2))
    else:
        zona_dijagonala = None
    lb, ub = get_bounds(wx, wy, z_min=z_min, z_max=z_max,
                        zona_dijagonala=zona_dijagonala, verzija=verzija)

    # Bounds lista za scipy: [(lb0,ub0), (lb1,ub1), ...]
    bounds = list(zip(lb, ub))

    # Wrapper koji prosleđuje kontekst
    def fitness(x: np.ndarray) -> float:
        return funkcija_cilja(x, ctx)

    try:
        rez: OptimizeResult = differential_evolution(
            fitness,
            bounds=bounds,
            maxiter=opcije.max_generacija,
            popsize=opcije.populacija,
            tol=opcije.tolerancija,
            recombination=opcije.rekombinacija,
            mutation=opcije.mutacija,
            strategy=opcije.strategija,
            seed=opcije.seed,
            polish=False,      # bez L-BFGS-B poliranja — isto kao MATLAB
            init="latinhypercube",  # uniformnija od random — ≈ gacreationuniform
            disp=False,
        )
    except Exception as e:
        return None

    fval = float(rez.fun)
    x = rez.x

    # MATLAB: if (fval > 0) → spremi
    if fval <= 0 or fval >= 40_000_000:
        return None

    wz_opt = float(x[0])
    k_opt  = float(x[1])

    # Ponovo izračunaj zapreminu i ekonomiju sa optimalnim parametrima
    def eko_fn(surf):
        return ekonomska_cijena(surf, ctx.dobre_zone)

    rez_kupe = zapremina_kupe(
        wx=wx, wy=wy, wz=wz_opt,
        ugao=ctx.ugao, k=k_opt,
        mnv=ctx.mnv,
        teren=ctx.teren,
        zona_x=ctx.zona_x,
        zona_y=ctx.zona_y,
        ekonomska_fn=eko_fn,
    )

    zapremina = rez_kupe.zapremina
    if zapremina >= 40_000_000:
        return None

    distanca = distanca_od_centra_masa(wx, wy, ctx.centar_masa)
    c1, c2, c3 = racunaj_troskove(zapremina, distanca, wz_opt, rez_kupe.ekonomska_cena,
                                   mnv=ctx.mnv)

    # Provjera unutar zone (unutarInteresneZone)
    unutar, _ = unutar_interesne_zone(
        zona_x=ctx.zona_x, zona_y=ctx.zona_y,
        wx=wx, wy=wy, wz=wz_opt,
        ugao=ctx.ugao, k=k_opt,
        mnv=ctx.mnv,
    )

    # Gornja i donja kontura za DXF
    xx1 = rez_kupe.gornja_kontura
    yy1 = rez_kupe.gornja_kontura_y
    zz1 = np.full(9, wz_opt)
    # Donja kontura — rekonstruisati iz k_opt
    from geometry import _kontura_kupe
    _, _, _, xx2, yy2, zz2 = _kontura_kupe(wx, wy, wz_opt, k_opt, ctx.mnv, ctx.ugao)

    return RezultatTacke(
        naziv=naziv,
        wx=wx, wy=wy, wz=wz_opt,
        k=k_opt,
        f_vrednost=fval,
        zapremina=zapremina,
        ugao=ctx.ugao,
        distanca=distanca,
        c1=c1, c2=c2, c3=c3,
        zone=rez_kupe.zone,
        unutar_zone=unutar,
        xx1=xx1, yy1=yy1, zz1=zz1,
        xx2=xx2, yy2=yy2, zz2=zz2,
    )


# ---------------------------------------------------------------------------
# Glavni GA pokretač  —  iterira po svim kandidat-tačkama
# ---------------------------------------------------------------------------

def pokreni_ga(
    tacke: np.ndarray,
    ctx: GAKontekst,
    opcije: Optional[GAOpcije] = None,
    verzija: str = "buvac",
    verbose: bool = True,
) -> tuple[list[RezultatTacke], list[RezultatTacke]]:
    """Pokreće GA za sve kandidat-tačke i vraća validne rezultate.

    MATLAB ekvivalent: petlja  for i=1:size(listaTacaka) + GA + post-procesiranje

    Tok:
    1. Za svaku tačku iz Monte Carlo faze pokreni GA
    2. Spremi sve rezultate gdje fval > 0
    3. Filtriraj: zadrži samo one gdje kupa ostaje unutar interesne zone

    Args:
        tacke:   (M, 3) array tačaka — X, Y, Z iz generiši_tačke()
        ctx:     GAKontekst
        opcije:  GAOpcije (default vrijednosti ako None)
        verzija: "buvac" ili "v1"
        verbose: ispisi napredak

    Returns:
        (svi_rezultati, validni_rezultati)
        svi_rezultati     → export_ga.xlsx  (MATLAB: tableDataFinal)
        validni_rezultati → export_ga_final.xlsx  (MATLAB: posleProvereUnutarZoneArray)
    """
    if opcije is None:
        opcije = GAOpcije(max_generacija=ctx.parametri.broj_generacija
                         if hasattr(ctx, "parametri") else 3)

    svi: list[RezultatTacke] = []
    start = time.time()

    n_tacaka = len(tacke)
    if verbose:
        print(f"\nPokretanje GA za {n_tacaka} kandidat-tačaka...")
        print(f"  Parametri: pop={opcije.populacija}, "
              f"gen={opcije.max_generacija}, tol={opcije.tolerancija}")
        print("-" * 55)

    for i, tacka in enumerate(tacke):
        wx, wy, wz_init = float(tacka[0]), float(tacka[1]), float(tacka[2])
        naziv = f"point_{i+1}"

        if verbose:
            print(f"  [{i+1:3d}/{n_tacaka}] ({wx:.0f}, {wy:.0f})  ", end="", flush=True)

        t0 = time.time()
        rez = optimizuj_tacku(naziv, wx, wy, ctx, opcije, verzija)
        dt = time.time() - t0

        if rez is not None:
            svi.append(rez)
            if verbose:
                print(f"✓  f={rez.f_vrednost:.4f}  V={rez.zapremina:,.0f}m³  ({dt:.1f}s)")
        else:
            if verbose:
                print(f"–  bez rješenja  ({dt:.1f}s)")

    # Filtriranje po unutar_zone (MATLAB: posleProvereUnutarZoneArray)
    validni = [r for r in svi if r.unutar_zone]

    elapsed = time.time() - start
    if verbose:
        print("-" * 55)
        print(f"Ukupno: {len(svi)} rezultata, {len(validni)} unutar zone  ({elapsed:.1f}s)\n")

    return svi, validni
