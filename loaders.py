"""
loaders.py  –  Korak 1 migracije: uvoz svih ulaznih podataka

Zamjenjuje MATLAB funkcije:
  uvozTerenaV3.m
  uvozEkonomskihZonaBuvac.m
  uvozCentarMasa.m
  uvozGraniceZoneInteresa.m  (+ hardkodirane koordinate iz zonaInteresaV3.m)
  uvozDodatnihUlaznihParametara.m

Umjesto uigetfile() dijaloga i globalnig getter/setter parova,
sve funkcije vraćaju Python objekte koji se prosleđuju eksplicitno.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial import Delaunay


# ---------------------------------------------------------------------------
# Strukture podataka  (zamjenjuje MATLAB struct-ove i globalne varijable)
# ---------------------------------------------------------------------------

@dataclass
class TerenSurface:
    """3D mreža terena dobivena Delaunay triangulacijom XYZ tačaka.

    Odgovara MATLAB struct-u: SurfaceTeren.vertices, SurfaceTeren.faces
    """
    vertices: np.ndarray   # shape (N, 3) — X, Y, Z
    faces: np.ndarray      # shape (M, 3) — indeksi trojki


@dataclass
class EkonomskaZona:
    """Jedna ekonomska zona sa poligonom, cijenom i površinom.

    Odgovara MATLAB struct-u: s.Name, s.Cena, s.Povrsina, s.xData, s.yData
    """
    naziv: str
    cena: float
    povrsina: float
    x_data: np.ndarray     # koordinate poligona (X)
    y_data: np.ndarray     # koordinate poligona (Y)


@dataclass
class GraniceZone:
    """Bounding box i poligon interesne zone.

    Odgovara povratnim vrijednostima uvozGraniceZoneInteresa.m
    """
    x_range: tuple[float, float]   # (x_min, x_max)
    y_range: tuple[float, float]   # (y_min, y_max)
    z_range: tuple[float, float]   # (z_min, z_max)
    x_poly: np.ndarray             # poligon interesne zone
    y_poly: np.ndarray


@dataclass
class DodatniParametri:
    """Ulazni parametri koji se čitaju iz DodatniUlazniParametri.txt.

    Odgovara uvozDodatnihUlaznihParametara.m + tri setter poziva.

    NAPOMENA: Defaults su neutralne vrijednosti (None) koje se
    automatski izračunavaju iz terena u ucitaj_sve() ako nisu
    eksplicitno zadane u fajlu. Na taj način kod radi za bilo koji teren.
    """
    nadmorska_visina: float = None     # mv — izračunava se iz z_min terena ako nije u fajlu
    broj_generacija: int = 3           # MaxGeneration za GA
    uslov_distance: float = None       # izračunava se iz dijagonale zone ako nije u fajlu


@dataclass
class UlazniPodaci:
    """Centralni kontejner svih učitanih podataka — zamjenjuje 39 get/set parova.

    Prosleđuje se eksplicitno kroz sve funkcije umjesto globalnih varijabli.
    """
    teren: Optional[TerenSurface] = None
    dobre_zone: list[EkonomskaZona] = field(default_factory=list)
    lose_zone: list[EkonomskaZona] = field(default_factory=list)
    centar_masa: Optional[np.ndarray] = None   # [X, Y, Z]
    granice: Optional[GraniceZone] = None
    parametri: DodatniParametri = field(default_factory=DodatniParametri)


# ---------------------------------------------------------------------------
# Uvoz terena  (zamjenjuje uvozTerenaV3.m)
# ---------------------------------------------------------------------------

def ucitaj_teren(putanja: str | Path) -> TerenSurface:
    """Čita XYZ oblak tačaka terena i gradi Delaunay triangulaciju.

    MATLAB ekvivalent:
        T1 = readtable(fileName);
        SurfaceTeren.vertices = [X', Y', Z'];
        SurfaceTeren.faces    = delaunay(X, Y);

    Format fajla: CSV bez zaglavlja, kolone X,Y,Z
    Primjer:   6413208.000000,4970630.500000,10.000000
    """
    putanja = Path(putanja)
    if not putanja.exists():
        raise FileNotFoundError(f"Fajl terena nije pronađen: {putanja}")

    print(f"  Učitavam teren: {putanja.name} ...", end=" ", flush=True)
    xyz = np.loadtxt(putanja, delimiter=",", dtype=np.float64)

    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"Neispravan format fajla terena — očekujem X,Y,Z kolone, dobio shape {xyz.shape}")

    # Uklanjanje duplikata (MATLAB readtable ih ne uklanja, ali Delaunay puca na duplikatima)
    _, unique_idx = np.unique(xyz[:, :2], axis=0, return_index=True)
    xyz_unique = xyz[unique_idx]

    X, Y, Z = xyz_unique[:, 0], xyz_unique[:, 1], xyz_unique[:, 2]

    # scipy.spatial.Delaunay — ekvivalent MATLAB delaunay(X, Y)
    tri = Delaunay(np.column_stack([X, Y]))

    print(f"OK ({len(xyz_unique):,} tačaka, {len(tri.simplices):,} trougao)")
    return TerenSurface(
        vertices=np.column_stack([X, Y, Z]),
        faces=tri.simplices,
    )


# ---------------------------------------------------------------------------
# Uvoz ekonomskih zona  (zamjenjuje uvozEkonomskihZonaBuvac.m)
# ---------------------------------------------------------------------------

def ucitaj_ekonomske_zone(putanja: str | Path) -> tuple[list[EkonomskaZona], list[EkonomskaZona]]:
    """Čita strukturirani tekstualni fajl sa ekonomskim zonama.

    MATLAB ekvivalent:
        [dobreEkonomskeZone, loseEkonomskeZone] = uvozEkonomskihZonaBuvac()

    Format fajla (5 linija po zoni):
        Z-1-1.1              ← naziv
        15                   ← cena (numerička)
        55737                ← površina (m²)
        6411777,6411577,...  ← X koordinate poligona odvojene zarezom
        4970315,4970315,...  ← Y koordinate poligona

    Klasifikacija zona (isto kao MATLAB kod):
        Z-1, Z-3, Z-4 → dobre zone (dobreEkonomskeZone)
        Z-5           → lose zone  (loseEkonomskeZone)
    """
    putanja = Path(putanja)
    if not putanja.exists():
        raise FileNotFoundError(f"Fajl ekonomskih zona nije pronađen: {putanja}")

    print(f"  Učitavam ekonomske zone: {putanja.name} ...", end=" ", flush=True)

    # Čitanje i normalizacija linija (strip \r\n, ignorisanje praznih)
    linije = putanja.read_text(encoding="utf-8", errors="replace").splitlines()
    linije = [l.strip() for l in linije]

    dobre: list[EkonomskaZona] = []
    lose: list[EkonomskaZona] = []

    i = 0
    while i < len(linije):
        linija = linije[i]

        # Zona počinje ako linija izgleda kao "Z-X..." pattern
        if re.match(r'^Z-\d', linija):
            naziv = linija
            try:
                cena     = float(linije[i + 1])
                povrsina = float(linije[i + 2])
                x_data   = np.fromstring(linije[i + 3], sep=",", dtype=np.float64)
                y_data   = np.fromstring(linije[i + 4], sep=",", dtype=np.float64)
            except (IndexError, ValueError) as e:
                print(f"\n  UPOZORENJE: Greška pri čitanju zone '{naziv}' (red {i}): {e}")
                i += 1
                continue

            zona = EkonomskaZona(naziv=naziv, cena=cena, povrsina=povrsina,
                                 x_data=x_data, y_data=y_data)

            # Klasifikacija: Z-5 su loše, sve ostalo (Z-1, Z-3, Z-4) su dobre
            if naziv.startswith("Z-5"):
                lose.append(zona)
            else:
                dobre.append(zona)

            i += 5  # preskači 5 linija zone
            continue

        i += 1

    print(f"OK ({len(dobre)} dobrih, {len(lose)} loših zona)")
    return dobre, lose


# ---------------------------------------------------------------------------
# Uvoz centra masa  (zamjenjuje uvozCentarMasa.m + setCentarMasa)
# ---------------------------------------------------------------------------

def ucitaj_centar_masa(putanja: str | Path) -> np.ndarray:
    """Čita koordinate centra masa iz fajla.

    MATLAB ekvivalent:
        fileData = readtable(fileName);
        centarMasaFile = [X, Y, Z];
        setCentarMasa(centarMasaFile);

    Format: jedna linija  X,Y,Z
    Primjer:  6413080,4970217,90
    """
    putanja = Path(putanja)
    if not putanja.exists():
        raise FileNotFoundError(f"Fajl centra masa nije pronađen: {putanja}")

    print(f"  Učitavam centar masa: {putanja.name} ...", end=" ", flush=True)
    cm = np.loadtxt(putanja, delimiter=",", dtype=np.float64, ndmin=1)

    if cm.ndim == 1 and len(cm) >= 2:
        cm = cm[:3]  # uzimamo X, Y, Z (ili X, Y ako nema Z)
    elif cm.ndim == 2:
        cm = cm[0, :3]

    print(f"OK (X={cm[0]:.1f}, Y={cm[1]:.1f})")
    return cm


# ---------------------------------------------------------------------------
# Uvoz granica zone interesa  (zamjenjuje uvozGraniceZoneInteresa.m + zonaInteresaV3.m)
# ---------------------------------------------------------------------------

def ucitaj_granice_zone(putanja: str | Path) -> GraniceZone:
    """Čita bounding box i poligon interesne zone iz fajla.

    MATLAB ekvivalent:
        [x, y, z, xrange, yrange, zrange] = uvozGraniceZoneInteresa()

    Format fajla (GranicaZonaBuvac.txt):
        Linija 1: X_min, Y_min, Z_min   ← donja granica bounding box-a
        Linija 2: X_max, Y_max, Z_max   ← gornja granica bounding box-a
        Linije 3+: X, Y, Z             ← tačke poligona interesne zone

    Primjer:
        6411177.27,4968315.083,150
        6414977.27,4972115.083,210
        6411177.27,4968315.083,150
        6414977.27,4968315.083,150
        ...
    """
    putanja = Path(putanja)
    if not putanja.exists():
        raise FileNotFoundError(f"Fajl granica zone nije pronađen: {putanja}")

    print(f"  Učitavam granice zone: {putanja.name} ...", end=" ", flush=True)
    podaci = np.loadtxt(putanja, delimiter=",", dtype=np.float64)

    if podaci.shape[0] < 3:
        raise ValueError(f"Fajl granica zone mora imati najmanje 3 linije, ima {podaci.shape[0]}")

    # Prvih 5 linija opisuje bounding box (min/max + 3 ugaone tačke)
    # Ostatak su tačke poligona interesne zone
    x_range = (podaci[0, 0], podaci[1, 0])
    y_range = (podaci[0, 1], podaci[1, 1])
    z_range = (podaci[0, 2], podaci[1, 2])

    # Tačke poligona — linije 3 do kraja
    x_poly = podaci[2:, 0]
    y_poly = podaci[2:, 1]

    print(f"OK (X: {x_range[0]:.0f}–{x_range[1]:.0f}, Y: {y_range[0]:.0f}–{y_range[1]:.0f}, {len(x_poly)} tačaka poligona)")
    return GraniceZone(
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        x_poly=x_poly,
        y_poly=y_poly,
    )


# ---------------------------------------------------------------------------
# Uvoz dodatnih parametara  (zamjenjuje uvozDodatnihUlaznihParametara.m)
# ---------------------------------------------------------------------------

def ucitaj_dodatne_parametre(putanja: str | Path) -> DodatniParametri:
    """Čita numeričke parametre iz konfiguracijskog fajla.

    MATLAB ekvivalent:
        uvozDodatnihUlaznihParametara() koji poziva:
          setNadmorskaVisina(), setBrojGeneracija(), setUslovDistanceTranskoporaMaterijala()

    Format fajla (DodatniUlazniParametri.txt):
        %% komentar
        140          ← nadmorska_visina (mv)
        %% komentar
        3            ← broj_generacija
        %% komentar
        2000         ← uslov_distance

    Linije koje počinju sa %% su komentari i ignorišu se.
    """
    putanja = Path(putanja)
    if not putanja.exists():
        raise FileNotFoundError(f"Fajl dodatnih parametara nije pronađen: {putanja}")

    print(f"  Učitavam dodatne parametre: {putanja.name} ...", end=" ", flush=True)

    brojevi: list[float] = []
    for linija in putanja.read_text(encoding="utf-8", errors="replace").splitlines():
        linija = linija.strip()
        if not linija or linija.startswith("%%") or linija.startswith("#"):
            continue
        try:
            brojevi.append(float(linija))
        except ValueError:
            pass  # ignorišemo nečitljive linije

    if len(brojevi) < 3:
        raise ValueError(f"Fajl parametara mora imati najmanje 3 numeričke vrijednosti, pronađeno: {len(brojevi)}")

    params = DodatniParametri(
        nadmorska_visina=brojevi[0],
        broj_generacija=int(brojevi[1]),
        uslov_distance=brojevi[2],
    )
    print(f"OK (mv={params.nadmorska_visina}, generacije={params.broj_generacija}, dist={params.uslov_distance})")
    return params


# ---------------------------------------------------------------------------
# Glavni loader — učitava sve odjednom
# ---------------------------------------------------------------------------

def ucitaj_sve(
    putanja_teren: str | Path,
    putanja_zone: str | Path,
    putanja_centar_masa: str | Path,
    putanja_granice: str | Path,
    putanja_parametri: str | Path,
) -> UlazniPodaci:
    """Učitava sve ulazne podatke i vraća jedan UlazniPodaci objekat.

    Zamjenjuje svih 5+ uigetfile() poziva i 39 getter/setter parova.

    Returns:
        UlazniPodaci sa svim učitanim podacima, spreman za dalju obradu.
    """
    print("Učitavanje ulaznih podataka...")
    print("-" * 50)

    podaci = UlazniPodaci()
    podaci.teren = ucitaj_teren(putanja_teren)
    podaci.dobre_zone, podaci.lose_zone = ucitaj_ekonomske_zone(putanja_zone)
    podaci.centar_masa = ucitaj_centar_masa(putanja_centar_masa)
    podaci.granice = ucitaj_granice_zone(putanja_granice)
    podaci.parametri = ucitaj_dodatne_parametre(putanja_parametri)

    print("-" * 50)

    # Automatski izračunaj mnv i uslov_distance iz podataka terena
    # ako nisu zadani u DodatniUlazniParametri.txt
    import math as _math
    if podaci.parametri.nadmorska_visina is None:
        if podaci.teren is not None:
            z_min_auto = float(podaci.teren.vertices[:, 2].min())
            podaci.parametri.nadmorska_visina = z_min_auto
            print(f"  Auto mnv: {z_min_auto:.1f} m (z_min terena)")
        else:
            podaci.parametri.nadmorska_visina = 0.0

    if podaci.parametri.uslov_distance is None:
        if podaci.granice is not None:
            dx = podaci.granice.x_range[1] - podaci.granice.x_range[0]
            dy = podaci.granice.y_range[1] - podaci.granice.y_range[0]
            dijagonala = _math.sqrt(dx**2 + dy**2)
            podaci.parametri.uslov_distance = dijagonala
            print(f"  Auto uslov_distance: {dijagonala:.0f} m (dijagonala zone)")
        else:
            podaci.parametri.uslov_distance = 999999.0

    print("Učitavanje završeno.\n")
    return podaci
