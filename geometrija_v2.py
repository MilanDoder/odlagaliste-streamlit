"""
geometrija_v2.py — Poboljšana geometrija presjeka kupe i terena.

ZAŠTO NOVA VERZIJA (problemi stare geometry.py):

  1. ConvexHull zapremina je POGREŠNA za nepravilan (brdovit) teren:
     donja strana tijela odlagališta prati teren, koji je konkavan —
     ConvexHull "popuni" doline ispod kupe i sistematski PRECJENJUJE
     zapreminu. Na brdovitom terenu greška lako pređe 10–40 %.

  2. Kupa je bila diskretizovana sa samo 18 tačaka (9 gore + 9 dolje),
     pa je presječna kriva bila izuzetno gruba.

  3. Möller trougao–trougao presjek je O(N_kupa × N_teren), bez prostorne
     akceleracije, i vraća samo NEPOVEZANE segmente — ne zatvorenu konturu,
     ne footprint, ne teren ispod kupe. Uz to se cijeli teren obrađivao
     ispočetka za SVAKU evaluaciju GA.

  4. Presjek kupe s brdovitim terenom može imati VIŠE odvojenih petlji
     (npr. padina kupe presiječe dva brda) — stari kod to nije modelovao.

NOVI PRISTUP (visinska polja):

  I kupa i teren su 2.5D površine: z = f(x, y).
  Definišimo razliku   d(x, y) = z_kupa(x, y) − z_teren(x, y).

    • Presječna kriva  =  nivo-kriva  d(x, y) = 0
      (marching squares / contourpy → zatvorene, orijentisane petlje,
       automatski i više petlji za nepravilan teren).

    • Zapremina tijela =  ∬ max(0, d) dA
      To je *matematički tačna* definicija zapremine između gornje
      površine (kupa) i donje površine (teren), ma koliko teren bio
      nepravilan — nikakve pretpostavke o konveksnosti.

    • Površina osnove  =  mjera skupa { d > 0 }  (footprint na terenu).

  Integracija: pravilna mreža + trapezno pravilo nad odsječenom funkcijom
  max(0, d), sa ADAPTIVNIM RAFINIRANJEM ćelija kroz koje prolazi granica
  d = 0. Konvergencija je O(h²); rafiniranje granice praktično uklanja
  dominantan izvor greške.

  BRZINA: interpolator terena (Delaunay + LinearNDInterpolator) se gradi
  JEDNOM po terenu i dijeli između svih GA evaluacija — umjesto da se
  Möller O(N×M) presjek računa ispočetka za svaku jedinku. Sama kupa je
  ANALITIČKA (nema mesh-a), pa je evaluacija d(x,y) čista vektorizovana
  numpy operacija.

Javni API:

    Teren.iz_tacaka(xyz)            — teren iz oblaka tačaka (Delaunay)
    Teren.ravan(z0, ...)            — ravan teren
    Kupa(wx, wy, wz, k, ugao, ...)  — analitička (zarubljena) kupa;
                                      profil "matlab" reprodukuje originalni
                                      8-režnjeviti oblik [s,k,u,k,...]
    presek_kupe_i_terena(kupa, teren, ...) → PresekRezultat
    zapremina_kupe_v2(...)          — drop-in zamjena za geometry.zapremina_kupe
    povrsine_po_zonama(...)         — tačnije površine footprinta po ekonomskim
                                      zonama (umjesto ConvexHull aproksimacije)

Zavisnosti: numpy, scipy, contourpy (dolazi uz matplotlib).
Opcionalno: trimesh + manifold3d za egzaktnu mesh-boolean verifikaciju.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.spatial import Delaunay, cKDTree

try:  # contourpy je zavisnost matplotliba — praktično uvijek prisutan
    from contourpy import contour_generator
    _IMA_CONTOURPY = True
except ImportError:  # pragma: no cover
    _IMA_CONTOURPY = False


# ===========================================================================
# TEREN — visinsko polje z_teren(x, y), gradi se JEDNOM i višekratno koristi
# ===========================================================================

class Teren:
    """Teren kao visinsko polje z(x, y).

    Interno: Delaunay triangulacija XY projekcije + barycentrična linearna
    interpolacija (identično onome što bi dala triangulisana TIN površina).
    Van konveksnog omotača tačaka koristi se najbliži susjed (KDTree), da
    upiti nikad ne vraćaju NaN.

    VAŽNO ZA BRZINU: instancu terena kreirati jednom (van GA petlje) i
    prosljeđivati u svaku evaluaciju.
    """

    def __init__(self, z_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 xy_granice: tuple[float, float, float, float],
                 z_granice: tuple[float, float],
                 opis: str = "analitički"):
        self._z_fn = z_fn
        self.xy_granice = xy_granice          # (x_min, x_max, y_min, y_max)
        self.z_min, self.z_max = z_granice
        self.opis = opis

    # -- konstruktori ------------------------------------------------------

    @classmethod
    def iz_tacaka(cls, xyz: np.ndarray) -> "Teren":
        """Teren iz oblaka tačaka (N, 3): Delaunay + linearna interpolacija."""
        xyz = np.asarray(xyz, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 3:
            raise ValueError("Teren traži bar 3 tačke oblika (N, 3)")

        xy = xyz[:, :2]
        z = xyz[:, 2]
        tri = Delaunay(xy)
        kdt = cKDTree(xy)

        # Predračunate barycentrične transformacije (scipy ih drži u tri.transform)
        def z_fn(px: np.ndarray, py: np.ndarray) -> np.ndarray:
            pts = np.column_stack([np.ravel(px), np.ravel(py)])
            simplex = tri.find_simplex(pts)
            out = np.empty(len(pts))

            unutra = simplex >= 0
            if np.any(unutra):
                s = simplex[unutra]
                T = tri.transform[s]                       # (m, 3, 2+ )
                delta = pts[unutra] - T[:, 2, :]
                bary2 = np.einsum("mij,mj->mi", T[:, :2, :], delta)
                bary = np.column_stack([bary2, 1.0 - bary2.sum(axis=1)])
                out[unutra] = np.einsum(
                    "mi,mi->m", bary, z[tri.simplices[s]])

            if np.any(~unutra):                            # van hull-a → NN
                _, idx = kdt.query(pts[~unutra])
                out[~unutra] = z[idx]

            return out.reshape(np.shape(px))

        granice = (xy[:, 0].min(), xy[:, 0].max(),
                   xy[:, 1].min(), xy[:, 1].max())
        t = cls(z_fn, granice, (float(z.min()), float(z.max())),
                opis=f"TIN ({len(xyz)} tačaka)")
        t.tacke = xyz
        t.tri = tri
        return t

    @classmethod
    def iz_surface(cls, surface) -> "Teren":
        """Kompatibilnost sa geometry.Surface (koristi samo vertices)."""
        return cls.iz_tacaka(np.asarray(surface.vertices, dtype=float))

    @classmethod
    def ravan(cls, z0: float,
              xy_granice: tuple[float, float, float, float] = (-1e9, 1e9, -1e9, 1e9)
              ) -> "Teren":
        """Savršeno ravan teren na visini z0."""
        return cls(lambda x, y: np.full(np.shape(x), float(z0), dtype=float),
                   xy_granice, (z0, z0), opis=f"ravan z={z0}")

    @classmethod
    def analiticki(cls, fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                   xy_granice: tuple[float, float, float, float],
                   z_granice: tuple[float, float]) -> "Teren":
        """Teren zadat proizvoljnom funkcijom z = fn(x, y) — za testove."""
        return cls(fn, xy_granice, z_granice, opis="analitički")

    # -- upit ---------------------------------------------------------------

    def z(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vektorizovan upit visine terena."""
        return self._z_fn(np.asarray(x, dtype=float), np.asarray(y, dtype=float))


# ===========================================================================
# KUPA — analitičko visinsko polje z_kupa(x, y) (bez mesh-a!)
# ===========================================================================

@dataclass
class Kupa:
    """Zarubljena kupa odlagališta kao analitičko visinsko polje.

    Geometrija (identična namjeri MATLAB koda, ali kontinualna):
      • gornji plato na visini wz, poluprečnika r_top(θ);
      • padina pod uglom `ugao` (stepeni) spušta se od ivice platoa;
      • z_kupa(x, y) = wz                        za r ≤ r_top(θ)
                     = wz − (r − r_top(θ))·tan(ugao)   inače.

    profil:
      "krug"   — r_top(θ) = k          (pravilna zarubljena kupa; za nju
                                         postoje egzaktne formule → testovi)
      "matlab" — r_top(θ) linearno interpolira originalni MATLAB obrazac
                 [s, k, u, k, s, k, u, k, s] na uglovima 0, π/4, …, 2π,
                 gdje je s = 1.4·k i u = 1.25·k (8-režnjeviti oblik).
    """
    wx: float
    wy: float
    wz: float
    k: float
    ugao: float                 # ugao kosine u STEPENIMA
    profil: str = "krug"

    _theta_cvorovi: np.ndarray = field(init=False, repr=False)
    _r_cvorovi: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if self.k <= 0:
            raise ValueError("Širina kupe k mora biti > 0")
        if not (0 < self.ugao < 90):
            raise ValueError("Ugao kosine mora biti u (0, 90) stepeni")
        s, u = 1.4 * self.k, 1.25 * self.k
        self._theta_cvorovi = np.linspace(0.0, 2.0 * np.pi, 9)
        if self.profil == "matlab":
            self._r_cvorovi = np.array([s, self.k, u, self.k,
                                        s, self.k, u, self.k, s])
        elif self.profil == "krug":
            self._r_cvorovi = np.full(9, float(self.k))
        else:
            raise ValueError(f"Nepoznat profil: {self.profil!r}")

    # -- geometrijski upiti --------------------------------------------------

    @property
    def tan_ugla(self) -> float:
        return float(np.tan(np.radians(self.ugao)))

    def r_top(self, theta: np.ndarray) -> np.ndarray:
        """Poluprečnik gornjeg platoa u pravcu θ (periodična lin. interpolacija)."""
        th = np.mod(np.asarray(theta, dtype=float), 2 * np.pi)
        return np.interp(th, self._theta_cvorovi, self._r_cvorovi)

    def z(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Visina površine kupe u tački (x, y) — vektorizovano."""
        dx = np.asarray(x, dtype=float) - self.wx
        dy = np.asarray(y, dtype=float) - self.wy
        r = np.hypot(dx, dy)
        rt = self.r_top(np.arctan2(dy, dx))
        return self.wz - np.maximum(0.0, r - rt) * self.tan_ugla

    def max_radijus(self, z_min: float) -> float:
        """Najveći mogući horizontalni doseg padine dok ne siđe do z_min."""
        r_top_max = float(self._r_cvorovi.max())
        pad = max(0.0, self.wz - z_min)
        return r_top_max + pad / self.tan_ugla

    def gornja_kontura(self, n: int = 128) -> tuple[np.ndarray, np.ndarray]:
        """Tačke gornje ivice platoa (za DXF / prikaz)."""
        th = np.linspace(0, 2 * np.pi, n, endpoint=True)
        rt = self.r_top(th)
        return self.wx + rt * np.cos(th), self.wy + rt * np.sin(th)


# ===========================================================================
# PRESJEK I ZAPREMINA — nivo-kriva d = 0 i integracija ∬ max(0, d) dA
# ===========================================================================

@dataclass
class PresekRezultat:
    """Kompletan rezultat presjeka kupe i terena."""
    zapremina: float                     # m³ — ∬ max(0, z_kupa − z_teren) dA
    povrsina_osnove: float               # m² — površina footprinta {d > 0}
    konture: list[np.ndarray]            # lista (K_i, 3) zatvorenih 3D petlji
    ima_preseka: bool                    # da li kupa uopšte "sjedi" na terenu
    broj_petlji: int                     # >1 kod nepravilnog terena je normalno
    max_debljina: float                  # m — najveća visina nasipa iznad terena
    procjena_greske: float               # m³ — |V_fino − V_grubo| (Richardson)
    rezolucija: int                      # osnovna rezolucija mreže
    granice_racuna: tuple[float, float, float, float]  # bbox integracije

    @property
    def kontura_glavna(self) -> Optional[np.ndarray]:
        """Najduža (glavna) presječna petlja ili None."""
        if not self.konture:
            return None
        return max(self.konture, key=len)


def _konture_nivoa_nula(gx: np.ndarray, gy: np.ndarray,
                        d: np.ndarray) -> list[np.ndarray]:
    """Zatvorene petlje nivo-krive d = 0 (marching squares)."""
    if _IMA_CONTOURPY:
        gen = contour_generator(x=gx, y=gy, z=d)
        linije = gen.lines(0.0)
        return [np.asarray(l) for l in linije if len(l) >= 3]
    # Rezervni put bez contourpy: matplotlib interni kontur
    import matplotlib.pyplot as plt  # pragma: no cover
    fig, ax = plt.subplots()
    cs = ax.contour(gx, gy, d, levels=[0.0])
    petlje = [np.asarray(p.vertices) for c in cs.collections
              for p in c.get_paths() if len(p.vertices) >= 3]
    plt.close(fig)
    return petlje


def presek_kupe_i_terena(
    kupa: Kupa,
    teren: Teren,
    rezolucija: int = 256,
    rafiniranje: int = 2,
    margina: float = 1.05,
) -> PresekRezultat:
    """Presjek kupe i terena + zapremina tijela između njih.

    Algoritam:
      1. Odredi kvadratni bbox oko kupe: centar (wx, wy), poluširina
         margina · max_radijus (doseg padine do najniže tačke terena).
      2. Na mreži (rezolucija+1)² izračunaj d = z_kupa − z_teren.
      3. Presječne krive = nivo-krive d = 0 (marching squares, contourpy);
         podignute na teren → 3D petlje. Više petlji = nepravilan teren.
      4. Zapremina = ∬ max(0, d) dA trapeznim pravilom po ćelijama,
         pri čemu se ćelije kroz koje prolazi granica (promjena znaka d)
         REKURZIVNO RAFINIRAJU `rafiniranje` puta (svaki nivo dijeli
         ćeliju na 4×4 pod-ćelije) — greška granice pada za ~16× po nivou.
      5. Procjena greške: Richardson — razlika integrala na punoj i
         duplo grubljoj mreži.

    Args:
        kupa:        Kupa objekat (analitička površina).
        teren:       Teren objekat (izgrađen JEDNOM, dijeli se kroz GA).
        rezolucija:  broj ćelija osnovne mreže po osi (256 je dobar default:
                     tačnost ≪ 0.1 % uz ~1–5 ms po evaluaciji).
        rafiniranje: broj nivoa 4×4 rafiniranja graničnih ćelija (0 = bez).
        margina:     faktor proširenja bbox-a (≥ 1).

    Returns:
        PresekRezultat
    """
    # --- 1. domen integracije ------------------------------------------------
    R = kupa.max_radijus(teren.z_min) * margina
    x0, x1 = kupa.wx - R, kupa.wx + R
    y0, y1 = kupa.wy - R, kupa.wy + R

    n = int(rezolucija)
    gx = np.linspace(x0, x1, n + 1)
    gy = np.linspace(y0, y1, n + 1)
    hx = (x1 - x0) / n
    hy = (y1 - y0) / n

    GX, GY = np.meshgrid(gx, gy)          # (n+1, n+1), red = y, kolona = x
    d = kupa.z(GX, GY) - teren.z(GX, GY)  # razlika visinskih polja

    poz = d > 0.0
    if not np.any(poz):
        return PresekRezultat(
            zapremina=0.0, povrsina_osnove=0.0, konture=[],
            ima_preseka=False, broj_petlji=0, max_debljina=0.0,
            procjena_greske=0.0, rezolucija=n,
            granice_racuna=(x0, x1, y0, y1))

    # --- 2. presječne krive (nivo d = 0), podignute na teren ---------------
    petlje_2d = _konture_nivoa_nula(gx, gy, d)
    konture_3d = []
    for p in petlje_2d:
        pz = teren.z(p[:, 0], p[:, 1])
        konture_3d.append(np.column_stack([p, pz]))

    # --- 3. zapremina: trapez nad max(0, d) po ćelijama ---------------------
    dc = np.maximum(d, 0.0)
    # kutne vrijednosti ćelija (n, n, 4)
    c00 = dc[:-1, :-1]; c01 = dc[:-1, 1:]
    c10 = dc[1:, :-1];  c11 = dc[1:, 1:]
    prosjek = 0.25 * (c00 + c01 + c10 + c11)
    V_celija = prosjek * hx * hy                    # (n, n)

    # ćelije sa promjenom znaka → kandidat za rafiniranje
    z00 = d[:-1, :-1] > 0; z01 = d[:-1, 1:] > 0
    z10 = d[1:, :-1] > 0;  z11 = d[1:, 1:] > 0
    suma_poz = (z00.astype(np.int8) + z01 + z10 + z11)
    granicne = (suma_poz > 0) & (suma_poz < 4)

    # površina osnove (footprint): udio pozitivnih uglova po ćeliji
    A_celija = (suma_poz / 4.0) * hx * hy

    V_grubo_ukupno = float(V_celija.sum())

    if rafiniranje > 0 and np.any(granicne):
        iy, ix = np.where(granicne)
        V_fine, A_fine = _rafiniraj_celije(
            kupa, teren, gx, gy, ix, iy, hx, hy, nivoa=rafiniranje)
        V_celija = V_celija.astype(float)
        V_celija[iy, ix] = V_fine
        A_celija[iy, ix] = A_fine

    zapremina = float(V_celija.sum())
    povrsina = float(A_celija.sum())

    # --- 4. Richardson procjena greške (poduzorkovana mreža koraka 2h) -----
    d2 = np.maximum(d[::2, ::2], 0.0)
    p2 = 0.25 * (d2[:-1, :-1] + d2[:-1, 1:] + d2[1:, :-1] + d2[1:, 1:])
    V_2h = float(p2.sum()) * (2 * hx) * (2 * hy)
    procjena_greske = abs(V_grubo_ukupno - V_2h) / 3.0   # O(h²) ekstrapolacija

    return PresekRezultat(
        zapremina=zapremina,
        povrsina_osnove=povrsina,
        konture=konture_3d,
        ima_preseka=True,
        broj_petlji=len(konture_3d),
        max_debljina=float(d.max()),
        procjena_greske=procjena_greske,
        rezolucija=n,
        granice_racuna=(x0, x1, y0, y1),
    )


def _rafiniraj_celije(kupa: Kupa, teren: Teren,
                      gx: np.ndarray, gy: np.ndarray,
                      ix: np.ndarray, iy: np.ndarray,
                      hx: float, hy: float, nivoa: int,
                      pod: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Rafinira granične ćelije pod×pod pod-mrežom, `nivoa` puta rekurzivno.

    Implementirano iterativno i POTPUNO vektorizovano: sve granične ćelije
    jednog nivoa evaluiraju se jednim numpy pozivom.

    Returns:
        (V, A) — zapremina i površina po ulaznoj ćeliji, oblika (len(ix),).
    """
    m = len(ix)
    # lokalne pod-mreže (pod+1) čvorova po osi u svakoj ćeliji
    fx = np.linspace(0.0, 1.0, pod + 1)
    FX, FY = np.meshgrid(fx, fx)                       # (pod+1, pod+1)

    X = gx[ix][:, None, None] + FX[None] * hx           # (m, pod+1, pod+1)
    Y = gy[iy][:, None, None] + FY[None] * hy

    d = kupa.z(X, Y) - teren.z(X, Y)
    shx, shy = hx / pod, hy / pod

    if nivoa == 1:
        dc = np.maximum(d, 0.0)
        pr = 0.25 * (dc[:, :-1, :-1] + dc[:, :-1, 1:]
                     + dc[:, 1:, :-1] + dc[:, 1:, 1:])
        V = pr.sum(axis=(1, 2)) * shx * shy
        poz = (d > 0).astype(np.int8)
        sp = (poz[:, :-1, :-1] + poz[:, :-1, 1:]
              + poz[:, 1:, :-1] + poz[:, 1:, 1:])
        A = (sp / 4.0).sum(axis=(1, 2)) * shx * shy
        return V, A

    # dublje rafiniranje: pod-ćelije bez promjene znaka → trapez;
    # pod-ćelije sa promjenom znaka → rekurzija
    poz = d > 0
    s00 = poz[:, :-1, :-1]; s01 = poz[:, :-1, 1:]
    s10 = poz[:, 1:, :-1];  s11 = poz[:, 1:, 1:]
    sp = s00.astype(np.int8) + s01 + s10 + s11          # (m, pod, pod)
    granicne = (sp > 0) & (sp < 4)

    dc = np.maximum(d, 0.0)
    pr = 0.25 * (dc[:, :-1, :-1] + dc[:, :-1, 1:]
                 + dc[:, 1:, :-1] + dc[:, 1:, 1:])
    V_sub = pr * shx * shy                              # (m, pod, pod)
    A_sub = (sp / 4.0) * shx * shy

    if np.any(granicne):
        mi, myi, mxi = np.where(granicne)
        sub_gx0 = gx[ix[mi]] + mxi * shx
        sub_gy0 = gy[iy[mi]] + myi * shy
        Vf, Af = _rafiniraj_celije_abs(
            kupa, teren, sub_gx0, sub_gy0, shx, shy, nivoa - 1, pod)
        V_sub[mi, myi, mxi] = Vf
        A_sub[mi, myi, mxi] = Af

    return V_sub.sum(axis=(1, 2)), A_sub.sum(axis=(1, 2))


def _rafiniraj_celije_abs(kupa: Kupa, teren: Teren,
                          x0: np.ndarray, y0: np.ndarray,
                          hx: float, hy: float, nivoa: int,
                          pod: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Isto kao _rafiniraj_celije, ali ćelije zadate apsolutnim donjim uglom."""
    fx = np.linspace(0.0, 1.0, pod + 1)
    FX, FY = np.meshgrid(fx, fx)
    X = x0[:, None, None] + FX[None] * hx
    Y = y0[:, None, None] + FY[None] * hy
    d = kupa.z(X, Y) - teren.z(X, Y)
    shx, shy = hx / pod, hy / pod

    dc = np.maximum(d, 0.0)
    pr = 0.25 * (dc[:, :-1, :-1] + dc[:, :-1, 1:]
                 + dc[:, 1:, :-1] + dc[:, 1:, 1:])
    V_sub = pr * shx * shy
    poz = d > 0
    sp = (poz[:, :-1, :-1].astype(np.int8) + poz[:, :-1, 1:]
          + poz[:, 1:, :-1] + poz[:, 1:, 1:])
    A_sub = (sp / 4.0) * shx * shy

    if nivoa > 1:
        granicne = (sp > 0) & (sp < 4)
        if np.any(granicne):
            mi, myi, mxi = np.where(granicne)
            Vf, Af = _rafiniraj_celije_abs(
                kupa, teren,
                x0[mi] + mxi * shx, y0[mi] + myi * shy,
                shx, shy, nivoa - 1, pod)
            V_sub[mi, myi, mxi] = Vf
            A_sub[mi, myi, mxi] = Af

    return V_sub.sum(axis=(1, 2)), A_sub.sum(axis=(1, 2))


# ===========================================================================
# ANALIZA DELOVA — kada presjek ima VIŠE odvojenih petlji (npr. rijeka
# presiječe footprint), svaki povezani deo {d > 0} se analizira zasebno:
# sopstvena zapremina, površina, konture i oznaka (A, B, C, ...).
# ===========================================================================

@dataclass
class DeoPreseka:
    """Jedan povezani deo footprinta {d > 0} — zaseban 'nasip'."""
    oznaka: str                  # "A", "B", "C", ... (A = najveća zapremina)
    zapremina: float             # m³ — zapremina samo ovog dela
    povrsina: float              # m² — površina osnove samo ovog dela
    konture: list                # lista (K_i, 3) 3D petlji koje pripadaju delu
    centar: tuple                # (x, y, z_teren) tačka najveće debljine —
                                 # pogodna za postavljanje oznake na prikazu
    max_debljina: float          # m — najveća visina nasipa u delu


def analiza_delova(
    kupa: Kupa,
    teren: Teren,
    rezolucija: int = 256,
    rafiniranje: int = 2,
    margina: float = 1.05,
) -> list[DeoPreseka]:
    """Podjela presjeka na povezane delove sa zapreminom po delu.

    Koristi ISTU mrežu i isti integracioni postupak kao
    presek_kupe_i_terena (trapez + rafiniranje graničnih ćelija), pa je
    zbir zapremina svih delova ≈ ukupna zapremina iz PresekRezultat.

    Povezanost: 8-susjedstvo nad maskom {d > 0} (scipy.ndimage.label).
    Svaka presječna petlja se dodjeljuje delu čije pozitivne čvorove
    dodiruje (većinsko glasanje po tjemenima petlje) — tako i eventualne
    'rupe' (ostrva terena) ostaju uz svoj deo.

    Returns:
        lista DeoPreseka, sortirana po zapremini opadajuće i označena
        slovima A, B, C, ... — prazna lista ako presjeka nema.
    """
    from scipy import ndimage

    # --- ista mreža kao u presek_kupe_i_terena ------------------------------
    R = kupa.max_radijus(teren.z_min) * margina
    x0, x1 = kupa.wx - R, kupa.wx + R
    y0, y1 = kupa.wy - R, kupa.wy + R
    n = int(rezolucija)
    gx = np.linspace(x0, x1, n + 1)
    gy = np.linspace(y0, y1, n + 1)
    hx = (x1 - x0) / n
    hy = (y1 - y0) / n

    GX, GY = np.meshgrid(gx, gy)
    d = kupa.z(GX, GY) - teren.z(GX, GY)
    poz = d > 0.0
    if not np.any(poz):
        return []

    # --- povezani delovi (8-susjedstvo) -------------------------------------
    labels, n_lab = ndimage.label(poz, structure=np.ones((3, 3), int))
    if n_lab == 0:
        return []

    # --- zapremina i površina po ćelijama (kao u presek_kupe_i_terena) ------
    dc = np.maximum(d, 0.0)
    c00 = dc[:-1, :-1]; c01 = dc[:-1, 1:]
    c10 = dc[1:, :-1];  c11 = dc[1:, 1:]
    V_celija = 0.25 * (c00 + c01 + c10 + c11) * hx * hy

    z00 = d[:-1, :-1] > 0; z01 = d[:-1, 1:] > 0
    z10 = d[1:, :-1] > 0;  z11 = d[1:, 1:] > 0
    suma_poz = (z00.astype(np.int8) + z01 + z10 + z11)
    granicne = (suma_poz > 0) & (suma_poz < 4)
    A_celija = (suma_poz / 4.0) * hx * hy

    if rafiniranje > 0 and np.any(granicne):
        iy, ix = np.where(granicne)
        V_fine, A_fine = _rafiniraj_celije(
            kupa, teren, gx, gy, ix, iy, hx, hy, nivoa=rafiniranje)
        V_celija = V_celija.astype(float)
        V_celija[iy, ix] = V_fine
        A_celija[iy, ix] = A_fine

    # oznaka ćelije = max oznaka njena 4 ugla (granične ćelije u praksi
    # dodiruju tačno jedan deo; unutrašnje sve uglove istog dela)
    lab_cell = np.maximum(np.maximum(labels[:-1, :-1], labels[:-1, 1:]),
                          np.maximum(labels[1:, :-1], labels[1:, 1:]))
    V_po_delu = np.bincount(lab_cell.ravel(), weights=V_celija.ravel(),
                            minlength=n_lab + 1)
    A_po_delu = np.bincount(lab_cell.ravel(), weights=A_celija.ravel(),
                            minlength=n_lab + 1)

    # --- konture i njihova pripadnost delovima ------------------------------
    petlje_2d = _konture_nivoa_nula(gx, gy, d)
    konture_po_delu: dict[int, list] = {L: [] for L in range(1, n_lab + 1)}
    for p in petlje_2d:
        # tjemena petlje → indeksi čvorova mreže → većinska (nenulta) oznaka
        jx = np.clip(np.searchsorted(gx, p[:, 0]) - 1, 0, n - 1)
        jy = np.clip(np.searchsorted(gy, p[:, 1]) - 1, 0, n - 1)
        okolne = np.concatenate([labels[jy, jx], labels[jy + 1, jx],
                                 labels[jy, jx + 1], labels[jy + 1, jx + 1]])
        okolne = okolne[okolne > 0]
        if len(okolne) == 0:
            continue
        L = int(np.bincount(okolne).argmax())
        pz = teren.z(p[:, 0], p[:, 1])
        konture_po_delu[L].append(np.column_stack([p, pz]))

    # --- sklapanje, sortiranje po zapremini, oznake A, B, C, ... -------------
    delovi = []
    for L in range(1, n_lab + 1):
        maska = labels == L
        d_masked = np.where(maska, d, -np.inf)
        iy_m, ix_m = np.unravel_index(np.argmax(d_masked), d.shape)
        cx, cy = float(gx[ix_m]), float(gy[iy_m])
        delovi.append(DeoPreseka(
            oznaka="?",
            zapremina=float(V_po_delu[L]),
            povrsina=float(A_po_delu[L]),
            konture=konture_po_delu.get(L, []),
            centar=(cx, cy, float(teren.z(cx, cy))),
            max_debljina=float(d[iy_m, ix_m]),
        ))
    delovi.sort(key=lambda deo: deo.zapremina, reverse=True)
    slova = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, deo in enumerate(delovi):
        deo.oznaka = slova[i] if i < len(slova) else f"D{i + 1}"
    return delovi


# ===========================================================================
# EGZAKTNE FORMULE (za validaciju) — zarubljena kupa na ravnom terenu
# ===========================================================================

def zapremina_zarubljene_kupe_ravan_teren(kupa: Kupa, z_teren: float) -> float:
    """Egzaktna zapremina kružne zarubljene kupe iznad ravnog terena.

    V = (π·h / 3) · (R² + R·r + r²),  r = k,  R = k + h/tan(ugao)

    Radi samo za profil "krug". Koristi se u testovima kao referenca.
    """
    if kupa.profil != "krug":
        raise ValueError("Egzaktna formula važi samo za profil 'krug'")
    h = kupa.wz - z_teren
    if h <= 0:
        return 0.0
    r = kupa.k
    R = r + h / kupa.tan_ugla
    return float(np.pi * h / 3.0 * (R * R + R * r + r * r))


# ===========================================================================
# OPCIONO: egzaktna mesh-boolean verifikacija (trimesh + manifold3d)
# ===========================================================================

def zapremina_mesh_boolean(kupa: Kupa, teren: Teren,
                           rezolucija_kupe: int = 256,
                           rezolucija_terena: int = 200) -> Optional[float]:
    """Referentna zapremina egzaktnim boolean presjekom čvrstih tijela.

    Zatvara kupu i teren u vodonepropusna (watertight) tijela i računa
    boolean presjek (manifold3d). SPORO — samo za verifikaciju, ne za GA.
    Vraća None ako trimesh/manifold3d nisu instalirani.
    """
    try:
        import trimesh
    except ImportError:
        return None

    z_dno = teren.z_min - 10.0

    # --- solid kupe: plato + padina + omotač do z_dno ----------------------
    th = np.linspace(0, 2 * np.pi, rezolucija_kupe, endpoint=False)
    rt = kupa.r_top(th)
    R_max = kupa.max_radijus(z_dno)
    prsten_gore = np.column_stack([kupa.wx + rt * np.cos(th),
                                   kupa.wy + rt * np.sin(th),
                                   np.full_like(th, kupa.wz)])
    r_dno = rt + (kupa.wz - z_dno) / kupa.tan_ugla
    prsten_dolje = np.column_stack([kupa.wx + r_dno * np.cos(th),
                                    kupa.wy + r_dno * np.sin(th),
                                    np.full_like(th, z_dno)])
    n = len(th)
    verts = np.vstack([prsten_gore, prsten_dolje,
                       [[kupa.wx, kupa.wy, kupa.wz]],
                       [[kupa.wx, kupa.wy, z_dno]]])
    i_vrh, i_dno = 2 * n, 2 * n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, i_vrh])                       # plato (lepeza)
        faces.append([i, n + i, n + j]); faces.append([i, n + j, j])  # omotač
        faces.append([n + j, n + i, i_dno])               # dno
    kupa_mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces))
    kupa_mesh.fix_normals()

    # --- solid terena: visinsko polje ekstrudirano do z_dno ----------------
    x0, x1, y0, y1 = (kupa.wx - R_max, kupa.wx + R_max,
                      kupa.wy - R_max, kupa.wy + R_max)
    m = rezolucija_terena
    gx = np.linspace(x0, x1, m); gy = np.linspace(y0, y1, m)
    GX, GY = np.meshgrid(gx, gy)
    GZ = teren.z(GX, GY)
    # gornje tjeme + donje tjeme
    top = np.column_stack([GX.ravel(), GY.ravel(), GZ.ravel()])
    bot = top.copy(); bot[:, 2] = z_dno
    V = np.vstack([top, bot])
    idx = np.arange(m * m).reshape(m, m)
    F = []
    for a, b, c, d_ in zip(idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(),
                           idx[1:, 1:].ravel(), idx[1:, :-1].ravel()):
        F += [[a, b, c], [a, c, d_]]                       # gornja ploha
        o = m * m
        F += [[a + o, c + o, b + o], [a + o, d_ + o, c + o]]  # donja ploha
    # bočne stranice — rub bez ponovljenih ugaonih indeksa (degenerisani tr.)
    o = m * m
    rub_raw = (list(idx[0, :]) + list(idx[:, -1])
               + list(idx[-1, ::-1]) + list(idx[::-1, 0]))
    rubovi = [rub_raw[0]]
    for r_ in rub_raw[1:]:
        if r_ != rubovi[-1]:
            rubovi.append(r_)
    if rubovi[-1] == rubovi[0]:
        rubovi.pop()
    for p, q in zip(rubovi, rubovi[1:] + rubovi[:1]):
        F += [[p, q, q + o], [p, q + o, p + o]]
    teren_mesh = trimesh.Trimesh(vertices=V, faces=np.array(F))
    teren_mesh.fix_normals()

    try:
        if not (kupa_mesh.is_watertight and teren_mesh.is_watertight):
            return None
        presjek = trimesh.boolean.difference(
            [kupa_mesh, teren_mesh], engine="manifold")
        return float(abs(presjek.volume))
    except Exception:
        return None


# ===========================================================================
# EKONOMIJA: tačnije površine footprinta po zonama (umjesto ConvexHull-a)
# ===========================================================================

def povrsine_po_zonama(rezultat: PresekRezultat,
                       zone: list,
                       rezolucija: int = 256) -> dict[str, float]:
    """Površina osnove kupe (footprinta) unutar svake ekonomske zone.

    Stara implementacija (ekonomija.py) je aproksimirala površinu kao
    ConvexHull tačaka presjeka unutar zone — što precjenjuje kod konkavnih
    footprinta i uopšte ne vidi teren ispod kupe. Ovdje se footprint
    rasterizuje i sabira po zonama → tačno i za više petlji.

    Args:
        rezultat: PresekRezultat sa konturama.
        zone:     lista objekata sa .naziv, .x_data, .y_data (kao u loaders).
        rezolucija: rezolucija rasterizacije.

    Returns:
        {naziv_zone: površina_m2} — samo zone sa presjekom > 0.
    """
    from matplotlib.path import Path as MplPath

    if not rezultat.konture:
        return {}

    x0, x1, y0, y1 = rezultat.granice_racuna
    n = rezolucija
    gx = np.linspace(x0, x1, n)
    gy = np.linspace(y0, y1, n)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    dA = (gx[1] - gx[0]) * (gy[1] - gy[0])

    # maska footprinta: unutar bar jedne petlje (even-odd preko XOR-a
    # bi bila strožija; petlje su vanjske granice + eventualne rupe)
    maska = np.zeros(len(pts), dtype=bool)
    for kont in rezultat.konture:
        p = MplPath(kont[:, :2])
        maska ^= p.contains_points(pts)     # even-odd: rupe se oduzimaju

    if not np.any(maska):
        return {}

    out: dict[str, float] = {}
    unutra_pts = pts[maska]
    for zona in zone:
        poly = np.column_stack([np.asarray(zona.x_data, float),
                                np.asarray(zona.y_data, float)])
        if not np.allclose(poly[0], poly[-1]):
            poly = np.vstack([poly, poly[0]])
        u_zoni = MplPath(poly).contains_points(unutra_pts)
        A = float(u_zoni.sum()) * dA
        if A > 0:
            out[getattr(zona, "naziv", str(zona))] = A
    return out


# ===========================================================================
# DROP-IN ZAMJENA za geometry.zapremina_kupe (isti potpis i povratni tip)
# ===========================================================================

def zapremina_kupe_v2(
    wx: float, wy: float, wz: float,
    ugao: float, k: float,
    mnv: float,
    teren,                       # geometry.Surface ILI geometrija_v2.Teren
    zona_x: np.ndarray, zona_y: np.ndarray,
    ekonomska_fn=None,
    rezolucija: int = 192,
    rafiniranje: int = 1,
    profil: str = "matlab",
):
    """Drop-in zamjena za geometry.zapremina_kupe — isti potpis, isti
    RezultatKupe, ali tačna zapremina i za brdovit teren.

    Razlike u ponašanju:
      • zapremina = ∬ max(0, z_kupa − z_teren) dA  (ne ConvexHull!)
      • donja_povrsina = tačke glavne presječne konture (gusto uzorkovane)
      • intersect_surface.vertices = sve 3D tačke svih presječnih petlji
      • mnv se koristi samo za provjeru interesne zone (kao ranije)

    PREPORUKA ZA GA: proslijediti već izgrađen `Teren` objekat (napravljen
    jednom sa Teren.iz_surface(surface)) umjesto Surface — tako se Delaunay
    terena ne gradi ponovo u svakoj evaluaciji.
    """
    from geometry import RezultatKupe, Surface, inpolygon, _kontura_kupe

    VELIKA_VREDNOST = 40_000_000.0

    XX1, YY1, ZZ1, XX2, YY2, ZZ2 = _kontura_kupe(wx, wy, wz, k, mnv, ugao)

    def _neuspjeh(isurf=None):
        return RezultatKupe(
            zapremina=VELIKA_VREDNOST, donja_povrsina=None,
            ekonomska_cena=VELIKA_VREDNOST, zone="",
            gornja_kontura=XX1, gornja_kontura_y=YY1, z_gore=wz,
            intersect_surface=isurf)

    # provjera interesne zone — identična staroj logici
    if not np.all(inpolygon(XX2, YY2, np.asarray(zona_x), np.asarray(zona_y))):
        return _neuspjeh()

    t = teren if isinstance(teren, Teren) else Teren.iz_surface(teren)
    kupa = Kupa(wx=wx, wy=wy, wz=wz, k=k, ugao=ugao, profil=profil)

    rez = presek_kupe_i_terena(kupa, t, rezolucija=rezolucija,
                               rafiniranje=rafiniranje)
    if not rez.ima_preseka or rez.zapremina <= 0:
        return _neuspjeh(Surface(vertices=np.empty((0, 3)),
                                 faces=np.empty((0, 3), dtype=int)))

    sve_tacke = np.vstack(rez.konture)
    isurf = Surface(vertices=sve_tacke, faces=np.empty((0, 3), dtype=int))

    glavna = rez.kontura_glavna
    donja = glavna[:, :2] if glavna is not None else sve_tacke[:, :2]

    cena, zone_str = 0.0, ""
    if ekonomska_fn is not None:
        try:
            cena, zone_str = ekonomska_fn(isurf)
        except Exception:
            cena, zone_str = 0.0, ""

    return RezultatKupe(
        zapremina=rez.zapremina,
        donja_povrsina=donja,
        ekonomska_cena=cena,
        zone=zone_str,
        gornja_kontura=XX1,
        gornja_kontura_y=YY1,
        z_gore=wz,
        intersect_surface=isurf,
    )

