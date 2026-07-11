"""
pipeline_v2.py — kompletan tok optimizacije odlagališta nad NOVOM geometrijom.

Zamjenjuje kombinaciju geometry.generiši_tačke + ga_funkcije + ga_pokretac,
ali sa geometrija_v2 (tačna zapremina i za nepravilan teren) i sa
ISPRAVLJENIM ograničenjima:

  • Monte Carlo izbor tačaka sada filtrira i:
      - tačke van pokrivenosti terena (nema podataka → "ispod terena i slično")
      - tačke dalje od uslov_distance od centra masa
        (u starom kodu ograničenje je postojalo, ali se NIJE prosljeđivalo
         u differential_evolution — sada se primjenjuje na izvoru)
  • Interesna zona se provjerava na STVARNOM footprintu (presječna kontura
    iz geometrija_v2), ne na gruboj 9-tačkastoj konturi.

Modul ne zavisi od Streamlita — koristi ga app_v2.py, a može se pokrenuti
i iz konzole / testova.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.optimize import differential_evolution

from geometrija_v2 import Kupa, Teren, presek_kupe_i_terena, povrsine_po_zonama
from ekonomija import cijena_zone, distanca_od_centra_masa, racunaj_troskove


VELIKA = 40_000_000.0


# ===========================================================================
# 1) MONTE CARLO IZBOR KANDIDAT-TAČAKA
# ===========================================================================

@dataclass
class MCTacke:
    """Rezultat Monte Carlo izbora tačaka, sa razlozima odbacivanja."""
    prihvacene: np.ndarray            # (M, 2) — X, Y
    odbacene: dict[str, np.ndarray]   # razlog -> (K, 2)
    n_generisano: int

    @property
    def statistika(self) -> dict[str, int]:
        s = {"generisano": self.n_generisano,
             "prihvaćeno": len(self.prihvacene)}
        for razlog, t in self.odbacene.items():
            s[f"odbačeno: {razlog}"] = len(t)
        return s


def monte_carlo_tacke(
    n: int,
    teren: Teren,
    zona_x: np.ndarray,
    zona_y: np.ndarray,
    centar_masa: Optional[np.ndarray] = None,
    uslov_distance: Optional[float] = None,
    lose_zone: Optional[list] = None,
    filtriraj_teren: bool = True,
    seed: Optional[int] = None,
) -> MCTacke:
    """Monte Carlo generisanje kandidat-tačaka sa svim filterima.

    Nasljednik geometry.generiši_tačke, ali:
      - generiše samo (X, Y) — wz nije slučajan, njega optimizuje GA
        strogo IZNAD terena, pa tačke "ispod terena" ne mogu nastati;
      - filtrira tačke van pokrivenosti terena (van Delaunay omotača
        oblaka tačaka — tamo kota terena nije definisana);
      - filtrira tačke dalje od uslov_distance od centra masa
        (transportno ograničenje — sada zaista primijenjeno);
      - filtrira tačke unutar loših (K) zona — kao i ranije;
      - filtrira tačke van interesne zone — kao i ranije.

    Redoslijed filtera je od jeftinijeg ka skupljem; svaka tačka se
    odbacuje s PRVIM razlogom koji je diskvalifikuje.

    Returns:
        MCTacke — prihvaćene tačke + odbačene grupisane po razlogu.
    """
    rng = np.random.default_rng(seed)

    # uzorkuj u bbox-u interesne zone (kao original)
    zx = np.asarray(zona_x, float)
    zy = np.asarray(zona_y, float)
    x = rng.uniform(zx.min(), zx.max(), n)
    y = rng.uniform(zy.min(), zy.max(), n)
    pts = np.column_stack([x, y])

    aktivna = np.ones(n, dtype=bool)
    odbacene: dict[str, np.ndarray] = {}

    def primijeni(maska_lose: np.ndarray, razlog: str):
        nonlocal aktivna
        lose = aktivna & maska_lose
        if np.any(lose):
            odbacene[razlog] = pts[lose]
        aktivna = aktivna & ~maska_lose

    # 1. interesna zona (poligon)
    poly = np.column_stack([zx, zy])
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    u_zoni = MplPath(poly).contains_points(pts)
    primijeni(~u_zoni, "van interesne zone")

    # 2. loše (Z-5) zone
    if lose_zone:
        u_losoj = np.zeros(n, dtype=bool)
        for zona in lose_zone:
            zp = np.column_stack([np.asarray(zona.x_data, float),
                                  np.asarray(zona.y_data, float)])
            if not np.allclose(zp[0], zp[-1]):
                zp = np.vstack([zp, zp[0]])
            u_losoj |= MplPath(zp).contains_points(pts)
        primijeni(u_losoj, "u lošoj (K) zoni")

    # 3. distanca od centra masa (transportno ograničenje)
    if centar_masa is not None and uslov_distance is not None and uslov_distance > 0:
        d = np.hypot(pts[:, 0] - float(centar_masa[0]),
                     pts[:, 1] - float(centar_masa[1]))
        primijeni(d > uslov_distance, "predaleko od centra masa")

    # 4. pokrivenost terena — tačka mora ležati na definisanom terenu
    if filtriraj_teren and hasattr(teren, "tri"):
        van = teren.tri.find_simplex(pts) < 0
        primijeni(van, "van pokrivenosti terena")

    return MCTacke(prihvacene=pts[aktivna], odbacene=odbacene, n_generisano=n)


# ===========================================================================
# 2) EKONOMIJA nad novom geometrijom
# ===========================================================================

def ekonomska_cijena_v2(rez, dobre_zone: list,
                        precizno: bool = False) -> tuple[float, str]:
    """Ekonomska cijena kupe iz PresekRezultat.

    precizno=False (za GA petlju): zona ulazi u cijenu ako presječna
        kontura prolazi kroz nju — isto kao original, ali sa gusto
        uzorkovanom konturom (stotine tačaka umjesto ~18).
    precizno=True (za finalni izvještaj): površina footprinta unutar
        svake zone se rasterizuje (povrsine_po_zonama) pa se cijena
        računa na STVARNU zahvaćenu površinu, ne na površinu cijele zone.
    """
    if not rez.ima_preseka or not rez.konture:
        return 0.0, ""

    if precizno:
        po_zonama = povrsine_po_zonama(rez, dobre_zone, rezolucija=256)
        ukupno = sum(cijena_zone(naziv, p) for naziv, p in po_zonama.items())
        zone_str = ",".join(po_zonama.keys()) + ("," if po_zonama else "")
        return float(ukupno), zone_str

    tacke = np.vstack(rez.konture)[:, :2]
    ukupno, zone_lista = 0.0, []
    for zona in dobre_zone:
        zp = np.column_stack([np.asarray(zona.x_data, float),
                              np.asarray(zona.y_data, float)])
        if not np.allclose(zp[0], zp[-1]):
            zp = np.vstack([zp, zp[0]])
        if np.any(MplPath(zp).contains_points(tacke)):
            ukupno += cijena_zone(zona.naziv, zona.povrsina)
            zone_lista.append(zona.naziv)
    return float(ukupno), ",".join(zone_lista) + ("," if zone_lista else "")


# ===========================================================================
# 3) PRORAČUN JEDNE TAČKE (fiksna kupa ili GA optimizacija wz, k)
# ===========================================================================

@dataclass
class KontekstV2:
    """Sve što proračun treba — gradi se jednom, dijeli kroz sve tačke."""
    teren: Teren
    zona_x: np.ndarray
    zona_y: np.ndarray
    dobre_zone: list
    centar_masa: np.ndarray
    mnv: float
    ugao: float = 37.0
    profil: str = "matlab"
    donja_granica_zapremine: float = 0.0
    gornja_granica_zapremine: float = 39_000_000.0
    uslov_distance: float = float("inf")
    rezolucija: int = 160
    rafiniranje: int = 1
    lose_zone: list = field(default_factory=list)

    _zona_path: MplPath = field(init=False, repr=False, default=None)
    _lose_paths: list = field(init=False, repr=False, default=None)

    def zona_path(self) -> MplPath:
        if self._zona_path is None:
            p = np.column_stack([np.asarray(self.zona_x, float),
                                 np.asarray(self.zona_y, float)])
            if not np.allclose(p[0], p[-1]):
                p = np.vstack([p, p[0]])
            self._zona_path = MplPath(p)
        return self._zona_path

    def lose_paths(self) -> list:
        if self._lose_paths is None:
            self._lose_paths = []
            for zona in (self.lose_zone or []):
                p = np.column_stack([np.asarray(zona.x_data, float),
                                     np.asarray(zona.y_data, float)])
                if not np.allclose(p[0], p[-1]):
                    p = np.vstack([p, p[0]])
                self._lose_paths.append((zona, MplPath(p)))
        return self._lose_paths


@dataclass
class RezultatTackeV2:
    naziv: str
    wx: float
    wy: float
    wz: float
    k: float
    f_vrednost: float
    zapremina: float
    povrsina_osnove: float
    broj_petlji: int
    ugao: float
    distanca: float
    c1: float
    c2: float
    c3: float
    ukupna_cena: float
    zone: str
    unutar_zone: bool
    konture: list = field(default_factory=list)
    trajanje_s: float = 0.0

    ZAGLAVLJE = ["Naziv_tacke", "X", "Y", "Z_vrha", "K", "Funkcija_cilja",
                 "Zapremina_m3", "Osnova_m2", "Petlji", "Ugao",
                 "Distanca_m", "c1_transport", "c2_visina", "c3_zemljiste",
                 "Ukupna_cena", "Zone", "Unutar_zone"]

    def kao_red(self) -> list:
        return [self.naziv, self.wx, self.wy, self.wz, self.k,
                self.f_vrednost, self.zapremina, self.povrsina_osnove,
                self.broj_petlji, self.ugao, self.distanca,
                self.c1, self.c2, self.c3, self.ukupna_cena,
                self.zone, self.unutar_zone]


def _evaluiraj(wz: float, k: float, wx: float, wy: float,
               ctx: KontekstV2, precizno: bool = False):
    """Jedna evaluacija: presjek + zapremina + ograničenja + cijena.

    Vraća (f, rez, eko, zone_str, distanca). f = VELIKA znači nedopustivo.
    """
    kupa = Kupa(wx=wx, wy=wy, wz=wz, k=k, ugao=ctx.ugao, profil=ctx.profil)
    rez = presek_kupe_i_terena(kupa, ctx.teren,
                               rezolucija=ctx.rezolucija,
                               rafiniranje=ctx.rafiniranje)
    if not rez.ima_preseka or rez.zapremina <= 0:
        return VELIKA, rez, 0.0, "", 0.0

    V = rez.zapremina
    # granice zapremine (kao MATLAB c(1), c(2))
    if V < ctx.donja_granica_zapremine or V > ctx.gornja_granica_zapremine:
        return VELIKA, rez, 0.0, "", 0.0

    # INTERESNA ZONA: stvarni footprint (sve presječne konture) mora biti
    # unutar poligona — nasljednik unutarInteresneZone, ali na pravoj konturi
    tacke = np.vstack(rez.konture)[:, :2]
    if not np.all(ctx.zona_path().contains_points(tacke)):
        return VELIKA, rez, 0.0, "", 0.0

    # LOŠE (K) ZONE: footprint kupe ne smije ZAHVATATI lošu zonu.
    # Dva testa: (a) presječna kontura ulazi u K zonu; (b) K zona je
    # cijela unutar footprinta (kontura je ne siječe, ali je površina
    # osnove prekriva) — provjera tjemena K zone unutar kontura.
    if ctx.lose_zone:
        konture_paths = [MplPath(k[:, :2]) for k in rez.konture
                         if len(k) >= 3]
        for zona, put in ctx.lose_paths():
            if np.any(put.contains_points(tacke)):
                return VELIKA, rez, 0.0, "", 0.0
            tjemena = np.column_stack([np.asarray(zona.x_data, float),
                                       np.asarray(zona.y_data, float)])
            for kp in konture_paths:
                if np.any(kp.contains_points(tjemena)):
                    return VELIKA, rez, 0.0, "", 0.0

    # DISTANCA od centra masa (transportno ograničenje, MATLAB c(3))
    distanca = distanca_od_centra_masa(wx, wy, ctx.centar_masa)
    if distanca > ctx.uslov_distance:
        return VELIKA, rez, 0.0, "", distanca

    eko, zone_str = ekonomska_cijena_v2(rez, ctx.dobre_zone, precizno=precizno)

    # funkcija cilja — identična originalu, ali sa mnv umjesto zakucanog 90
    c1 = (distanca / 1000.0) * 0.8 * V
    # max(0, ...): kota vrha ispod mnv NE smije davati negativan (nagradni)
    # trošak dizanja — bez ovoga GA 'bježi' u kop da naplati kopanje
    c2 = V * ((max(wz - ctx.mnv, 0.0) / 0.08 * 1.6) / 1000.0) * 1.2
    f = (c1 + c2 + eko) / V
    return f, rez, eko, zone_str, distanca


def provjeri_footprint(rez, ctx: KontekstV2) -> list[str]:
    """Provjere footprinta (presječnih kontura) prema ograničenjima:
    interesna zona i loše (K) zone. Vraća listu prekršaja (prazna = OK).

    Ista logika kao u _evaluiraj, ali kao javna funkcija za upotrebu
    van GA petlje (npr. tab 'Stepenasti prikaz' — širenje dna pri
    povećanju visine mora ostati unutar zone i van K zona).
    """
    prekrsaji: list[str] = []
    if not rez.ima_preseka or not rez.konture:
        return ["kupa nema presjeka s terenom"]

    tacke = np.vstack(rez.konture)[:, :2]
    if not np.all(ctx.zona_path().contains_points(tacke)):
        prekrsaji.append("footprint izlazi iz interesne zone")

    if ctx.lose_zone:
        konture_paths = [MplPath(k[:, :2]) for k in rez.konture
                         if len(k) >= 3]
        for zona, put in ctx.lose_paths():
            pogodak = np.any(put.contains_points(tacke))
            if not pogodak:
                tjemena = np.column_stack([np.asarray(zona.x_data, float),
                                           np.asarray(zona.y_data, float)])
                pogodak = any(np.any(kp.contains_points(tjemena))
                              for kp in konture_paths)
            if pogodak:
                prekrsaji.append(f"footprint zahvata lošu zonu {zona.naziv}")
    return prekrsaji


def proracun_tacke(
    naziv: str, wx: float, wy: float, ctx: KontekstV2,
    mod: str = "ga",
    wz_fiksno: Optional[float] = None,
    k_fiksno: float = 120.0,
    populacija: int = 20,
    max_generacija: int = 3,
    seed: Optional[int] = None,
) -> Optional[RezultatTackeV2]:
    """Proračun jedne kandidat-tačke.

    mod="fiksno": kupa zadatih dimenzija (wz_fiksno ili teren+40, k_fiksno)
                  — brz pregled svih tačaka bez optimizacije.
    mod="ga":     differential_evolution optimizuje (wz, k) minimizujući
                  funkciju cilja — nasljednik MATLAB ga() poziva.

    Returns:
        RezultatTackeV2 ili None ako je tačka nedopustiva.
    """
    t0 = time.perf_counter()
    z_tu = float(ctx.teren.z(wx, wy))

    if mod == "fiksno":
        wz = wz_fiksno if wz_fiksno is not None else z_tu + 40.0
        k = k_fiksno
        f, rez, eko, zone_str, dist = _evaluiraj(wz, k, wx, wy, ctx,
                                                 precizno=True)
        if f >= VELIKA:
            return None
    else:
        # bounds izvedeni iz terena (kao get_bounds "auto")
        zr = ctx.teren.z_max - ctx.teren.z_min
        wz_lo = max(z_tu + 5.0, ctx.teren.z_min + 0.1 * zr)
        wz_hi = ctx.teren.z_max + 0.5 * zr
        dij = float(np.hypot(np.ptp(ctx.zona_x), np.ptp(ctx.zona_y)))
        k_lo, k_hi = max(10.0, 0.02 * dij), 0.35 * dij

        def cilj(x):
            return _evaluiraj(float(x[0]), float(x[1]), wx, wy, ctx)[0]

        try:
            out = differential_evolution(
                cilj, bounds=[(wz_lo, wz_hi), (k_lo, k_hi)],
                maxiter=max_generacija, popsize=populacija,
                mutation=(0.5, 1.0), recombination=0.8,
                strategy="best1bin", init="latinhypercube",
                seed=seed, polish=False, disp=False)
        except Exception:
            return None
        if not np.isfinite(out.fun) or out.fun >= VELIKA:
            return None
        wz, k = float(out.x[0]), float(out.x[1])
        # finalna evaluacija sa preciznom ekonomijom
        f, rez, eko, zone_str, dist = _evaluiraj(wz, k, wx, wy, ctx,
                                                 precizno=True)
        if f >= VELIKA:
            return None

    c1, c2, c3 = racunaj_troskove(rez.zapremina, dist, wz, eko, mnv=ctx.mnv)
    return RezultatTackeV2(
        naziv=naziv, wx=wx, wy=wy, wz=wz, k=k,
        f_vrednost=f, zapremina=rez.zapremina,
        povrsina_osnove=rez.povrsina_osnove, broj_petlji=rez.broj_petlji,
        ugao=ctx.ugao, distanca=dist, c1=c1, c2=c2, c3=c3,
        ukupna_cena=c1 + c2 + c3,
        zone=zone_str, unutar_zone=True, konture=rez.konture,
        trajanje_s=time.perf_counter() - t0)


def proracun_svih_tacaka(
    tacke: np.ndarray, ctx: KontekstV2,
    mod: str = "ga",
    callback: Optional[Callable[[int, int, Optional[RezultatTackeV2]], None]] = None,
    **kwargs,
) -> list[RezultatTackeV2]:
    """Proračun svih MC tačaka; callback(i, n, rezultat) za progres bar."""
    rezultati = []
    n = len(tacke)
    for i, (wx, wy) in enumerate(tacke):
        r = proracun_tacke(f"point_{i + 1}", float(wx), float(wy), ctx,
                           mod=mod, **kwargs)
        if r is not None:
            rezultati.append(r)
        if callback:
            callback(i + 1, n, r)
    rezultati.sort(key=lambda r: r.f_vrednost)
    return rezultati

