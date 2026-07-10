"""
monte_karlo.py — Monte Carlo zapremina tijela između kupe i terena.

Zašto postoji: NEZAVISNA provjera mrežne metode iz geometrija_v2.py.
Dvije metode koje dijele isti kod dijele i iste greške; MC koristi potpuno
drugačiji princip (slučajno uzorkovanje umjesto kvadrature), pa ako se
slože, rezultat je vjerodostojan.

Za produkciju (GA) mrežna metoda je BOLJA — vidi poređenje na dnu docstringa.
MC ovdje služi kao kontrola i kao osnova ako se ikad pređe na pravi 3D
(previsi), gdje visinsko polje više ne važi.

Tri varijante:

  1. "hit"  — hit-or-miss: bacaj tačke u 3D kvadar, broji koliko ih upadne
              između terena i kupe.  V = udio · V_kvadra
              Greška ~ O(1/√N), najsporija konvergencija. Didaktički jasna.

  2. "mean" — srednja vrijednost (default): bacaj tačke u 2D pravougaonik,
              usrednji max(0, z_kupa − z_teren).  V = A · srednja_vrijednost
              Ista složenost, ali VIŠESTRUKO manja varijansa — jer se
              integrira glatka funkcija umjesto indikatora 0/1.

  3. "qmc"  — kvazi-Monte Carlo: isto kao "mean", ali umjesto slučajnih
              koristi Sobolov niz s malom diskrepancom.
              Greška ~ O(log^d N / N) — praktično red veličine bolja.

Sve tri vraćaju i procjenu standardne greške, pa znaš koliko da vjeruješ.

Upotreba:

    from geometrija_v2 import Kupa, Teren
    from monte_karlo import zapremina_monte_carlo

    r = zapremina_monte_carlo(kupa, teren, n=2_000_000, metoda="qmc")
    print(r.zapremina, "±", r.std_greska)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometrija_v2 import Kupa, Teren


@dataclass
class MCRezultat:
    zapremina: float          # m³
    std_greska: float         # m³ — 1σ procjena (za "qmc" konzervativna)
    povrsina_osnove: float    # m² — mjera skupa {d > 0}
    n_uzoraka: int
    metoda: str
    granice: tuple[float, float, float, float]

    @property
    def rel_greska(self) -> float:
        return self.std_greska / self.zapremina if self.zapremina > 0 else float("inf")

    def __str__(self) -> str:
        return (f"{self.metoda:5s} N={self.n_uzoraka:>10,}  "
                f"V = {self.zapremina:14,.0f} m³  ± {self.std_greska:10,.0f}  "
                f"({self.rel_greska*100:.3f} %)")


def _bbox(kupa: Kupa, teren: Teren, margina: float = 1.05):
    R = kupa.max_radijus(teren.z_min) * margina
    return (kupa.wx - R, kupa.wx + R, kupa.wy - R, kupa.wy + R)


def zapremina_monte_carlo(kupa: Kupa, teren: Teren,
                          n: int = 1_000_000,
                          metoda: str = "mean",
                          seed: int | None = 12345,
                          blok: int = 500_000) -> MCRezultat:
    """Monte Carlo zapremina tijela između kupe i terena.

    Args:
        kupa, teren: isti objekti kao u geometrija_v2.
        n:           broj uzoraka (za "qmc" se zaokružuje na stepen dvojke).
        metoda:      "hit" | "mean" | "qmc".
        seed:        za ponovljivost.
        blok:        koliko uzoraka odjednom (kontrola memorije).
    """
    x0, x1, y0, y1 = _bbox(kupa, teren)
    A = (x1 - x0) * (y1 - y0)

    if metoda == "qmc":
        return _qmc(kupa, teren, n, seed, x0, x1, y0, y1, A)
    if metoda == "hit":
        return _hit_or_miss(kupa, teren, n, seed, x0, x1, y0, y1, blok)
    if metoda == "mean":
        return _mean_value(kupa, teren, n, seed, x0, x1, y0, y1, A, blok)
    raise ValueError(f"Nepoznata metoda: {metoda!r}")


def _mean_value(kupa, teren, n, seed, x0, x1, y0, y1, A, blok):
    """V = A · E[max(0, d)];  greška = A · s / √n."""
    rng = np.random.default_rng(seed)
    suma = suma_kv = 0.0
    poz = 0
    ostalo = n
    while ostalo > 0:
        m = min(blok, ostalo)
        px = rng.uniform(x0, x1, m)
        py = rng.uniform(y0, y1, m)
        d = np.maximum(kupa.z(px, py) - teren.z(px, py), 0.0)
        suma += d.sum()
        suma_kv += (d * d).sum()
        poz += int((d > 0).sum())
        ostalo -= m

    sr = suma / n
    var = max(suma_kv / n - sr * sr, 0.0)
    return MCRezultat(
        zapremina=A * sr,
        std_greska=A * np.sqrt(var / n),
        povrsina_osnove=A * poz / n,
        n_uzoraka=n, metoda="mean", granice=(x0, x1, y0, y1))


def _hit_or_miss(kupa, teren, n, seed, x0, x1, y0, y1, blok):
    """V = V_kvadra · udio tačaka između terena i kupe."""
    rng = np.random.default_rng(seed)
    z0, z1 = teren.z_min, kupa.wz
    if z1 <= z0:
        return MCRezultat(0.0, 0.0, 0.0, n, "hit", (x0, x1, y0, y1))
    V_kvadar = (x1 - x0) * (y1 - y0) * (z1 - z0)

    pogodaka = 0
    poz = 0
    ostalo = n
    while ostalo > 0:
        m = min(blok, ostalo)
        px = rng.uniform(x0, x1, m)
        py = rng.uniform(y0, y1, m)
        pz = rng.uniform(z0, z1, m)
        zt = teren.z(px, py)
        zk = kupa.z(px, py)
        unutra = (pz > zt) & (pz < zk)
        pogodaka += int(unutra.sum())
        poz += int((zk > zt).sum())
        ostalo -= m

    p = pogodaka / n
    # varijansa Bernoullija: p(1-p)/n
    return MCRezultat(
        zapremina=V_kvadar * p,
        std_greska=V_kvadar * np.sqrt(max(p * (1 - p), 0.0) / n),
        povrsina_osnove=(x1 - x0) * (y1 - y0) * poz / n,
        n_uzoraka=n, metoda="hit", granice=(x0, x1, y0, y1))


def _qmc(kupa, teren, n, seed, x0, x1, y0, y1, A):
    """Sobolov niz umjesto slučajnih tačaka — brža konvergencija."""
    from scipy.stats import qmc

    m = int(np.ceil(np.log2(max(n, 2))))
    n_stv = 2 ** m
    sob = qmc.Sobol(d=2, scramble=True, seed=seed)
    u = sob.random_base2(m)
    px = x0 + u[:, 0] * (x1 - x0)
    py = y0 + u[:, 1] * (y1 - y0)

    d = np.maximum(kupa.z(px, py) - teren.z(px, py), 0.0)
    sr = d.mean()
    # Za QMC klasična formula za grešku ne važi; koristimo je kao KONZERVATIVNU
    # gornju ogradu (stvarna greška je obično znatno manja).
    std = A * d.std(ddof=1) / np.sqrt(n_stv)
    return MCRezultat(
        zapremina=A * sr, std_greska=std,
        povrsina_osnove=A * float((d > 0).mean()),
        n_uzoraka=n_stv, metoda="qmc", granice=(x0, x1, y0, y1))


# ===========================================================================
# Poređenje sa mrežnom metodom — pozovi za nezavisnu verifikaciju
# ===========================================================================

def uporedi_sa_mrezom(kupa: Kupa, teren: Teren,
                      n_mc: int = 4_000_000,
                      rezolucija: int = 512,
                      rafiniranje: int = 2) -> None:
    """Ispiše mrežnu i sve tri MC procjene za istu kupu i teren."""
    import time
    from geometrija_v2 import presek_kupe_i_terena

    t0 = time.perf_counter()
    g = presek_kupe_i_terena(kupa, teren, rezolucija, rafiniranje)
    t_g = time.perf_counter() - t0
    print(f"mreza n={rezolucija} raf={rafiniranje}   "
          f"V = {g.zapremina:14,.0f} m³  ± {g.procjena_greske:10,.0f}  "
          f"({g.procjena_greske/g.zapremina*100:.3f} %)   {t_g*1000:7.0f} ms")

    for met in ("hit", "mean", "qmc"):
        t0 = time.perf_counter()
        r = zapremina_monte_carlo(kupa, teren, n=n_mc, metoda=met)
        dt = time.perf_counter() - t0
        odst = (r.zapremina - g.zapremina) / g.zapremina * 100
        print(f"{r}   {dt*1000:7.0f} ms   odst. od mreze: {odst:+.3f} %")


if __name__ == "__main__":
    from geometrija_v2 import zapremina_zarubljene_kupe_ravan_teren

    print("=== Ravan teren: postoji EGZAKTNA formula ===")
    k = Kupa(wx=0, wy=0, wz=180, k=100, ugao=37, profil="krug")
    t = Teren.ravan(140.0)
    V_egz = zapremina_zarubljene_kupe_ravan_teren(k, 140.0)
    print(f"egzaktno                     V = {V_egz:14,.0f} m³\n")
    uporedi_sa_mrezom(k, t, n_mc=2_000_000)

    print("\n=== Brdovit teren (sinusni reljef) ===")
    brdo = lambda x, y: (145.0 + 8.0*np.sin((x)/60.0)*np.cos((y)/45.0)
                         + 5.0*np.sin((x)/23.0 + 1.0))
    tb = Teren.analiticki(brdo, (-500, 500, -500, 500), (130.0, 160.0))
    uporedi_sa_mrezom(k, tb, n_mc=2_000_000)

    print("\n=== Konvergencija: greška vs broj uzoraka (brdovit teren) ===")
    from geometrija_v2 import presek_kupe_i_terena
    V_ref = presek_kupe_i_terena(k, tb, 2048, 2).zapremina
    print(f"{'N':>10}  {'hit':>12}  {'mean':>12}  {'qmc':>12}   (|greška| u %)")
    for n in (10_000, 100_000, 1_000_000):
        red = [f"{n:>10,}"]
        for met in ("hit", "mean", "qmc"):
            r = zapremina_monte_carlo(k, tb, n=n, metoda=met)
            red.append(f"{abs(r.zapremina-V_ref)/V_ref*100:>12.4f}")
        print("  ".join(red))
