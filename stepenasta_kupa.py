"""
stepenasta_kupa.py — stepenasta (etažna) kupa odlagališta.

Standardna rudarska geometrija: umjesto jedne kontinualne kosine, tijelo
se gradi u ETAŽAMA — kosina visine `korak` pod uglom `ugao`, pa horizontalna
BERMA širine `berma`, pa sljedeća kosina... Tlocrtni oblik (r_top(θ) profil)
je očuvan; svaki niži nivo je isti oblik proširen radijalno.

Ključno: stepenasta kupa je i dalje VISINSKO POLJE z(x, y), pa sav postojeći
aparat iz geometrija_v2 (presek_kupe_i_terena — zapremina, presječne
konture, površina osnove) radi bez ijedne izmjene: objekat implementira
isti interfejs (.z i .max_radijus) kao Kupa.

Radijalni profil, gledano od ivice platoa (u = r − r_top(θ) ≥ 0):

    ciklus  = kosina širine  fw = korak / tan(ugao)   (pad za `korak` m)
            + berma  širine  berma                     (pad 0)

    pad(u):  k = floor(u / (fw + berma));  rem = u − k·(fw + berma)
             rem ≤ fw  →  pad = k·korak + rem·tan(ugao)
             rem >  fw  →  pad = (k + 1)·korak

    z(x, y) = wz − pad(u)      (wz za u < 0, tj. na platou)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from geometrija_v2 import Kupa


@dataclass
class StepenastaKupa:
    """Etažna kupa — isti interfejs kao geometrija_v2.Kupa (z, max_radijus).

    Args:
        wx, wy, wz: vrh (kota platoa)
        k:          širina platoa (kao kod Kupa)
        ugao:       ugao kosine etaže u STEPENIMA
        profil:     "krug" ili "matlab" (tlocrtni oblik, kao kod Kupa)
        korak:      visina jedne etaže (m)
        berma:      širina horizontalne berme između etaža (m)
    """
    wx: float
    wy: float
    wz: float
    k: float
    ugao: float
    korak: float = 10.0
    berma: float = 4.0
    profil: str = "matlab"

    _kupa: Kupa = field(init=False, repr=False)

    def __post_init__(self):
        if self.korak <= 0:
            raise ValueError("korak (visina etaže) mora biti > 0")
        if self.berma < 0:
            raise ValueError("berma ne može biti negativna")
        # unutrašnja glatka kupa — daje r_top(θ) profil i validaciju
        self._kupa = Kupa(wx=self.wx, wy=self.wy, wz=self.wz,
                          k=self.k, ugao=self.ugao, profil=self.profil)

    # -- geometrija ----------------------------------------------------------

    @property
    def tan_ugla(self) -> float:
        return self._kupa.tan_ugla

    @property
    def sirina_kosine(self) -> float:
        """Horizontalna širina jedne kosine (fw = korak / tan)."""
        return self.korak / self.tan_ugla

    @property
    def sirina_ciklusa(self) -> float:
        return self.sirina_kosine + self.berma

    def r_top(self, theta):
        return self._kupa.r_top(theta)

    def gornja_kontura(self, n: int = 128):
        return self._kupa.gornja_kontura(n)

    def _pad(self, u: np.ndarray) -> np.ndarray:
        """Vertikalni pad profila na horizontalnoj udaljenosti u ≥ 0
        od ivice platoa (vektorizovano)."""
        u = np.maximum(np.asarray(u, dtype=float), 0.0)
        cw = self.sirina_ciklusa
        fw = self.sirina_kosine
        ciklusa = np.floor(u / cw)
        rem = u - ciklusa * cw
        pad_kosine = np.minimum(rem, fw) * self.tan_ugla
        return ciklusa * self.korak + pad_kosine

    def z(self, x, y) -> np.ndarray:
        """Visina stepenaste površine u (x, y) — vektorizovano."""
        dx = np.asarray(x, dtype=float) - self.wx
        dy = np.asarray(y, dtype=float) - self.wy
        r = np.hypot(dx, dy)
        rt = self.r_top(np.arctan2(dy, dx))
        return self.wz - self._pad(r - rt)

    def max_radijus(self, z_min: float) -> float:
        """Najveći doseg profila dok ne padne do z_min (berme ga šire
        u odnosu na glatku kupu)."""
        pad = max(0.0, self.wz - z_min)
        n_etaza = int(np.ceil(pad / self.korak))
        r_top_max = float(self._kupa._r_cvorovi.max())
        return (r_top_max + n_etaza * self.sirina_ciklusa)

    # -- ivice etaža za prikaz / DXF -----------------------------------------

    def ivice_etaza(self, teren=None, n_theta: int = 180,
                    max_etaza: int = 200) -> list[dict]:
        """3D prstenovi ivica etaža (za crtanje ili DXF).

        Za svaku etažu vraća do tri prstena: vrh kosine, dno kosine
        (= unutrašnja ivica berme) i vanjska ivica berme. Ako je zadan
        teren, dijelovi prstena ispod terena se maskiraju (NaN).

        Returns:
            lista {"z": kota, "x": (n,), "y": (n,), "tip": str}
        """
        th = np.linspace(0.0, 2.0 * np.pi, n_theta)
        rt = self.r_top(th)
        cw, fw = self.sirina_ciklusa, self.sirina_kosine

        prstenovi = []

        def dodaj(u_pomak: float, kota: float, tip: str):
            x = self.wx + (rt + u_pomak) * np.cos(th)
            y = self.wy + (rt + u_pomak) * np.sin(th)
            z = np.full_like(x, kota)
            if teren is not None:
                zt = teren.z(x, y)
                maska = kota <= zt          # ispod terena → sakrij
                if maska.all():
                    return False
                x = np.where(maska, np.nan, x)
                y = np.where(maska, np.nan, y)
                z = np.where(maska, np.nan, z)
            prstenovi.append({"x": x, "y": y, "z": z, "kota": kota, "tip": tip})
            return True

        # ivica platoa
        ziv = dodaj(0.0, self.wz, "plato")
        for k_i in range(max_etaza):
            kota_dna = self.wz - (k_i + 1) * self.korak
            a = dodaj(k_i * cw + fw, kota_dna, "dno kosine")
            b = True
            if self.berma > 0:
                b = dodaj((k_i + 1) * cw, kota_dna, "vanjska ivica berme")
            if not (a or b):
                break                        # cijela etaža ispod terena
        return prstenovi
