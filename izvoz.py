"""
izvoz.py  –  Korak 5: izvoz rezultata u Excel i DXF

Zamjenjuje MATLAB funkcije:
  writetable() + array2table()  → izvezi_excel()   (pandas)
  izvozCADga.m                  → izvezi_dxf()     (ezdxf / fallback: ručni DXF)
  dxf_open/close/polymesh/polyline/print_vertex/...  → sve zamijenjeno

Izvozi se dva Excel fajla:
  {timestamp}_export_ga.xls       → svi rezultati GA
  {timestamp}_export_ga_final.xls → samo validni (unutar interesne zone)

Izvozi se jedan DXF fajl:
  {timestamp}_line_export_ga_24.dxf → konture kupa (gornja + donja linija)
"""

from __future__ import annotations

import struct
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from ga_pokretac import RezultatTacke


# ---------------------------------------------------------------------------
# Timestamp  (identičan format kao MATLAB)
# ---------------------------------------------------------------------------

def timestamp_string() -> str:
    """Vraća timestamp string u MATLAB formatu: DD-Mon-YYYY_HH_MM_SS"""
    return datetime.now().strftime("%d-%b-%Y_%H_%M_%S")


# ---------------------------------------------------------------------------
# Excel izvoz  (zamjenjuje writetable + array2table)
# ---------------------------------------------------------------------------

def izvezi_excel(
    rezultati: list[RezultatTacke],
    putanja: str | Path,
) -> Path:
    """Izvozi listu rezultata u Excel (.xlsx) fajl.

    MATLAB ekvivalent:
        outputFinal = array2table(tableDataFinal);
        outputFinal.Properties.VariableNames = cellstr(headerFinal);
        writetable(outputFinal, titleFinal);

    13 kolona identičnih MATLAB headerFinal:
        Naziv_tacke, X_koordinata, Y_koordinata, Z_koordinata, K,
        Funkcija_cilja, Zapremina, Ugao, distanca, c1, c2, c3, Zone

    Args:
        rezultati: lista RezultatTacke objekata
        putanja:   putanja do output fajla (.xlsx)

    Returns:
        Path do zapisanog fajla
    """
    try:
        import csv
        putanja = Path(putanja)
        # Koristimo CSV kao fallback ako pandas nije dostupan
        with open(putanja.with_suffix('.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(RezultatTacke.ZAGLAVLJE)
            for r in rezultati:
                writer.writerow(r.kao_red())
        return putanja.with_suffix('.csv')
    except Exception:
        pass

    # Pokušaj sa pandas ako je dostupan
    try:
        import pandas as pd  # type: ignore
        putanja = Path(putanja)
        if not str(putanja).endswith(('.xlsx', '.xls')):
            putanja = putanja.with_suffix('.xlsx')

        redovi = [r.kao_red() for r in rezultati]
        df = pd.DataFrame(redovi, columns=RezultatTacke.ZAGLAVLJE)

        # Numeričke kolone — eksplicitno pretvoriti
        num_cols = ["X_koordinata", "Y_koordinata", "Z_koordinata", "K",
                    "Funkcija_cilja", "Zapremina", "Ugao",
                    "distanca", "c1", "c2", "c3"]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.to_excel(putanja, index=False, engine='openpyxl')
        return putanja

    except ImportError:
        # Fallback: CSV umjesto Excel
        import csv
        putanja = Path(str(putanja).replace('.xlsx', '.csv').replace('.xls', '.csv'))
        with open(putanja, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(RezultatTacke.ZAGLAVLJE)
            for r in rezultati:
                writer.writerow(r.kao_red())
        return putanja


def izvezi_oba_excela(
    svi: list[RezultatTacke],
    validni: list[RezultatTacke],
    izlazni_direktorij: str | Path = ".",
    ts: Optional[str] = None,
) -> tuple[Optional[Path], Optional[Path]]:
    """Izvozi oba Excel fajla identično MATLAB kodu.

    MATLAB ekvivalent:
        fileNameGA      = datumTimeString + '_export_ga.xls'
        fileNameGAFinal = datumTimeString + '_export_ga_final.xls'

    Returns:
        (putanja_svi, putanja_validni) — None ako je lista prazna
    """
    if ts is None:
        ts = timestamp_string()
    izlaz = Path(izlazni_direktorij)
    izlaz.mkdir(parents=True, exist_ok=True)

    putanja_svi = None
    putanja_validni = None

    if svi:
        putanja_svi = izvezi_excel(svi, izlaz / f"{ts}_export_ga.xlsx")
        print(f"  Excel (svi):    {putanja_svi.name}  ({len(svi)} redova)")

    if validni:
        putanja_validni = izvezi_excel(validni, izlaz / f"{ts}_export_ga_final.xlsx")
        print(f"  Excel (validni): {putanja_validni.name}  ({len(validni)} redova)")

    return putanja_svi, putanja_validni


# ---------------------------------------------------------------------------
# DXF izvoz  —  ručna implementacija (zamjenjuje svih 10 dxf_*.m fajlova)
# ---------------------------------------------------------------------------

class DxfWriter:
    """Minimalni DXF pisač kompatibilan sa AutoCAD DXF R12 formatom.

    Zamjenjuje 10 MATLAB dxf_*.m fajlova:
      dxf_open, dxf_close, dxf_set, dxf_polymesh,
      dxf_polyline, dxf_print_layer, dxf_print_point,
      dxf_print_vertex, dxf_print_seqend, dxf_rgb2aci

    Generira ASCII DXF koji otvara AutoCAD, BricsCAD, LibreCAD.
    """

    def __init__(self, putanja: str | Path, jedinice: str = "m"):
        self.putanja = Path(putanja)
        self._f = open(self.putanja, "w", encoding="ascii")
        self._sloj = 0
        self._boja = 7   # bijela
        self._pisati_zaglavlje()

    def _pisati_zaglavlje(self):
        f = self._f
        f.write("0\nSECTION\n2\nHEADER\n")
        f.write("9\n$ACADVER\n1\nAC1009\n")   # R12 — maksimalna kompatibilnost
        f.write("9\n$INSUNITS\n70\n6\n")       # 6 = metri
        f.write("0\nENDSEC\n")
        f.write("0\nSECTION\n2\nENTITIES\n")

    def postavi(self, sloj: int = None, boja: tuple = None):
        """Zamjenjuje dxf_set(FID, 'Layer', j, 'Color', [R G B])"""
        if sloj is not None:
            self._sloj = sloj
        if boja is not None:
            # RGB [0..1] → ACI indeks (aproksimacija)
            r, g, b = boja
            if r > 0.5 and g < 0.3 and b < 0.3:
                self._boja = 1   # crvena
            elif r < 0.3 and g < 0.3 and b > 0.5:
                self._boja = 5   # plava
            elif r < 0.3 and g > 0.5 and b < 0.3:
                self._boja = 3   # zelena
            else:
                self._boja = 7   # bijela/default

    def _print_sloj(self):
        self._f.write(f"8\n{self._sloj}\n62\n{self._boja}\n")

    def _print_tacka(self, n: int, x: float, y: float, z: float):
        """Zamjenjuje dxf_print_point"""
        self._f.write(f"1{n}\n{x:.8g}\n2{n}\n{y:.8g}\n3{n}\n{z:.8g}\n")

    def polimreža(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ):
        """Zapisuje 3D polyface mesh.  Zamjenjuje dxf_polymesh.m

        Args:
            vertices: (N, 3) array X, Y, Z
            faces:    (M, 3) array indeksa (0-based)
        """
        f = self._f
        f.write("0\nPOLYLINE\n")
        self._print_sloj()
        f.write("66\n1\n")
        self._print_tacka(0, 0.0, 0.0, 0.0)
        f.write("70\n64\n")   # polyface mesh flag

        # Upisivanje vrhova (tip 192 = 128+64)
        for v in vertices:
            f.write("0\nVERTEX\n")
            self._print_sloj()
            self._print_tacka(0, float(v[0]), float(v[1]), float(v[2]))
            f.write("70\n192\n")

        # Upisivanje lica (indeksi, 1-based u DXF)
        for face in faces:
            f.write("0\nVERTEX\n")
            self._print_sloj()
            self._print_tacka(0, 0.0, 0.0, 0.0)
            f.write("70\n128\n")
            f.write(f"71\n{int(face[0]) + 1}\n")
            f.write(f"72\n{int(face[1]) + 1}\n")
            f.write(f"73\n{int(face[2]) + 1}\n")
            f.write("74\n0\n")

        f.write(f"0\nSEQEND\n8\n{self._sloj}\n")

    def polilinija(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ):
        """Zapisuje 3D poliliniju.  Zamjenjuje dxf_polyline.m

        Args:
            x, y, z: 1D array koordinata tačaka linije
        """
        f = self._f
        f.write("0\nPOLYLINE\n")
        self._print_sloj()
        f.write("66\n1\n")
        self._print_tacka(0, 0.0, 0.0, 0.0)
        f.write("70\n8\n")   # 3D polyline flag

        for xi, yi, zi in zip(x, y, z):
            f.write("0\nVERTEX\n")
            self._print_sloj()
            self._print_tacka(0, float(xi), float(yi), float(zi))
            f.write("70\n32\n")

        f.write(f"0\nSEQEND\n8\n{self._sloj}\n")

    def zatvori(self):
        """Zamjenjuje dxf_close.m"""
        self._f.write("0\nENDSEC\n0\nEOF\n")
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.zatvori()


# ---------------------------------------------------------------------------
# DXF izvoz kupa  (zamjenjuje izvozCADga.m)
# ---------------------------------------------------------------------------

def izvezi_dxf(
    rezultati: list[RezultatTacke],
    ts: str,
    izlazni_direktorij: str | Path = ".",
) -> Optional[Path]:
    """Izvozi konture kupa u DXF fajl.

    MATLAB ekvivalent: izvozCADga(posleProvereUnutarZoneArray, datumTimeString)

    Generiše jedan DXF fajl sa:
      - gornjom konturom svake kupe (polilinija, sloj j)
      - donjom konturom svake kupe (polilinija, sloj j)
      - svaka kupa na zasebnom sloju (Layer = redni broj)

    MATLAB je generirao 2 DXF fajla (_tri i _line). Ovdje generiramo
    jedan kombinirani _line fajl koji sadrži iste entitete.

    Args:
        rezultati: lista validnih RezultatTacke (sa geometrijom)
        ts:        timestamp string
        izlazni_direktorij: gdje snimiti fajl

    Returns:
        Path do DXF fajla, ili None ako nema rezultata
    """
    if not rezultati:
        print("  DXF: nema rezultata za izvoz")
        return None

    izlaz = Path(izlazni_direktorij)
    izlaz.mkdir(parents=True, exist_ok=True)
    dxf_putanja = izlaz / f"{ts}_line_export_ga_24.dxf"

    with DxfWriter(dxf_putanja) as dxf:
        for j, rez in enumerate(rezultati, start=1):
            dxf.postavi(sloj=j, boja=(0.0, 0.0, 1.0))   # plava = linija

            # Gornja kontura (XX1, YY1, ZZ1)
            if rez.xx1 is not None:
                dxf.polilinija(rez.xx1, rez.yy1, rez.zz1)

            # Donja kontura (XX2, YY2, ZZ2)
            if rez.xx2 is not None:
                dxf.polilinija(rez.xx2, rez.yy2, rez.zz2)

    print(f"  DXF:             {dxf_putanja.name}  ({len(rezultati)} kupa)")
    return dxf_putanja


# ---------------------------------------------------------------------------
# Sve odjednom
# ---------------------------------------------------------------------------

def izvezi_sve(
    svi: list[RezultatTacke],
    validni: list[RezultatTacke],
    izlazni_direktorij: str | Path = ".",
) -> dict[str, Optional[Path]]:
    """Izvozi Excel i DXF jednim pozivom.

    MATLAB ekvivalent: blok nakon GA petlje u IzvrsniKodBuvac.m

    Returns:
        Dict sa ključevima 'excel_svi', 'excel_validni', 'dxf'
    """
    ts = timestamp_string()
    izlaz = Path(izlazni_direktorij)

    print(f"\nIzvoz rezultata [{ts}]...")
    p_svi, p_validni = izvezi_oba_excela(svi, validni, izlaz, ts)
    p_dxf = izvezi_dxf(validni, ts, izlaz)

    return {
        "excel_svi": p_svi,
        "excel_validni": p_validni,
        "dxf": p_dxf,
    }


# ---------------------------------------------------------------------------
# DXF izvoz za NOVE (pipeline_v2) rezultate — kupe svih dopustivih tačaka
# ---------------------------------------------------------------------------

def izvezi_dxf_v2(
    rezultati: list,
    teren,
    profil: str = "matlab",
    putanja: str | Path = "kupe_export.dxf",
    n_izvodnica: int = 24,
) -> Path:
    """Izvozi kupe svih dopustivih tačaka (RezultatTackeV2) u DXF R12.

    AutoCAD otvara fajl direktno (File → Open) i može ga snimiti kao .dwg.

    Za svaku tačku, na zasebnom sloju "point_N":
      • gornja kontura platoa (zatvorena 3D polilinija na koti wz) — plavo
      • sve presječne konture kupa–teren (3D polilinije na terenu) — crveno
      • izvodnice kosine: radijalne linije od ivice platoa do presjeka
        s terenom (daju CAD-u čitljiv 3D oblik kupe) — bijelo

    Args:
        rezultati:    lista RezultatTackeV2 (moraju imati .konture)
        teren:        geometrija_v2.Teren (za izvodnice)
        profil:       profil kupe korišten u proračunu
        putanja:      izlazni .dxf fajl
        n_izvodnica:  broj radijalnih linija po kupi

    Returns:
        Path do zapisanog fajla.
    """
    from geometrija_v2 import Kupa

    putanja = Path(putanja)
    with DxfWriter(putanja) as dxf:
        for i, r in enumerate(rezultati, start=1):
            sloj = f"point_{i}"
            kupa = Kupa(wx=r.wx, wy=r.wy, wz=r.wz, k=r.k,
                        ugao=r.ugao, profil=profil)

            # 1) gornja kontura platoa (plavo, ACI 5)
            dxf._sloj, dxf._boja = sloj, 5
            kx, ky = kupa.gornja_kontura(n=96)
            dxf.polilinija(kx, ky, np.full_like(kx, r.wz))

            # 2) presječne konture na terenu (crveno, ACI 1)
            dxf._boja = 1
            for kont in (r.konture or []):
                kont = np.asarray(kont)
                if len(kont) < 3:
                    continue
                # zatvori petlju ako nije zatvorena
                if not np.allclose(kont[0], kont[-1]):
                    kont = np.vstack([kont, kont[0]])
                dxf.polilinija(kont[:, 0], kont[:, 1], kont[:, 2])

            # 3) izvodnice kosine (bijelo, ACI 7): od ivice platoa niz
            #    kosinu do prvog presjeka s terenom (bisekcija po d=0)
            dxf._boja = 7
            th = np.linspace(0.0, 2.0 * np.pi, n_izvodnica, endpoint=False)
            rt = kupa.r_top(th)
            r_max = kupa.max_radijus(teren.z_min) * 1.02
            for t_i, rt_i in zip(th, rt):
                ux, uy = np.cos(t_i), np.sin(t_i)
                rr = np.linspace(rt_i, r_max, 80)
                px = r.wx + rr * ux
                py = r.wy + rr * uy
                d = kupa.z(px, py) - teren.z(px, py)
                if d[0] <= 0:          # plato već ispod terena u tom pravcu
                    continue
                neg = np.nonzero(d <= 0)[0]
                if len(neg) == 0:      # kosina ne dodiruje teren u dosegu
                    continue
                j = neg[0]
                # linearna interpolacija nule između j-1 i j
                f0, f1 = d[j - 1], d[j]
                w = f0 / (f0 - f1)
                rx = px[j - 1] + w * (px[j] - px[j - 1])
                ry = py[j - 1] + w * (py[j] - py[j - 1])
                rz = float(kupa.z(rx, ry))
                dxf.polilinija(
                    np.array([r.wx + rt_i * ux, rx]),
                    np.array([r.wy + rt_i * uy, ry]),
                    np.array([r.wz, rz]))
    return putanja
