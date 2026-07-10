"""
prikaz_3d.py — interaktivni 3D prikaz kupe, terena i njihovog presjeka.

Generiše samostalan HTML fajl (Plotly) koji se otvara duplim klikom u
browseru — bez Pythona, bez interneta. Mišem: rotacija (lijevi klik),
pomjeranje (desni klik), zum (točkić). Klik na stavku legende
pali/gasi sloj (teren, kupa, konture, profili).

Ne mijenja geometrija_v2.py — samo ga koristi.

Pokretanje (fajlovi u podaci/ ili zadaj putanje kao u test_buvac_podaci.py):

    py prikaz_3d.py
    py prikaz_3d.py --wx 6412750 --wy 4970450 --wz 175 --k 120
    py prikaz_3d.py --rezolucija 512 --izlaz moja_kupa.html

Slojevi u prikazu:
    • teren (obojen po visini)
    • tijelo kupe (samo dio IZNAD terena — stvarno tijelo nasipa)
    • crvene presječne konture (kupa ∩ teren; više petlji na nepravilnom terenu)
    • plavi gornji plato kupe + vrh
    • dva vertikalna profila kroz vrh (I–Z i S–J) — klasični rudarski presjeci
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import plotly.graph_objects as go

from geometrija_v2 import Kupa, Teren, presek_kupe_i_terena
from test_buvac_podaci import (
    ucitaj_teren, ucitaj_granice, ucitaj_centar_masa, ucitaj_parametre,
    nadji_fajl, auto_lokacija,
)


def profil_kroz_vrh(kupa: Kupa, teren: Teren, ugao_stepeni: float,
                    R: float, n: int = 400):
    """Tačke vertikalnog profila kroz vrh kupe pod zadatim azimutom.

    Vraća (x, y, z_teren, z_kupa) duž linije dužine 2R kroz (wx, wy).
    """
    a = np.radians(ugao_stepeni)
    t = np.linspace(-R, R, n)
    x = kupa.wx + t * np.cos(a)
    y = kupa.wy + t * np.sin(a)
    return x, y, teren.z(x, y), kupa.z(x, y)


def main():
    ap = argparse.ArgumentParser(description="Interaktivni 3D prikaz presjeka")
    ap.add_argument("--teren")
    ap.add_argument("--granice")
    ap.add_argument("--cm")
    ap.add_argument("--params")
    ap.add_argument("--wx", type=float)
    ap.add_argument("--wy", type=float)
    ap.add_argument("--wz", type=float)
    ap.add_argument("--k", type=float, default=150.0)
    ap.add_argument("--ugao", type=float, default=37.0)
    ap.add_argument("--profil", default="matlab", choices=["matlab", "krug"])
    ap.add_argument("--rezolucija", type=int, default=384)
    ap.add_argument("--mreza", type=int, default=200,
                    help="rezolucija mreže za CRTANJE površina (default 200)")
    ap.add_argument("--izlaz", default="buvac_3d.html")
    a = ap.parse_args()

    # --- učitavanje (isti formati kao u test_buvac_podaci.py) ---------------
    f_teren = nadji_fajl(a.teren, "podaci/001-Teren-3-Buvac.txt",
                         "001-Teren-3-Buvac.txt")
    f_gran = nadji_fajl(a.granice, "podaci/001GranicaZonaBuvac.txt",
                        "001GranicaZonaBuvac.txt")
    f_cm = nadji_fajl(a.cm, "podaci/001CentarMasaBuvac.txt",
                      "001CentarMasaBuvac.txt")

    xyz = ucitaj_teren(f_teren)
    _, _, poligon = ucitaj_granice(f_gran)
    cm = ucitaj_centar_masa(f_cm)

    print(f"Teren: {len(xyz):,} tačaka. Gradim interpolator...")
    teren = Teren.iz_tacaka(xyz)

    # --- kupa ----------------------------------------------------------------
    if a.wx is not None and a.wy is not None:
        wx, wy = a.wx, a.wy
        z_tu = float(teren.z(wx, wy))
    else:
        wx, wy, z_tu = auto_lokacija(teren, poligon)
        print(f"Auto-lokacija kupe: ({wx:.0f}, {wy:.0f}), teren = {z_tu:.1f} m")
    wz = a.wz if a.wz is not None else z_tu + 40.0
    kupa = Kupa(wx=wx, wy=wy, wz=wz, k=a.k, ugao=a.ugao, profil=a.profil)

    # --- presjek ---------------------------------------------------------------
    t0 = time.perf_counter()
    rez = presek_kupe_i_terena(kupa, teren, rezolucija=a.rezolucija,
                               rafiniranje=2)
    if not rez.ima_preseka:
        sys.exit("Kupa nema presjeka s terenom — spusti wz ili pomjeri lokaciju.")
    print(f"Zapremina: {rez.zapremina:,.0f} m³ | osnova: "
          f"{rez.povrsina_osnove:,.0f} m² | petlji: {rez.broj_petlji} "
          f"| {(time.perf_counter()-t0)*1000:.0f} ms")

    # --- mreža za crtanje -------------------------------------------------------
    x0, x1, y0, y1 = rez.granice_racuna
    pad = 0.30 * (x1 - x0)
    vx0, vx1, vy0, vy1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    n = a.mreza
    gx = np.linspace(vx0, vx1, n)
    gy = np.linspace(vy0, vy1, n)
    GX, GY = np.meshgrid(gx, gy)
    ZT = teren.z(GX, GY)
    ZK = kupa.z(GX, GY)
    tijelo = np.where(ZK > ZT + 1e-9, ZK, np.nan)   # samo dio iznad terena

    fig = go.Figure()

    # 1) teren
    fig.add_trace(go.Surface(
        x=GX, y=GY, z=ZT, name="teren", showlegend=True,
        colorscale="Earth", opacity=1.0,
        colorbar=dict(title="kota (m)", len=0.6, x=1.02),
        hovertemplate="x: %{x:.0f}<br>y: %{y:.0f}<br>teren: %{z:.1f} m<extra></extra>",
    ))

    # 2) tijelo kupe (iznad terena)
    fig.add_trace(go.Surface(
        x=GX, y=GY, z=tijelo, name="kupa (nasip)", showlegend=True,
        colorscale=[[0, "peru"], [1, "burlywood"]], showscale=False,
        opacity=0.75,
        hovertemplate="x: %{x:.0f}<br>y: %{y:.0f}<br>kupa: %{z:.1f} m<extra></extra>",
    ))

    # 3) presječne konture
    for i, kont in enumerate(rez.konture):
        fig.add_trace(go.Scatter3d(
            x=kont[:, 0], y=kont[:, 1], z=kont[:, 2] + 0.4,
            mode="lines", line=dict(color="red", width=7),
            name="presjek kupa–teren", legendgroup="presjek",
            showlegend=(i == 0),
            hovertemplate="presjek<br>x: %{x:.0f}<br>y: %{y:.0f}<br>z: %{z:.1f} m<extra></extra>",
        ))

    # 4) gornji plato + vrh
    kx, ky = kupa.gornja_kontura()
    fig.add_trace(go.Scatter3d(
        x=kx, y=ky, z=np.full_like(kx, kupa.wz) + 0.4,
        mode="lines", line=dict(color="royalblue", width=5),
        name="gornji plato",
    ))
    fig.add_trace(go.Scatter3d(
        x=[kupa.wx], y=[kupa.wy], z=[kupa.wz + 1.0],
        mode="markers+text", text=[f"vrh {kupa.wz:.0f} m"],
        textposition="top center",
        marker=dict(color="royalblue", size=5, symbol="diamond"),
        name="vrh kupe", showlegend=False,
    ))

    # 5) vertikalni profili kroz vrh (I–Z i S–J) — rudarski presjeci
    R_prof = 0.5 * (x1 - x0)
    for az, ime, boja in [(0.0, "profil I–Z", "black"),
                          (90.0, "profil S–J", "dimgray")]:
        px, py, pzt, pzk = profil_kroz_vrh(kupa, teren, az, R_prof)
        gornja = np.where(pzk > pzt, pzk, pzt)      # linija po kupi/terenu
        fig.add_trace(go.Scatter3d(
            x=px, y=py, z=gornja + 0.4, mode="lines",
            line=dict(color=boja, width=4, dash="dash"),
            name=ime, visible="legendonly",          # uključi klikom u legendi
        ))

    # 6) centar masa
    fig.add_trace(go.Scatter3d(
        x=[cm[0]], y=[cm[1]], z=[float(teren.z(cm[0], cm[1])) + 1],
        mode="markers", marker=dict(color="magenta", size=5, symbol="x"),
        name="centar masa", visible="legendonly",
    ))

    fig.update_layout(
        title=(f"Buvac — kupa ({wx:.0f}, {wy:.0f}, {wz:.0f}), k={a.k:.0f} m, "
               f"ugao={a.ugao:.0f}°  |  V = {rez.zapremina:,.0f} m³, "
               f"osnova = {rez.povrsina_osnove:,.0f} m², "
               f"petlji presjeka: {rez.broj_petlji}"),
        scene=dict(
            aspectmode="data",
            zaxis=dict(title="z (m)"),
            xaxis=dict(title="x"), yaxis=dict(title="y"),
            camera=dict(eye=dict(x=1.4, y=-1.4, z=0.7)),
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    # realniji odnos visine (teren je plitak u odnosu na širinu) — data aspект
    # zadržava stvarne proporcije; ako želiš "razvučeno" po z, koristi:
    #   scene.aspectmode="manual", scene.aspectratio=dict(x=1, y=1, z=0.4)

    fig.write_html(a.izlaz, include_plotlyjs=True, full_html=True)
    print(f"Sačuvano: {a.izlaz}  (otvori duplim klikom u browseru)")


if __name__ == "__main__":
    main()
