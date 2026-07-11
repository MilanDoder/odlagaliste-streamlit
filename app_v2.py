"""
app_v2.py — Streamlit aplikacija za optimizaciju odlagališta (nova geometrija).

Nasljednik deployovane app.py (odlagaliste-ga.streamlit.app), ali nad
geometrija_v2 (tačna zapremina i za nepravilan/brdovit teren) i sa
kompletnim Monte Carlo izborom tačaka:

  1. PODACI      — upload terena, centra masa, granice interesne zone,
                   ekonomskih zona i dodatnih parametara (ili ugrađeni Buvac)
  2. MC TAČKE    — Monte Carlo generisanje kandidat-tačaka sa filterima:
                   interesna zona, loše (K) zone, max distanca od centra
                   masa, pokrivenost terena
  3. PRORAČUN    — za svaku prihvaćenu tačku: zapremina, osnova, ekonomske
                   zone, distanca, funkcija cilja; fiksna kupa ili GA (wz, k)
  4. REZULTATI   — tabela, CSV, 3D prikaz najbolje kupe s presjekom

Pokretanje:  py -m streamlit run app_v2.py
"""

from __future__ import annotations

import io
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from loaders import (ucitaj_teren as l_teren, ucitaj_ekonomske_zone,
                     ucitaj_centar_masa, ucitaj_granice_zone,
                     ucitaj_dodatne_parametre)
from geometrija_v2 import Kupa, Teren, presek_kupe_i_terena
from stepenasta_kupa import StepenastaKupa
from pipeline_v2 import (KontekstV2, MCTacke, RezultatTackeV2,
                         monte_carlo_tacke, proracun_svih_tacaka)

st.set_page_config(page_title="Optimizacija odlagališta v2", layout="wide")


# ---------------------------------------------------------------------------
# Pomoćno: upload → temp fajl; keširano učitavanje
# ---------------------------------------------------------------------------

def _spremi(uploaded) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    tmp.write(uploaded.getbuffer())
    tmp.close()
    return tmp.name


@st.cache_resource(show_spinner="Gradim interpolator terena (radi se jednom)...")
def _teren_iz_fajla(putanja: str, mtime: float) -> tuple[Teren, np.ndarray]:
    ts = l_teren(putanja)
    return Teren.iz_tacaka(ts.vertices), ts.vertices


def _ucitaj_podatke():
    """Sidebar sa uploadima; vraća (teren, vertices, dobre, lose, cm, granice, par)."""
    st.sidebar.title("1 · Ulazni podaci")
    ugradjeni = st.sidebar.checkbox("Koristi ugrađene Buvac podatke (podaci/)",
                                    value=True)

    if ugradjeni:
        p = {
            "teren": "podaci/001-Teren-3-Buvac.txt",
            "zone": "podaci/001EkonomskeZoneBuvac.txt",
            "cm": "podaci/001CentarMasaBuvac.txt",
            "granice": "podaci/001GranicaZonaBuvac.txt",
            "par": "podaci/DodatniUlazniParametri.txt",
        }
        # dozvoli izbor između više teren fajlova (npr. usko brdo)
        import glob
        tereni = sorted(glob.glob("podaci/*Teren*.txt"))
        if tereni:
            p["teren"] = st.sidebar.selectbox(
                "Teren fajl", tereni, format_func=os.path.basename)
        for kljuc, put in p.items():
            if not os.path.exists(put):
                st.sidebar.error(f"Nedostaje: {put}")
                st.stop()
    else:
        up = {}
        up["teren"] = st.sidebar.file_uploader("Teren (X,Y,Z po liniji)", type="txt")
        up["zone"] = st.sidebar.file_uploader("Ekonomske zone", type="txt")
        up["cm"] = st.sidebar.file_uploader("Centar masa", type="txt")
        up["granice"] = st.sidebar.file_uploader("Granica interesne zone", type="txt")
        up["par"] = st.sidebar.file_uploader("Dodatni parametri (opciono)", type="txt")
        if not all(up[k] for k in ("teren", "zone", "cm", "granice")):
            st.info("Učitaj sva 4 obavezna fajla u sidebar-u (parametri su opcioni), "
                    "ili uključi ugrađene Buvac podatke.")
            st.stop()
        p = {k: _spremi(v) for k, v in up.items() if v is not None}

    teren, vertices = _teren_iz_fajla(p["teren"], os.path.getmtime(p["teren"]))
    dobre, lose = ucitaj_ekonomske_zone(p["zone"])
    cm = ucitaj_centar_masa(p["cm"])
    granice = ucitaj_granice_zone(p["granice"])
    if "par" in p:
        par = ucitaj_dodatne_parametre(p["par"])
    else:
        from loaders import DodatniParametri
        par = DodatniParametri()
    if par.nadmorska_visina is None:
        par.nadmorska_visina = float(teren.z_min)
    if par.uslov_distance is None:
        par.uslov_distance = float(np.hypot(np.ptp(granice.x_poly),
                                            np.ptp(granice.y_poly)))
    return teren, vertices, dobre, lose, cm, granice, par


# ---------------------------------------------------------------------------
# Prikazi
# ---------------------------------------------------------------------------

def _teren_grid(teren: Teren, n: int = 150):
    x0, x1, y0, y1 = teren.xy_granice
    GX, GY = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    return GX, GY, teren.z(GX, GY)


BOJE_ZONA = [("Z-1", "orange"), ("Z-2", "green"), ("Z-3", "dodgerblue"),
             ("Z-4", "hotpink"), ("K", "purple"), ("Z-5", "purple")]


def _boja_zone(naziv: str) -> str:
    for prefiks, boja in BOJE_ZONA:
        if naziv.startswith(prefiks):
            return boja
    return "white"


def fig_pregled(teren, cm, granice, dobre, lose):
    """Tab Podaci: teren, centar masa, granica interesne zone i sve zone
    interesa obojene po prefiksu (Z-1 narandžasta, Z-2 zelena, Z-3 plava,
    Z-4 roza, K ljubičasta). Zone istog prefiksa su jedan trace (poligoni
    razdvojeni None tačkama) — brzo i sa jednom stavkom u legendi.
    """
    GX, GY, ZT = _teren_grid(teren)
    f = go.Figure()
    f.add_trace(go.Surface(x=GX, y=GY, z=ZT, colorscale="Earth",
                           showscale=False, name="teren", opacity=0.95))

    # granica interesne zone (žuto)
    zx = np.append(granice.x_poly, granice.x_poly[0])
    zy = np.append(granice.y_poly, granice.y_poly[0])
    f.add_trace(go.Scatter3d(x=zx, y=zy, z=teren.z(zx, zy) + 2.5,
                             mode="lines",
                             line=dict(color="yellow", width=7),
                             name="granica interesne zone"))

    # zone interesa — zajednička funkcija (ista kao u MC tabu)
    _dodaj_zone_na_teren(f, teren, dobre, lose)

    f.add_trace(go.Scatter3d(x=[cm[0]], y=[cm[1]],
                             z=[float(teren.z(cm[0], cm[1])) + 4],
                             mode="markers",
                             marker=dict(color="red", size=7,
                                         symbol="diamond"),
                             name="centar masa"))
    f.update_layout(scene=dict(aspectmode="data"), height=560,
                    margin=dict(l=0, r=0, t=0, b=0),
                    legend=dict(orientation="h", y=-0.04))
    return f


def _dodaj_zone_na_teren(f, teren, dobre, lose, opacity_pomak=2.0):
    """Docrtava sve zone interesa (grupisane po prefiksu, boje iz
    BOJE_ZONA) položene na teren — dijeli se između tabova Podaci i MC."""
    sve_zone = list(dobre or []) + list(lose or [])
    if not sve_zone:
        return
    for prefiks, boja in BOJE_ZONA:
        xs, ys = [], []
        n_zona = 0
        for zona in sve_zone:
            if not zona.naziv.startswith(prefiks):
                continue
            px = np.asarray(zona.x_data, float)
            py = np.asarray(zona.y_data, float)
            if not (np.isclose(px[0], px[-1]) and np.isclose(py[0], py[-1])):
                px = np.append(px, px[0])
                py = np.append(py, py[0])
            xs.extend(px.tolist() + [None])
            ys.extend(py.tolist() + [None])
            n_zona += 1
        if n_zona == 0:
            continue
        xa = np.array([np.nan if v is None else v for v in xs])
        ya = np.array([np.nan if v is None else v for v in ys])
        za = np.full_like(xa, np.nan)
        ok = ~np.isnan(xa)
        za[ok] = teren.z(xa[ok], ya[ok]) + opacity_pomak
        f.add_trace(go.Scatter3d(
            x=xa, y=ya, z=za, mode="lines",
            line=dict(color=boja, width=5),
            name=f"{prefiks} ({n_zona})", legendgroup=prefiks,
            connectgaps=False))


def fig_mc_3d(teren, granice, mc: MCTacke, cm, uslov,
              prikazi_odbacene: bool = True,
              dobre=None, lose=None, prikazi_zone: bool = True):
    """3D prikaz Monte Carlo tačaka na terenu.

    Teren kao površina; interesna zona žuta linija na terenu; prihvaćene
    tačke zelene, odbačene obojene po razlogu; narandžasti isprekidani
    krug = max distanca od centra masa (položen na teren).
    """
    GX, GY, ZT = _teren_grid(teren, 140)
    f = go.Figure()
    f.add_trace(go.Surface(x=GX, y=GY, z=ZT, colorscale="Earth",
                           showscale=False, opacity=0.95, name="teren",
                           hovertemplate="x: %{x:.0f}<br>y: %{y:.0f}"
                                         "<br>teren: %{z:.1f} m<extra></extra>"))

    # interesna zona — označena na terenu
    zx = np.append(granice.x_poly, granice.x_poly[0])
    zy = np.append(granice.y_poly, granice.y_poly[0])
    f.add_trace(go.Scatter3d(x=zx, y=zy, z=teren.z(zx, zy) + 2.5,
                             mode="lines",
                             line=dict(color="yellow", width=8),
                             name="interesna zona"))

    if prikazi_zone:
        _dodaj_zone_na_teren(f, teren, dobre, lose)

    boje = {"van interesne zone": "gray", "u lošoj (K) zoni": "black",
            "predaleko od centra masa": "orange",
            "van pokrivenosti terena": "saddlebrown"}
    if prikazi_odbacene:
        for razlog, t in mc.odbacene.items():
            if len(t) == 0:
                continue
            f.add_trace(go.Scatter3d(
                x=t[:, 0], y=t[:, 1], z=teren.z(t[:, 0], t[:, 1]) + 1.5,
                mode="markers",
                marker=dict(color=boje.get(razlog, "red"), size=2.5,
                            opacity=0.5),
                name=f"✗ {razlog} ({len(t)})"))

    if len(mc.prihvacene):
        px, py = mc.prihvacene[:, 0], mc.prihvacene[:, 1]
        pz = teren.z(px, py)
        f.add_trace(go.Scatter3d(
            x=px, y=py, z=pz + 2.0, mode="markers",
            marker=dict(color="lime", size=4,
                        line=dict(color="darkgreen", width=1)),
            name=f"✓ prihvaćene ({len(mc.prihvacene)})",
            hovertemplate="x: %{x:.0f}<br>y: %{y:.0f}"
                          "<br>teren: %{z:.1f} m<extra></extra>"))

    # krug max distance od centra masa — položen na teren
    th = np.linspace(0, 2 * np.pi, 160)
    kx = cm[0] + uslov * np.cos(th)
    ky = cm[1] + uslov * np.sin(th)
    x0, x1, y0, y1 = teren.xy_granice
    u = (kx >= x0) & (kx <= x1) & (ky >= y0) & (ky <= y1)
    f.add_trace(go.Scatter3d(x=kx[u], y=ky[u],
                             z=teren.z(kx[u], ky[u]) + 2.5, mode="lines",
                             line=dict(color="orange", width=5, dash="dash"),
                             name=f"max distanca ({uslov:.0f} m)"))
    f.add_trace(go.Scatter3d(x=[cm[0]], y=[cm[1]],
                             z=[float(teren.z(cm[0], cm[1])) + 4],
                             mode="markers",
                             marker=dict(color="red", size=7, symbol="x"),
                             name="centar masa"))

    f.update_layout(scene=dict(aspectmode="data",
                               camera=dict(eye=dict(x=1.3, y=-1.3, z=0.9))),
                    height=620, margin=dict(l=0, r=0, t=0, b=0),
                    legend=dict(orientation="h", y=-0.04))
    return f


def fig_tacka(teren, r: RezultatTackeV2, ctx: KontekstV2,
              cijela_kupa: bool = True, z_uvecanje: float = 2.0,
              dobre=None, lose=None):
    """3D prikaz jedne tačke: teren + kupa + crveni presjek.

    cijela_kupa=True crta CIJELU površinu kupe (providno i dio ispod
    terena), pa se jasno vidi gdje kupa 'ulazi' u teren; presjek je
    crvena kriva. Prozor se kadrira oko stvarnog footprinta, ne oko
    cijelog integracionog bbox-a. z_uvecanje > 1 razvlači visinu da
    reljef i kupa budu vidljivi (1 = stvarne proporcije).
    """
    kupa = Kupa(wx=r.wx, wy=r.wy, wz=r.wz, k=r.k, ugao=r.ugao,
                profil=ctx.profil)
    rez = presek_kupe_i_terena(kupa, teren, rezolucija=256, rafiniranje=2)

    # kadar: bbox svih presječnih kontura + 25% margine
    if rez.konture:
        sve = np.vstack(rez.konture)
        x0, x1 = sve[:, 0].min(), sve[:, 0].max()
        y0, y1 = sve[:, 1].min(), sve[:, 1].max()
    else:
        x0, x1, y0, y1 = rez.granice_racuna
    pad = 0.25 * max(x1 - x0, y1 - y0, 2 * r.k)
    vx0, vx1, vy0, vy1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad

    n = 150
    GX, GY = np.meshgrid(np.linspace(vx0, vx1, n),
                         np.linspace(vy0, vy1, n))
    ZT = teren.z(GX, GY)
    ZK = kupa.z(GX, GY)

    f = go.Figure()
    f.add_trace(go.Surface(x=GX, y=GY, z=ZT, colorscale="Earth",
                           showscale=False, name="teren", opacity=1.0))

    if cijela_kupa:
        # cijela površina kupe, ali samo do dna kadra (da ne prekrije sve)
        z_pod = float(np.nanmin(ZT)) - 2.0
        ZK_cr = np.where(ZK >= z_pod, ZK, np.nan)
        f.add_trace(go.Surface(x=GX, y=GY, z=ZK_cr, opacity=0.45,
                               colorscale=[[0, "#c46a1b"], [1, "#e8a15c"]],
                               showscale=False, name="kupa (cijela)"))
    # tijelo iznad terena — punije, da se vidi šta je stvarni nasip
    tijelo = np.where(ZK > ZT + 1e-9, ZK, np.nan)
    f.add_trace(go.Surface(x=GX, y=GY, z=tijelo, opacity=0.95,
                           colorscale=[[0, "peru"], [1, "burlywood"]],
                           showscale=False, name="nasip (iznad terena)"))

    if getattr(r, "delovi", None):
        # VIŠE DELOVA: uzeti deo crveno, odbačeni sivo + oznake A, B, C...
        prvi_uzet, prvi_odb = True, True
        for deo in r.delovi:
            boja = "red" if deo["uzet"] else "gray"
            ime = ("presjek — uzeti deo" if deo["uzet"]
                   else "presjek — odbačeni deo")
            for kont in deo["konture"]:
                kont = np.asarray(kont)
                f.add_trace(go.Scatter3d(
                    x=kont[:, 0], y=kont[:, 1], z=kont[:, 2] + 0.4,
                    mode="lines", line=dict(color=boja, width=8),
                    name=ime,
                    showlegend=(prvi_uzet if deo["uzet"] else prvi_odb)))
                if deo["uzet"]:
                    prvi_uzet = False
                else:
                    prvi_odb = False
            cx, cy, cz = deo["centar"]
            status = "✓" if deo["uzet"] else "✗"
            f.add_trace(go.Scatter3d(
                x=[cx], y=[cy], z=[cz + 8],
                mode="text", text=[f"{deo['oznaka']} {status}"],
                textfont=dict(size=22, color=boja),
                name=f"deo {deo['oznaka']}", showlegend=False,
                hovertext=(f"Deo {deo['oznaka']}: "
                           f"V = {deo['zapremina']:,.0f} m³, "
                           f"P = {deo['povrsina']:,.0f} m²"
                           + ("" if deo["uzet"]
                              else f" — odbačen ({deo.get('razlog', '')})")),
                hoverinfo="text"))
    else:
        for i, kont in enumerate(rez.konture):
            f.add_trace(go.Scatter3d(x=kont[:, 0], y=kont[:, 1],
                                     z=kont[:, 2] + 0.4, mode="lines",
                                     line=dict(color="red", width=8),
                                     name="presjek kupa–teren",
                                     showlegend=(i == 0)))
    _dodaj_zone_na_teren(f, teren, dobre, lose)

    kx, ky = kupa.gornja_kontura()
    f.add_trace(go.Scatter3d(x=kx, y=ky, z=np.full_like(kx, r.wz) + 0.4,
                             mode="lines",
                             line=dict(color="royalblue", width=5),
                             name="gornji plato"))
    f.add_trace(go.Scatter3d(x=[r.wx], y=[r.wy], z=[r.wz + 1],
                             mode="markers",
                             marker=dict(color="royalblue", size=4,
                                         symbol="diamond"),
                             name="vrh", showlegend=False))

    # proporcije: data = stvarne; z_uvecanje razvlači samo visinu
    dx, dy = vx1 - vx0, vy1 - vy0
    zmin = min(float(np.nanmin(ZT)), r.wz) - 2
    zmax = max(float(np.nanmax(ZT)), r.wz) + 2
    dz = max(zmax - zmin, 1.0)
    m = max(dx, dy)
    f.update_layout(
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(x=dx / m, y=dy / m,
                             z=(dz / m) * float(z_uvecanje)),
            xaxis=dict(range=[vx0, vx1]),
            yaxis=dict(range=[vy0, vy1]),
            zaxis=dict(range=[zmin, zmax], title="z (m)"),
            camera=dict(eye=dict(x=1.3, y=-1.3, z=0.8)),
        ),
        height=600, margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", y=0.02))
    return f


# ---------------------------------------------------------------------------
# Glavna aplikacija
# ---------------------------------------------------------------------------

teren, vertices, dobre, lose, cm, granice, par = _ucitaj_podatke()

st.title("Optimizacija odlagališta — v2 (nova geometrija)")

tab1, tab2, tab3, tab4 = st.tabs(["1 · Podaci", "2 · Monte Carlo tačke",
                                  "3 · Proračun i rezultati",
                                  "4 · Stepenasti prikaz odlagališta"])

# =========================== TAB 1: PODACI =================================
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tačaka terena", f"{len(vertices):,}")
    c2.metric("Kote terena", f"{teren.z_min:.0f}–{teren.z_max:.0f} m")
    c3.metric("Ekonomskih zona", f"{len(dobre)} dobrih / {len(lose)} loših")
    c4.metric("Centar masa", f"({cm[0]:.0f}, {cm[1]:.0f})")
    st.caption(f"mnv = {par.nadmorska_visina:.0f} m · "
               f"uslov distance = {par.uslov_distance:.0f} m · "
               f"GA generacija = {par.broj_generacija}")
    st.plotly_chart(fig_pregled(teren, cm, granice, dobre, lose),
                    use_container_width=True)

# ======================= TAB 2: MONTE CARLO ================================
with tab2:
    st.subheader("Monte Carlo izbor kandidat-tačaka")
    cA, cB, cC = st.columns(3)
    n_mc = cA.number_input("Broj generisanih tačaka", 50, 20000, 500, step=50)
    seed = cB.number_input("Seed (ponovljivost)", 0, 99999, 42)
    uslov = cC.number_input("Max distanca od centra masa (m)", 100.0, 50000.0,
                            float(par.uslov_distance), step=100.0)

    cD, cE, cF = st.columns(3)
    f_zona = cD.checkbox("Filter: interesna zona", value=True)
    f_lose = cE.checkbox("Filter: loše (K) zone", value=True,
                         disabled=(len(lose) == 0),
                         help="U ovom setu podataka nema K zona." if not lose else None)
    f_teren = cF.checkbox("Filter: pokrivenost terena", value=True,
                          help="Odbacuje tačke van oblaka tačaka terena — "
                               "tamo kota terena nije definisana.")

    if st.button("Generiši tačke", type="primary"):
        mc = monte_carlo_tacke(
            n=int(n_mc), teren=teren,
            zona_x=granice.x_poly if f_zona else np.array(
                [granice.x_range[0], granice.x_range[1],
                 granice.x_range[1], granice.x_range[0]]),
            zona_y=granice.y_poly if f_zona else np.array(
                [granice.y_range[0], granice.y_range[0],
                 granice.y_range[1], granice.y_range[1]]),
            centar_masa=cm, uslov_distance=float(uslov),
            lose_zone=lose if f_lose else None,
            filtriraj_teren=f_teren, seed=int(seed))
        st.session_state["mc"] = mc
        st.session_state["uslov"] = float(uslov)
        st.session_state.pop("rezultati", None)

    if "mc" in st.session_state:
        mc: MCTacke = st.session_state["mc"]
        st.write({k: v for k, v in mc.statistika.items()})

        cO1, cO2 = st.columns(2)
        prikazi_odb = cO1.checkbox("Prikaži i odbačene tačke", value=True,
                                   help="Odbačene tačke obojene po razlogu; "
                                        "isključi za pregledniji prikaz.")
        prikazi_zone = cO2.checkbox("Prikaži zone interesa", value=True)
        st.plotly_chart(fig_mc_3d(teren, granice, mc, cm,
                                  st.session_state["uslov"],
                                  prikazi_odbacene=prikazi_odb,
                                  dobre=dobre, lose=lose,
                                  prikazi_zone=prikazi_zone),
                        use_container_width=True)

        if len(mc.prihvacene) == 0:
            st.warning("Nijedna tačka nije prošla filtere — povećaj broj "
                       "tačaka ili opusti max distancu.")
        else:
            st.subheader(f"Prihvaćene tačke ({len(mc.prihvacene)})")
            px, py = mc.prihvacene[:, 0], mc.prihvacene[:, 1]
            df_mc = pd.DataFrame({
                "Tačka": [f"point_{i+1}" for i in range(len(px))],
                "X": px, "Y": py,
                "Kota_terena_m": teren.z(px, py),
                "Distanca_od_CM_m": np.hypot(px - cm[0], py - cm[1]),
            })
            st.dataframe(df_mc.style.format({
                "X": "{:.1f}", "Y": "{:.1f}",
                "Kota_terena_m": "{:.1f}",
                "Distanca_od_CM_m": "{:.0f}"}),
                use_container_width=True, height=320)
            st.download_button(
                "Preuzmi prihvaćene tačke (CSV)",
                df_mc.to_csv(index=False).encode("utf-8"),
                "mc_tacke.csv", "text/csv")

# ==================== TAB 3: PRORAČUN I REZULTATI ==========================
with tab3:
    if "mc" not in st.session_state or len(st.session_state["mc"].prihvacene) == 0:
        st.info("Prvo generiši Monte Carlo tačke u tabu 2.")
        st.stop()

    mc: MCTacke = st.session_state["mc"]
    st.subheader("Proračun zapremine i parametara za prihvaćene tačke")

    cA, cB, cC, cD = st.columns(4)
    mod = cA.radio("Režim", ["GA optimizacija (wz, k)", "Fiksna kupa (brzo)"])
    izbor = cB.radio("Koje tačke obraditi",
                     [f"Sve MC prihvaćene ({len(mc.prihvacene)})",
                      "Slučajni podskup"])
    if izbor.startswith("Slučajni"):
        max_tacaka = cB.number_input("Veličina podskupa", 1,
                                     len(mc.prihvacene),
                                     min(20, len(mc.prihvacene)))
        seed_pod = cB.number_input("Seed podskupa", 0, 99999, 7)
    else:
        max_tacaka = len(mc.prihvacene)
        seed_pod = 0
    ugao = cC.number_input("Ugao kosine (°)", min_value=5.0,
                           max_value=60.0, value=37.0, step=1.0)
    profil = cD.selectbox("Profil kupe", ["matlab", "krug"])

    cE, cF, cG, cH = st.columns(4)
    v_min = cE.number_input("Min zapremina (m³)", 0.0, 1e8, 100_000.0,
                            step=50_000.0, format="%.0f")
    v_max = cF.number_input("Max zapremina (m³)", 1e5, 1e9, 39_000_000.0,
                            step=1e6, format="%.0f")
    if mod.startswith("GA"):
        popul = cG.number_input("GA populacija", 5, 100, 20)
        gener = cH.number_input("GA generacija", 1, 50,
                                int(par.broj_generacija))
        wz_fix, k_fix = None, 120.0
    else:
        h_fix = cG.number_input("Visina platoa iznad terena (m)",
                                min_value=5.0, max_value=200.0,
                                value=40.0, step=5.0)
        k_fix = cH.number_input("k — širina platoa (m)", min_value=20.0,
                                max_value=500.0, value=120.0, step=10.0)
        popul, gener = 20, 3

    rez_slider = st.select_slider(
        "Rezolucija proračuna (veće = tačnije, sporije)",
        [128, 160, 192, 256], value=160)

    if st.button("Pokreni proračun", type="primary"):
        ctx = KontekstV2(
            teren=teren, zona_x=granice.x_poly, zona_y=granice.y_poly,
            dobre_zone=dobre, centar_masa=cm,
            mnv=float(par.nadmorska_visina), ugao=float(ugao),
            profil=profil,
            donja_granica_zapremine=float(v_min),
            gornja_granica_zapremine=float(v_max),
            uslov_distance=st.session_state["uslov"],
            rezolucija=int(rez_slider), rafiniranje=1,
            lose_zone=lose)

        bar = st.progress(0.0, text="Proračun...")
        t0 = time.perf_counter()

        def cb(i, n, r):
            bar.progress(i / n, text=f"Tačka {i}/{n} — "
                         f"{'✓' if r else '✗ nedopustiva'}")

        if izbor.startswith("Slučajni") and int(max_tacaka) < len(mc.prihvacene):
            rng = np.random.default_rng(int(seed_pod))
            idx = rng.choice(len(mc.prihvacene), size=int(max_tacaka),
                             replace=False)
            tacke = mc.prihvacene[np.sort(idx)]
        else:
            tacke = mc.prihvacene
        if mod.startswith("GA"):
            rezultati = proracun_svih_tacaka(
                tacke, ctx, mod="ga", callback=cb,
                populacija=int(popul), max_generacija=int(gener), seed=1)
        else:
            rezultati = proracun_svih_tacaka(
                tacke, ctx, mod="fiksno", callback=cb,
                wz_fiksno=None, k_fiksno=float(k_fix))
            # wz = teren + h_fix po tački:
            # (proracun_tacke sa wz_fiksno=None koristi teren+40; ako je
            #  h_fix različit, ponovi sa eksplicitnim wz)
            if abs(h_fix - 40.0) > 1e-9:
                rezultati = []
                for i, (wx, wy) in enumerate(tacke):
                    from pipeline_v2 import proracun_tacke
                    wz_i = float(teren.z(wx, wy)) + float(h_fix)
                    r = proracun_tacke(f"point_{i+1}", float(wx), float(wy),
                                       ctx, mod="fiksno", wz_fiksno=wz_i,
                                       k_fiksno=float(k_fix))
                    if r:
                        rezultati.append(r)
                    cb(i + 1, len(tacke), r)
                rezultati.sort(key=lambda r: r.f_vrednost)

        bar.progress(1.0, text=f"Gotovo za {time.perf_counter()-t0:.1f} s — "
                     f"{len(rezultati)} dopustivih od {len(tacke)}")
        st.session_state["rezultati"] = rezultati
        st.session_state["ctx"] = ctx

    if "rezultati" in st.session_state:
        rezultati: list[RezultatTackeV2] = st.session_state["rezultati"]
        ctx: KontekstV2 = st.session_state["ctx"]
        if not rezultati:
            st.warning("Nijedna tačka nije dala dopustivo rješenje — "
                       "opusti granice zapremine ili povećaj max distancu.")
            st.stop()

        df = pd.DataFrame([r.kao_red() for r in rezultati],
                          columns=RezultatTackeV2.ZAGLAVLJE)
        st.dataframe(df.style.format({
            "X": "{:.0f}", "Y": "{:.0f}", "Z_vrha": "{:.1f}", "K": "{:.1f}",
            "Funkcija_cilja": "{:.4f}", "Zapremina_m3": "{:,.0f}",
            "Osnova_m2": "{:,.0f}", "Distanca_m": "{:.0f}",
            "c1_transport": "{:,.0f}", "c2_visina": "{:,.0f}",
            "c3_zemljiste": "{:,.0f}", "Ukupna_cena": "{:,.0f}"}), use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        cD1, cD2 = st.columns(2)
        cD1.download_button("Preuzmi rezultate (CSV)", csv,
                            "rezultati_odlagaliste_v2.csv", "text/csv")

        # DXF izvoz svih dopustivih kupa (AutoCAD: File → Open, pa po
        # želji Save As → .dwg — DWG je zatvoren format i ne piše se direktno)
        import tempfile as _tf
        from izvoz import izvezi_dxf_v2
        _tmp = _tf.NamedTemporaryFile(delete=False, suffix=".dxf")
        _tmp.close()
        izvezi_dxf_v2(rezultati, teren, profil=ctx.profil, putanja=_tmp.name)
        with open(_tmp.name, "rb") as _fh:
            dxf_bytes = _fh.read()
        cD2.download_button(
            f"Preuzmi kupe svih tačaka (DXF za AutoCAD, {len(rezultati)})",
            dxf_bytes, "kupe_odlagaliste_v2.dxf", "application/dxf",
            help="Otvara se direktno u AutoCAD-u; za .dwg koristi Save As "
                 "u AutoCAD-u. Svaka tačka je na svom sloju point_N: plavi "
                 "plato, crveni presjek s terenom, bijele izvodnice kosine.")

        best = rezultati[0]
        st.subheader("3D prikaz tačke")
        cP, cQ, cR = st.columns([2, 1, 1])
        opcije_t = [f"{r.naziv}  (f={r.f_vrednost:.3f}, "
                    f"V={r.zapremina:,.0f} m³)" for r in rezultati]
        i_sel = cP.selectbox("Tačka za prikaz (sortirano po funkciji cilja)",
                             range(len(rezultati)),
                             format_func=lambda i: opcije_t[i])
        cijela = cQ.checkbox("Prikaži cijelu kupu", value=True,
                             help="Providno crta i dio kupe ispod terena — "
                                  "jasno se vidi gdje kupa ulazi u teren.")
        z_uv = cR.slider("Uveličanje visine", 1.0, 5.0, 2.0, 0.5,
                         help="1 = stvarne proporcije; veće razvlači visinu.")
        sel: RezultatTackeV2 = rezultati[int(i_sel)]

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Funkcija cilja", f"{sel.f_vrednost:.4f}")
        m2.metric("Zapremina", f"{sel.zapremina:,.0f} m³")
        m3.metric("Vrh / k", f"{sel.wz:.1f} m / {sel.k:.0f} m")
        m4.metric("Distanca od CM", f"{sel.distanca:.0f} m")
        _uzeti = getattr(sel, "uzeti_deo", "")
        _delovi = getattr(sel, "delovi", None) or []
        m5.metric("Petlji presjeka", sel.broj_petlji,
                  delta=f"uzet deo {_uzeti}" if _uzeti else None,
                  delta_color="off")
        m6.metric("Ukupna cijena", f"{sel.ukupna_cena:,.0f}")
        if sel.zone:
            st.caption(f"Ekonomske zone: {sel.zone}")
        if _delovi:
            st.info("Presjek ima više odvojenih delova — analiziran je "
                    "svaki zasebno, a zapremina/cijena se odnose SAMO na "
                    f"uzeti deo **{_uzeti}**:\n\n"
                    + "\n".join(
                        f"- **{d['oznaka']}**: V = {d['zapremina']:,.0f} m³, "
                        f"P = {d['povrsina']:,.0f} m² — "
                        + ("✅ **uzet**" if d["uzet"]
                           else f"❌ odbačen ({d.get('razlog') or 'lošiji'})")
                        for d in _delovi))
        st.plotly_chart(fig_tacka(teren, sel, ctx, cijela_kupa=cijela,
                                  z_uvecanje=z_uv, dobre=dobre, lose=lose),
                        use_container_width=True)


# ================= TAB 4: STEPENASTI PRIKAZ ODLAGALIŠTA ====================
with tab4:
    if "rezultati" not in st.session_state or not st.session_state["rezultati"]:
        st.info("Prvo pokreni proračun u tabu 3 — ovdje se biraju dobre "
                "(dopustive) kupe iz tih rezultata.")
    else:
        rezultati = st.session_state["rezultati"]
        ctx = st.session_state["ctx"]

        st.subheader("Stepenasta (etažna) kupa")
        opcije = [f"{r.naziv}  (f={r.f_vrednost:.3f}, V={r.zapremina:,.0f} m³)"
                  for r in rezultati]
        i_k = st.selectbox("Izaberi kupu (dopustive tačke iz taba 3)",
                           range(len(rezultati)),
                           format_func=lambda i: opcije[i], key="step_sel")
        rk = rezultati[int(i_k)]

        z_apex = float(teren.z(rk.wx, rk.wy))
        # "visina terena gdje se sijeku kupa i teren" — najviša kota presjeka
        if rk.konture:
            z_presjek_max = float(max(np.asarray(kk)[:, 2].max()
                                      for kk in rk.konture))
        else:
            z_presjek_max = z_apex
        visina_kupe = rk.wz - z_apex

        c1_, c2_, c3_, c4_ = st.columns(4)
        c1_.metric("Kota vrha kupe", f"{rk.wz:.1f} m")
        c2_.metric("Visina kupe (iznad terena u vrhu)", f"{visina_kupe:.1f} m")
        c3_.metric("Kote presjeka s terenom",
                   f"{min(np.asarray(kk)[:, 2].min() for kk in rk.konture):.1f}"
                   f"–{z_presjek_max:.1f} m" if rk.konture else "—")
        c4_.metric("Širina platoa k", f"{rk.k:.0f} m")

        min_wz = float(np.ceil(z_presjek_max + 1.0))
        cA_, cB_, cC_ = st.columns(3)
        wz_novo = cA_.number_input(
            "Nova kota vrha (m)", min_value=min_wz,
            max_value=float(teren.z_max + 300.0),
            value=float(max(rk.wz, min_wz)), step=5.0,
            help=f"Ne može ispod najviše kote presjeka kupe i terena "
                 f"({z_presjek_max:.1f} m).")
        korak = cB_.number_input("Visina etaže — korak (m)", min_value=2.0,
                                 max_value=100.0, value=10.0, step=1.0)
        berma = cC_.number_input("Širina berme — suženje po nivou (m)",
                                 min_value=0.0, max_value=50.0, value=4.0,
                                 step=0.5)

        rast = st.radio(
            "Način rasta kupe",
            ["Fiksno dno (dno kao original, plato se sužava)",
             "Fiksni plato (k kao original, dno se širi)"],
            horizontal=True,
            help="Fiksno dno: footprint ostaje kao kod originalne kupe, "
                 "visina raste sužavanjem platoa. Fiksni plato: k ostaje "
                 "isti, pa svaka dodatna visina širi dno za ΔH/tan(ugao) "
                 "+ berme.")

        k_novo = rk.k
        if rast.startswith("Fiksno dno") and rk.konture:
            # cilj: max poluprečnik footprinta = kao kod originalne kupe
            svek0 = np.vstack(rk.konture)
            r_cilj = float(np.hypot(svek0[:, 0] - rk.wx,
                                    svek0[:, 1] - rk.wy).max())

            def _r_max(k_probe: float) -> float:
                sk_ = StepenastaKupa(wx=rk.wx, wy=rk.wy, wz=float(wz_novo),
                                     k=float(k_probe), ugao=rk.ugao,
                                     korak=float(korak), berma=float(berma),
                                     profil=ctx.profil)
                r_ = presek_kupe_i_terena(sk_, teren, rezolucija=160,
                                          rafiniranje=1)
                if not r_.ima_preseka or not r_.konture:
                    return 0.0
                sv = np.vstack(r_.konture)
                return float(np.hypot(sv[:, 0] - rk.wx,
                                      sv[:, 1] - rk.wy).max())

            k_lo, k_hi = 2.0, rk.k + max(0.0, (rk.wz - wz_novo)
                                         / skupa_tan(rk.ugao)) + 5.0                 if False else (2.0, rk.k * 1.5 + 50.0)
            k_lo, k_hi = 2.0, rk.k * 1.5 + 50.0
            if _r_max(k_lo) > r_cilj + 1.0:
                tan_ef = float(korak) / (float(korak)
                                         / np.tan(np.radians(rk.ugao))
                                         + float(berma))
                wz_max = rk.wz + (rk.k - k_lo) * tan_ef
                st.error(f"Tražena visina je prevelika za fiksno dno — "
                         f"plato bi morao ispod {k_lo:.0f} m. Maksimalna "
                         f"kota vrha uz ovo dno je ≈ {wz_max:.0f} m "
                         f"(ili smanji korak/bermu).")
                st.stop()
            for _ in range(10):                       # bisekcija po k
                k_mid = 0.5 * (k_lo + k_hi)
                if _r_max(k_mid) > r_cilj:
                    k_hi = k_mid
                else:
                    k_lo = k_mid
            k_novo = 0.5 * (k_lo + k_hi)
            st.caption(f"Fiksno dno: plato sužen sa k = {rk.k:.0f} m na "
                       f"**k = {k_novo:.1f} m** (ciljni poluprečnik dna "
                       f"{r_cilj:.0f} m).")

        skupa = StepenastaKupa(wx=rk.wx, wy=rk.wy, wz=float(wz_novo),
                               k=float(k_novo), ugao=rk.ugao,
                               korak=float(korak),
                               berma=float(berma), profil=ctx.profil)
        rez_s = presek_kupe_i_terena(skupa, teren, rezolucija=256,
                                     rafiniranje=2)
        glatka = Kupa(wx=rk.wx, wy=rk.wy, wz=float(wz_novo),
                      k=float(k_novo),
                      ugao=rk.ugao, profil=ctx.profil)
        rez_g = presek_kupe_i_terena(glatka, teren, rezolucija=256,
                                     rafiniranje=1)

        if not rez_s.ima_preseka:
            st.warning("Stepenasta kupa nema presjeka s terenom — podigni "
                       "novu kotu vrha.")
        else:
            from pipeline_v2 import provjeri_footprint
            prekrsaji = provjeri_footprint(rez_s, ctx)
            if prekrsaji:
                st.error("Novo (šire) dno krši ograničenja: "
                         + "; ".join(prekrsaji)
                         + ". Smanji visinu, koristi 'Fiksno dno' ili "
                           "smanji bermu.")
            else:
                st.success("Footprint stepenaste kupe je unutar interesne "
                           "zone i van loših (K) zona.")
            m1_, m2_, m3_, m4_ = st.columns(4)
            m1_.metric("Zapremina (stepenasta)",
                       f"{rez_s.zapremina:,.0f} m³",
                       delta=f"{rez_s.zapremina - rez_g.zapremina:+,.0f} "
                             f"vs glatka", delta_color="off")
            m2_.metric("Površina osnove", f"{rez_s.povrsina_osnove:,.0f} m²")
            m3_.metric("Broj etaža",
                       int(np.ceil((wz_novo - z_apex) / korak)))
            m4_.metric("Petlji presjeka", rez_s.broj_petlji)

            # ---- 3D prikaz ----
            cO1_, cO2_ = st.columns(2)
            z_uv_s = cO1_.slider("Uveličanje visine", 1.0, 5.0, 2.0, 0.5,
                                 key="step_zuv")
            zone_s = cO2_.checkbox("Prikaži zone interesa", value=False,
                                   key="step_zone")

            x0s, x1s, y0s, y1s = rez_s.granice_racuna
            if rez_s.konture:
                svek = np.vstack(rez_s.konture)
                x0s, x1s = svek[:, 0].min(), svek[:, 0].max()
                y0s, y1s = svek[:, 1].min(), svek[:, 1].max()
            pads = 0.22 * max(x1s - x0s, y1s - y0s, 2 * rk.k)
            vx0s, vx1s = x0s - pads, x1s + pads
            vy0s, vy1s = y0s - pads, y1s + pads
            ns = 170
            GXs, GYs = np.meshgrid(np.linspace(vx0s, vx1s, ns),
                                   np.linspace(vy0s, vy1s, ns))
            ZTs = teren.z(GXs, GYs)
            ZKs = skupa.z(GXs, GYs)
            tijelo_s = np.where(ZKs > ZTs + 1e-9, ZKs, np.nan)

            fs = go.Figure()
            fs.add_trace(go.Surface(x=GXs, y=GYs, z=ZTs, colorscale="Earth",
                                    showscale=False, name="teren"))
            fs.add_trace(go.Surface(x=GXs, y=GYs, z=tijelo_s, opacity=0.9,
                                    colorscale=[[0, "peru"],
                                                [1, "burlywood"]],
                                    showscale=False, name="stepenasta kupa"))
            if zone_s:
                _dodaj_zone_na_teren(fs, teren, dobre, lose)
            for i2, kont in enumerate(rez_s.konture):
                fs.add_trace(go.Scatter3d(
                    x=kont[:, 0], y=kont[:, 1], z=kont[:, 2] + 0.4,
                    mode="lines", line=dict(color="red", width=7),
                    name="presjek s terenom", showlegend=(i2 == 0)))
            prvi_prsten = True
            for pr in skupa.ivice_etaza(teren=teren):
                fs.add_trace(go.Scatter3d(
                    x=pr["x"], y=pr["y"], z=pr["z"] + 0.3, mode="lines",
                    line=dict(color="royalblue", width=3),
                    name="ivice etaža", legendgroup="etaze",
                    showlegend=prvi_prsten, connectgaps=False))
                prvi_prsten = False

            dxs, dys = vx1s - vx0s, vy1s - vy0s
            zmin_s = float(np.nanmin(ZTs)) - 2
            zmax_s = float(wz_novo) + 2
            dzs = max(zmax_s - zmin_s, 1.0)
            ms_ = max(dxs, dys)
            fs.update_layout(
                scene=dict(aspectmode="manual",
                           aspectratio=dict(x=dxs / ms_, y=dys / ms_,
                                            z=(dzs / ms_) * float(z_uv_s)),
                           xaxis=dict(range=[vx0s, vx1s]),
                           yaxis=dict(range=[vy0s, vy1s]),
                           zaxis=dict(range=[zmin_s, zmax_s],
                                      title="z (m)")),
                height=620, margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(orientation="h", y=0.02))
            st.plotly_chart(fs, use_container_width=True)

            # radijalni profil kroz vrh — da se stepenice jasno vide
            azp = st.slider("Azimut profila (°)", 0, 180, 0, 5,
                            key="step_az")
            ap_ = np.radians(azp)
            Rp = 0.55 * (x1s - x0s) + pads
            tp = np.linspace(-Rp, Rp, 800)
            pxp = rk.wx + tp * np.cos(ap_)
            pyp = rk.wy + tp * np.sin(ap_)
            ztp, zkp = teren.z(pxp, pyp), skupa.z(pxp, pyp)
            fp2 = go.Figure()
            fp2.add_trace(go.Scatter(x=tp, y=np.where(zkp > ztp, zkp, ztp),
                                     mode="lines",
                                     line=dict(color="peru", width=0),
                                     showlegend=False))
            fp2.add_trace(go.Scatter(x=tp, y=ztp, mode="lines",
                                     fill="tonexty",
                                     fillcolor="rgba(205,133,63,0.55)",
                                     line=dict(color="black", width=2),
                                     name="teren"))
            fp2.add_trace(go.Scatter(x=tp, y=zkp, mode="lines",
                                     line=dict(color="royalblue", width=2),
                                     name="stepenasta kupa"))
            fp2.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                              xaxis_title="rastojanje od vrha (m)",
                              yaxis_title="kota (m)",
                              legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fp2, use_container_width=True)


st.caption("Geometrija: geometrija_v2 (visinska polja, ∬ max(0, z_kupa − z_teren) dA) · "
           "MC filteri: interesna zona, K zone, distanca od CM, pokrivenost terena · "
           "Funkcija cilja: (c1 + c2 + eko) / V")
