"""
geometry.py  –  Korak 2 migracije: geometrija odlagališta (kupa)

Zamjenjuje MATLAB funkcije:
  pol2cart()                      → pol2cart() u numpy
  delaunay()                      → scipy.spatial.Delaunay
  SurfaceIntersection.m           → surface_intersection() — Möller 3D trougao/trougao
  convhull()                      → scipy.spatial.ConvexHull
  zapreminaKupeGenetskiAlgoritam  → zapremina_kupe()
  unutarInteresneZone.m           → unutar_interesne_zone()
  inpolygon()                     → matplotlib.path.Path.contains_points()

Sve funkcije primaju i vraćaju eksplicitne argumente — nema globalnih varijabli.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull, Delaunay


# ---------------------------------------------------------------------------
# Tipovi
# ---------------------------------------------------------------------------

@dataclass
class Surface:
    """Triangulisana 3D površina.

    Ekvivalent MATLAB struct-a sa poljima .vertices i .faces
    """
    vertices: np.ndarray   # (N, 3)  — X, Y, Z
    faces: np.ndarray      # (M, 3)  — indeksi trouglova


@dataclass
class RezultatKupe:
    """Izlaz iz zapremina_kupe() — sve što GA i post-procesiranje trebaju."""
    zapremina: float                     # m³  (ConvexHull)
    donja_povrsina: Optional[np.ndarray] # (K, 2) XY tačke presječišta kupe i terena
    ekonomska_cena: float                # placeholder — puni se u ekonomija.py
    zone: str                            # zone koje kupa pokriva
    gornja_kontura: np.ndarray           # (9,)  XX1  gornja ivica kupe
    gornja_kontura_y: np.ndarray         # (9,)  YY1
    z_gore: float                        # wz  (visina vrha kupe)
    intersect_surface: Optional[Surface] # presječišna površina (za DXF)


# ---------------------------------------------------------------------------
# pol2cart  —  MATLAB pol2cart ekvivalent
# ---------------------------------------------------------------------------

def pol2cart(theta: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Konvertuje polarne u Kartezijanske koordinate.

    MATLAB ekvivalent:  [p, t, q] = pol2cart(theta, r, z)
    Ovdje ignorišemo z jer se dodaje direktno kao wz u pozivu.

    Args:
        theta: array kutova (radijani)
        r:     array poluprečnika

    Returns:
        (x, y) tuple numpy arraya
    """
    return r * np.cos(theta), r * np.sin(theta)


# ---------------------------------------------------------------------------
# inpolygon  —  MATLAB inpolygon ekvivalent
# ---------------------------------------------------------------------------

def inpolygon(px: np.ndarray, py: np.ndarray,
              poly_x: np.ndarray, poly_y: np.ndarray) -> np.ndarray:
    """Vraća bool masku: True za tačke koje su unutar poligona.

    MATLAB ekvivalent:  mask = inpolygon(dots(:,1), dots(:,2), x, y)

    Args:
        px, py:           koordinate tačaka koje se testiraju  (N,)
        poly_x, poly_y:   koordinate poligona  (M,)

    Returns:
        bool array shape (N,)
    """
    points = np.column_stack([px, py])
    polygon = np.column_stack([poly_x, poly_y])

    # Zatvori poligon ako nije zatvoren
    if not np.allclose(polygon[0], polygon[-1]):
        polygon = np.vstack([polygon, polygon[0]])

    path = MplPath(polygon)
    return path.contains_points(points)


# ---------------------------------------------------------------------------
# Kontura kupe  —  gornja i donja ivica
# ---------------------------------------------------------------------------

def _kontura_kupe(wx: float, wy: float, wz: float, k: float,
                  mnv: float, ugao: float
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                              np.ndarray, np.ndarray, np.ndarray]:
    """Računa gornju i donju konturu konusnog odlagališta.

    MATLAB ekvivalent (iz zapreminaKupeGenetskiAlgoritam.m):
        k=0.001 (u originalnom), ali ovdje k je parametar GA (širina)
        s=1.4*k;  u=1.25*k
        r=[s k u k s k u k s]
        theta=(0:pi/4:2*pi)
        [p,t,q]=pol2cart(theta,r,z)
        XX1=wx+p;  YY1=wy+t;  ZZ1=[wz wz wz wz wz wz wz wz wz]
        l=H1*1/tand(ugao)
        r1=[s+l k+l u+l k+l s+l k+l u+l k+l s+l]
        [p1,t1,q1]=pol2cart(theta,r1,z2)
        XX2=wx+p1;  YY2=wy+t1;  ZZ2=[H H H H H H H H H]

    Args:
        wx, wy, wz: koordinate vrha kupe
        k:          parametar širine (polupречnik) — varijabla GA
        mnv:        nadmorska visina baze (DodatniParametri.nadmorska_visina)
        ugao:       ugao kosine odlagališta (u stepenima)

    Returns:
        (XX1, YY1, ZZ1, XX2, YY2, ZZ2) — gornja i donja kontura
    """
    s = 1.4 * k
    u = 1.25 * k
    r = np.array([s, k, u, k, s, k, u, k, s])
    theta = np.arange(0, 2 * np.pi + np.pi / 4, np.pi / 4)  # 9 tačaka

    # Gornja površina (na visini wz)
    px, py = pol2cart(theta, r)
    XX1 = wx + px
    YY1 = wy + py
    ZZ1 = np.full(9, wz)

    # Donja površina — spuštena za mnv, proširena po uglu
    H1 = wz - mnv
    H = wz - H1          # = mnv  (visina baze)
    l = H1 / np.tan(np.radians(ugao))
    r1 = r + l            # širi se po uglu
    px1, py1 = pol2cart(theta, r1)
    XX2 = wx + px1
    YY2 = wy + py1
    ZZ2 = np.full(9, H)

    return XX1, YY1, ZZ1, XX2, YY2, ZZ2


# ---------------------------------------------------------------------------
# Surface Intersection  —  Möller triangle-triangle intersection
# ---------------------------------------------------------------------------

def surface_intersection(surface1: Surface, surface2: Surface) -> Surface:
    """Računa presječišnu površinu između dvije triangulisane 3D mreže.

    Implementacija Möller-ovog algoritma za trougao–trougao presječište.
    Ekvivalent MATLAB SurfaceIntersection.m (Tomas Möller, 1997).

    Vraća presječišnu površinu kao Surface objekat.
    Ako nema presječišta, surface.vertices je prazan array.

    Reference:
        Möller, T. (1997). A Fast Triangle-Triangle Intersection Test.
        Journal of Graphics Tools, 2(2), 1997.
    """
    v1 = surface1.vertices
    f1 = surface1.faces
    v2 = surface2.vertices
    f2 = surface2.faces

    # -----------------------------------------------------------------------
    # Pomoćne lambda funkcije — identično MATLAB kodu
    # -----------------------------------------------------------------------
    def cross_prod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.column_stack([
            a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
            a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
        ])

    def dot_prod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a * b).sum(axis=1)

    def normalize(V: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return V / norms

    epsilon = np.finfo(float).eps

    # Tačke trouglova surface1
    V1 = v1[f1[:, 0]]
    V2 = v1[f1[:, 1]]
    V3 = v1[f1[:, 2]]
    N1 = normalize(cross_prod(V2 - V1, V3 - V1))
    d1 = dot_prod(N1, V1)

    # Tačke trouglova surface2
    U1 = v2[f2[:, 0]]
    U2 = v2[f2[:, 1]]
    U3 = v2[f2[:, 2]]
    N2 = normalize(cross_prod(U2 - U1, U3 - U1))
    d2 = dot_prod(N2, U1)

    nf1 = len(f1)
    nf2 = len(f2)

    # -----------------------------------------------------------------------
    # Faza 1: Grubi rez — rastojanja vrhova do ravni
    # Odgovara Stage 1 u MATLAB kodu
    # -----------------------------------------------------------------------

    # Rastojanja svih vrhova surface2 od ravni svakog trougla surface1
    # du shape: (nf1, nv2) — MATLAB: du = zeros(nFace1, nVert2)
    nv2 = len(v2)
    du = (N1 @ v2.T) - d1[:, np.newaxis]      # (nf1, nv2)
    du[np.abs(du) < epsilon] = 0.0

    du1 = du[:, f2[:, 0]]   # (nf1, nf2)
    du2 = du[:, f2[:, 1]]
    du3 = du[:, f2[:, 2]]

    # int_matrix: -2 = nepoznato, 0 = nema presječišta, 1 = ima
    int_matrix = np.full((nf1, nf2), -2, dtype=np.int8)
    int_matrix[(du1 * du2 > 0) & (du1 * du3 > 0)] = 0  # svi isti znak
    int_matrix[(du1 == 0) & (du2 == 0) & (du3 == 0)] = -1  # koplanarni

    if np.all(int_matrix == 0):
        return Surface(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=int))

    # Rastojanja svih vrhova surface1 od ravni svakog trougla surface2
    nv1 = len(v1)
    dv = (N2 @ v1.T) - d2[:, np.newaxis]      # (nf2, nv1)
    dv[np.abs(dv) < epsilon] = 0.0

    dv1 = dv[:, f1[:, 0]].T   # (nf1, nf2)
    dv2 = dv[:, f1[:, 1]].T
    dv3 = dv[:, f1[:, 2]].T

    int_matrix[(dv1 * dv2 > 0) & (dv1 * dv3 > 0)] = 0
    int_matrix[(dv1 == 0) & (dv2 == 0) & (dv3 == 0)] = -1

    if np.all(int_matrix == 0):
        return Surface(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=int))

    # -----------------------------------------------------------------------
    # Faza 2: Möller test za ne-koplanarne parove
    # Odgovara Stage 2 / TriangleIntersection3D_Moller u MATLAB kodu
    # -----------------------------------------------------------------------

    t_mask = (int_matrix == -2)
    face1_idx, face2_idx = np.where(t_mask)

    all_vertices = []
    all_faces = []
    n_verts = 0

    if len(face1_idx) > 0:
        # Lookup table za određivanje "odd" vrha
        lut = np.array([0, 3, 3, 2, 1, 3, 2, 2, 1, 1, 2, 3, 3, 0, 3,
                         3, 2, 1, 1, 2, 2, 3, 1, 2, 3, 3, 0])

        f1i = face1_idx
        f2i = face2_idx

        # Grupirani podaci za parne trouglove
        Av = V1[f1i]; Bv = V2[f1i]; Cv = V3[f1i]
        Au = U1[f2i]; Bu = U2[f2i]; Cu = U3[f2i]
        n1i = N1[f1i]; d1i = d1[f1i]
        n2i = N2[f2i]; d2i = d2[f2i]

        # Rastojanja za ovaj podskup parova
        dv_sub = np.column_stack([dv1[f1i, f2i], dv2[f1i, f2i], dv3[f1i, f2i]])
        du_sub = np.column_stack([du1[f1i, f2i], du2[f1i, f2i], du3[f1i, f2i]])

        # Smjer linije presječišta dvaju ravni  D = N1 × N2
        D = cross_prod(n1i, n2i)
        max_dim = np.argmax(np.abs(D), axis=1)   # (n,) indeks najveće komponente

        # Normalizuj D i nađi tačku na liniji (rješava sistem N*O = d)
        D_norm = D / np.linalg.norm(D, axis=1, keepdims=True)
        n_pairs = len(f1i)
        O = np.zeros((n_pairs, 3))
        for r in range(n_pairs):
            N_sys = np.array([n1i[r], n2i[r], np.zeros(3)])
            N_sys[2, max_dim[r]] = 1.0
            rhs = np.array([d1i[r], d2i[r], 0.0])
            try:
                O[r] = np.linalg.solve(N_sys, rhs)
            except np.linalg.LinAlgError:
                O[r] = 0.0

        # Projekcije trouglova na liniju presječišta
        Vp = np.column_stack([
            dot_prod(Av - O, D_norm),
            dot_prod(Bv - O, D_norm),
            dot_prod(Cv - O, D_norm),
        ])
        Up = np.column_stack([
            dot_prod(Au - O, D_norm),
            dot_prod(Bu - O, D_norm),
            dot_prod(Cu - O, D_norm),
        ])

        # Nađi "odd" vrh za svaki trougao koristeći LUT
        n = n_pairs
        rows = np.arange(n)

        def odd_vertex(d_arr: np.ndarray) -> np.ndarray:
            """Vraća indeks 'odd' vrha (0-based) koristeći Möller LUT."""
            signs = np.sign(d_arr).astype(int)
            key = signs @ np.array([9, 3, 1]) + 13   # +13 da bude 0-based u lut
            key = np.clip(key, 0, len(lut) - 1)
            return lut[key] - 1   # pretvori u 0-based

        a1 = odd_vertex(dv_sub)
        a2 = odd_vertex(du_sub)

        # Ostala dva vrha (bc parovi)
        b1 = (a1 + 1) % 3
        c1 = (a1 + 2) % 3
        b2 = (a2 + 1) % 3
        c2 = (a2 + 2) % 3

        # Intervali presječišta surface1 i surface2 s linijom
        def interval_t(Vp_arr, a, b, c, d_arr):
            va = Vp_arr[rows, a]
            vb = Vp_arr[rows, b]
            vc = Vp_arr[rows, c]
            da = d_arr[rows, a]
            db = d_arr[rows, b]
            dc = d_arr[rows, c]
            denom_b = db - da
            denom_c = dc - da
            # Zaštita od dijeljenja nulom
            denom_b = np.where(np.abs(denom_b) < 1e-12, 1e-12, denom_b)
            denom_c = np.where(np.abs(denom_c) < 1e-12, 1e-12, denom_c)
            t1 = va - (vb - va) * da / denom_b
            t2 = va - (vc - va) * da / denom_c
            return t1, t2

        t1, t2 = interval_t(Vp, a1, b1, c1, dv_sub)
        s1, s2 = interval_t(Up, a2, b2, c2, du_sub)

        # Sortiraj intervale
        swap = t2 < t1
        t1[swap], t2[swap] = t2[swap].copy(), t1[swap].copy()
        swap = s2 < s1
        s1[swap], s2[swap] = s2[swap].copy(), s1[swap].copy()

        # Test preklapanja intervala
        overlap = (s1 < t2) & (t1 < s2)
        int_matrix[t_mask] = np.where(overlap, 1, 0)

        # Izračunaj tačke presječišta za parove koji se sijeku
        ovlp_idx = np.where(overlap)[0]
        if len(ovlp_idx) > 0:
            p1 = D_norm[ovlp_idx] * np.maximum(t1[ovlp_idx], s1[ovlp_idx])[:, np.newaxis] + O[ovlp_idx]
            p2 = D_norm[ovlp_idx] * np.minimum(t2[ovlp_idx], s2[ovlp_idx])[:, np.newaxis] + O[ovlp_idx]
            m = len(ovlp_idx)
            new_verts = np.vstack([p1, p2])
            new_faces = np.column_stack([
                np.arange(m),
                np.arange(m, 2 * m),
                np.arange(m, 2 * m),
            ])
            all_vertices.append(new_verts)
            all_faces.append(new_faces + n_verts)
            n_verts += len(new_verts)

    # -----------------------------------------------------------------------
    # Sastavi izlaznu površinu
    # -----------------------------------------------------------------------
    if not all_vertices:
        return Surface(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=int))

    vertices = np.vstack(all_vertices)
    faces = np.vstack(all_faces)

    # Ukloni duplikate (kao u MATLAB kodu: PointRoundingTol = 1e6)
    tol = 1e6
    P_rounded = np.round(vertices * tol) / tol
    _, ia, ic = np.unique(P_rounded, axis=0, return_index=True, return_inverse=True)
    vertices_unique = vertices[ia]
    faces_unique = ic[faces]

    return Surface(vertices=vertices_unique, faces=faces_unique)


# ---------------------------------------------------------------------------
# Zapremina kupe  —  glavni geometrijski proračun
# ---------------------------------------------------------------------------

def zapremina_kupe(
    wx: float, wy: float, wz: float,
    ugao: float, k: float,
    mnv: float,
    teren: "Surface",
    zona_x: np.ndarray, zona_y: np.ndarray,
    ekonomska_fn=None,
) -> RezultatKupe:
    """Računa zapreminu konusnog odlagališta presjekom sa terenom.

    MATLAB ekvivalent: zapreminaKupeGenetskiAlgoritam(wx, wy, wz, ugao, k)

    Algoritam:
    1. Konstruiši gornju (XX1, YY1) i donju (XX2, YY2) konturu kupe
    2. Provjeri da li je donja kontura unutar interesne zone
    3. Izračunaj Delaunay mesh kupe
    4. Nađi presječište kupe sa terenom (SurfaceIntersection)
    5. Izračunaj ConvexHull zapreminu presječišta + gornje konture
    6. Izračunaj ekonomsku cijenu (poziva ekonomska_fn ako je data)

    Args:
        wx, wy, wz:    koordinate vrha kupe
        ugao:          ugao kosine (stepeni)
        k:             širina — varijabla GA
        mnv:           nadmorska visina baze (iz DodatniParametri)
        teren:         Surface objekat terena
        zona_x/y:      poligon interesne zone (za provjeru)
        ekonomska_fn:  funkcija(surface_intersection) → (cijena, zone_str)
                       Ako None, cena=0, zone=""

    Returns:
        RezultatKupe sa zapreminom, donjom površinom, cijenom i konturama
    """
    # Defaultna vrijednost ako nema presječišta
    VELIKA_VREDNOST = 40_000_000.0

    s = 1.4 * k
    u_val = 1.25 * k

    # --- Konstruisanje kontura kupe ---
    XX1, YY1, ZZ1, XX2, YY2, ZZ2 = _kontura_kupe(wx, wy, wz, k, mnv, ugao)

    # --- Provjera: donja kontura unutar zone interesa ---
    if not np.all(inpolygon(XX2, YY2, zona_x, zona_y)):
        return RezultatKupe(
            zapremina=VELIKA_VREDNOST,
            donja_povrsina=None,
            ekonomska_cena=VELIKA_VREDNOST,
            zone="",
            gornja_kontura=XX1,
            gornja_kontura_y=YY1,
            z_gore=wz,
            intersect_surface=None,
        )

    # --- Delaunay mesh kupe ---
    x_kupa = np.concatenate([XX1, XX2])
    y_kupa = np.concatenate([YY1, YY2])
    z_kupa = np.concatenate([ZZ1, ZZ2])

    try:
        tri = Delaunay(np.column_stack([x_kupa, y_kupa]))
        kupa = Surface(
            vertices=np.column_stack([x_kupa, y_kupa, z_kupa]),
            faces=tri.simplices,
        )
    except Exception:
        return RezultatKupe(
            zapremina=VELIKA_VREDNOST,
            donja_povrsina=None,
            ekonomska_cena=VELIKA_VREDNOST,
            zone="",
            gornja_kontura=XX1,
            gornja_kontura_y=YY1,
            z_gore=wz,
            intersect_surface=None,
        )

    # --- Presječište kupe i terena ---
    intersect_surf = surface_intersection(kupa, teren)

    if intersect_surf.vertices.shape[0] == 0:
        return RezultatKupe(
            zapremina=VELIKA_VREDNOST,
            donja_povrsina=None,
            ekonomska_cena=VELIKA_VREDNOST,
            zone="",
            gornja_kontura=XX1,
            gornja_kontura_y=YY1,
            z_gore=wz,
            intersect_surface=intersect_surf,
        )

    x_int = intersect_surf.vertices[:, 0]
    y_int = intersect_surf.vertices[:, 1]
    z_int = intersect_surf.vertices[:, 2]

    # --- ConvexHull zapremina (presječište + gornja kontura) ---
    x_full = np.concatenate([x_int, XX1])
    y_full = np.concatenate([y_int, YY1])
    z_full = np.concatenate([z_int, ZZ1])

    try:
        hull = ConvexHull(np.column_stack([x_full, y_full, z_full]))
        zapremina = hull.volume
    except Exception:
        return RezultatKupe(
            zapremina=VELIKA_VREDNOST,
            donja_povrsina=None,
            ekonomska_cena=VELIKA_VREDNOST,
            zone="",
            gornja_kontura=XX1,
            gornja_kontura_y=YY1,
            z_gore=wz,
            intersect_surface=intersect_surf,
        )

    # --- Donja površina = XY tačke presječišta ---
    donja_povrsina = np.column_stack([x_int, y_int])

    # --- Ekonomska cijena ---
    ekonomska_cena = 0.0
    zone_str = ""
    if ekonomska_fn is not None:
        try:
            ekonomska_cena, zone_str = ekonomska_fn(intersect_surf)
        except Exception:
            ekonomska_cena = 0.0
            zone_str = ""

    return RezultatKupe(
        zapremina=zapremina,
        donja_povrsina=donja_povrsina,
        ekonomska_cena=ekonomska_cena,
        zone=zone_str,
        gornja_kontura=XX1,
        gornja_kontura_y=YY1,
        z_gore=wz,
        intersect_surface=intersect_surf,
    )


# ---------------------------------------------------------------------------
# Provjera unutar interesne zone  —  zamjenjuje unutarInteresneZone.m
# ---------------------------------------------------------------------------

def unutar_interesne_zone(
    zona_x: np.ndarray, zona_y: np.ndarray,
    wx: float, wy: float, wz: float,
    ugao: float, k: float, mnv: float,
) -> tuple[bool, np.ndarray]:
    """Provjerava da li donja površina kupe ostaje unutar granice zone.

    MATLAB ekvivalent: unutarInteresneZone(x, y, z, wx, wy, wz, ugao, k, 0)

    Returns:
        (unutra, donja_povrsina_xy)
        unutra = True ako sve tačke donje konture su unutar zone
    """
    _, _, _, XX2, YY2, _ = _kontura_kupe(wx, wy, wz, k, mnv, ugao)
    mask = inpolygon(XX2, YY2, zona_x, zona_y)
    donja_povrsina = np.column_stack([XX2, YY2])
    return bool(np.all(mask)), donja_povrsina


# ---------------------------------------------------------------------------
# Generisanje i filtriranje nasumičnih tačaka  —  Faza 2 izvrsnog koda
# ---------------------------------------------------------------------------

def generiši_tačke(
    n: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    zona_x: np.ndarray,
    zona_y: np.ndarray,
    lose_zone: list,
) -> np.ndarray:
    """Generiše nasumične tačke unutar zone i filtrira loše zone.

    MATLAB ekvivalent (iz IzvrsniKodBuvac.m):
        dots = rand(numDots, 3)
        dots(:,1) = dots(:,1)*diff(xrange) + xrange(1)
        ...
        dotsIn = inpolygon(dots(:,1), dots(:,2), x, y)
        goodPoint = dotsIn  (+ filtriranje loših zona)

    Args:
        n:          broj nasumičnih tačaka
        x/y/z_range: opsezi koordinata
        zona_x/y:   poligon interesne zone
        lose_zone:  lista EkonomskaZona objekata (Z-5 zone)

    Returns:
        np.ndarray shape (M, 3) — samo dobre tačke (X, Y, Z)
    """
    # Generiši nasumične tačke u bounding boxu
    dots = np.random.rand(n, 3)
    dots[:, 0] = dots[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
    dots[:, 1] = dots[:, 1] * (y_range[1] - y_range[0]) + y_range[0]
    dots[:, 2] = dots[:, 2] * (z_range[1] - z_range[0]) + z_range[0]

    # Filtriranje: samo tačke unutar poligona interesne zone
    maska = inpolygon(dots[:, 0], dots[:, 1], zona_x, zona_y)

    # Izbacivanje iz loših zona (Z-5)
    for zona in lose_zone:
        u_losoj = inpolygon(dots[:, 0], dots[:, 1], zona.x_data, zona.y_data)
        maska = maska & ~u_losoj

    return dots[maska]
