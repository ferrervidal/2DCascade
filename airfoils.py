import numpy as np
from scipy.optimize import least_squares


def _cross2(a, b):
    """2-D scalar cross product."""
    return a[0] * b[1] - a[1] * b[0]


def _arc_points(center, p0, p1, npts, side):
    """
    Circular arc from p0 to p1.

    side = "pressure" selects the lower arc.
    side = "suction" selects the upper arc.
    """

    a0 = np.arctan2(p0[1] - center[1], p0[0] - center[0])
    a1 = np.arctan2(p1[1] - center[1], p1[0] - center[0])

    candidates = []

    for k in [-1, 0, 1]:
        delta = a1 - a0 + 2.0 * np.pi * k
        angles = np.linspace(a0, a0 + delta, npts)
        xmid = center[0] + np.cos(angles[len(angles) // 2]) * np.linalg.norm(p0 - center)
        ymid = center[1] + np.sin(angles[len(angles) // 2]) * np.linalg.norm(p0 - center)

        candidates.append((ymid, abs(delta), angles))

    if side == "pressure":
        # Select the candidate lying below the chord
        valid = [c for c in candidates if c[0] < 0.0]
    else:
        # Select the candidate lying above the chord
        valid = [c for c in candidates if c[0] > 0.0]

    if not valid:
        valid = candidates

    _, _, angles = min(valid, key=lambda item: item[1])

    radius = np.linalg.norm(p0 - center)

    return np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])


def _edge_point(center, tangent, normal, radius, phi, side):
    """
    Point on a circular edge blend.

    The edge nose is located at the point opposite the center for the LE,
    and at the point in the tangent direction for the TE.
    """

    if side == "pressure":
        sign = -1.0
    else:
        sign = 1.0

    radial = -np.cos(phi) * tangent + sign * np.sin(phi) * normal

    return center + radius * radial


def DCA(
    beta_in_deg,
    beta_out_deg,
    n_points=201,
    t_max=0.08,
    r_le=0.002,
    r_te=0.002,
    thickness_tolerance=1e-5
):
    """
    Generate a canonical rounded DCA blade section.

    Parameters
    ----------
    beta_in_deg : float
        Mean-line inlet metal angle measured from the global x-axis.

    beta_out_deg : float
        Mean-line outlet metal angle measured from the global x-axis.

    t_max : float
        Maximum total thickness divided by chord.

    r_le : float
        Leading-edge radius divided by chord.

    r_te : float
        Trailing-edge radius divided by chord.

    n_points : int
        Approximate total number of output coordinates.

    Returns
    -------
    X, Y : tuple of ndarray
        Clockwise coordinates ordered as:

        TE -> pressure surface -> LE -> suction surface -> TE
    """

    if not (0.0 < t_max < 1.0):
        raise ValueError("t_max must be a fraction of chord between 0 and 1.")

    if not (0.0 < r_le < 0.5):
        raise ValueError("r_le must be a positive fraction of chord.")

    if not (0.0 < r_te < 0.5):
        raise ValueError("r_te must be a positive fraction of chord.")

    if n_points < 40:
        raise ValueError("Use at least 40 coordinate points.")

    beta_in = np.deg2rad(beta_in_deg)
    beta_out = np.deg2rad(beta_out_deg)

    # Chord stagger is the mean of the inlet and outlet metal angles
    stagger = 0.5 * (beta_in + beta_out)

    # Angles relative to the chord
    theta_in = beta_in - stagger
    theta_out = beta_out - stagger

    # Chord-frame end points
    p_le = np.array([0.0, 0.0])
    p_te = np.array([1.0, 0.0])

    t_le = np.array([np.cos(theta_in), np.sin(theta_in)])
    n_le = np.array([-np.sin(theta_in), np.cos(theta_in)])

    t_te = np.array([np.cos(theta_out), np.sin(theta_out)])
    n_te = np.array([-np.sin(theta_out), np.cos(theta_out)])

    # Leading-edge and trailing-edge circle centers
    c_le = p_le + r_le * t_le
    c_te = p_te - r_te * t_te

    # Initial guesses for edge transition angles
    phi_le_0 = 0.25
    phi_te_0 = 0.25

    # Initial guesses for main circular-arc centers and radii
    z0 = np.array([
        phi_le_0,
        phi_te_0,

        0.5,
        -0.5,
        0.5,

        0.5,
        0.5,
        0.5
    ])

    def geometry_from_unknowns(z):
        phi_le, phi_te = z[0], z[1]

        c_pressure = np.array([z[2], z[3]])
        r_pressure = z[4]

        c_suction = np.array([z[5], z[6]])
        r_suction = z[7]

        # Main-arc transition points
        p_le_pressure = _edge_point(
            c_le, t_le, n_le, r_le, phi_le, "pressure"
        )

        p_le_suction = _edge_point(
            c_le, t_le, n_le, r_le, phi_le, "suction"
        )

        # At the TE, the radial direction is toward the TE point
        p_te_pressure = (
            c_te
            + r_te * (
                np.cos(phi_te) * t_te
                - np.sin(phi_te) * n_te
            )
        )

        p_te_suction = (
            c_te
            + r_te * (
                np.cos(phi_te) * t_te
                + np.sin(phi_te) * n_te
            )
        )

        return (
            c_pressure,
            r_pressure,
            c_suction,
            r_suction,
            p_le_pressure,
            p_te_pressure,
            p_le_suction,
            p_te_suction
        )

    def residual(z):
        (
            cp,
            rp,
            cs,
            rs,
            ple_p,
            pte_p,
            ple_s,
            pte_s
        ) = geometry_from_unknowns(z)

        residuals = []

        # Pressure-surface circularity and tangency conditions
        residuals.extend([
            np.linalg.norm(cp - ple_p) - rp,
            np.linalg.norm(cp - pte_p) - rp,
            _cross2(cp - ple_p, ple_p - c_le),
            _cross2(cp - pte_p, pte_p - c_te)
        ])

        # Suction-surface circularity and tangency conditions
        residuals.extend([
            np.linalg.norm(cs - ple_s) - rs,
            np.linalg.norm(cs - pte_s) - rs,
            _cross2(cs - ple_s, ple_s - c_le),
            _cross2(cs - pte_s, pte_s - c_te)
        ])

        # Enforce maximum thickness at x = 0.5
        def circle_y_values(center, radius, x):
            value = radius**2 - (x - center[0])**2

            if value <= 0.0:
                return None

            root = np.sqrt(value)
            return center[1] - root, center[1] + root

        y_pressure = circle_y_values(cp, rp, 0.5)
        y_suction = circle_y_values(cs, rs, 0.5)

        if y_pressure is None or y_suction is None:
            residuals.append(10.0)
        else:
            yp = min(y_pressure)
            ys = max(y_suction)
            residuals.append((ys - yp) - t_max)

        return np.asarray(residuals)

    result = least_squares(
        residual,
        z0,
        bounds=(
            [
                1e-5, 1e-5,
                -10.0, -10.0, 1e-5,
                -10.0, -10.0, 1e-5
            ],
            [
                np.pi - 1e-5, np.pi - 1e-5,
                10.0, 10.0, 10.0,
                10.0, 10.0, 10.0
            ]
        ),
        max_nfev=5000
    )

    if not result.success:
        raise ValueError(
            "DCA geometry solver failed: " + result.message
        )

    if np.max(np.abs(residual(result.x))) > thickness_tolerance:
        raise ValueError(
            "Requested DCA parameters are geometrically incompatible. "
            f"Maximum residual = {np.max(np.abs(residual(result.x))):.3e}"
        )

    (
        cp,
        rp,
        cs,
        rs,
        ple_p,
        pte_p,
        ple_s,
        pte_s
    ) = geometry_from_unknowns(result.x)

    # Number of points assigned to each section
    n_main = max(8, n_points // 4)
    n_edge = max(5, n_points // 8)

    # Pressure-side circular arc: TE transition -> LE transition
    pressure = _arc_points(
        cp,
        pte_p,
        ple_p,
        n_main,
        side="pressure"
    )

    # Suction-side circular arc: LE transition -> TE transition
    suction = _arc_points(
        cs,
        ple_s,
        pte_s,
        n_main,
        side="suction"
    )

    # Leading-edge blend: pressure transition -> LE nose -> suction transition
    a_p = np.pi + result.x[0]
    a_s = np.pi - result.x[0]

    angles_le_1 = np.linspace(a_p, np.pi, n_edge)
    angles_le_2 = np.linspace(np.pi, a_s, n_edge)


    def edge_arc(center, tangent, normal, radius,
                angle_start, angle_end, npts):

        angles = np.linspace(angle_start, angle_end, npts)

        return (
            center
            + radius * (
                np.cos(angles)[:, None] * tangent
                + np.sin(angles)[:, None] * normal
            )
        )

    phi_le = result.x[0]

    # Pressure-side transition to the LE nose
    le_pressure = edge_arc(
        c_le,
        t_le,
        n_le,
        r_le,
        -np.pi + phi_le,
        -np.pi,
        n_edge
    )

    # LE nose to suction-side transition
    le_suction = edge_arc(
        c_le,
        t_le,
        n_le,
        r_le,
        -np.pi,
        -np.pi - phi_le,
        n_edge
    )

    pressure_to_le = le_pressure[1:]
    le_to_suction = le_suction[1:]

    phi_te = result.x[1]

    # TE point to pressure-side transition
    te_pressure = edge_arc(
        c_te,
        t_te,
        n_te,
        r_te,
        0.0,
        -phi_te,
        n_edge
    )

    # Suction-side transition to TE point
    te_suction = edge_arc(
        c_te,
        t_te,
        n_te,
        r_te,
        phi_te,
        0.0,
        n_edge
    )

    # Assemble in clockwise order:
    # TE -> pressure -> LE -> suction -> TE
    points = np.vstack([
        p_te,
        te_pressure[1:],
        pressure[1:],
        pressure_to_le,
        le_to_suction,
        suction[1:],
        te_suction[1:]
    ])

    # Rotate from chord coordinates to global coordinates
    rotation = np.array([
        [np.cos(stagger), -np.sin(stagger)],
        [np.sin(stagger),  np.cos(stagger)]
    ])

    # Remove consecutive duplicate points
    tol = 1e-12

    keep = [True]
    for i in range(1, len(points)):
        keep.append(np.linalg.norm(points[i] - points[i - 1]) > tol)

    points = points[np.asarray(keep)]

    # Remove duplicated closing point if it equals the first point
    if np.linalg.norm(points[-1] - points[0]) <= tol:
        points = points[:-1]

    points_global = points @ rotation.T

    X = points_global[:, 0]
    Y = points_global[:, 1]



    return (X, Y)


def quadCamber(beta_in, beta_out, thick, numpts, thickness_model="NACA4", x_mid=None, beta_mid=None):
    """
    Generate a cambered airfoil with a selectable thickness distribution.

    Coordinates are returned clockwise:
        trailing edge -> pressure side -> leading edge
        -> suction side -> trailing edge

    Parameters
    ----------
    beta_in : float
        Camberline tangent angle at x = 0.

    beta_out : float
        Camberline tangent angle at x = 1.

    x_mid : list or None
        Intermediate camberline locations as fractions of chord.
        Must be strictly increasing.

    beta_mid : list or None
        Intermediate camberline tangent angles in degrees.
        Must have the same length as x_mid.

    thick : float
        Maximum thickness as a percentage of chord.

    numpts : int
        Total number of returned coordinates.

    thickness_model : str
        Thickness distribution selector. Currently supported:
        "NACA4".

    """
    thick=thick/100

    if numpts < 8:
        raise ValueError("numpts must be at least 8")


    # Use an even number of points
    numpts = 2*(numpts // 2)
    nside = numpts // 2

    # Check intermediate-point inputs
    has_x_mid = x_mid is not None
    has_beta_mid = beta_mid is not None

    if has_x_mid != has_beta_mid:
        raise ValueError(
            "x_mid and beta_mid must either both be supplied or both be None."
        )

    if not has_x_mid:
        x_knots = np.array([0.0, 1.0])
        beta_knots = np.array([beta_in, beta_out])

    else:
        x_mid = np.asarray(x_mid, dtype=float).ravel()
        beta_mid = np.asarray(beta_mid, dtype=float).ravel()

        if len(x_mid) != len(beta_mid):
            raise ValueError(
                "x_mid and beta_mid must have the same length."
            )

        if len(x_mid) == 0:
            x_knots = np.array([0.0, 1.0])
            beta_knots = np.array([beta_in, beta_out])

        else:
            if np.any(x_mid <= 0.0) or np.any(x_mid >= 1.0):
                raise ValueError(
                    "All x_mid values must lie between 0 and 1."
                )

            if np.any(np.diff(x_mid) <= 0.0):
                raise ValueError(
                    "x_mid values must be strictly increasing."
                )

            x_knots = np.concatenate((
                [0.0],
                x_mid,
                [1.0]
            ))

            beta_knots = np.concatenate((
                [beta_in],
                beta_mid,
                [beta_out]
            ))

    # Convert all metal angles to slopes
    beta_knots = np.deg2rad(beta_knots)
    slope_knots = np.tan(beta_knots)

    # Cosine-spaced points from leading edge to trailing edge
    x = 0.5*(1.0 - np.cos(np.linspace(0.0, np.pi, nside + 1)))

    yc = np.zeros_like(x)
    dydx = np.zeros_like(x)

    # Camberline height at the beginning of each segment
    y_start = 0.0

    for n in range(len(x_knots) - 1):

        xa = x_knots[n]
        xb = x_knots[n + 1]
        segment_length = xb - xa

        slope_a = slope_knots[n]
        slope_b = slope_knots[n + 1]

        if n == len(x_knots) - 2:
            segment = (x >= xa) & (x <= xb)
        else:
            segment = (x >= xa) & (x < xb)

        t = (x[segment] - xa)/segment_length

        # Quadratic segment with prescribed slopes at both ends
        yc[segment] = y_start + segment_length*(
            slope_a*t
            + 0.5*(slope_b - slope_a)*t**2
        )

        # Linear slope variation over the quadratic segment
        dydx[segment] = (
            slope_a
            + (slope_b - slope_a)*t
        )

        # Camberline height at the next knot
        y_start += segment_length*0.5*(slope_a + slope_b)

    # Camberline tangent angle
    theta = np.arctan(dydx)

    # Unit normal to camberline
    nx = -np.sin(theta)
    ny =  np.cos(theta)

    # Thickness distribution
    thickness_model = thickness_model.upper()

    if thickness_model == "NACA4":

        yt = 5.0*thick*(
            0.2969*np.sqrt(x)
            - 0.1260*x
            - 0.3516*x**2
            + 0.2843*x**3
            - 0.1036*x**4
        )

    else:
        raise ValueError(
            f"Unknown thickness model: {thickness_model}"
        )

    # Suction and pressure surfaces
    xu = x + yt*nx
    yu = yc + yt*ny

    xl = x - yt*nx
    yl = yc - yt*ny

    # Clockwise ordering:
    # TE -> pressure side -> LE -> suction side -> near TE
    X = np.concatenate((xl[:0:-1], xu[:-1]))
    Y = np.concatenate((yl[:0:-1], yu[:-1]))

    return X, Y

def NACA4(maxc, maxp, thick, numpts):
    # NACA 4-digit airfoil generator.
    
    pts=np.floor(numpts/2)*2 # Even number of points.
    k=np.linspace(0,np.pi,int(pts/2))
    x_c=np.zeros_like(k)
    y_c=np.zeros_like(k)
    y_u=np.zeros_like(k)
    y_l=np.zeros_like(k)

    thck=thick/100
    m=maxc/100
    p=maxp/10; 


    thckDstr=np.zeros_like(k)
    for n in range(len(k)):
        x_c[n]=(1-np.cos(k[n]))/2.0
        thckDstr[n]=(thck/0.2)*(.2969*np.sqrt(x_c[n])-0.126*x_c[n]-0.35160*x_c[n]**2+.2843*x_c[n]**3-0.1036*x_c[n]**4)
        if x_c[n]<=p:
            y_c[n]=m*(1/p)**2*(2*p*x_c[n]-x_c[n]**2)
        else:
            y_c[n]=m*(1/(p-1))**2*((1-2*p)+2*p*x_c[n]-x_c[n]**2)
        y_u[n]=y_c[n]+thckDstr[n]
        y_l[n]=y_c[n]-thckDstr[n]

    y_l=np.flip(y_l[1:])
    x_l=np.flip(x_c[1:])

    y_u=y_u[:-1]
    x_u=x_c[:-1]

    if m==0:
        y_u[0]=0

    # Concatenate the SS and PS coordinates.
    X=np.concatenate((x_l, x_u))
    Y=np.concatenate((y_l, y_u))

    return (X,Y)