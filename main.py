from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

# ----------------- Модель Лотки–Вольтерри -----------------
@dataclass
class Params:
    a: float = 0.01
    b: float = 0.004
    g: float = 0.3
    s: float = 0.2

def rhs(y: float, x: float, p: Params):
    dy = p.g * y - p.a * y * x
    dx = -p.s * x + p.b * x * y
    return dy, dx

def step_euler(x: float, y: float, h: float, p: Params):
    dy, dx = rhs(y, x, p)
    y_next = y + h * dy
    x_next = x + h * dx
    return max(x_next, 0.0), max(y_next, 0.0)

def step_rk4(x: float, y: float, h: float, p: Params):
    k1y, k1x = rhs(y, x, p)
    k2y, k2x = rhs(y + 0.5*h*k1y, x + 0.5*h*k1x, p)
    k3y, k3x = rhs(y + 0.5*h*k2y, x + 0.5*h*k2x, p)
    k4y, k4x = rhs(y + h*k3y,  x + h*k3x,  p)
    y_next = y + (h/6.0)*(k1y + 2*k2y + 2*k3y + k4y)
    x_next = x + (h/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
    return max(x_next, 0.0), max(y_next, 0.0)

def simulate(x0: float, y0: float, t0: float, t1: float, dt: float,
             p: Params, method: str = "rk4"):
    n = int(np.floor((t1 - t0) / dt)) + 1
    t = t0 + np.arange(n) * dt
    x = np.empty(n); y = np.empty(n)
    x[0], y[0] = float(x0), float(y0)
    stepper = step_rk4 if method == "rk4" else step_euler

    MAX_VAL = 1e6
    valid_len = n
    for i in range(1, n):
        xn, yn = stepper(x[i-1], y[i-1], dt, p)
        if not np.isfinite(xn) or not np.isfinite(yn) or xn > MAX_VAL or yn > MAX_VAL:
            valid_len = i
            break
        x[i], y[i] = xn, yn

    if valid_len < n:
        t = t[:valid_len]; x = x[:valid_len]; y = y[:valid_len]
    return t, x, y

# --------------------------- DASH APP ---------------------------
app = Dash(__name__)
app.title = "Хижаки–жертви: динаміка і фазовий простір"

GRAPH_HEIGHT = 430

def slider(id_, min_, max_, step_, value_, label_):
    return html.Div(
        className="ctrl",
        children=[
            html.Div(className="ctrl-label", children=[
                html.Span(label_),
                html.Span(className="ctrl-value", id=f"{id_}-label", children=str(value_))
            ]),
            dcc.Slider(
                id=id_, min=min_, max=max_, step=step_, value=value_,
                marks=None, tooltip={"placement": "bottom"}
            ),
        ],
    )

def base_fig(title: str, x_title: str, y_title: str):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=GRAPH_HEIGHT,
        template="plotly_dark",  # під темний фон сайту
        margin=dict(l=48, r=28, t=56, b=44),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)"
        ),
        font=dict(family="Inter, system-ui, sans-serif", size=13),
        # узгоджена палітра
        colorway=["#ff7a1a", "#4da3ff", "#9b6dff", "#39d98a", "#f2c94c"],
        plot_bgcolor="#10151f",
        paper_bgcolor="#10151f",
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,.06)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,.06)", zeroline=False)
    return fig

def warning_figure(message: str):
    fig = base_fig("", "", "")
    fig.add_annotation(
        text=message, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=14, color="#a5adbb")
    )
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return fig

app.layout = html.Div(
    className="container",
    children=[
        html.Div(className="header", children=[
            html.H2("Система «хижаки–жертви» (Лотка–Вольтерра)", className="h-title"),
            html.Span("Interactive demo", className="h-badge"),
        ]),

        html.Div(className="card", children=[
            dcc.Markdown(
                "dy/dt = g·y − a·y·x  \n"
                "dx/dt = −s·x + b·x·y  \n\n"
                "де *y* — жертви, *x* — хижаки.",
                className="formula"
            ),
            html.Div(id="equil-info", className="center-note")
        ]),

        html.Div(className="grid", children=[
            html.Div(className="card", children=[ dcc.Graph(id="time-series", config={"displayModeBar": True}) ]),
            html.Div(className="card", children=[ dcc.Graph(id="phase-portrait", config={"displayModeBar": True}) ]),
        ]),

        html.Details(className="details", open=False, children=[
            html.Summary("Параметри та умови (натисни, щоб розгорнути)"),
            html.Div(className="controls", children=[
                html.Div(children=[
                    slider("a", 0.001, 0.05, 0.001, 0.01, "a (вплив хижаків на жертв)"),
                    slider("b", 0.0005, 0.02, 0.0005, 0.004, "b (приріст хижаків від зустрічей)"),
                    slider("g", 0.05, 1.0, 0.01, 0.30, "g (приріст жертв)"),
                    slider("s", 0.05, 1.0, 0.01, 0.20, "s (смертність хижаків)"),
                ]),
                html.Div(children=[
                    slider("x0", 1.0, 200.0, 1.0, 20.0, "x₀ (хижаки, початково)"),
                    slider("y0", 1.0, 500.0, 1.0, 100.0, "y₀ (жертви, початково)"),
                    slider("t0", 0.0, 50.0, 0.5, 0.0, "t₀ (початок часу)"),
                    slider("t1", 10.0, 400.0, 5.0, 120.0, "T (кінець часу)"),
                    slider("dt", 0.005, 0.5, 0.005, 0.05, "Δt (крок інтегрування)"),
                    html.Div(className="ctrl", children=[
                        html.Div(className="ctrl-label", children=[
                            html.Span("Чисельний метод"),
                            html.Span(className="ctrl-value", id="method-label")
                        ]),
                        dcc.Dropdown(
                            id="method",
                            options=[
                                {"label": "Runge–Kutta 4 (рекомендовано)", "value": "rk4"},
                                {"label": "Euler (демонстраційно)", "value": "euler"},
                            ],
                            value="rk4", clearable=False, style={"marginTop": 6}
                        )
                    ]),
                ]),
            ])
        ]),
    ]
)

# --------------------------- Callbacks ---------------------------
@app.callback(
    [Output("time-series", "figure"),
     Output("phase-portrait", "figure"),
     Output("equil-info", "children"),
     Output("a-label", "children"), Output("b-label", "children"),
     Output("g-label", "children"), Output("s-label", "children"),
     Output("x0-label", "children"), Output("y0-label", "children"),
     Output("t0-label", "children"), Output("t1-label", "children"),
     Output("dt-label", "children"), Output("method-label", "children")],
    [Input("a", "value"), Input("b", "value"), Input("g", "value"), Input("s", "value"),
     Input("x0", "value"), Input("y0", "value"),
     Input("t0", "value"), Input("t1", "value"), Input("dt", "value"),
     Input("method", "value")]
)
def update(a, b, g, s, x0, y0, t0, t1, dt, method):
    # live labels
    live_labels = [f"{a:.3f}", f"{b:.4f}", f"{g:.2f}", f"{s:.2f}",
                   f"{x0:.0f}", f"{y0:.0f}", f"{t0:.1f}", f"{t1:.1f}",
                   f"{dt:.3f}", ("Runge–Kutta 4" if method=="rk4" else "Euler")]

    params = Params(a=a, b=b, g=g, s=s)
    t0 = float(t0); dt = float(dt); t1 = float(max(t1, t0 + dt))
    t, x, y = simulate(x0=x0, y0=y0, t0=t0, t1=t1, dt=dt, p=params, method=method)

    if len(t) < 3:
        msg = "Чисельне рішення нестійке. Оберіть RK4 та/або зменшіть Δt."
        warn = warning_figure(msg)
        return warn, warn, msg, *live_labels

    # стаціонарна точка та нуль-лінії
    x_star = s / b if b > 0 else None
    y_star = g / a if a > 0 else None

    # часові ряди
    fig_time = base_fig("Динаміка популяцій у часі", "t", "чисельність")
    fig_time.add_trace(go.Scatter(x=t, y=y, name="жертви y(t)", mode="lines", line=dict(width=2)))
    fig_time.add_trace(go.Scatter(x=t, y=x, name="хижаки x(t)", mode="lines", line=dict(width=2)))
    fig_time.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.2,  # нижче осі X
            xanchor="left", x=0
        )
    )
    # фазовий портрет
    fig_phase = base_fig("Фазовий простір (x — хижаки, y — жертви)", "хижаки x", "жертви y")
    fig_phase.add_trace(go.Scatter(x=x, y=y, name="траєкторія", mode="lines", line=dict(width=2)))
    fig_phase.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.2,  # нижче осі X
            xanchor="left", x=0
        )
    )
    xmax = max(np.max(x), x0) * 1.12
    ymax = max(np.max(y), y0) * 1.12
    if a > 0:
        fig_phase.add_trace(go.Scatter(x=[g/a, g/a], y=[0, ymax], mode="lines",
                                       name="dy/dt = 0 (x = g/a)", line=dict(dash="dash")))
    if b > 0:
        fig_phase.add_trace(go.Scatter(x=[0, xmax], y=[s/b, s/b], mode="lines",
                                       name="dx/dt = 0 (y = s/b)", line=dict(dash="dash")))
    if x_star is not None and y_star is not None:
        fig_phase.add_trace(go.Scatter(x=[x_star], y=[y_star], mode="markers",
                                       name="стаціонарна точка (x*, y*)",
                                       marker=dict(size=9, symbol="x")))

    eq_text = (f"Метод: {'Runge–Kutta 4' if method=='rk4' else 'Euler'} · "
               f"x* = s/b = {x_star:.3f}, y* = g/a = {y_star:.3f} · "
               f"Нуль-лінії: x = g/a, y = s/b.")

    return fig_time, fig_phase, eq_text, *live_labels

if __name__ == "__main__":
    app.run(debug=True)
