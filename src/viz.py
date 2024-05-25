from typing import Dict, List
import numpy as np
from plotly.subplots import make_subplots

import plotly.graph_objects as go


def plot_sensor_data(
    data: Dict,
    signals2plot: List[str],
    trial: int,
    title: str = "plot",
) -> None:
    """
    Plot sensor data.

    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Dictionary containing sensor data arrays. Assumes data[sensor][trial] contains a pandas DataFrame
    signals2plot : List[str]
        List of signal names to plot.
    trial : int
        Index of the trial to plot.
    title : str
        Title of the plot.

    Returns
    -------
    go.Figure
        The plotly figure object.

    """
    subplots = make_subplots(
        rows=7,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"{s} sensor" for s in data.keys()],
        vertical_spacing=0.01,
    )
    for r, s in enumerate(data.keys()):
        if data[s][trial] is None:
            continue
        elif data[s][trial].shape[0] == 0:
            continue
        samples = np.arange(0, data[s][trial].shape[0], 1)
        for signal in signals2plot:
            subplots.add_trace(
                go.Scatter(
                    x=samples,
                    y=data[s][trial][signal],
                    mode="lines",
                    name=f"{s} : {signal}",
                ),
                row=r + 1,
                col=1,
            )
    subplots.update_layout(title_text=title, height=1200)
    subplots.show()
    return None
