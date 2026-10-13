from __future__ import annotations

import polars as pl
from rich.console import Console
from rich.table import Table

console = Console()

def render_frame(frame:pl.DataFrame,title:str)->None:
    table = Table(title=title, show_lines=False)

    for column in frame.columns:
        table.add_column(column)

    for row in frame.iter_rows():
        table.add_row(*(str(value) for value in row))

    console.print(table)