from abc import abstractmethod

from rich.console import Console
from rich.table import Table


class Report:
    @abstractmethod
    def repr_info(self):
        return {}

    def show(self):
        info = self.repr_info()
        name = info.get("method_name", "N/A")
        attributes = info.get("attributes", {})

        table = Table(title=name, show_header=True, header_style="bold cyan")
        table.add_column("Parameter", style="bold")
        table.add_column("Value")

        for key, value in attributes.items():
            if isinstance(value, list):
                value = ", ".join(map(str, value))
            table.add_row(str(key), str(value))

        console = Console()
        console.print(table)

    def __repr__(self):
        info = self.repr_info()
        name = info.get("method_name", "N/A")
        attributes = info.get("attributes", {})
        lines = [f"[{name}]"]
        for key, value in attributes.items():
            if isinstance(value, list):
                value = ", ".join(map(str, value))
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def _repr_mimebundle_(self, **_):
        return {"text/plain": repr(self)}
