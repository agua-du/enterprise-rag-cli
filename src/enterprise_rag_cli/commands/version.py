from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version

from enterprise_rag_cli.presentation import console


def show_version() -> None:
    """显示当前安装的 Enterprise RAG CLI 版本。"""

    try:
        value = distribution_version("enterprise-rag-cli")
    except PackageNotFoundError:
        value = "unknown"

    console.print(f"Enterprise RAG CLI {value}")