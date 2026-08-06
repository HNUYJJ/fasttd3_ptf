from .manifest import SourceManifest

__all__ = ["SourceManifest", "build_source_bank", "export_source_manifest"]


def __getattr__(name: str):
    if name == "build_source_bank":
        from .builder import build_source_bank

        return build_source_bank
    if name == "export_source_manifest":
        from .exporter import export_source_manifest

        return export_source_manifest
    raise AttributeError(name)
