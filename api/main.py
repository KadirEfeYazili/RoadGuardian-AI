"""
RoadGuardian-CV - FastAPI Uygulamasi

Trafik ve surucu modullerinin urettigi verileri dis dunyaya (UI dashboard,
diger servisler) sunan REST API'nin temel iskeleti.

Calistirmak icin (proje kok dizininden):
    uvicorn api.main:app --reload
"""

import sys
from pathlib import Path

from fastapi import FastAPI

# Proje kokunu sys.path'e ekle ki "core" paketi import edilebilsin.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.config import settings  # noqa: E402

app = FastAPI(
    title="RoadGuardian-CV API",
    description="Yol guvenligi ve yapay zeka sistemi icin REST API.",
    version="0.1.0",
)


@app.get("/")
def root():
    """Kok endpoint - kisa karsilama mesaji."""
    return {"message": "RoadGuardian-CV API'ye hos geldiniz.", "docs": "/docs"}


@app.get("/status")
def status():
    """Sistemin calistigini dogrulayan basit saglik kontrolu endpoint'i."""
    return {
        "status": "ok",
        "service": "RoadGuardian-CV",
        "version": app.version,
        "modules": {
            "traffic_module": "ready",
            "driver_module": "ready",
        },
        "debug": settings.DEBUG,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
