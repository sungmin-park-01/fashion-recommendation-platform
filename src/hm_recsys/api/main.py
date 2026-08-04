from fastapi import FastAPI

from hm_recsys import __version__

app = FastAPI(title="H&M Recommendation API", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
