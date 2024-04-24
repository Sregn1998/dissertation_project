import uvicorn
from fastapi import FastAPI
from routers import route
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.include_router(route.router, prefix = "/prediction")
app.mount("/static", StaticFiles(directory="./neural_network/main/images"), name="static")
if __name__ == '__main__':
    uvicorn.run("main:app", host = '0.0.0.0', port = 8000, reload = True, workers = 3)
