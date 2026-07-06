from fastapi import FastAPI
from app.routes import router

# 1️⃣ Create FastAPI application
app = FastAPI(
    title="Spotify Recommender System",
    description="A ML + SWE hybrid system using embeddings for music recommendation",
    version="1.0.0"
)

# 2️⃣ 注册路由（所有 API 都在 routes.py 里）
app.include_router(router)


# 3️⃣ 根路径（用来测试服务是否启动成功）
@app.get("/")
def home():
    return {
        "message": "Spotify Recommender API is running 🎧",
        "status": "OK"
    }
