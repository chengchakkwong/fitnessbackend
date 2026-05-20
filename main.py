from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Project CheekyCat AI Backend", version="1.0.0")

# 設定 CORS 安全白名單，允許你的前端 Next.js 來溝通
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 測試階段允許所有來源，之後部署再改成 Vercel 網址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "message": "賴皮貓後端大腦已上線！",
        "character": "CheekyCat"
    }

@app.get("/api/cat-status")
async def get_cat_status():
    # 預留給賴皮貓減肥狀態的趣味 API
    return {
        "cat_name": "CheekyCat",
        "current_weight_kg": 8.5,
        "target_weight_kg": 5.0,
        "mood": "Angry (Because of calorie deficit)",
        "cheat_attempts_today": 3
    }