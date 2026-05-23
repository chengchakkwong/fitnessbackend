import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel
from dotenv import load_dotenv

# 1. 載入 .env 保險箱裡的密鑰
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ 找不到 GEMINI_API_KEY！請檢查 .env 檔案")

# 2. 初始化 2026 最新版 Google GenAI 客戶端
client = genai.Client(api_key=api_key)

app = FastAPI(title="Project CheekyCat AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 定義 Pydantic 規格：強迫 Gemini 必須吐出這種嚴格的格式，不然就打槍
class NutritionalAnalysis(BaseModel):
    dish_name: str
    calories_kcal: int
    protein_g: float
    carbs_g: float
    fat_g: float
    cheeky_cat_comment: str  # 讓 Gemini 用賴皮貓的機車語氣吐槽這道菜

@app.get("/")
async def root():
    return {"status": "healthy", "message": "賴皮貓大腦已就位，VPN 管線暢通！"}

# 4. 核心：接收前端傳來的美食照片並進行 AI 分析
@app.post("/api/analyze-food")
async def analyze_food(file: UploadFile = File(...)):
    try:
        # 讀取前端傳過來的圖片二進位資料
        image_bytes = await file.read()
        
        # 設定給 Gemini 的 AI 限制提示詞（System Instruction）
        prompt = (
            "你是一隻叫 CheekyCat（賴皮貓）的毒舌健身教練貓。請分析這張圖片中的食物。"
            "你必須精準估算它的熱量與三大營養素（蛋白質、碳水、脂肪）。"
            "特別注意：如果這是香港特有的茶餐廳外食（例如菠蘿油、沙嗲牛麵、西多士），"
            "請根據香港當地的道地配方來估算熱量，並在 cheeky_cat_comment 欄位中，"
            "用極度傲嬌、機車、幽默的語氣，吐槽使用者今天是不是又想偷吃美食、阻礙減肥。"
        )

        # 呼叫 Gemini 2.5 Flash 模型（2026年性價比之王，支援結構化輸出）
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=file.content_type),
                prompt
            ],
            # 🚀 這裡就是精髓：強迫 Gemini 必須完全符合我們在上面定義的 Pydantic 格式
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=NutritionalAnalysis,
                temperature=0.7
            ),
        )

        # 因為設定了 response_schema，Gemini 吐出來的會是完美符合格式的 JSON 字串
        # 我們直接將其解析並回傳給前端
        import json
        result_json = json.loads(response.text)
        return result_json

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 分析失敗: {str(e)}")