import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Annotated, Any, List
from uuid import UUID

import jwt
from jwt import PyJWKClient
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from supabase import Client, create_client
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
import google.generativeai as genai

logger = logging.getLogger(__name__)

# =====================================================================
# 1. 初始化與環境變數載入
# =====================================================================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("❌ 錯誤：找不到 GEMINI_API_KEY，請檢查環境變數！")

genai.configure(api_key=API_KEY)

# CORS 網域防禦守門員配置
DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000,"
    "http://127.0.0.1:3000,"
    "https://boompala.vercel.app"
)
_raw_origins = os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
CORS_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# 🚀 降落回最穩定、絕對不會報 404 找不到的真實模型版本
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
# 🚀 預設開啟 DEBUG，讓前端直接能看到報錯真兇
DEBUG = os.getenv("DEBUG", "true").lower() in ("1", "true", "yes")

SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").rstrip("/")
SUPABASE_JWT_ISSUER = f"{SUPABASE_URL}/auth/v1" if SUPABASE_URL else ""
_jwks_client: PyJWKClient | None = (
    PyJWKClient(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json")
    if SUPABASE_URL
    else None
)

SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
_supabase_admin: Client | None = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    _supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

bearer_scheme = HTTPBearer(auto_error=False)

MEALS_DEFAULT_LIMIT = 20
MEALS_MAX_LIMIT = 100

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
EXTENSION_TO_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".heic": "image/heic",
    ".heif": "image/heif",
}

app = FastAPI(title="Project CheekyCat - AI Fitness Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# 🛠️ 輔助工具函數
# =====================================================================
def resolve_image_mime_type(content_type: str | None, filename: str | None) -> str | None:
    if content_type and content_type.startswith("image/"):
        return content_type
    if filename:
        ext = Path(filename).suffix.lower()
        if ext in EXTENSION_TO_MIME:
            return EXTENSION_TO_MIME[ext]
    return None


async def read_upload_with_limit(upload: UploadFile, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail="圖片太大喵！請上傳較小的檔案。")
        chunks.append(chunk)
    return b"".join(chunks)


def parse_gemini_json_response(response) -> str:
    if not response.candidates:
        raise HTTPException(
            status_code=422,
            detail="圖片無法分析（模型未回傳有效內容）",
        )
    candidate = response.candidates[0]
    finish_reason = getattr(candidate, "finish_reason", None)
    if finish_reason and str(finish_reason).upper() in ("SAFETY", "RECITATION", "BLOCKED"):
        raise HTTPException(
            status_code=422,
            detail="圖片無法分析（內容被過濾或無法處理）",
        )
    try:
        text = response.text
    except (ValueError, AttributeError) as exc:
        logger.warning("Gemini response has no text: %s", exc)
        raise HTTPException(
            status_code=422,
            detail="圖片無法分析（無有效文字回應）",
        ) from exc
    if not text or not text.strip():
        raise HTTPException(
            status_code=422,
            detail="圖片無法分析（回應為空）",
        )
    
    # 🚀 自動防禦清理：剔除 ```json 標籤
    clean_json = text.strip()
    if clean_json.startswith("```json"):
        clean_json = clean_json.split("```json")[1].split("```")[0].strip()
    elif clean_json.startswith("```"):
        clean_json = clean_json.split("```")[1].split("```")[0].strip()
        
    return clean_json


def get_current_user_id(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ],
) -> str:
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="未提供登入憑證")
    if _jwks_client is None:
        logger.error("SUPABASE_URL is not configured")
        raise HTTPException(status_code=503, detail="伺服器尚未設定 JWT 驗證")

    token = credentials.credentials
    try:
        signing_key = _jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256"],
            audience="authenticated",
            issuer=SUPABASE_JWT_ISSUER,
        )
    except jwt.PyJWTError as exc:
        logger.warning("JWT validation failed: %s", exc)
        raise HTTPException(status_code=401, detail="登入憑證無效或已過期") from exc

    user_id = payload.get("sub")
    if not user_id or not isinstance(user_id, str):
        raise HTTPException(status_code=401, detail="登入憑證缺少使用者識別")

    return user_id


def get_supabase_admin() -> Client:
    if _supabase_admin is None:
        logger.error("Supabase service role is not configured")
        raise HTTPException(
            status_code=503,
            detail="伺服器尚未設定資料庫連線（SUPABASE_SERVICE_ROLE_KEY）",
        )
    return _supabase_admin


def _meal_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "user_id": str(row["user_id"]),
        "created_at": row["created_at"],
        "dish_name": row["dish_name"],
        "calories_kcal": row["calories_kcal"],
        "protein_g": row["protein_g"],
        "carbs_g": row["carbs_g"],
        "fat_g": row["fat_g"],
        "visual_clues": row.get("visual_clues") or [],
        "assumption_and_blindspots": row["assumption_and_blindspots"],
        "confidence_score": row["confidence_score"],
        "cheeky_cat_comment": row["cheeky_cat_comment"],
        "user_correction_note": row.get("user_correction_note"),
        "analysis_source": row["analysis_source"],
        "reused_from_meal_id": (
            str(row["reused_from_meal_id"]) if row.get("reused_from_meal_id") else None
        ),
        "image_path": row.get("image_path"),
    }


def _fetch_meal_for_user(
    supabase: Client, meal_id: UUID, user_id: str
) -> dict[str, Any] | None:
    response = (
        supabase.table("meals")
        .select("*")
        .eq("id", str(meal_id))
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    rows = response.data or []
    return rows[0] if rows else None


# =====================================================================
# 2. 鋼鐵 Pydantic 數據模型
# =====================================================================
class NutritionalAnalysis(BaseModel):
    dish_name: str = Field(description="食物的英文與繁體中文名稱")
    calories_kcal: int = Field(description="整盤食物的預估總熱量卡路里")
    protein_g: float = Field(description="總蛋白質克數")
    carbs_g: float = Field(description="總碳水化合物克數")
    fat_g: float = Field(description="總脂肪克數")
    visual_clues: List[str] = Field(description="你在圖片中看到了哪些關鍵食材與視覺線索")
    assumption_and_blindspots: str = Field(description="你在估算時做了什麼份量假設？有哪些物理盲點")
    confidence_score: float = Field(ge=0.0, le=1.0, description="你對這次辨識結果的信心指數（0.0 到 1.0）")
    cheeky_cat_comment: str = Field(description="毒舌健身教練貓 CheekyCat 的繁體中文機車吐槽")


class MealCreate(NutritionalAnalysis):
    user_correction_note: str | None = Field(
        default=None,
        description="使用者修正提示詞（可選）",
    )


class MealOut(BaseModel):
    id: str
    user_id: str
    created_at: datetime
    dish_name: str
    calories_kcal: int
    protein_g: float
    carbs_g: float
    fat_g: float
    visual_clues: List[str]
    assumption_and_blindspots: str
    confidence_score: float
    cheeky_cat_comment: str
    user_correction_note: str | None = None
    analysis_source: str
    reused_from_meal_id: str | None = None
    image_path: str | None = None


# =====================================================================
# 3. AI 拍照辨識核心路由
# =====================================================================
@app.post("/api/analyze-food", response_model=NutritionalAnalysis)
async def analyze_food(file: UploadFile = File(...)):
    mime_type = resolve_image_mime_type(file.content_type, file.filename)
    if not mime_type:
        raise HTTPException(status_code=400, detail="這不是照片喵！請上傳正確的圖片格式。")

    try:
        # 強制重置檔案流指標
        await file.seek(0)
        image_bytes = await read_upload_with_limit(file, MAX_UPLOAD_BYTES)
        if not image_bytes:
            raise HTTPException(status_code=400, detail="上傳的檔案是空的喵！")

        image_part = {"mime_type": mime_type, "data": image_bytes}
        model = genai.GenerativeModel(GEMINI_MODEL)

        prompt = (
            "你是一隻叫 CheekyCat 的毒舌健身教練貓，同時也是一位極其嚴格的臨床營養學專家。\n"
            "請仔細審查這張圖片中的食物，並根據以下 JSON 結構回傳報告，不要有任何多餘的文字。\n\n"
            "【必須包含的 JSON 鍵值】：\n"
            '{"dish_name": "字串", "calories_kcal": 整數, "protein_g": 浮點數, "carbs_g": 浮點數, "fat_g": 浮點數, "visual_clues": ["字串列表"], "assumption_and_blindspots": "字串", "confidence_score": 浮點數, "cheeky_cat_comment": "字串"}\n\n'
            "【三大鋼鐵審計指令】：\n"
            "1. 科學基準定錨：遵循每 100g 標準成分進行還原估算。\n"
            "2. 誠實揭露盲點：老實交代 2D 俯拍照片帶來的物理限制，不准隱瞞誤差！\n"
            "3. 注入機車貓魂：`cheeky_cat_comment` 必須極度毒舌，狠狠吐槽使用者的罪惡熱量，並用『喵～』作為傲嬌語氣的靈魂結尾。"
        )

        response = model.generate_content(
            [prompt, image_part],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
            ),
        )

        return NutritionalAnalysis.model_validate_json(parse_gemini_json_response(response))

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("analyze_food failed")
        detail = "後端大腦抽筋了，請稍後再試。"
        # 🚀 確保 DEBUG 模式下一定會印出真正的錯誤原因！
        if DEBUG:
            detail = f"{detail} 🚨 真兇: {type(e).__name__} -> {str(e)}"
        raise HTTPException(status_code=500, detail=detail) from e


# =====================================================================
# 4. 日記 API（P1：Supabase meals + service role）
# =====================================================================
@app.post("/api/meals", response_model=MealOut, status_code=201)
async def create_meal(
    body: MealCreate,
    user_id: Annotated[str, Depends(get_current_user_id)],
):
    supabase = get_supabase_admin()
    row = {
        "user_id": user_id,
        "dish_name": body.dish_name,
        "calories_kcal": body.calories_kcal,
        "protein_g": body.protein_g,
        "carbs_g": body.carbs_g,
        "fat_g": body.fat_g,
        "visual_clues": body.visual_clues,
        "assumption_and_blindspots": body.assumption_and_blindspots,
        "confidence_score": body.confidence_score,
        "cheeky_cat_comment": body.cheeky_cat_comment,
        "user_correction_note": body.user_correction_note,
        "analysis_source": "gemini_fresh",
    }
    try:
        response = supabase.table("meals").insert(row).execute()
    except Exception as exc:
        logger.exception("create_meal failed")
        detail = "無法儲存到日記，請稍後再試。"
        if DEBUG:
            detail = f"{detail} ({type(exc).__name__}: {exc})"
        raise HTTPException(status_code=500, detail=detail) from exc

    if not response.data:
        raise HTTPException(status_code=500, detail="儲存失敗（無回傳資料）")
    return MealOut.model_validate(_meal_row_to_dict(response.data[0]))


@app.get("/api/meals", response_model=list[MealOut])
async def list_meals(
    user_id: Annotated[str, Depends(get_current_user_id)],
    limit: int = Query(default=MEALS_DEFAULT_LIMIT, ge=1, le=MEALS_MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
):
    supabase = get_supabase_admin()
    try:
        response = (
            supabase.table("meals")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
    except Exception as exc:
        logger.exception("list_meals failed")
        detail = "無法讀取日記列表。"
        if DEBUG:
            detail = f"{detail} ({type(exc).__name__}: {exc})"
        raise HTTPException(status_code=500, detail=detail) from exc

    rows = response.data or []
    return [MealOut.model_validate(_meal_row_to_dict(r)) for r in rows]


@app.get("/api/meals/{meal_id}", response_model=MealOut)
async def get_meal(
    meal_id: UUID,
    user_id: Annotated[str, Depends(get_current_user_id)],
):
    supabase = get_supabase_admin()
    try:
        row = _fetch_meal_for_user(supabase, meal_id, user_id)
    except Exception as exc:
        logger.exception("get_meal failed")
        detail = "無法讀取餐點詳情。"
        if DEBUG:
            detail = f"{detail} ({type(exc).__name__}: {exc})"
        raise HTTPException(status_code=500, detail=detail) from exc

    if row is None:
        raise HTTPException(status_code=404, detail="找不到此餐點")
    return MealOut.model_validate(_meal_row_to_dict(row))


@app.delete("/api/meals/{meal_id}", status_code=204)
async def delete_meal(
    meal_id: UUID,
    user_id: Annotated[str, Depends(get_current_user_id)],
):
    supabase = get_supabase_admin()
    row = _fetch_meal_for_user(supabase, meal_id, user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="找不到此餐點")

    try:
        supabase.table("meals").delete().eq("id", str(meal_id)).eq(
            "user_id", user_id
        ).execute()
    except Exception as exc:
        logger.exception("delete_meal failed")
        detail = "無法刪除此餐點。"
        if DEBUG:
            detail = f"{detail} ({type(exc).__name__}: {exc})"
        raise HTTPException(status_code=500, detail=detail) from exc

    return None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)