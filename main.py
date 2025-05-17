from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends#, UploadFile, File
# from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from db import get_db
from models import UserChatLog, FAQQuestion, UserProfilePreference
from schema import ChatRequest, ChatResponse, Message
from prompt import build_prompt
from qwen import call_qwen
from vectorstore import get_top_faqs, init_faq_index_from_db
from typing import AsyncGenerator
from vectorstore import faq_collection

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    # Only initialize FAQ index if it's empty
    if len(faq_collection.get()['ids']) == 0:
        with next(get_db()) as db:
            faqs = db.query(FAQQuestion).all()
            init_faq_index_from_db(faqs)
    else:
        print("[Startup] FAQ embeddings already exist in ChromaDB.")
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    user_id = request.user_id
    user_input = request.message

    # Load user chat history from DB
    db_history = db.query(UserChatLog).filter_by(user_id=user_id).order_by(UserChatLog.created_at.desc()).limit(10).all()
    history = []
    for log in reversed(db_history):
        history.append(Message(role="user", text=log.question))
        history.append(Message(role="assistant", text=log.answer))
    print(history)
    # Retrieve top 3 matching FAQs with fallback
    try:
        faqs = get_top_faqs(user_input, top_k=3)
    except Exception as e:
        faqs = []
        print(f"[Error] get_top_faqs failed: {e}")
    print(faqs)
    # Fetch profession from user_profile_preferences
    profession_pref = (
        db.query(UserProfilePreference)
        .filter_by(user_id=user_id, question="profession")
        .first()
    )
    profession = profession_pref.answer if profession_pref else ""
    if profession:
        profession = f"Freelance {profession}"

    # Build context-aware prompt with FAQs
    prompt = build_prompt(history, user_input, faqs, profession)

    # Get response from Qwen with fallback
    try:
        answer = call_qwen(prompt)
    except Exception as e:
        print(f"[Error] call_qwen failed: {e}")
        answer = "Sorry, something went wrong while generating a response."

    # Save to DB (intent placeholder left for future use)
    log = UserChatLog(user_id=user_id, question=user_input, answer=answer, intent="")
    db.add(log)
    db.commit()

    return ChatResponse(answer=answer)



# @app.post("/process-file")
# async def process_file(file: UploadFile = File(...)):
#     content = await file.read()
#
#     # Pass content to your model pipeline
#     result = your_model_ocr_and_classify(content, file.filename)
#
#     return JSONResponse(content=result)