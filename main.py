from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from multi import process_question
import os

app = FastAPI()


class QuestionRequest(BaseModel):
    question: str
    account_id: str


@app.post("/process_question")
async def api_process_question(request: QuestionRequest):
    try:
        query, answer = process_question(request.question, request.account_id)
        return {"query": query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
