from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse,FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import json
import asyncio
import concurrent.futures

load_dotenv()

from graph import run_graph
from prompts import detect_language

app = FastAPI(title="AI Code Assistant — LangGraph Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


class AnalyzeRequest(BaseModel):
    code: str
    language: str = "auto"


class DetectRequest(BaseModel):
    code: str


@app.get("/")
def root():

    return FileResponse("index.html")


@app.post("/detect-language")
def detect(req: DetectRequest):
    return {"language": detect_language(req.code)}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    if not req.code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty.")

    language = req.language
    if language == "auto":
        language = detect_language(req.code)

    async def stream():
        loop = asyncio.get_event_loop()

       
        result_holder = {}

        def run():
            result = run_graph(req.code, language)
            result_holder["result"] = result

        future = loop.run_in_executor(executor, run)

        # Send a "working" ping while graph runs
        yield f"data: {json.dumps({'type': 'status', 'text': 'Graph running...'})}\n\n"
        await future

        result = result_holder.get("result", {})

        # Stream the agent trace first
        for trace_line in result.get("agent_trace", []):
            yield f"data: {json.dumps({'type': 'trace', 'text': trace_line})}\n\n"
            await asyncio.sleep(0.05)

        # Then stream the final output word by word for a nice effect
        final = result.get("final_output", "No output generated.")
        words = final.split(" ")
        chunk = ""
        for i, word in enumerate(words):
            chunk += word + " "
            if (i + 1) % 8 == 0:
                yield f"data: {json.dumps({'type': 'output', 'text': chunk})}\n\n"
                chunk = ""
                await asyncio.sleep(0.01)
        if chunk:
            yield f"data: {json.dumps({'type': 'output', 'text': chunk})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":


    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)