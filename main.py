from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from translator import *

app = FastAPI()

class TranslationRequest(BaseModel):
    source_lang: str
    target_lang: str
    src_text: List[str]

print("You are translating from ENGLISH to FRENCH")
source_lang = "en"
target_lang = "fr"
model_path = "model.pth"  # Path to save the model
tokenizer, model = load_tokenizer_and_model(source_lang=source_lang, target_lang=target_lang)
model = load_model(model, model_path, source_lang=source_lang, target_lang=target_lang)
cuda_availability, device = check_gpu_availability_and_move(model=model)

@app.post("/translate/")
async def translate_text(request: TranslationRequest):
    # Here we assume that request contains source_lang, target_lang and src_text
    # Now let's add the input to original dictionary src_text

    source_lang = request.source_lang
    target_lang = request.target_lang
    src_text = request.src_text

    output = translate(tokenizer=tokenizer, model=model, src_text=src_text)

    return output
