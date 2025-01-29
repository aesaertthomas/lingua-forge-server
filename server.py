from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from translator import *

app = FastAPI()

class TranslationRequest(BaseModel):
    source_lang: str
    target_lang: str
    src_text: List[str]

def reload_model(isource_lang : str, itarget_lang : str):
    global source_lang
    global target_lang
    global model_path

    #Updating global vars to new langs (First time indeed bit redundant, but not worth fussing out over)
    source_lang = isource_lang
    target_lang = itarget_lang

    tokenizer, model = load_tokenizer_and_model(source_lang=source_lang, target_lang=target_lang)
    model = load_model(model, model_path, source_lang=source_lang, target_lang=target_lang)
    cuda_availability, device = check_gpu_availability_and_move(model=model)

    return tokenizer, model

source_lang = "en" #Set to beginning because not possible to input beginning as a model, so when loading for first time, model gets reloaded, and not get error model not available 👍
target_lang = "fr"
model_path = "model.pth"

tokenizer, model = reload_model(isource_lang=source_lang, itarget_lang=target_lang)


@app.post("/translate/")
async def translate_text(request: TranslationRequest):
    # Here we assume that request contains source_lang, target_lang and src_text
    # Now let's add the input to original dictionary src_text
    global source_lang
    global target_lang
    global model_path
    global tokenizer
    global model


    inputted_source_lang = request.source_lang
    inputted_target_lang = request.target_lang

    if ((inputted_source_lang != source_lang) or (inputted_target_lang != target_lang)):
        tokenizer, model = reload_model(isource_lang=inputted_source_lang, itarget_lang=inputted_target_lang)

    src_text = request.src_text

    output = translate(tokenizer=tokenizer, model=model, src_text=src_text)

    return output
