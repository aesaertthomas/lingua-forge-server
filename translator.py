import torch
import random

from transformers import MarianMTModel, MarianTokenizer


    # Specify the model name


def load_tokenizer_and_model(source_lang : str, target_lang : str):
    # Load the tokenizer and model
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"

    tokenizer = MarianTokenizer.from_pretrained(model_name, use_fast=True)
    model = MarianMTModel.from_pretrained(model_name)

    return tokenizer, model

def check_gpu_availability_and_move(model):
    # Check if GPU is available
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    #logging
    if cuda_available:
        print("CUDA is available:", torch.cuda.is_available())
        print("Device being used:", next(model.parameters()).device)

        #Moving model
        print(f"Moving model to GPU: {device}")
        try:
            model = model.to(device)
            print(f"Model moved to GPU {device}")
        except:
            print("Something went wrong moving the model, not using CUDA device")

    return cuda_available, device

def tokenize_inputs(src_text: list, tokenizer):
    inputs = tokenizer(src_text, return_tensors="pt", padding=True, truncation=True)
    print() #Whitespace
    for key, value in inputs.items():
        inputs[key] = value.to("cuda")

    return inputs

def translate_tokens(tokenized_inputs, tokenizer, model):
    with torch.no_grad():
        translated = model.generate(**tokenized_inputs)
    outputs = []
    print() #Whitespace
    for t in translated:
        decoded = tokenizer.decode(t, skip_special_tokens=True)
        outputs.append(decoded)

    return outputs

def generate_varied_benchmark_entries():
    subjects = ["A cat", "A dog", "The airplane", "The scientist", "An artist", "The engineer",
                "A programmer", "The farmer", "A teacher", "The student", "An astronaut", "A bird"]
    actions = ["is exploring", "is building", "is drawing", "is coding", "is observing",
               "is flying", "is learning about", "is teaching", "is testing", "is preparing",
               "is imagining", "is repairing"]
    objects = ["a new planet", "an innovative robot", "a mathematical formula", "a high-tech spacecraft",
               "a complex puzzle", "an artistic masterpiece", "a language model", "a weather prediction",
               "a breakthrough in science", "an artificial intelligence", "a sustainable farm",
               "a challenging game"]
    extensions = ["to improve humanity.", "for future generations.", "with impressive precision.",
                  "to solve global problems.", "to inspire others.", "in a groundbreaking way.",
                  "with remarkable creativity.", "for scientific exploration.", "for space missions.",
                  "with unparalleled dedication.", "to explore new ideas.", "with cutting-edge technology."]

    entries = []
    for _ in range(150):
        subject = random.choice(subjects)
        action = random.choice(actions)
        obj = random.choice(objects)
        extension = random.choice(extensions)
        sentence = f"{subject} {action} {obj} {extension}"
        entries.append(sentence)

    return entries

def save_model(model, model_path="model.pth"):
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def load_model(model, model_path="model.pth", source_lang="en", target_lang="fr"):

    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    model = MarianMTModel.from_pretrained(model_name)
    save_model(model, model_path)  # Save the model after loading it from Hugging Face
    return model

def translate(tokenizer, model, src_text : list):
    tokenized_inputs = tokenize_inputs(src_text=src_text, tokenizer=tokenizer)

    translated_tokens = translate_tokens(tokenizer=tokenizer, model=model, tokenized_inputs=tokenized_inputs)

    return translated_tokens

# print("You are translating from ENGLISH to FRENCH")
# source_lang = "en"
# target_lang = "fr"
# model_path = "model.pth"  # Path to save the model
# tokenizer, model = load_tokenizer_and_model(source_lang=source_lang, target_lang=target_lang)
# model = load_model(model, model_path, source_lang=source_lang, target_lang=target_lang)
# cuda_availability, device = check_gpu_availability_and_move(model=model)

# to_continue = True
# while to_continue:


#     print(benchmark_text[i])

#     outputs = main(source_lang="en", target_lang="fr", src_text=benchmark_text[i])

#     print()
#     for output in outputs:
#         print(output)
