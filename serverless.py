import runpod
from transformers import MBartForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import AlbertTokenizer, AutoTokenizer
# Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("text2text-generation", model="ai4bharat/IndicBARTSS")

def handler_old(job):
    job_input=job["input"]
    return f"Hello {job_input['name']}!"



def handler(job):
    input_text=job["input"]["text"]
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBARTSS", do_lower_case=False, use_fast=False, keep_accents=True)

    # Or use tokenizer = AlbertTokenizer.from_pretrained("ai4bharat/IndicBARTSS", do_lower_case=False, use_fast=False, keep_accents=True)

    model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/IndicBARTSS")

    # Or use model = MBartForConditionalGeneration.from_pretrained("ai4bharat/IndicBARTSS")

    # Some initial mapping
    bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
    eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
    pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")
    # To get lang_id use any of ['<2as>', '<2bn>', '<2en>', '<2gu>', '<2hi>', '<2kn>', '<2ml>', '<2mr>', '<2or>', '<2pa>', '<2ta>', '<2te>']

    # First tokenize the input and outputs. The format below is how IndicBARTSS was trained so the input should be "Sentence </s> <2xx>" where xx is the language code. Similarly, the output should be "<2yy> Sentence </s>". 
    inp = tokenizer("I am a boy </s> <2en>", add_special_tokens=False, return_tensors="pt", padding=True).input_ids # tensor([[  466,  1981,    80, 25573, 64001, 64004]])

    # out = tokenizer("<2hi> मैं  एक लड़का हूँ </s>", add_special_tokens=False, return_tensors="pt", padding=True).input_ids # tensor([[64006,   942,    43, 32720,  8384, 64001]])

    # model_outputs=model(input_ids=inp, decoder_input_ids=out[:,0:-1], labels=out[:,1:])

    # # For loss
    # model_outputs.loss ## This is not label smoothed.

    # # For logits
    # model_outputs.logits

    # # For generation. Pardon the messiness. Note the decoder_start_token_id.

    model.eval() # Set dropouts to zero

    model_output=model.generate(inp, use_cache=True, num_beams=4, max_length=20, min_length=1, early_stopping=True, pad_token_id=pad_id, bos_token_id=bos_id, eos_token_id=eos_id, decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc("<2hi>"))


    # Decode to get output strings

    decoded_output=tokenizer.decode(model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return decoded_output # I am a boy

runpod.serverless.start({"handler": handler})