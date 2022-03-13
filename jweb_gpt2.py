import torch
from transformers import T5Tokenizer, AutoModelForCausalLM

tokenizer = T5Tokenizer.from_pretrained("jweb/japanese-soseki-gpt2-1b")
model = AutoModelForCausalLM.from_pretrained("jweb/japanese-soseki-gpt2-1b")

if torch.cuda.is_available():
    model = model.to("cuda")

text = "夏目漱石は、"
token_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(
        token_ids.to(model.device),
        max_length=128,
        min_length=40,
        do_sample=True,
        repetition_penalty= 1.6,
        early_stopping= True,
        num_beams= 5,
        temperature= 1.0,
        top_k=500,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

output = tokenizer.decode(output_ids.tolist()[0])
print(output)  
# sample output: 夏目漱石は、明治時代を代表する文豪です。夏目漱石の代表作は「吾輩は猫である」や「坊っちゃん」、「草枕」「三四郎」、それに「虞美人草(ぐびじんそう)」などたくさんあります。
