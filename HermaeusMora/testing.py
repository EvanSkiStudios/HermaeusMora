from Hermaeus.HermaMora import HermaeusMora

HermaeusMora = HermaeusMora()
HermaeusMora.create()

while True:
    prompt = input("> ")
    response = HermaeusMora.chat(prompt)
    print(response)
