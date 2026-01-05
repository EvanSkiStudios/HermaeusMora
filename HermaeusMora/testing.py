from apocrypha.EpistolaryAcumen import RecallKnowledge
from hermaeus.HermaMora import HermaeusMora

HermaeusMora = HermaeusMora()
HermaeusMora.create()

while True:
    prompt = input("> ")

    results = RecallKnowledge(prompt)
    context = []
    for item in results:
        print("=" * 10)
        print(item["distance"])
        print(item["content"])
        print("=" * 10)
        context.append(item["content"])
    context_info = "\n".join(context)

    response = HermaeusMora.chat(prompt, context_info)
    print(response)
