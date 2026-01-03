from apocrypha.EpistolaryAcumen import RecallKnowledge
from hermaeus.HermaMora import HermaeusMora
from test_format import to_markdown

HermaeusMora = HermaeusMora()
HermaeusMora.create()

while True:
    prompt = input("> ")

    results = RecallKnowledge(prompt)
    context = []
    for item in results:
        context.append(item["content"])
    context_info = "\n".join(context)

    response = HermaeusMora.chat(prompt, context_info)
    print(response)
