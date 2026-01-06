from apocrypha.EpistolaryAcumen import RecallKnowledge
from hermaeus.HermaMora import HermaeusMora
from seekers.web_pages.test_cleanup import clean_wikipedia_html_file

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
