with open('trabalho.tex', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('\u2212', '-')

with open('trabalho.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print("Unicode characters replaced successfully!")
