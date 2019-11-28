tokenize_texts = [['은행', '성', '산', '팀장', '입니다', '행복', '한', '주', '말', '되', '세요']]

for i in tokenize_texts:
    print(type(i))
    print(i)

    tokenize_texts = ['[CLS]'] + i + ['[SEP]']

    

print(tokenize_texts)