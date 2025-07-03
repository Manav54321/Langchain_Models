score = [0.4, 0.3, 0.5, 0.2, 0.1]

score = list(enumerate(score))

document_index, score = sorted(score, key=lambda x: x[1])[-1]

print(f"Document Index: {document_index}, Score: {score}")

