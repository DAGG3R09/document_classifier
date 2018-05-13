
with open("../training_data/englishText_0_10000", 'r') as f:

    doc = f.read().lower()
    f.close()

    doc = doc.split("endofarticle.")

    with open("../training_data/data", 'w') as f:
        for d in doc[:1000]:
            f.write(d)
        f.close()