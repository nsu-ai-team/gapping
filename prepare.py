from codecs import open
from csv import DictReader, DictWriter, field_size_limit
from sys import maxsize
import tokenization
from collections import OrderedDict



field_size_limit(maxsize)

def prep_data(data):
    with open(data, encoding="utf8") as data:
        reader = DictReader(data, delimiter='\t')
        out = []
        tokenizer = tokenization.FullTokenizer(vocab_file='data/vocab.txt', do_lower_case=True)
        for row in reader:
            line = {"cV":"", "cR1":"", "cR2":"", "V":"", "R1":"", "R2":""}
            if row["class"] == "1":
                indexes = {}
                for key in line.keys():
                    if key.startswith("c"):
                        if row[key]: indexes[key + "_start"], indexes[key + "_end"] = row[key].split(":")
                    else:
                        tmp = row[key].split(" ")
                        if row[key]:
                            for i in range(len(tmp)): indexes[key + "_start" + str(i)], indexes[key + "_end" + str(i)] = tmp[i].split(":")
                for key in indexes: indexes[key] = int(indexes[key])
                indexes = OrderedDict(sorted(indexes.items(), key=lambda t: -t[1]))
                text = list(row["text"])
                for i in indexes: text.insert(indexes[i], i)
                tokens = []
                s = ""
                for word in text:
                    if len(word)>1:
                        if s:
                            tokens.extend(tokenizer.tokenize(s))
                            s = ""
                        tokens.append(word)
                    else: s += word
                final = []
                c = -1
                for i, token in enumerate(tokens):
                    c += 1
                    if "_start" in token or "_end" in token:
                        if token[-1].isdigit(): token = token[:-1]
                        line[token[:token.index("_")]] += str(i-c) + " "
                    else:                    
                        final.append(token)
                        c -= 1
                for key in line: line[key] = line[key][:-1].replace(" ", ":")
                line["class"] = "1"
            else:
                final = tokenizer.tokenize(row["text"])
                line["class"] = "0"
            if len(final) > 1000:
                final = tokenizer.tokenize(row["text"])[:1000]
                line["class"] = "0"
            line["text"] = tokenizer.convert_tokens_to_ids(["[CLS]"] + final + ["[SEP]"])
            line["text"] += [0] * (1000 - len(line["text"])) + [len(line["text"])]
            out.append(line)
        return out


def save(out, data):                
    with open(data, "w", encoding="utf8") as new:            
        writer = DictWriter(new, delimiter="\t", fieldnames=["text", "class", "cV", "cR1", "cR2", "V", "R1", "R2"])
        writer.writeheader()
        for i in out:
            writer.writerow(i)
            
            
            
if __name__ == '__main__':
    save(prep_data("data/train.csv") + prep_data("add/train.csv"), "data/train_prepared.csv")
    save(prep_data("data/dev.csv"), "data/dev_prepared.csv")
    save(prep_data("data/test.csv"), "data/test_prepared.csv")
