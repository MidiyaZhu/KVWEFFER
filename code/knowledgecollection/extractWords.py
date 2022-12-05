from requests_html import HTMLSession
import ast
import json

def collect_knowledge(query):
    ans_dict = {}
    session = HTMLSession()
    url = 'https://relatedwords.org/relatedto/'
    url = url + query
    r = session.get(url)
    sel = '#preloadedDataEl'
    results = r.html.find(sel)
    ans = results[0].text
    ans = ast.literal_eval(ans)
    ans_dict[ans["query"]] = ans["terms"]
    return ans_dict

output_file = 'entity_knowledge.json'
count = 0
# read process_label dict
label_dict = 'entity.csv'
with open(label_dict) as f:
    with open(output_file, "w") as fout:
        for line in f.readlines():
            line = line.rstrip().split('\t')
            label_name = line[0]
            ans = collect_knowledge(label_name)
            json_str = json.dumps(ans)
            fout.write(json_str + '\n')
            count += 1
            print('count:',count)
# print('all collected label:',count)



