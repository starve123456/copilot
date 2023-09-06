import json


def add_lang_by_task(target_str, task, sub_task):
    if task == 'summarize':
        target_str = '<en> ' + target_str
    elif task == 'refine':
        target_str = '<java> ' + target_str
    elif task == 'translate':
        if sub_task == 'java-cs':
            target_str = '<c_sharp> ' + target_str
        else:
            target_str = '<java> ' + target_str
    elif task == 'concode':
        target_str = '<java> ' + target_str
    elif task == 'defect':
        target_str = target_str
    return target_str

def convert_defect_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source
    code = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    return DefectInputFeatures(example_index, code,example.target, example.domain_label)


class DefectInputFeatures(object):

    def __init__(self,
                 example_id,
                 source_ids,
                 label,
                 domain_label
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label
        self.domain_label = domain_label

class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url


class Example(object):
    """A single training/test example."""
    def __init__(self,
                 domain,
                 domain_label,
                 idx,
                 source,
                 target,
                 url=None,
                 task='',
                 sub_task=''
                 ):
        self.domain = domain
        self.domain_label = domain_label
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task

def read_test2_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['function'].split())
            examples.append(
                Example(
                    domain=js['domain'],
                    domain_label=js['domain_label'],
                    idx=js['index'],
                    source=code,
                    target=js['target'],
                )
            )
            if idx + 1 == data_num:
                break
    return examples

def split_by_query(examples):
    result_dict = dict()
    doma = list()
    for cur_tw in examples:
        if cur_tw.domain not in result_dict:
            result_dict[cur_tw.domain] = list()
        result_dict[cur_tw.domain].append(cur_tw)
    dict_items = list(result_dict.items())
    dict_items.sort(key=lambda entry: entry[0])
    result_list = [entry[1] for entry in dict_items]
    doma = [i[0].domain for i in result_list]
    return result_list,doma