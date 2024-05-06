import json

with open('/node_data/hyun/data/task/files/all/allinst_val_text_mix_new_laughtype_norev_r_revise2.json', 'r') as f:
    data = json.load(f)
    
newlst = []
    
for d in data:
    # inst, inp = d['conversations'][0]['value'].split('The video clip is')
    # inp = 'The video clip is' + inp
    
    all_ = d['conversations'][0]['value']
    if 'The video clip is' in all_:
        # inst, inp = all_.split('The video clip is')
        # inp = 'The video clip is' + inp
        idx = all_.rfind('The video clip is')
        inst = all_[:idx]
        
        inp = all_[idx:]
        if 'task' not in d:
            d['task'] = 'detect' if 'detect' in all_ else 'classify' if 'classification' in all_ else 'reason' if 'reasoning' in all_ else 'none'
        
    elif 'Input:' in all_:
        all_ = all_.rstrip('\n')
        all_ = all_.replace('Output:', '')
        idx = all_.rfind('Input:')
        # inst, inp = all_.split('Input:')
        inst = all_[:idx]
        inp = all_[idx:]
        task = all_.split('Task:')[0]#.split('Output:')[0]
        # print(task)
        d['task'] = task
        
    else:
        # print(all_)
        all_ = all_.rstrip('\n')
        inst = all_.replace('Output:', '')
        inp = ''
        task = all_.lower().split('task:')[0]
        d['task'] = task
    
    if d['task']=='detect':
        task_type = 0
    
    elif d['task']=='classify':
        task_type = 1
    
    elif d['task']=='reason':
        task_type = 2
        
    else:
        # print('hh', d['task'], all_)
        print(f"^^{d['task']}^^{all_}")
        task_type = 3
        
    output = d['conversations'][1]['value']
    
    newlst.append({
        'instruction': inst,
        'input': inp,
        'output': output,
        'task_type': task_type
    })
    
with open('/node_data/mok/module/LoRAMoE/data/laugh_instruction/val.json', 'w') as f:
    json.dump(newlst, f, indent=2)