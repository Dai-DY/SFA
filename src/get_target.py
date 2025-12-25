import spacy
import re

# 加载英语模型
nlp = spacy.load("en_core_web_sm")

def extract_action_objects(instruction):
    """
    提取指令中的源对象、目标对象和动作
    返回字典包含: source_obj, dest_obj, actions
    """
    doc = nlp(instruction)
    
    source_obj = None   # 源对象（被操作的物体）
    dest_obj = None     # 目标对象（目标位置）
    actions = []        # 动作列表
    
    # 收集所有动作动词
    for token in doc:
        if token.pos_ == "VERB":
            actions.append(token.lemma_)
    
    # 分析依存关系找到源对象和目标对象
    for token in doc:
        # 处理拾取类动作
        if token.lemma_ in ['pick', 'take', 'grab', 'lift', 'get', 'move']:
            # 寻找直接宾语（被拾取的物体）
            for child in token.children:
                if child.dep_ == "dobj":  # 直接宾语
                    source_obj = get_full_phrase(child)
        
        # 处理放置类动作  
        elif token.lemma_ in ['insert', 'put', 'place', 'drop', 'move']:
            # 寻找介词短语中的目标对象
            for child in token.children:
                if child.dep_ == "prep" and child.text in ['into', 'in', 'to', 'on', 'at']:
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":  # 介词宾语
                            dest_obj = get_full_phrase(grandchild)
    
    # 如果没有明确区分，尝试从名词短语中推断
    if not source_obj or not dest_obj:
        noun_phrases = []
        for chunk in doc.noun_chunks:
            noun_phrases.append(chunk.text)
        
        # 简单策略：第一个重要名词作为源，第二个作为目标
        important_keywords = ['cup', 'box', 'bottle', 'object', 'item', 'block']
        filtered_phrases = [phrase for phrase in noun_phrases 
                          if any(keyword in phrase.lower() for keyword in important_keywords)]
        
        if not source_obj and len(filtered_phrases) >= 1:
            source_obj = filtered_phrases[0]
        if not dest_obj and len(filtered_phrases) >= 2:
            dest_obj = filtered_phrases[1]
    
    return {
        'source_object': source_obj,
        'destination_object': dest_obj, 
        'actions': actions,
        'raw_instruction': instruction
    }

def get_full_phrase(token):
    """获取包含修饰词的完整短语"""
    phrase_parts = []
    
    # 收集修饰词（按顺序：限定词、形容词等）
    modifiers = []
    for child in token.children:
        if child.dep_ in ['det', 'amod', 'compound', 'nummod']:  # 限定词、形容词、复合词、数词
            modifiers.append((child.i, child.text))  # 保存原始位置以便排序
    
    # 按原文顺序排列修饰词
    modifiers.sort(key=lambda x: x[0])
    phrase_parts.extend([mod[1] for mod in modifiers])
    
    # 添加中心词
    phrase_parts.append(token.text)
    
    return ' '.join(phrase_parts)

# 单独测试你的例子
print("\n" + "="*50)
result = extract_action_objects("Please pick up the green cup and insert it into the pink cup")
print(f"源对象: {result['source_object']}")
print(f"目标对象: {result['destination_object']}")
print(f"动作: {result['actions']}")