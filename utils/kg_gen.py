from typing import Union, List
import ast
import re
from utils.gpt import gpt_chat
from KnowledgeGraph import KGRelation, KGEntity


def split_text(input: str, window_size: int=6000, overlap: Union[int, None]=1500, delimiter: str='\n') -> List[str]:
    """
    Split the text based on number of characters. It would only breaks a document if there is a line
    break, if not, it would go back until hits a line break. You can also set the overlap by the "stride"
    argument.

    Parameters:
    input (str): the input document to be split
    window_size (int): the length of each chunk of split in number of characters
    overlap (int or None): e.g. if overlap=3000, the distance between the starts of two chunks are 3000 characters.

    Returns:
    list[str]: the split
    """
    stride = window_size - overlap
    start = 0
    end = 0
    result = list()
    while start + window_size <= len(input):
        end = start + window_size
        while input[end - 1] != delimiter and end > start:
            end -= 1
        if end <= start:
            end = start + window_size
        result.append(input[start:end])
        if stride == 0 or stride is None:
            start = end
        else:
            start_tmp = start
            start += stride
            while input[start - 1] != delimiter and start > start_tmp:
                start -= 1
            if start <= start_tmp:
                start = start_tmp + stride
    
    if start + window_size > len(input):
        result.append(input[start:len(input)])
    return result


def format_json_answer(s: str) -> str:
    """
    If a string contains "```json" and "```", remove them and return the stuff at the middle fo them, 
    else return the string itself

    Parameters:
    a (str): the input string containing a JSON of interest

    Returns:
    str: JSON as a string
    """
    start_tag = "```json"
    end_tag = "```"

    if start_tag in s and end_tag in s:
        s = s[s.index(start_tag) + len(start_tag): s.rindex(end_tag)]

    # remove all the line breaks
    s = s.replace("\n", "")
    
    new_s = ""
    for i in range(0, len(s)):
        if s[i] == "\'":
            is_quote_inside_value = True
            j = i + 1
            while j < len(s):
                if s[j] == " ":
                    j += 1
                else:
                    if s[j] == "," or s[j] == ":" or s[j] == "{" or s[j] == "}" or s[j] == "[" or s[j] == "]":
                        is_quote_inside_value = False
                    break
            j = i - 1
            if is_quote_inside_value:
                while j >= 0:

                    if s[j] == " ":
                        j -= 1
                    else:
                        if s[j] == "\\" or s[j] == "," or s[j] == ":" or s[j] == "{" or s[j] == "}" or s[j] == "[" or s[j] == "]":
                            is_quote_inside_value = False
                        break
            if is_quote_inside_value:
                new_s += "\\\'"
            else:
                new_s += "\'"
            
        elif s[i] == "\"":
            is_quote_inside_value = True
            j = i + 1
            while j < len(s):
                if s[j] == " ":
                    j += 1
                else:
                    if s[j] == "," or s[j] == ":" or s[j] == "{" or s[j] == "}" or s[j] == "[" or s[j] == "]":
                        is_quote_inside_value = False
                    break
            j = i - 1
            if is_quote_inside_value:
                while j >= 0:
                    if s[j] == " ":
                        j -= 1
                    else:
                        if s[j] == "\\" or s[j] == "," or s[j] == ":" or s[j] == "{" or s[j] == "}" or s[j] == "[" or s[j] == "]":
                            is_quote_inside_value = False
                        break
            if is_quote_inside_value:
                new_s += "\\\""
            else:
                new_s += "\""
            
        else:
            new_s += s[i]
    return new_s

    
def format_list_answer(s: str) -> str:
    """
    If the answer contains text other than list we want, extract the list from the answer

    Parameters:
    a (str): the input string containing a list of interest

    Returns:
    str: list as a string
    """
    start_index = s.find('[')
    end_index = s.rfind(']')

    if start_index != -1 and end_index != -1 and start_index < end_index:
        return s[start_index: end_index + 1]
    else:
        return None


def entity_disambiguation(text: str) -> str:
    """
    Given a chunk of text, replace all the pronouns, abbreviations, acronyms, and last names with the 
    names to which they refer to in the context of the paragraph.

    Parameters:
    text (str): the input document
    
    Returns:
    str: the result text
    """
    system_prompt : str = 'Given a chunk of text, replace all the pronouns, abbreviations, acronyms, and last names with the names to which they refer to in the context of the paragraph. For example: "Tiger Wang is an excellent student at UIUC, he is also the Co-founder of Geni. At 2019, Wang founded Geni with his friends." Should become "Tiger Wang is an excellent student at UIUC, Tiger Wang is also the Co-founder of Geni. At 2019, Tiger Wang founded Geni with Tiger Wang\'s friends."'
    prompt : str = f"Given Text:\n{text}"

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    response = gpt_chat(messages=messages, model="gpt-3.5-turbo-1106", max_tokens=2048)
    return response


def entity_extract(text: str, entity_question: bool=False) -> List[KGEntity]:
    """
    Extract all the entities in the given text chunk, since the entities that GPT extract are sometimes 
    attributes, and it is difficult to prevent that, I also added a validation process.

    Parameters:
    text (str): the input text chunk
    entity_question (bool): This parameter is only used in the question-answering part, since there is a
    special entity [ENTITY] in the QA part that represent the unknown entity of our interest, and the prompt
    also need to be changed a little bit for GPT to understand what [ENTITY] is

    Returns:
    list[KGEntity]: a list of extracted entities
    """
    system_prompt : str = "You are an expert in linguistics and knowledge graph. Your task is to examine the given text and extract all identifiable entities. An 'entity' in this context refers to any distinct and identifiable concept, person, place, or thing in the text with specific attributes and relationships. For instance, 'Elon Musk' is an entity as it defines a distinct person.\n\nIn the case of compound entities, such as 'Chancellor of UIUC', consider 'UIUC' as the entity and do not consider 'Chancellor of' as a separate entity but as a relationship between entities.\n\nFor each identified entity, provide a short description based on its context in the text and classify them into one or more types such as person, place, organization, event, etc. Please note that an entity like 'BMW' can be categorized as a 'Car Manufacturer', 'Organization', and 'Thing'.\n\nThe returned output should be in a well-formatted JSON structure: {'Entity_Name': {'description': 'brief description', 'types': ['type1', 'type2',...]}}, ensure that the entities are clearly identified.\n\nDo not incorporate information about the entities from outside the given text. Aim for a comprehensive extraction of all possible entities in the text, and remember to treat compound entities and date ranges as multiple separate entities."
    prompt : str = f"\n\nGiven Text:\n{text}\n\n" + ("(treat [Entity] as an actual entity, use \"an unknown entity\" as its description)" if entity_question else "")

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    response = gpt_chat(messages=messages, model="gpt-4-1106-preview", max_tokens=2048)
    
    result = ast.literal_eval(format_json_answer(response))

    print(response)

    for key, value in result.items():
        if key[0] == '\'' and key[-1] == '\'':
            key = f'"key[1:-1]"'
    for res in result:
        if res[0] == '\'' and res[-1] == '\'':
            res = f'"res[1:-1]"'

    final_entities = []

    for entity_name in result:
        # entity validation, split into 2 parts since I found it hard for GPT to output True/False in 1 step
        phrase = phrase_selection(entity_name, text)

        # validation part
        messages = [{"role": "system", "content": ""}, {"role": "user", "content": f"In the following text chunk \"{phrase}\", is \"{entity_name}\" a distinct entity or an attribute of a relation"}]
        response = gpt_chat(messages=messages, model = "gpt-4-1106-preview", max_tokens=1024)
        print("validation: ", response)

        # convert the answer to True/False
        messages = [{"role": "system", "content": ""}, {"role": "user", "content": f"Does the response \"{response}\" mean \"{entity_name}\" is a distinct entity? If so, output True, else False. Only output True/False and nothing else"}]
        response = gpt_chat(messages=messages, model = "gpt-3.5-turbo-1106", max_tokens=1024)
        print(response)
        is_distinct_entity = ast.literal_eval(response)

        if is_distinct_entity:
            # This entity is valid, continue extract data property
            next_system_prompt: str = "You are an expert in linguistics and knowledge graph. You are given an entity and a text chunk. You will help extract the attributes of this entity. Do not add any external information outside of the given text chunk. Your output should be a well-formatted JSON that has all property names and their respective values in this format: {'property_name': 'value'}\n\nFor example given:\nEntity: Christine\nText: Christine, an 89 year old woman, has breast tumor.\n\nYou should output:\n{'age': '89', 'breast_tumor': 'true'}\n\nFor another example:\nEntity: Barbara Wilson\nText: Barbara Wilson is the Chancellor of UIUC from 2015-2016\n\nYou should output:\n{} Since everything is about the \"Chancellor of\" relation."
            
            next_prompt: str = f"Entity: {result[entity_name]}\nText: {phrase}"
            next_messages = [{"role": "system", "content": next_system_prompt}, {"role": "user", "content": next_prompt}]
            
            next_response = gpt_chat(messages=next_messages, model = "gpt-4-1106-preview", max_tokens=1024)
            print("Attributes: ", next_response)
            try:
                attributes = ast.literal_eval(format_json_answer(next_response))
            except:
                attributes = {}

            entity_to_add = KGEntity(name=entity_name, data_properties=attributes ,description=result[entity_name]["description"], types=result[entity_name]["types"], relations=[])

            final_entities.append(entity_to_add)

    print("Entities: ")
    for entity in final_entities:
        print(entity)
    return final_entities


def phrase_selection(entity: str, text_chunk: str) -> str:
    """
    Extracting all the sentences that contains the given entity.

    Parameters:
    entity (str): the target entity
    text_chunk (str): the given text chunk

    Returns:
    str: concatenation of all the selected sentences
    """
    sentences = re.split('[.?!;]', text_chunk)
    sentences = [sentence.strip() for sentence in sentences]
    result = ""
    for sentence in sentences:
        if entity in sentence:
            result += sentence + '. '
    return result.strip()


def mention_recognition(entity_list: List[str], text_chunk: str) -> List[str]:
    """
    Given a list of entities and a text chunk, extract all the entities occurred in the text chunk

    Parameters:
    entity_list (list[str]): the list of possible entities
    text chunk (str): the input text chunk

    Returns:
    list[str]: the list of occurred entities
    """
    other_entities = []
    for entity in entity_list:
        if entity in text_chunk:
            other_entities.append(entity)
    return other_entities


def predicate_extract(text: str, entities: List[KGEntity], entity_question: bool=False) -> List[KGRelation]:
    """
    Extract all the triples of [head_entity, relation, tail_entity] in the given text chunk, as well as 
    data properties corresponding to that relation, and put them together into KGRelation object.

    Parameters:
    text (str): the input chunk of text
    entities (list[KGEntity]): all the entities occurred in the text chunk
    entity_question (bool): This parameter is only used in the question-answering part, since there is a
    special entity [ENTITY] in the QA part that represent the unknown entity of our interest, and the prompt
    also need to be changed a little bit for GPT to understand what [ENTITY] is

    Returns:
    list[KGRelation]: the result list of relations
    """
    entity_name_list = [entity.name for entity in entities]
    text_chunks = split_text(text, 3000, 750, '.')
    
    def relation_extraction(entity: str, entity_list: List[str], text_chunk: str) -> List[KGRelation]:
        """
        Helper function, used to extract relation with given entity as the head entity

        Parameters:
        entity (str): the head entity
        entity_list (list[str]): the list of all the entities that may relate to the head entity
        text_chunk (str): the input chunk of text
       
        Returns:
        list[KGRelation]: the result list of relations
        """
        result_relations: list[KGRelation] = []
        
        # extract all the triplets in the text chunk
        system_prompt: str = "You are an expert in linguistics and knowledge graph. You are given a target entity, a list of entities, and a text chunk. You help extract relations between the target entity and each of the entities in the list. Do not add any external information outside of the text to the relations. Your output should be a list of triplets in this 2d-list format: [['Head_entity', 'relation', 'Tail_entity'], ...]"
        prompt: str = f"Target Entity: {entity}\nEntities: {entity_list}\nText: {text_chunk}\n\n" + ("(treat [Entity] as an actual entity)" if entity_question else "")

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        response = gpt_chat(messages=messages, model = "gpt-4-1106-preview", max_tokens=2048)
        print("Relations", response)
        
        triplets: List = ast.literal_eval(format_list_answer(response))

        # filter the result so that now triplets only contains valid triplets where the first value in the triplet is the input head entity and the last one is in the input entity list
        triplets_tmp = []
        for triplet in triplets:
            if triplet[0] == entity and triplet[2] in entity_list and triplet[2] in text_chunk:
                triplets_tmp.append(triplet)
        triplets = triplets_tmp
       
       # extract data properties of all the triplets
        for triplet in triplets:
            head = ''
            relation = ''
            tail = ''
            if len(triplet) != 3:
                # handles the case where output of GPT doesn't follow the format
                head, relation = triplet
                tail = ''
            elif len(triplet) == 3: head, relation, tail = triplet

            # possible_sentences will contains only sentences that has both head and tail entities
            sentences = re.split('[.?!;]', text_chunk)
            sentences = [sentence.strip() for sentence in sentences]
            possible_sentences = ""
            for sentence in sentences:
                if head in sentence and tail in sentence:
                    possible_sentences += sentence + '. '
            
            this_system_prompt: str = "You are an expert in linguistics and knowledge graph. You are given a head entity, relation, tail entity, and a text chunk. You help extract the only relevant text that may describe that relation between head entity and tail entity. Do not add any external information outside of the given text chunk. Your output should has a brief description of the relation and the excerpt about the relation in this format: {'description': 'brief description', 'source': 'excerpt abut the relation'}"

            this_prompt: str = f"Head Entity: {head}\nRelation: {relation}\nTail Entity: {tail}\nText: {possible_sentences}\n\n" + ("(treat [Entity] as an actual entity)" if entity_question else "")

            messages = [{"role": "system", "content": this_system_prompt}, {"role": "user", "content": this_prompt}]
            response = gpt_chat(messages=messages, model = "gpt-3.5-turbo-1106", max_tokens=1024)
            print("Relation description:", response)
            this_result = ast.literal_eval(format_json_answer(response))

            relation_text_chunk = this_result["source"]
            
            # extract data properties of relation
            next_system_prompt: str = "You are an expert in linguistics and knowledge graph. You are given a relation, and a text chunk. You will help extract the attributes of the relation. Do not add any external information outside of the given text chunk. Your output should be a well-formatted JSON that has all property names and their respective values in this format: {'property_name': 'value'}\n\nFor example given:\nHead Entity of Relation: Barbara Wilson\nRelation: Chancellor_of_Relation\nTail Entity of Relation: UIUC\nText: Barbara Wilson is the Chancellor of UIUC from 2015 to 2016\n\nThis 'Chancellor_of_Relation' should have attributes for example 'start_time' and 'end_time'; so, you should output:\n{'start_time': '2015', 'end_time': '2016'}\n\nNote: Only include attribute of the relation, do not include attribute of the entities.\n\nFor example, for the sentence \"Philip is 25 years old, and he is the teacher of Isaac\", 25 years old is the attribute of the entity \"Philip\", not attribute of the relation \"teacher_of\"."
            
            next_prompt: str = f"Head Entity: {head}\nRelation: {relation}\nTail Entity: {tail}\nText: {relation_text_chunk}\n\n" + ("(treat [Entity] as an actual entity)" if entity_question else "")
            next_messages = [{"role": "system", "content": next_system_prompt}, {"role": "user", "content": next_prompt}]
            
            next_response = gpt_chat(messages=next_messages, model = "gpt-4-1106-preview", max_tokens=1024)
            print("Relation attributes: ", next_response)
            try:
                attributes = ast.literal_eval(format_json_answer(next_response))
            except:
                attributes = {}

            triplet[1] = f'{triplet[1].replace(" ", "_")}_Relation'
            
            relation_to_add = KGRelation(name=triplet[1], head_entity=triplet[0], tail_entity=triplet[2], data_properties=attributes, description=this_result["description"], source=this_result["source"])
            print("Relation to add: ", relation_to_add)
            result_relations.append(relation_to_add)

        return result_relations

    result = []
    for entity in entity_name_list:
        for text_chunk in text_chunks:
            other_entities = entity_name_list.copy()
            other_entities.remove(entity)
            phrases : str = phrase_selection(entity, text_chunk)
            mentioned_list : List[str] = mention_recognition(other_entities, phrases)
            result += relation_extraction(entity, mentioned_list, text_chunk)
    return result
