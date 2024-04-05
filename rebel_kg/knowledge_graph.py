import math
import torch
import copy

class KG():
    def __init__(self):
        self.entities = set()
        self.relations = []
        
    def are_relations_equal(self, r1, r2):
        return all(r1[attr] == r2[attr] for attr in ["head", "type", "tail"])

    def exists_relation(self, r1):
        return any(self.are_relations_equal(r1, r2) for r2 in self.relations)
    
    def merge_with_kb(self, kb2):
        for r in kb2.relations:
            self.add_relation(r)
        
    def add_entity(self, e):
        self.entities.add(e)
    
    def merge_relations(self, r2):
        r1 = [r for r in self.relations if self.are_relations_equal(r2, r)][0]
        existing_srcs = r1["source"]
        
        all_new_sources = r2["source"]
        for article_id in all_new_sources:
            article_sentences = all_new_sources[article_id]
            if article_id not in existing_srcs:
                existing_srcs[article_id] = article_sentences
            else:
                existing_srcs[article_id].extend(article_sentences)

    def add_relation(self, r):
        # manage new entities
        entities = [r["head"], r["tail"]]
        for e in entities:
            self.add_entity(e)

        # manage new relation
        if not self.exists_relation(r):
            self.relations.append(r)
        else:
            self.merge_relations(r)
    
    def get_json_representation(self):
        result = copy.deepcopy(self.relations)
        for idx in range(len(result)):
            curr_relationship = result[idx]
            curr_source = curr_relationship.pop("source")
            reformatted_sources = []
            for article_id in curr_source:
                reformatted_sources.append({
                    "article_id" : article_id,
                    "txt_used" : curr_source[article_id]
                })
            
            curr_relationship["sources"] = reformatted_sources

        return result

# extract relations for each span and put them together in a knowledge base
relation_name_mapper = {
    "att_lithology" : ("has lithology of", "lithology", "lith attribute lithology"),
    "att_sed_structure" : ("has sedimentary structure", "lithology", "lith attribute sedimentary structure"),
    "strat_name_to_lith" : ("strat has lithology", "strat_name", "lithology"),
    "lith_to_lith_group" : ("lithology is part of group", "lithology", "lithology group"),
    "lith_to_lith_type" : ("lithology has type of", "lithology", "lithology type"),
    "att_grains" : ("has grains of", "lithology", "lith attribute grains"),
    "att_color" : ("has color of", "lithology", "lith attribute color"),
    "att_bedform" : ("has bedform of", "lithology", "lith attribute bedform"),
    "att_structure" : ("has structure of", "lithology", "lith attribute structure"),
}
def get_kg_for_line(model, line, article_id, span_length=128):
    kg = KG()
    all_relations = model.get_relations_in_line(line)

    for relation in all_relations:
        # Add in metadata if it is a relationship we care about
        relationship_type = relation["type"]
        if relationship_type not in relation_name_mapper:
            continue

        human_name, src_type, dst_type = relation_name_mapper[relationship_type]
        relation["human_type"] = human_name
        relation["src_type"] = src_type
        relation["dst_type"] = dst_type
        relation["source"] = {
            article_id: [line],
        }
        kg.add_relation(relation)        

    return kg