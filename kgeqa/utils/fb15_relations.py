"""Filtering the relations of FB15K dataset"""
import pandas as pd

# -- filtered domain relations
film = [
    # FB15 total relations: 95
    # KEYWORD: /film/
    '/film/film/starring./film/performance/actor',  # 9466
    '/film/actor/film./film/performance/film',  # 9494
    '/film/film/starring./film/performance/character',  # 64
    '/film/film/language',  # 2570
    '/film/film/country',  # 2407
    '/film/film/genre',  # 7268
    '/film/production_company/films',  # 1537
    '/film/writer/film',  # 807
    '/film/film/produced_by',  # 1285
    '/film/film/directed_by',  # 850
    '/film/director/film',  # 859
    '/film/film/written_by',  # 787
]
location = [
    # FB15 total relations: 63
    # KEYWORD: /location/
    '/location/location/containedby',  # 5186
    '/location/location/contains',  # 5204
    '/location/country/capital',  # 142
    '/location/location/time_zones',  # 1151
    '/location/country/languages_spoken',  # 334
    '/location/country/official_language',  # 225
    '/location/country/form_of_government',  # 298
    '/location/country/currency_used',  # 48
]
computer = [
    # FB15 total relations: 24
    # KEYWORD: /computer/
    '/computer/operating_system/parent_os',  # 10
    '/computer/programming_language/influenced',  # 26
    '/computer/programming_language/influenced_by',  # 25
    '/computer/software/license',  # 8
]
people = [
    # FB15 total relations: 47
    # KEYWORD: /people/
    '/people/person/ethnicity',  # 2030         person -> ethnicity
    '/people/ethnicity/people',  # 2073         ethnicity -> person
    '/people/person/profession',  # 11636
    '/people/person/place_of_birth',  # 2468
    '/people/deceased_person/place_of_death',  # 697
    '/people/person/education./education/education/institution',  # 2591
    '/people/person/nationality',  # 4198
    '/location/location/people_born_here',  # 2485
    '/people/person/languages',  # 783
    '/people/person/religion',  # 1086
]

# -- how they were filtered
SOURCE_FB15K = '/Users/Aziz/Downloads/rel-embedding/RelatedWork/fastText/scripts/kbcompletion/data/FB15k/freebase_mtr100_mte100-train.txt'
df = pd.read_csv(SOURCE_FB15K, sep='\t', names=['h', 'r', 't'])


# -- filtering FB15K on a given relationship keyword
def relations_contains(s='/location/', df=df):
    return df['r'][df['r'].str.contains(s)].unique()


# -- choosing the candidate relations for a given DOMAIN
def candidate_relations(filter_on='/people/'):
    candidates = relations_contains(filter_on)
    print(len(candidates))
    for r_ in candidates:
        total = df[df['r'] == r_].shape[0]
        print(total, '\t', f"'{r_}', #", f"{total}")


if __name__ == '__name__':
    # -- how to use
    print(candidate_relations('/computer/'))
