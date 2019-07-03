import yaml

loader = yaml.FullLoader

class Monster:

    def __init__(self, name, hp, ac, attacks):
        self.name = name
        self.hp = hp
        self.ac = ac
        self.attacks = attacks
    def __repr__(self):
        return "%s(name=%r, hp=%r, ac=%r, attacks=%r)" % (
            self.__class__.__name__, self.name, self.hp, self.ac, self.attacks
        )

def monster_constructor(loader, node):
    mapping = loader.construct_mapping(node)
    return Monster(**mapping)

yaml.add_constructor(u'!Monster', monster_constructor, Loader=yaml.SafeLoader)

x = yaml.safe_load(stream="""
--- !Monster
name: Cave spider
hp: [2,6]    # 2d6
ac: 16
attacks: [BITE, HURT]
""")

# yaml.add_path_resolver('!ColumnDefn', ['ColumnDefn'], dict)

col_defn = ColumnDefn('my col', 'my_col', 'int64')
defs = [col_defn, col_defn]
x = yaml.dump_all(defs)
dfn = list(yaml.load_all(stream=x, Loader=yaml.Loader))
pass