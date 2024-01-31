import random
import traceback

import sqlglot
from sqlglot.expressions import *
from sqlglot.tokens import Tokenizer as Tokens
from sqlglot.optimizer.scope import *
from sqlglot.optimizer.qualify import qualify

MAX_TABLES = 12
EXTRA_KEYWORDS = ['IF', 'SUM', 'AVG', 'COUNT', 'MIN', 'MAX', 'NULLIF',
                  'COALESCE', 'CONCAT', 'ORDER', 'GROUP', 'BY', 'ONLY']


def is_valid_identifier(s):
    if not s:
        return False

    if not (s[0].isascii() and (s[0].isalpha() or s[0] == '_')):
        return False

    if not all(c.isascii() and (c.isalnum() or c == '_') for c in s):
        return False

    if str.upper(s) in Tokens.KEYWORDS.keys() or str.upper(s) in EXTRA_KEYWORDS:
        return False

    return True


def generate_random_string(length):
    return ''.join(random.sample(TOKENS, length))


def gen_valid_random_identifier(exclude_items, max_len=12):
    while True:
        identifier = ''.join(random.sample(TOKENS, random.randint(1, 3)))[:max_len].lower()
        if is_valid_identifier(identifier) and identifier not in exclude_items:
            break
    return identifier


def remove_table(expression):
    remove_table_flag = False
    for scope in traverse_scope(expression):
        select = scope.expression
        tables = scope.tables
        sources = scope.sources

        if len(tables) < 3:
            continue
        remove_table_cnt = random.randint(1, max(1, int(len(tables) / 2)))

        removed_joins = random.sample(select.args['joins'], remove_table_cnt)
        for join in removed_joins:
            if isinstance(sources[select.args['from'].this.alias_or_name], Scope):
                continue

            remove_table_flag = True
            select.args['joins'].remove(join)
            table_source = sources[join.this.alias_or_name]
            if isinstance(table_source, Scope):
                if table_source.is_cte:
                    cte_node = table_source.expression.parent
                    with_node = cte_node.parent
                    cte_node.pop()
                    if with_node is not None and len(with_node.expressions) <= 0:
                        with_node.pop()

            for column in scope.columns:
                if column.table == join.this.alias_or_name:
                    column.args['table'] = to_identifier(select.args['from'].this.alias_or_name)

    if not remove_table_flag:
        return None

    return expression


def add_table(expression):
    MAX_ADD_TABLES = 4
    add_table_cnt = 0
    original_names = {_.name for _ in expression.find_all(exp.Identifier)}
    add_table_probability = 0.4

    for scope in traverse_scope(expression):
        select = scope.expression
        tables = scope.tables
        columns = scope.columns

        if len(tables) == 0 \
                or len(columns) == 0 \
                or len(tables) >= MAX_TABLES \
                or random.random() > add_table_probability:
            continue

        new_table_name = gen_valid_random_identifier(original_names)
        new_alias = gen_valid_random_identifier(original_names)
        original_names.add(new_table_name)
        original_names.add(new_alias)
        new_table = table_(table=new_table_name, alias=new_alias)
        range_operators = ['=', '!=', '>=', '<=', '>', '<']
        condition_parts = [f"{new_alias}.id",
                           f"{random.choice(range_operators)}",
                           f"{random.choice(columns)}"]
        select.join(new_table, on=' '.join(condition_parts), copy=False)
        add_table_cnt += 1

        if add_table_cnt >= MAX_ADD_TABLES:
            break

    if add_table_cnt == 0:
        return None

    return expression


def modify_table(expression):
    replace_probability = 1.0
    original_names = {_.name for _ in expression.find_all(exp.Identifier)}
    updated_table_cnt = 0
    for scope in traverse_scope(expression):
        tables = scope.tables
        sources = scope.sources
        columns = scope.columns

        if len(tables) == 0:
            continue

        for table in tables:
            if random.random() < replace_probability and isinstance(sources[table.alias_or_name], exp.Table):
                new_table_name = gen_valid_random_identifier(original_names)
                original_names.add(new_table_name)
                new_alias = gen_valid_random_identifier(original_names)
                original_names.add(new_alias)
                original_alias = table.alias
                updated_table_cnt += 1
                table = table.replace(table_(table=new_table_name, alias=new_alias))
                for col in columns:
                    if col.table == original_alias:
                        col.args['table'] = to_identifier(new_alias)

    if updated_table_cnt == 0:
        return None

    for scope in traverse_scope(expression):
        sources = scope.sources
        columns = scope.columns
        if len(scope.tables) == 0:
            continue
        for col in columns:
            if random.random() < replace_probability and col.table != '' and isinstance(sources[col.table], exp.Table):
                new_col_name = gen_valid_random_identifier(original_names)
                original_names.add(new_col_name)
                col = col.replace(column(col=new_col_name, table=col.table))

    return expression


def gen_positive(expression):
    modify_literal_probability = 0.7
    modify_alias_probability = 0.7
    replace_probability = 0.4
    success_flag = False

    if random.random() < modify_literal_probability:
        literals = expression.find_all(exp.Literal)
        for literal in literals:
            if random.random() < replace_probability:
                success_flag = True
                if literal.parent is not None and isinstance(literal.parent, Cast) and literal.parent.to in ['DATE', 'TIMESTAMP']:
                    random_date = f"'{random.randint(1900, 2100)}-{random.randint(1, 12)}-{random.randint(1, 28)}'"
                    literal = literal.replace(Literal.string(random_date))
                elif literal.is_string:
                    random_string = generate_random_string(random.randint(1, 5))
                    literal = literal.replace(Literal.string(random_string))
                else:
                    literal = literal.replace(Literal.number(random.randint(1, 1000)))

    if random.random() < modify_alias_probability:
        original_names = {_.name for _ in expression.find_all(exp.Identifier)}

        for scope in traverse_scope(expression):
            tables = scope.tables
            columns = scope.columns
            sources = scope.sources

            for table in tables:
                if random.random() < replace_probability:
                    original_table_name = table.alias_or_name
                    table_source = sources[original_table_name]
                    new_alias = gen_valid_random_identifier(original_names)
                    original_names.add(new_alias)
                    if isinstance(table_source, Scope):
                        if table_source.is_cte:
                            cte_node = table_source.expression.parent
                            if original_table_name != cte_node.alias:
                                continue
                            else:
                                cte_node.args['alias'] = to_identifier(new_alias)
                                table = table.replace(table_(table=new_alias, alias=table.alias))
                                if table.alias != '':
                                    original_table_name = None
                        elif table_source.is_subquery:
                            table = table.replace(subquery(expression=table_source.expression, alias=new_alias))
                    else:
                        table = table.replace(table_(table=table.name, alias=new_alias))
                    for col in columns:
                        if col.table == original_table_name:
                            col.args['table'] = to_identifier(new_alias)

                    success_flag = True

            for table in tables:
                original_table_name = table.alias_or_name
                table_source = sources[original_table_name]
                if isinstance(table_source, Scope) and table_source.is_cte:
                    cte_node = table_source.expression.parent
                    if original_table_name != cte_node.alias:
                        table = table.replace(table_(table=cte_node.alias, alias=table.alias))

    if not success_flag:
        return None

    return expression


def augment_data(sql, num_neg=6):
    parsed = sqlglot.parse_one(sql, read='spark')

    while True:
        try:
            positive = gen_positive(parsed.copy())
            if positive is not None:
                qualify(positive.copy())
                positive = positive.sql(dialect='spark')
                break
        except Exception as exc:
            print(f"error in positive, source = {sql}, negative = {positive.sql() if positive is not None else ''}")
            traceback.print_exception(type(exc), exc, exc.__traceback__)

    negatives = []
    negative_probs = [0.0, 0.0, 1.0]
    while True:
        rand = random.random()
        try:
            if rand < negative_probs[0]:
                print("To remove table...")
                negative = remove_table(parsed.copy())
            elif rand < sum(negative_probs[0: 2]):
                print("To add table...")
                negative = add_table(parsed.copy())
            else:
                negative = modify_table(parsed.copy())
            if negative is not None:
                qualify(negative.copy())
                negatives.append(negative.sql(dialect='spark'))
        except Exception as exc:
            print(f"error in negative, source = {sql}, negative = {negative.sql() if negative is not None else ''}")
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        if len(negatives) >= num_neg:
            break

    return positive, negatives


def init_tokens(tokens):
    global TOKENS
    tokens.remove("\\")
    TOKENS = tokens

