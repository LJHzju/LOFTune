import sqlglot
from sqlglot.optimizer.optimizer import optimize
from sqlglot.optimizer.pushdown_predicates import pushdown_predicates
from sqlglot.optimizer.pushdown_projections import pushdown_projections
from sqlglot.optimizer.normalize import normalize
from sqlglot.optimizer.optimize_joins import optimize_joins
from sqlglot.optimizer.eliminate_subqueries import eliminate_subqueries
from sqlglot.optimizer.merge_subqueries import merge_subqueries
from sqlglot.optimizer.eliminate_joins import eliminate_joins
from sqlglot.optimizer.eliminate_ctes import eliminate_ctes
from sqlglot.optimizer.canonicalize import canonicalize
from sqlglot.optimizer.simplify import simplify
from sql_encoder.optimizer.my_qualify import qualify
from sql_encoder.optimizer.my_simplify import further_simplify
from sql_encoder.optimizer.my_unnest import unnest_subqueries

rules = (
    qualify,
    pushdown_projections,
    normalize,
    unnest_subqueries,
    pushdown_predicates,
    optimize_joins,
    eliminate_subqueries,
    lambda x: merge_subqueries(x, leave_tables_isolated=True),
    eliminate_joins,
    eliminate_ctes,
    canonicalize,
    simplify,
    further_simplify
)

simplified_rules = (
    qualify,
    pushdown_projections,
    normalize,
    unnest_subqueries,
    pushdown_predicates,
    canonicalize,
    simplify,
    further_simplify
)


def rewrite_query(sql, schema):
    try:
        optimized = optimize(sqlglot.parse_one(sql, read='spark'),
                             rules=rules,
                             dialect='spark',
                             schema=schema).sql(pretty=True, dialect='spark')
    except Exception as e:
        optimized = optimize(sqlglot.parse_one(sql, read='spark'),
                             rules=simplified_rules,
                             dialect='spark',
                             schema=schema).sql(pretty=True, dialect='spark')
    return optimized

