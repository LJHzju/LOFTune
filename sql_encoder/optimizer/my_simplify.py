import sqlglot
from sqlglot import exp
from sqlglot.expressions import *
from dateutil.relativedelta import relativedelta
from datetime import datetime


def further_simplify(expression):
    _simplify_in(expression)
    _simplify_interval(expression)
    _replace_try_cast(expression)

    return expression


def _simplify_in(expression):
    ins = expression.find_all(exp.In)
    for in_exp in ins:
        if 'expressions' in in_exp.args:
            items = in_exp.expressions
            if len(set([_.is_string for _ in items])) == 2:
                items = [_ for _ in items if _.is_string]
            unnest_items = [eval(_.this) if not _.is_string else _.this for _ in items if not isinstance(_, Null)]
            sorted_items = sorted(list(set(unnest_items)))
            sorted_items = [Literal.string(_) if isinstance(_, str) else Literal.number(_) for _ in sorted_items]
            in_exp.args['expressions'] = sorted_items


def _simplify_interval(expression):
    intervals = expression.find_all(exp.Interval)
    for interval in intervals:
        ancestor = interval.find_ancestor(exp.Binary)
        base_time = ancestor.this.find(exp.Literal)
        if base_time is None:
            continue
        base_time = base_time.this

        target_type = 'date'
        if ' ' in base_time:
            # If there's a space, assume it includes time
            base_time = datetime.strptime(base_time, "%Y-%m-%d %H:%M:%S")
            target_type = 'time'
            output_format = "%Y-%m-%d %H:%M:%S"
        else:
            # If there's no space, assume it's just a date
            base_time = datetime.strptime(base_time, "%Y-%m-%d")
            target_type = 'date'
            output_format = "%Y-%m-%d"

        unit = interval.unit.this
        num = int(interval.this.this)
        if isinstance(ancestor, exp.Sub):
            num = -1 * num
        if unit.find("YEAR") != -1:
            result_date = base_time + relativedelta(years=num)
        elif unit.find("MONTH") != -1:
            result_date = base_time + relativedelta(months=num)
        elif unit.find("DAY") != -1:
            result_date = base_time + relativedelta(days=num)
        elif unit.find("HOUR") != -1:
            result_date = base_time + relativedelta(hours=num)
        elif unit.find("MINUTE") != -1:
            result_date = base_time + relativedelta(minutes=num)
        elif unit.find("SECOND") != -1:
            result_date = base_time + relativedelta(seconds=num)

        result_date = result_date.strftime(output_format)
        ancestor = ancestor.replace(exp.cast(expression=Literal.string(result_date), to=target_type))


def _replace_try_cast(expression):
    for trycast in expression.find_all(exp.TryCast):
        trycast = trycast.replace(exp.cast(expression=trycast.this, to=trycast.to))
