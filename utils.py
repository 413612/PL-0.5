def op_priority(t):
    if t in ['MUL', 'DIV']:
        return 1
    elif t in ['ADD', 'SUB']:
        return 2
    elif t in ['GT', 'GE', 'LT', 'LE']:
        return 3
    elif t in ['NE', 'EQ']:
        return 4
    elif t in ['AND', 'OR', 'NOT']:
        return 5


def is_number(t):
    return t in ['INT_LIT', 'FLOAT_LIT']


def is_type(t):
    return t in ['INT', 'FLOAT', 'CHAR']


def is_operator(t):
    return t in ['MUL', 'DIV', 'ADD', 'SUB', 'GT', 'GE', 'LT', 'LE', 'NE', 'EQ', 'AND', 'OR', 'NOT']


def is_arithmetic_operator(t):
    return t in ['MUL', 'DIV', 'ADD', 'SUB']


def is_logic_operator(t):
    return t in ['GT', 'GE', 'LT', 'LE', 'NE', 'EQ', 'AND', 'OR', 'NOT']
