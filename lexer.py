import ply.lex as lex


ESCAPE_SEQUENCES = {
	r'\a': 7,  # bell
	r'\b': 8,  # backspace

	r'\t': 9,  # h tab
	r'\n': 10, # new line
	r'\v': 11, # v tab
	r'\f': 12, # new page
	r'\r': 13, # carriage return

	r'\"': 34, # "
	r'\'': 39, # '
	r'\\': 92, # \
}


class Lexer:
	keywords = {
		'if'     :'IF',
		'elif'   :'ELIF',
		'else'   :'ELSE',
		'while'  :'WHILE',
		'do'     :'DO',
		'end'    :'END',
		'proc'   :'PROC',
		'func'   :'FUNC',
		'return' :'RETURN',
		'not'    :'NOT',
		'and'    :'AND',
		'or'     :'OR',
		'var'    :'VAR',
		'int'    :'INT',
		'float'  :'FLOAT',
		'char'   :'CHAR'
	}

	tokens = [
		'DOT','COLON','COMMA','SEMI',
		'LPAREN','RPAREN','LBRACE','RBRACE','LBRACK','RBRACK',
		'NE','LE','GE','EQ','LT','GT',
		'ADD','SUB','MUL','DIV','MOD',
		'INT_LIT','FLOAT_LIT','CHAR_LIT','STR_LIT','ID'
	]

	tokens += list(keywords.values())

	t_ignore = ' \t'
	t_ignore_COMMENT = r'[#].*'

	t_DOT   = r'\.'
	t_COLON = r':'
	t_COMMA = r','
	t_SEMI  = r';'

	t_LPAREN = r'\('
	t_RPAREN = r'\)'

	t_LBRACE = r'\{'
	t_RBRACE = r'\}'

	t_LBRACK = r'\['
	t_RBRACK = r'\]'

	t_NE = r'<>'
	t_LE = r'<='
	t_GE = r'>='

	t_EQ = r'='
	t_LT = r'<'
	t_GT = r'>'

	t_ADD = r'\+'
	t_SUB = r'-'

	t_MUL = r'\*'
	t_DIV = r'/'
	t_MOD = r'%'

	def t_CHAR_LIT(self, t):
		r'\'([^\\\n]|(\\[abtnvfre\\\'\"])|(\\[0-3][0-7][0-7]))\''
		value = t.value[1:-1]
		t.value = value
		return t

	def t_invalid_char_lit(self, t):
		r'\'([^\\\n]|(\\.))*\''
		column = t.lexer.lexpos - t.lexer.linestart - len(t.value)
		print("Invalid char literal", t.value, "at line", t.lineno, "column", column)

	def t_unclosed_char_lit(self, t):
		r'\'([^\\\n]|(\\.))*'
		column = t.lexer.lexpos - t.lexer.linestart - len(t.value)
		print("Unclosed char literal", t.value, "at line", t.lineno, "column", column)

	def t_STR_LIT(self, t):
		r'\"([^\\\n]|(\\[abtnvfre\\\'\"])|(\\[0-3][0-7][0-7]))*\"'
		value = t.value[1:-1]
		t.value = value
		return t

	def t_invalid_str_lit(self, t):
		r'\"([^\\\n]|(\\.))*\"'
		column = t.lexer.lexpos - t.lexer.linestart - len(t.value)
		print("Invalid string literal", t.value, "at line", t.lineno, "column", column)

	def t_unclosed_str_lit(self, t):
		r'\"([^\\\n]|(\\.))*'
		column = t.lexer.lexpos - t.lexer.linestart - len(t.value)
		print("Unclosed string literal", t.value, "at line", t.lineno, "column", column)

	def t_FLOAT_LIT(self, t):
		r'\d+\.\d+'
		value = float(t.value)
		t.value = value
		return t

	def t_INT_LIT(self, t):
		r'\d+'
		value = int(t.value)
		t.value = value
		return t

	def t_ID(self, t):
		r'[a-zA-Z_][a-zA-Z_0-9]*'
		t.type = Lexer.keywords.get(t.value, 'ID')
		return t

	def t_newline(self, t):
		r'\n'
		t.lexer.lineno += 1
		t.lexer.linestart = t.lexer.lexpos

	def t_error(self, t):
		err_val  = t.value[0]
		err_line = t.lexer.lineno
		err_col  = t.lexer.lexpos - t.lexer.linestart - len(err_val)
		print("Illegal character", err_val, "at line", err_line, "column", err_col)
		t.lexer.skip(1)

	def build(self, **kwargs):
		self.lexer = lex.lex(module=self, **kwargs)
		self.lexer.linestart = self.lexer.lexpos

	def lex(self, input):
		self.lexer.input(input)
		for token in self.lexer:
			yield token

	def __init__(self):
		self.build()

if __name__ == '__main__':
	lexer = Lexer()

	f1 = "examples\\incorrect\\literals.txt"
	with open(f1, 'r', encoding='utf-8') as f:
		print(f1)
		code = f.read()
		for t in lexer.lex(code):
			print("Token", t.type, "value =", t.value, "at line", t.lineno)

	print()

	f2 = "examples\\correct\\1.txt"
	with open(f2, 'r', encoding='utf-8') as f:
		print(f2)
		code = f.read()
		for t in lexer.lex(code):
			print("Token", t.type, "value =", t.value, "at line", t.lineno)
