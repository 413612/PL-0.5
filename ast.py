import utils


class BaseAST:
    def __init__(self, parent=None):
        self.parent = parent
        self.children = {}

    def add_child(self, name, obj):
        self.children[name] = obj

    def get_children(self, name):
        return self.children.get(name) or self.parent and self.parent.get_children(name)

    def set_parent(self, value):
        self.parent = value

    def get_parent(self):
        return self.parent

    def code_gen(self, module):
        pass


class VarDecAST(BaseAST):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.dim = 0
        self.name = ""
        self.type = None
        self.value = None

    def set_dim(self, value):
        self.dim = value
    
    def get_dim(self):
        return self.dim

    def set_name(self, value):
        self.name = value
    
    def get_name(self):
        return self.name

    def set_type(self, value):
        self.type = value

    def set_value(self, value):
        self.value = value

    def get_type(self):
        return self.type


class VarDefAST(BaseAST):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.var_dec = None
        self.value = None

    def set_declaration(self, obj):
        self.var_dec = obj

    def set_value(self, value):
        self.value = value

    def get_type(self):
        return self.var_dec.type


class IntLiteralAST(BaseAST):
    def __init__(self, value: int, parent=None):
        super().__init__(parent=parent)
        self.value = value

    def get_type(self):
        return 'INT'


class FloatLiteralAST(BaseAST):
    def __init__(self, value: float, parent=None):
        super().__init__(parent=parent)
        self.value = value

    def get_type(self):
        return 'FLOAT'


class CompoundExpression(BaseAST):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.order_operations = []

    def set_child(self, obj):
        obj.set_parent(self)
        self.order_operations.append(obj)

    def get_var_def(self, name):
        ops = self.order_operations.copy()
        ops.reverse()
        for o in ops:
            if isinstance(o, BinaryAST) and o.operator == 'EQ' and isinstance(o.lhs, VarDefAST) and o.lhs.var_dec.name == name:
                return o.lhs
        return None


class ProcedureDefAST(CompoundExpression):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.name = ""
        self.args = []
        self.body = None
        self.proc = None

    def set_name(self, value):
        self.name = value

    def set_body(self, obj):
        self.body = obj

    def add_arg(self, arg):
        self.args.append(arg)
        self.add_child(arg.name, arg)


class FunctionDefAST(CompoundExpression):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.name = ""
        self.args = []
        self.return_value = None
        self.type = None
        self.body = None
        self.func = None

    def set_name(self, value):
        self.name = value

    def set_body(self, obj):
        self.body = obj

    def set_type(self, t):
        self.type = t

    def add_arg(self, arg):
        self.args.append(arg)
        self.add_child(arg.name, arg)

    def set_return_value(self, obj):
        self.return_value = obj


class ReturnAst(BaseAST):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.value = None

    def set_value(self, value):
        self.value = value


class ProcedureCallAST(BaseAST):
    def __init__(self, proc, args, parent=None):
        super().__init__(parent=parent)
        self.proc_callee = proc
        self.args = args

    def set_parent(self, value):
        self.parent = value


class FunctionCallAST(BaseAST):
    def __init__(self, func, args, parent=None):
        super().__init__(parent=parent)
        self.func_callee = func
        self.args = args
        self.ret = None

    def set_parent(self, value):
        self.parent = value

    def set_ret_name(self, name):
        self.ret = name

    def get_type(self):
        t = self.func_callee.type
        if t in ['int', 'INT']:
            return 'INT'
        elif t in ['float', 'FLOAT']:
            return 'FLOAT'


class AssignmentAST(BaseAST):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.lval = None
        self.rval = None

    def set_lval(self, value):
        self.lval = value

    def set_rval(self, value):
        self.rval = value

    def is_valid(self):
        if self.lval.get_type() == self.rval.get_type():
            return True
        else:
            return False


class BinaryAST(BaseAST):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.operator = None
        self.lhs = None
        self.rhs = None

    def set_lhs(self, value):
        self.lhs = value

    def set_rhs(self, value):
        self.rhs = value

    def set_op(self, value):
        self.operator = value

    def is_valid(self):
        if utils.is_operator(self.operator) and (self.lhs is not None) and (self.rhs is not None):
            return True
        else:
            return False

    def get_type(self):
        t1 = self.lhs.get_type()
        t2 = self.rhs.get_type()
        if t1 == t2 and utils.is_arithmetic_operator(self.operator):
            return t1
        else:
            return None


class ExprIfAST(CompoundExpression):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.expression = None
        self.then_body = None
        self.else_body = None

    def set_expression(self, expr):
        self.expression = expr

    def set_then(self, obj):
        self.then_body = obj

    def set_else(self, obj):
        self.else_body = obj


class ExprWhileAST(CompoundExpression):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.expression = None
        self.body = None

    def set_expression(self, expr):
        self.expression = expr

    def set_body(self, obj):
        self.body = obj


class ExprDoWhileAST(CompoundExpression):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.expression = None
        self.body = None

    def set_expression(self, expr):
        self.expression = expr

    def set_body(self, obj):
        self.body = obj
